from typing import TypedDict, Optional, Any
import pandas as pd
from sqlalchemy.engine import Engine
import logging
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# ---------------- STATE ----------------

class AgentState(TypedDict):
    user_query: str
    cleaned_query: Optional[str]
    generated_sql: Optional[str]
    validated_sql: Optional[str]
    query_result: Optional[Any]
    final_response: Optional[str]
    error: Optional[str]


# ---------------- INPUT HANDLER ----------------

def input_guardrail_node(state: AgentState):
    try:
        query = state.get("user_query", "")

        if not query or len(query.strip()) < 3:
            return {"error": "Query is too short or empty."}

        query = query.strip()

        blocked_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "EXEC"]
        if any(keyword in query.upper() for keyword in blocked_keywords):
            return {"error": "Query contains potentially dangerous operations."}

        # FIX #2: Expanded to include natural language words like "details", "show", "list" etc.
        inventory_keywords = [
            "stock", "inventory", "material", "plant",
            "shelf", "demand", "sop", "product", "value",
            "wip", "git", "family", "group", "many", "count",
            "total", "sum", "average", "highest", "lowest",
            "doh", "days", "safety", "abc", "category",
            "application", "mrp", "purchasing", "uom", "type",
            "detail", "details", "show", "list", "find", "get",
            "which", "what", "how", "where", "display", "give",
            "semifinished", "finished", "raw", "trading", "packaging",
            "sensor", "fiber", "cable", "wire",
        ]

        if not any(word in query.lower() for word in inventory_keywords):
            return {"error": "Query seems unrelated to inventory data."}

        cleaned_query = " ".join(query.split())
        return {"cleaned_query": cleaned_query}

    except Exception as e:
        return {"error": str(e)}


# ---------------- SQL GENERATOR ----------------

def create_sql_generator(llm: ChatOpenAI):

    SYSTEM_PROMPT = """
You are a strict SQLite SQL generator for an inventory database.
Your ONLY job is to convert user questions into CORRECT SQL queries.

════════════════════════════════════════════════════════════════
RULE 0 — TABLE NAME
════════════════════════════════════════════════════════════════
- The ONLY table is: inventory
- ALWAYS use FROM inventory
- NEVER use a column name as a table name (e.g. FROM "Material Name" is WRONG)

════════════════════════════════════════════════════════════════
RULE 1 — COLUMN QUOTING
════════════════════════════════════════════════════════════════
- ALL column names MUST be wrapped in double quotes: "Column Name"
- Table name inventory does NOT need quotes
- "Product Family " has a TRAILING SPACE — use exactly: "Product Family "

════════════════════════════════════════════════════════════════
RULE 2 — WIP COLUMNS: quantity vs value (FIX)
════════════════════════════════════════════════════════════════
There are TWO separate WIP columns — never confuse them:

  "WIP"     → WIP QUANTITY (number of units)
  "WIP($)"  → WIP VALUE in dollars

Mapping:
- "WIP value", "WIP dollar", "WIP cost", "WIP amount", "WIP ($)"
  → Use "WIP($)" with ROUND("WIP($)", 2)

- "WIP quantity", "WIP units", "WIP count", "WIP stock", plain "WIP"
  → Use "WIP" (no ROUND — whole number)

- Ambiguous "WIP" → show BOTH:
  SELECT "WIP" AS wip_quantity, ROUND("WIP($)", 2) AS wip_value

Same rule for GIT:
  "GIT"     → quantity  |  "GIT ($)" → dollar value

Same rule for Shelf Stock:
  "Shelf Stock"     → quantity  |  "Shelf Stock ($)" → dollar value

════════════════════════════════════════════════════════════════
RULE 3 — MATERIAL TYPE EXACT VALUES (FIX: no SAP codes!)
════════════════════════════════════════════════════════════════
CRITICAL: "Material Type" stores FULL TEXT. NEVER use SAP codes.

  User says                        → SQL value
  ─────────────────────────────────────────────────────
  finished, finished goods, FG     → 'Finished products'
  semifinished, semi-finished      → 'Semifinished products'
  raw, raw material, RM            → 'Raw materials'
  trading, trading goods           → 'Trading goods'
  packaging, packing               → 'Packaging'
  tools, production resources      → 'Prod. resources/tools'
  operating supplies               → 'Operating supplies-NON VA'
  spare parts                      → 'Spare parts'
  services                         → 'Services'

  ❌ 'FERT'  → ✅ 'Finished products'
  ❌ 'HALB'  → ✅ 'Semifinished products'
  ❌ 'SEMIF' → ✅ 'Semifinished products'
  ❌ 'ROH'   → ✅ 'Raw materials'
  ❌ 'HAWA'  → ✅ 'Trading goods'
  ❌ 'VERP'  → ✅ 'Packaging'

════════════════════════════════════════════════════════════════
RULE 4 — DOH QUERIES (FIX: exact equality and all comparisons)
════════════════════════════════════════════════════════════════
DOH is a numeric float column. Map user language EXACTLY:

  "DOH = 0" / "DOH is 0" / "DOH equals 0" / "zero DOH"
    → WHERE "DOH" = 0

  "DOH = 30" / "DOH is 30" / "DOH equal to 30"
    → WHERE "DOH" = 30

  "DOH under 30" / "DOH less than 30" / "DOH below 30" / "DOH < 30"
    → WHERE "DOH" < 30

  "DOH above 30" / "DOH greater than 30" / "DOH more than 30"
    → WHERE "DOH" > 30

  "DOH between 30 and 60"
    → WHERE "DOH" BETWEEN 30 AND 60

  "DOH at least 30" / "DOH >= 30"
    → WHERE "DOH" >= 30

  "DOH at most 30" / "DOH <= 30"
    → WHERE "DOH" <= 30

STRICT RULES FOR DOH:
  ❌ NEVER convert equality to range (DOH=0 must stay DOH=0, not BETWEEN 0 AND 1)
  ❌ NEVER stack two DOH conditions: WHERE "DOH" < 30 AND "DOH" = 0 is ALWAYS WRONG
  ❌ NEVER use AVG when listing/counting materials — only for "average DOH"
  ❌ NEVER use SUM("DOH") anywhere — use AVG("DOH") only
  ✅ Use exactly ONE DOH condition per query matching what the user asked
  ✅ Use ROUND("DOH", 1) in SELECT output only, not in WHERE

════════════════════════════════════════════════════════════════
RULE 5 — "DETAILS" AND LISTING SYNONYMS (FIX)
════════════════════════════════════════════════════════════════
These words all mean "give me individual rows" — generate a row-level SELECT:
  "details", "detail", "show", "list", "display",
  "give me", "find", "get", "fetch", "what are", "which are"

Example:
  "Show details of materials in plant 2001"
  → SELECT "Material", "Material Name", "Material Type",
           "Shelf Stock", ROUND("Shelf Stock ($)", 2) AS "Shelf Stock ($)",
           ROUND("DOH", 1) AS "DOH", "ABC"
     FROM inventory
     WHERE "Plant" = '2001' AND "Material Name" IS NOT NULL
     ORDER BY "Shelf Stock ($)" DESC
     LIMIT 50;

════════════════════════════════════════════════════════════════
RULE 6 — OTHER COLUMN MAPPINGS
════════════════════════════════════════════════════════════════
- demand → "Demand"
- zero demand / demand = 0 → WHERE "Demand" = 0  (exact 0, NOT NULL)
- safety stock / buffer stock → "Safety Stock" (never SUM — use AVG or individual)
- ABC class → "ABC"  values: 'A', 'B', 'C' (uppercase only)
- product family (no SOP) → "Product Family "  ← trailing space!
- SOP family → "SOP Family"
- product group → "Product Group"
- material group → "Material Group"
- category → "Product Category"
- material application → "Material Application"
- sub application → "Sub Application"
- MRP / planner → "MRP Controller Text"
- buyer / purchasing → "Purchasing Group Text"
- UOM / unit → "UOM"
- WIP / wip / Work-in-progress -> "WIP"

════════════════════════════════════════════════════════════════
RULE 7 — AGGREGATION
════════════════════════════════════════════════════════════════
- COUNT rows → COUNT(*)
- COUNT materials → COUNT(DISTINCT "Material")
- SUM money → ROUND(SUM("column ($)"), 2)
- AVG DOH → ROUND(AVG("DOH"), 1)
- NEVER SUM("DOH")
- NEVER SUM quantities across different materials

════════════════════════════════════════════════════════════════
RULE 8 — NULL FILTERS
════════════════════════════════════════════════════════════════
- Always add "Material Name" IS NOT NULL when listing/counting materials
- Always add IS NOT NULL on GROUP BY columns
- SOP Family queries: also add "SOP Family" IS NOT NULL

════════════════════════════════════════════════════════════════
RULE 9 — SORTING & LIMITS
════════════════════════════════════════════════════════════════
- "top N" → ORDER BY value DESC LIMIT N
- "bottom N" / "lowest" → ORDER BY value ASC LIMIT N
- "highest" → ORDER BY value DESC LIMIT 1
- "lowest" → ORDER BY value ASC LIMIT 1

════════════════════════════════════════════════════════════════
AVAILABLE COLUMNS
════════════════════════════════════════════════════════════════
"Plant", "Material", "Material Name", "Material Type", "UOM",
"Shelf Stock", "Shelf Stock ($)",
"GIT", "GIT ($)",
"WIP", "WIP($)",
"DOH", "Safety Stock", "Demand",
"Product Family ", "SOP Family", "Product Group", "Material Group",
"Product Category", "Material Application", "Sub Application",
"ABC", "MRP Controller Text", "Purchasing Group Text"

════════════════════════════════════════════════════════════════
QUERY EXAMPLES
════════════════════════════════════════════════════════════════

── DETAILS / LISTING ──
"Show details of semifinished products in plant 2001"
→ SELECT "Material", "Material Name", "Material Type",
         "Shelf Stock", ROUND("Shelf Stock ($)", 2) AS "Shelf Stock ($)",
         ROUND("DOH", 1) AS "DOH", "ABC"
   FROM inventory
   WHERE "Material Type" = 'Semifinished products'
     AND "Plant" = '2001' AND "Material Name" IS NOT NULL
   ORDER BY "Shelf Stock ($)" DESC LIMIT 50;

── WIP QUANTITY vs VALUE ──
"Show WIP quantity by plant"
→ SELECT "Plant", SUM("WIP") AS total_wip_quantity
   FROM inventory WHERE "Material Name" IS NOT NULL
   GROUP BY "Plant" ORDER BY total_wip_quantity DESC;

"Show WIP value by plant"
→ SELECT "Plant", ROUND(SUM("WIP($)"), 2) AS total_wip_value
   FROM inventory WHERE "Material Name" IS NOT NULL
   GROUP BY "Plant" ORDER BY total_wip_value DESC;

"Show WIP for plant 2001"
→ SELECT "Material", "Material Name",
         "WIP" AS wip_quantity, ROUND("WIP($)", 2) AS wip_value
   FROM inventory
   WHERE "Plant" = '2001' AND "Material Name" IS NOT NULL
   ORDER BY "WIP($)" DESC LIMIT 50;

── MATERIAL TYPE ──
"How many semifinished products are there?"
→ SELECT COUNT(DISTINCT "Material") AS material_count
   FROM inventory
   WHERE "Material Type" = 'Semifinished products'
     AND "Material Name" IS NOT NULL;

"Semifinished products in plant 2001 with DOH under 30"
→ SELECT "Material", "Material Name",
         ROUND("DOH", 1) AS "DOH",
         ROUND("Shelf Stock ($)", 2) AS "Shelf Stock ($)"
   FROM inventory
   WHERE "Material Type" = 'Semifinished products'
     AND "Plant" = '2001' AND "DOH" < 30 AND "Material Name" IS NOT NULL
   ORDER BY "DOH" ASC;

── DOH EXACT EQUALITY ──
"Which materials have DOH = 0?"
→ SELECT "Material", "Material Name", "Plant",
         ROUND("Shelf Stock ($)", 2) AS "Shelf Stock ($)"
   FROM inventory
   WHERE "DOH" = 0 AND "Material Name" IS NOT NULL
   ORDER BY "Shelf Stock ($)" DESC;

"How many materials have DOH equal to 0?"
→ SELECT COUNT(*) AS material_count
   FROM inventory
   WHERE "DOH" = 0 AND "Material Name" IS NOT NULL;

── DOH COMPARISONS ──
"Materials with DOH below 30"
→ SELECT "Material", "Material Name",
         ROUND("DOH", 1) AS "DOH",
         ROUND("Shelf Stock ($)", 2) AS "Shelf Stock ($)"
   FROM inventory
   WHERE "DOH" < 30 AND "Material Name" IS NOT NULL
   ORDER BY "DOH" ASC;

"Materials with DOH above 90"
→ SELECT "Material", "Material Name",
         ROUND("DOH", 1) AS "DOH",
         ROUND("Shelf Stock ($)", 2) AS "Shelf Stock ($)"
   FROM inventory
   WHERE "DOH" > 90 AND "Material Name" IS NOT NULL
   ORDER BY "DOH" DESC;

── DEMAND ──
"How many materials have zero demand?"
→ SELECT COUNT(*) AS material_count
   FROM inventory
   WHERE "Demand" = 0 AND "Material Name" IS NOT NULL;

── SOP FAMILY ──
"Shelf stock for SENSORS SOP family by plant"
→ SELECT "Plant",
         ROUND(SUM("Shelf Stock"), 2) AS total_shelf_stock_qty,
         ROUND(SUM("Shelf Stock ($)"), 2) AS total_shelf_stock_value
   FROM inventory
   WHERE "SOP Family" LIKE '%SENSOR%' AND "Material Name" IS NOT NULL
   GROUP BY "Plant" ORDER BY total_shelf_stock_value DESC;

── ABC ──
"Average DOH for ABC class A"
→ SELECT ROUND(AVG("DOH"), 1) AS avg_doh
   FROM inventory
   WHERE "ABC" = 'A' AND "Material Name" IS NOT NULL;

── TOP N ──
"Show top 10 materials by shelf stock value"
→ SELECT "Material", "Material Name",
         ROUND(SUM("Shelf Stock ($)"), 2) AS total_value
   FROM inventory WHERE "Material Name" IS NOT NULL
   GROUP BY "Material", "Material Name"
   ORDER BY total_value DESC LIMIT 10;

════════════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════════════
Return ONLY the SQL query.
No explanation. No backticks. No markdown. No preamble. No comments.
"""

    def node(state: AgentState):
        try:
            res = llm.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state["cleaned_query"]}
            ])
            sql = res.content.strip()
            if sql.startswith("```"):
                sql = sql.replace("```sql", "").replace("```", "").strip()
            logger.info(f"Generated SQL: {sql}")
            return {"generated_sql": sql}
        except Exception as e:
            logger.error(f"SQL generation error: {str(e)}")
            return {"error": f"Failed to generate SQL: {str(e)}"}

    return node


# ---------------- SQL VALIDATOR ----------------

def create_sql_validator(llm: ChatOpenAI):

    VALIDATION_PROMPT = """
You are a strict SQL validator for an inventory database. Table: inventory

Check and fix ALL errors below. Return ONLY the corrected SQL.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] TABLE NAME
  ❌ FROM "Material Name" / FROM "Material" / FROM any column name
  ✅ FROM inventory  ← only valid table name

[2] WIP COLUMNS
  "WIP"    = quantity (no ROUND)
  "WIP($)" = dollar value (ROUND to 2)
  Fix if wrong column used for context.

[3] MATERIAL TYPE — replace all SAP codes with full text:
  'FERT'  → 'Finished products'
  'HALB'  → 'Semifinished products'
  'SEMIF' → 'Semifinished products'
  'ROH'   → 'Raw materials'
  'HAWA'  → 'Trading goods'
  'VERP'  → 'Packaging'

[4] DOH — CRITICAL RULES
  ❌ WHERE "DOH" BETWEEN 0 AND 1  (when user meant DOH = 0) → ✅ WHERE "DOH" = 0
  ❌ SUM("DOH") → ✅ AVG("DOH")
  ❌ AVG("DOH") in WHERE clause

  MOST IMPORTANT: DO NOT add any extra "DOH" condition if one already exists.
  ❌ WHERE "DOH" < 30 AND "DOH" = 0   ← WRONG: contradictory, returns only DOH=0
  ✅ WHERE "DOH" < 30                  ← CORRECT: keep only the original condition

  If the SQL already has a DOH filter (e.g. "DOH" < 30, "DOH" > 90, "DOH" = 0,
  "DOH" BETWEEN x AND y), do NOT add another DOH condition.
  One DOH condition per query — never stack them.

[5] COLUMN QUOTES
  All column names must have double quotes.
  "Product Family " must include trailing space.

[6] NULL FILTERS
  Add "Material Name" IS NOT NULL when listing/counting materials.
  Add IS NOT NULL on GROUP BY columns.

[7] COUNT
  ❌ COUNT() → ✅ COUNT(*)  or  COUNT(DISTINCT "column")

[8] ROUNDING
  Monetary → ROUND(value, 2)
  DOH output → ROUND("DOH", 1)

[9] ABC
  Values must be uppercase: 'A', 'B', 'C'

[10] DEMAND ZERO
  ❌ WHERE "Demand" IS NULL
  ✅ WHERE "Demand" = 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If VALID → return EXACT SAME SQL
If INVALID → fix and return ONLY corrected SQL
No explanation. No comments. No markdown. No backticks.
"""

    def node(state: AgentState):
        try:
            res = llm.invoke([
                {"role": "system", "content": VALIDATION_PROMPT},
                {"role": "user", "content": state["generated_sql"]}
            ])
            sql = res.content.strip()
            if sql.startswith("```"):
                sql = sql.replace("```sql", "").replace("```", "").strip()
            logger.info(f"Validated SQL: {sql}")
            return {"validated_sql": sql}
        except Exception as e:
            logger.error(f"SQL validation error: {str(e)}")
            return {"error": f"Failed to validate SQL: {str(e)}"}

    return node


# ---------------- SQL EXECUTOR ----------------

def create_sql_executor(engine: Engine):

    def node(state: AgentState):
        try:
            logger.info(f"Executing SQL: {state['validated_sql']}")
            df = pd.read_sql(state["validated_sql"], engine)
            logger.info(f"Query returned {len(df)} rows")
            return {"query_result": df}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"SQL execution error: {error_msg}")
            if "no such table" in error_msg.lower():
                return {"error": f"Table not found. Details: {error_msg}"}
            elif "no such column" in error_msg.lower():
                return {"error": f"Column not found — check for typos or trailing spaces. Details: {error_msg}"}
            else:
                return {"error": f"Query execution failed: {error_msg}"}

    return node


# ---------------- RESPONSE FORMATTER ----------------

def create_response_formatter(llm: ChatOpenAI):

    FORMAT_PROMPT = """
You are a precise inventory data analyst.

Format the response using this EXACT structure:

1. **Direct Answer** (one clear line)
   - State the TOTAL count/result from the full dataset — NOT the number of displayed rows
   - If the result has 986 rows but only 20 are shown, say "986 materials" NOT "20 materials"
   - The display limit is just for formatting — always report the REAL total in the Direct Answer

2. **SQL Query** (in code block)
   ```sql
   [The SQL query used]
   ```

3. **Data Summary** (clean table or key metrics)
   - Single number: show it clearly
   - Table: format cleanly (max 20 rows), use | for table formatting
   - If total rows > 20, clearly state: "Showing top 20 of X total results"

4. **Key Insights** (1-2 bullet points)
   - Business interpretation
   - DOH = 0: flag as no inventory coverage
   - DOH < 30: flag as potential stockout risk
   - Safety Stock: flag materials below safety stock
   - ABC A: highlight value concentration

Keep it concise, professional, and actionable.
"""

    def node(state: AgentState):
        try:
            result_df = state["query_result"]

            total_rows = len(result_df)
            if total_rows == 0:
                result_str = "No results found."
            elif total_rows == 1 and len(result_df.columns) == 1:
                result_str = f"Result: {result_df.iloc[0, 0]}"
            else:
                result_str = f"TOTAL ROWS IN FULL RESULT: {total_rows}\n\n"
                result_str += result_df.head(20).to_string(index=False)
                if total_rows > 20:
                    result_str += f"\n\n(Showing top 20 of {total_rows} total rows — use TOTAL ROWS count in Direct Answer)"

            res = llm.invoke([
                {"role": "system", "content": FORMAT_PROMPT},
                {"role": "user", "content": f"""
User Question: {state['cleaned_query']}
SQL Query: {state['validated_sql']}
Query Result: {result_str}
"""}
            ])

            logger.info("Response formatted successfully")
            return {"final_response": res.content}

        except Exception as e:
            logger.error(f"Response formatting error: {str(e)}")
            return {"error": f"Failed to format response: {str(e)}"}

    return node


# ---------------- ERROR HANDLER ----------------

def error_handler(state: AgentState):
    error_msg = state.get('error', 'Unknown error occurred')
    logger.error(f"Error in workflow: {error_msg}")
    return {"final_response": f"❌ **Error**: {error_msg}\n\nPlease try rephrasing your question."}


# ---------------- ROUTER ----------------

def route(state: AgentState):
    if state.get("error"):
        return "error"
    return "ok"


# ---------------- GRAPH BUILDER ----------------

def build_graph(engine: Engine, api_key: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    builder = StateGraph(AgentState)

    builder.add_node("input_handler", input_guardrail_node)
    builder.add_node("sql_generator", create_sql_generator(llm))
    builder.add_node("sql_validator", create_sql_validator(llm))
    builder.add_node("sql_executor", create_sql_executor(engine))
    builder.add_node("formatter", create_response_formatter(llm))
    builder.add_node("error", error_handler)

    builder.set_entry_point("input_handler")

    builder.add_conditional_edges("input_handler", route, {"ok": "sql_generator", "error": "error"})
    builder.add_conditional_edges("sql_generator", route, {"ok": "sql_validator", "error": "error"})
    builder.add_conditional_edges("sql_validator", route, {"ok": "sql_executor", "error": "error"})
    builder.add_conditional_edges("sql_executor", route, {"ok": "formatter", "error": "error"})
    builder.add_conditional_edges("formatter", route, {"ok": END, "error": "error"})
    builder.add_edge("error", END)

    return builder.compile()


def run_query(graph, user_query: str) -> str:
    result = graph.invoke({"user_query": user_query})
    return result.get("final_response", "No response generated")


