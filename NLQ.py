import os
import logging
import sys
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from graph import build_graph

# ---------------- CONFIG ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "inventory.db")
EXCEL_PATH = os.path.join(BASE_DIR, "Current_Inventory.xlsx")

st.set_page_config(
    page_title="Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    /* ── Global font & background ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
    }

    .main-header p {
        color: #94a3b8;
        margin: 0;
        font-size: 0.95rem;
    }

    /* ── Stat cards ── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }

    .stat-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, border-color 0.2s;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99,102,241,0.5);
    }

    .stat-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 8px;
    }

    .stat-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-card .icon {
        font-size: 1.5rem;
        margin-bottom: 8px;
    }

    /* ── Chat area ── */
    .chat-container {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }

    /* ── Messages ── */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        padding: 12px 16px !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: #e2e8f0 !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInputTextArea"] {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stChatInputTextArea"]::placeholder {
        color: #64748b !important;
    }

    [data-testid="stChatInputContainer"] {
        background: rgba(15,12,41,0.8) !important;
        border-top: 1px solid rgba(255,255,255,0.08) !important;
        padding: 12px !important;
        backdrop-filter: blur(10px) !important;
    }

    /* ── Quick question buttons ── */
    .stButton > button {
        background: rgba(99,102,241,0.12) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        color: #a78bfa !important;
        border-radius: 10px !important;
        font-size: 0.8rem !important;
        padding: 8px 12px !important;
        text-align: left !important;
        transition: all 0.2s !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }

    .stButton > button:hover {
        background: rgba(99,102,241,0.25) !important;
        border-color: rgba(99,102,241,0.6) !important;
        color: #c4b5fd !important;
        transform: translateX(4px) !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }

    [data-testid="metric-container"] label {
        color: #64748b !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-weight: 700 !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] > div {
        border-top-color: #a78bfa !important;
    }

    /* ── Alerts ── */
    .stAlert {
        border-radius: 10px !important;
    }

    /* ── Sidebar section labels ── */
    .sidebar-section {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #475569;
        margin: 20px 0 8px 0;
        font-weight: 600;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(52,211,153,0.12);
        border: 1px solid rgba(52,211,153,0.3);
        color: #34d399;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #34d399;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── Hide Streamlit branding ── */
    /* ── Hide Streamlit branding but keep sidebar toggle ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Make sidebar toggle arrow white */
button[kind="header"] svg {
    fill: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = None
if "db_stats" not in st.session_state:
    st.session_state.db_stats = None


# ---------------- API KEY (Streamlit Secrets) ----------------

def get_api_key() -> str:
    """
    Reads the OpenAI API key from Streamlit secrets.
    Add this to your .streamlit/secrets.toml:

        OPENAI_API_KEY = "sk-..."

    Or set it in the Streamlit Cloud dashboard under App Settings → Secrets.
    """
    try:
        return st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error(
            "⚠️ **OpenAI API key not found.**\n\n"
            "Please add your key to Streamlit secrets:\n\n"
            "1. Locally: create `.streamlit/secrets.toml` and add:\n"
            "   `OPENAI_API_KEY = \"sk-...\"`\n\n"
            "2. On Streamlit Cloud: go to **App Settings → Secrets** and paste the key."
        )
        st.stop()


# ---------------- LOAD DATA ----------------

def load_excel_to_sqlite():
    """Load Excel into SQLite with proper data cleaning."""
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
        logger.info("SQLite DB already exists, skipping load.")
        return

    if not os.path.exists(EXCEL_PATH):
        st.error(f"❌ Excel file not found at: `{EXCEL_PATH}`")
        st.stop()

    with st.spinner("⚙️ Loading inventory data…"):
        logger.info(f"Loading Excel from {EXCEL_PATH}")
        df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
        df.columns = [col.strip() for col in df.columns]

        critical_cols = [
            "Material Name", "SOP Family", "Product Family",
            "Material Type", "Product Group", "Material Application",
            "Sub Application"
        ]
        for col in critical_cols:
            if col in df.columns:
                df[col] = df[col].replace('', None).replace(' ', None)

        numeric_cols = [
            "Shelf Stock", "Shelf Stock ($)", "GIT", "GIT ($)",
            "WIP", "WIP($)", "DOH", "Safety Stock", "Demand"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        before_count = len(df)
        df = df[df["Material Name"].notna()]
        logger.info(f"Removed {before_count - len(df)} rows with NULL Material Name")

        engine = create_engine(f"sqlite:///{DB_PATH}")
        df.to_sql("inventory", engine, if_exists="replace", index=False)

        with engine.connect() as conn:
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_material_name ON inventory("Material Name")'))
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_sop_family ON inventory("SOP Family")'))
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_plant ON inventory("Plant")'))
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_shelf_stock ON inventory("Shelf Stock ($)")'))
            conn.commit()

        engine.dispose()
        logger.info(f"Data written to {DB_PATH}")


# ---------------- GRAPH INITIALIZATION ----------------

@st.cache_resource
def initialize_graph():
    load_excel_to_sqlite()
    engine = create_engine(f"sqlite:///{DB_PATH}")
    api_key = get_api_key()
    graph = build_graph(engine, api_key)
    logger.info("LangGraph pipeline initialized.")
    return graph, engine


# ---------------- DB STATS ----------------

def load_db_stats(engine):
    if st.session_state.db_stats is not None:
        return st.session_state.db_stats
    try:
        stats = pd.read_sql("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(DISTINCT "Material Name") as unique_materials,
                ROUND(SUM("Shelf Stock ($)"), 2) as total_value
            FROM inventory
        """, engine)
        st.session_state.db_stats = {
            "rows": stats["total_rows"][0],
            "materials": stats["unique_materials"][0],
            "value": stats["total_value"][0],
        }
    except Exception:
        st.session_state.db_stats = {"rows": "—", "materials": "—", "value": "—"}
    return st.session_state.db_stats


# ---------------- LOGGING ----------------

def validate_and_log_query(user_query: str, result: dict):
    logger.info("=" * 80)
    logger.info(f"USER QUERY: {user_query}")
    logger.info(f"GENERATED SQL:\n{result.get('generated_sql')}")
    logger.info(f"VALIDATED SQL:\n{result.get('validated_sql')}")
    logger.info(f"ERROR: {result.get('error')}")
    logger.info("=" * 80)


# ---------------- SIDEBAR ----------------

def render_sidebar(engine):
    with st.sidebar:
        # Branding
        st.markdown("""
        <div style="padding: 16px 0 8px 0;">
            <div style="font-size:1.3rem; font-weight:700; background:linear-gradient(135deg,#a78bfa,#60a5fa); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                📦 InventoryIQ
            </div>
            <div style="margin-top:6px;">
                <span class="status-badge">
                    <span class="status-dot"></span> System Online
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # DB Stats in sidebar
        stats = load_db_stats(engine)
        st.markdown('<div class="sidebar-section">Database Overview</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Records", f"{stats['rows']:,}" if isinstance(stats['rows'], int) else stats['rows'])
        with col2:
            st.metric("Materials", f"{stats['materials']:,}" if isinstance(stats['materials'], int) else stats['materials'])

        val = stats['value']
        if isinstance(val, (int, float)):
            st.metric("Total Value", f"${val:,.0f}")
        else:
            st.metric("Total Value", val)

        st.divider()

        # Quick questions
        st.markdown('<div class="sidebar-section">Quick Questions</div>', unsafe_allow_html=True)

        example_questions = [
            "🔝  Top 10 materials by shelf stock value",
            "💰  Materials with highest shelf stock value?",
            "📡  What is the DOH for material 924689-000",
            "🏭  What’s in transit for 10XL2-ZH",
            "⚠️  Any wip for 908689-000",
            "📊  Total inventory value by plant",
        ]

        for q in example_questions:
            clean_q = q[3:].strip()
            if st.button(q, key=q, use_container_width=True):
                st.session_state.prefill_query = clean_q
                st.rerun()

        st.divider()

        # Clear chat
        if st.button("🗑️  Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("""
        <div style="margin-top:auto; padding-top:24px; font-size:0.72rem; color:#334155; text-align:center;">
            Powered by LangGraph + OpenAI<br/>
            <span style="color:#1e293b;">v2.0 • Production</span>
        </div>
        """, unsafe_allow_html=True)


# ---------------- SCHEMA EXPANDER ----------------

def show_schema_expander(engine):
    with st.expander("🔍 Explore Database Schema & Sample Records", expanded=False):
        try:
            df_preview = pd.read_sql("SELECT * FROM inventory LIMIT 10", engine)
            st.dataframe(df_preview, use_container_width=True)
        except Exception as e:
            st.warning(f"Preview unavailable: {e}")


# ---------------- MAIN ----------------

def main():
    graph, engine = initialize_graph()

    render_sidebar(engine)

    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>📦 Inventory Intelligence</h1>
        <p>Ask questions about your inventory in plain English — powered by AI-driven natural language querying.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Schema expander ──
    show_schema_expander(engine)

    # ── Chat history ──
    for msg in st.session_state.chat_history:
        avatar = "🧑‍💼" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Empty state ──
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding:40px 20px; color:#475569;">
            <div style="font-size:3rem; margin-bottom:12px;">💬</div>
            <div style="font-size:1rem; font-weight:500; color:#64748b; margin-bottom:6px;">
                No conversation yet
            </div>
            <div style="font-size:0.85rem; color:#334155;">
                Ask a question below or pick one from the sidebar →
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input ──
    user_input = st.chat_input("Ask anything about your inventory… e.g. 'Which plants have excess WIP?'")

    if st.session_state.prefill_query:
        user_input = st.session_state.prefill_query
        st.session_state.prefill_query = None

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Querying your inventory…"):
                try:
                    result = graph.invoke({"user_query": user_input})
                    validate_and_log_query(user_input, result)
                    response = result.get("final_response", "No response generated.")
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                except Exception as e:
                    logger.error(str(e), exc_info=True)
                    st.error(f"**Error:** {str(e)}")


if __name__ == "__main__":
    main()
