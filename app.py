import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AirSense AI | Environmental Intelligence & NO₂ Calibration",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# 2. LUXURY ENVIRONMENTAL INTELLIGENCE DESIGN SYSTEM
#    Palette: Midnight Forest, Botanical Emerald & Champagne Gold
#    Theme-Aware: Adapts seamlessly to Streamlit Dark & Light Modes
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&family=Space+Grotesk:wght@600;700&display=swap');

    /* Default Tokens: Midnight Forest (Dark Mode) */
    :root {
        --lux-bg: #101A1B;
        --lux-sidebar: #142322;
        --lux-card: #1B2A28;
        --lux-card-elevated: #213431;
        --lux-ink: #EDF3EF;
        --lux-muted: #A8B7AE;
        --lux-emerald: #49C6A5;
        --lux-teal: #73BCA8;
        --lux-gold: #D5B477;
        --lux-border: #30423C;
        --lux-shadow: 0 10px 28px rgba(4, 10, 10, 0.32);
        --lux-nav-active-bg: linear-gradient(90deg, rgba(73, 198, 165, 0.18) 0%, rgba(213, 180, 119, 0.06) 100%);
        --lux-nav-hover-bg: rgba(73, 198, 165, 0.08);
    }

    /* Automatic Adaptation when Streamlit Native Light Theme is Selected */
    @container style(--background-color: #ffffff) {
        :root {
            --lux-bg: #F5F4EF;
            --lux-sidebar: #EAECE6;
            --lux-card: #FFFFFF;
            --lux-card-elevated: #FAF9F5;
            --lux-ink: #1B2926;
            --lux-muted: #64736D;
            --lux-emerald: #176B57;
            --lux-teal: #267F70;
            --lux-gold: #B99557;
            --lux-border: #E1E5DC;
            --lux-shadow: 0 8px 24px rgba(27, 41, 38, 0.06);
            --lux-nav-active-bg: linear-gradient(90deg, rgba(23, 107, 87, 0.14) 0%, rgba(185, 149, 87, 0.08) 100%);
            --lux-nav-hover-bg: rgba(23, 107, 87, 0.07);
        }
    }
    @container style(--background-color: rgb(255, 255, 255)) {
        :root {
            --lux-bg: #F5F4EF;
            --lux-sidebar: #EAECE6;
            --lux-card: #FFFFFF;
            --lux-card-elevated: #FAF9F5;
            --lux-ink: #1B2926;
            --lux-muted: #64736D;
            --lux-emerald: #176B57;
            --lux-teal: #267F70;
            --lux-gold: #B99557;
            --lux-border: #E1E5DC;
            --lux-shadow: 0 8px 24px rgba(27, 41, 38, 0.06);
            --lux-nav-active-bg: linear-gradient(90deg, rgba(23, 107, 87, 0.14) 0%, rgba(185, 149, 87, 0.08) 100%);
            --lux-nav-hover-bg: rgba(23, 107, 87, 0.07);
        }
    }

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background-color: var(--lux-bg);
        color: var(--lux-ink);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1260px;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', system-ui, sans-serif !important;
        color: var(--lux-ink) !important;
        letter-spacing: -0.02em;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--lux-sidebar);
        border-right: 1px solid var(--lux-border);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.15rem;
    }

    /* Sidebar Brand Identity Block */
    .brand-header {
        background: linear-gradient(145deg, #0E1D1C 0%, #16332E 60%, #1C423B 100%);
        border: 1px solid rgba(213, 180, 119, 0.35);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
    }
    .brand-title-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.55rem;
    }
    .brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.24rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #EDF3EF;
        margin: 0;
    }
    .brand-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        letter-spacing: 0.08em;
        color: #D5B477;
        text-align: center;
        margin: 0.35rem 0 0 0;
        text-transform: uppercase;
    }
    .brand-status {
        display: flex;
        justify-content: center;
        margin-top: 0.65rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        background: rgba(73, 198, 165, 0.14);
        border: 1px solid rgba(73, 198, 165, 0.35);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.63rem;
        font-weight: 700;
        color: #73BCA8;
        letter-spacing: 0.06em;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #49C6A5;
    }

    /* Sidebar Section Labels */
    .nav-group-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.67rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        color: var(--lux-gold);
        margin: 0.95rem 0 0.42rem 0.35rem;
    }

    /* Sidebar Navigation Buttons */
    section[data-testid="stSidebar"] div.stButton {
        margin-bottom: -0.35rem;
    }
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 0.58rem 0.9rem !important;
        border-radius: 10px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background: transparent !important;
        color: var(--lux-ink) !important;
        opacity: 0.86;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
        background: var(--lux-nav-hover-bg) !important;
        border-color: var(--lux-border) !important;
        opacity: 1;
        transform: translateX(3px);
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: var(--lux-nav-active-bg) !important;
        color: var(--lux-ink) !important;
        font-weight: 700 !important;
        border: 1px solid var(--lux-border) !important;
        border-left: 3.5px solid var(--lux-gold) !important;
        opacity: 1;
    }

    /* Sidebar Telemetry Card */
    .sidebar-telemetry {
        background: var(--lux-card);
        border: 1px solid var(--lux-border);
        border-top: 2px solid var(--lux-gold);
        border-radius: 12px;
        padding: 0.85rem 0.95rem;
        margin-top: 0.9rem;
        font-size: 0.79rem;
        color: var(--lux-muted);
    }
    .tel-head {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: var(--lux-gold);
        margin-bottom: 0.4rem;
    }
    .tel-row {
        display: flex;
        justify-content: space-between;
        padding: 0.24rem 0;
        border-bottom: 1px solid var(--lux-border);
    }
    .tel-row:last-child {
        border-bottom: none;
    }
    .tel-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: var(--lux-emerald);
    }

    /* Top Enterprise Metadata Strip */
    .meta-bar {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 0.6rem;
        background: var(--lux-card);
        border: 1px solid var(--lux-border);
        border-radius: 10px;
        padding: 0.48rem 1rem;
        margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.71rem;
        color: var(--lux-muted);
    }
    .meta-bar b {
        color: var(--lux-ink);
    }
    .meta-bar .gold-tag {
        color: var(--lux-gold);
        font-weight: 700;
    }

    /* Luxury Emerald Hero Banner */
    .lux-hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #0D1C1B 0%, #14332D 52%, #1B463E 100%);
        border: 1px solid rgba(213, 180, 119, 0.34);
        border-radius: 16px;
        padding: 2.1rem 2.3rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 14px 34px rgba(6, 14, 13, 0.28);
    }
    .hero-grid {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.5rem;
    }
    .hero-copy {
        max-width: 780px;
        z-index: 1;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #D5B477;
        background: rgba(213, 180, 119, 0.12);
        border: 1px solid rgba(213, 180, 119, 0.35);
        padding: 0.26rem 0.8rem;
        border-radius: 999px;
        margin-bottom: 0.75rem;
    }
    .hero-h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.55rem, 2.8vw, 2.15rem);
        font-weight: 700;
        color: #EDF3EF;
        margin: 0 0 0.5rem 0;
        line-height: 1.18;
        letter-spacing: -0.02em;
    }
    .hero-desc {
        font-size: 0.98rem;
        color: #C4D4CC;
        margin: 0;
        line-height: 1.58;
    }
    .hero-art {
        flex-shrink: 0;
        opacity: 0.92;
    }
    @media (max-width: 820px) {
        .hero-art { display: none; }
    }

    /* Executive KPI Cards */
    .kpi-box {
        background: var(--lux-card);
        border: 1px solid var(--lux-border);
        border-top: 3px solid var(--lux-emerald);
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        box-shadow: var(--lux-shadow);
        transition: transform 0.22s ease, border-color 0.22s ease;
    }
    .kpi-box.gold-accent {
        border-top-color: var(--lux-gold);
    }
    .kpi-box:hover {
        transform: translateY(-3px);
        border-color: var(--lux-gold);
    }
    .kpi-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .kpi-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--lux-muted);
    }
    .kpi-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--lux-ink);
        line-height: 1.15;
        margin: 0;
    }
    .kpi-num span.unit {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--lux-emerald);
    }
    .kpi-sub {
        font-size: 0.79rem;
        color: var(--lux-muted);
        margin-top: 0.32rem;
    }

    /* Luxury Content Cards */
    .content-card {
        background: var(--lux-card);
        border: 1px solid var(--lux-border);
        border-left: 3.5px solid var(--lux-emerald);
        border-radius: 14px;
        padding: 1.3rem 1.45rem;
        box-shadow: var(--lux-shadow);
        margin-bottom: 1rem;
        height: 100%;
    }
    .content-card.gold {
        border-left-color: var(--lux-gold);
    }
    .card-head {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--lux-ink);
        margin: 0 0 0.55rem 0;
    }
    .content-card p, .content-card li {
        color: var(--lux-muted);
        font-size: 0.93rem;
        line-height: 1.6;
    }
    .content-card b {
        color: var(--lux-ink);
    }

    /* 4-Stage Pipeline Strip */
    .arch-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.9rem;
        margin-top: 0.5rem;
    }
    .arch-step {
        background: var(--lux-card);
        border: 1px solid var(--lux-border);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        box-shadow: var(--lux-shadow);
    }
    .arch-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--lux-gold);
        letter-spacing: 0.08em;
    }
    .arch-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--lux-ink);
        margin: 0.22rem 0;
    }
    .arch-desc {
        font-size: 0.81rem;
        color: var(--lux-muted);
        line-height: 1.45;
    }

    /* High-DPI Figure Presentation Frame */
    .figure-frame {
        background: #FFFFFF;
        border: 1px solid var(--lux-border);
        border-radius: 14px;
        padding: 0.85rem;
        box-shadow: var(--lux-shadow);
        margin-top: 0.6rem;
    }

    /* Group Header inside Estimator Form */
    .form-group-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--lux-gold);
        border-bottom: 1px solid var(--lux-border);
        padding-bottom: 0.4rem;
        margin-bottom: 0.75rem;
    }

    /* Estimator Result Card */
    .est-card {
        background: linear-gradient(145deg, #0E1D1C 0%, #16332E 65%, #1D433C 100%);
        border: 1px solid rgba(213, 180, 119, 0.45);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin: 1.1rem 0;
        box-shadow: 0 14px 32px rgba(4, 12, 11, 0.28);
        color: #EDF3EF;
    }
    .est-kicker {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #D5B477;
    }
    .est-reading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.75rem;
        font-weight: 700;
        margin: 0.2rem 0 0.45rem 0;
        line-height: 1.1;
    }
    .est-pill {
        display: inline-block;
        padding: 0.32rem 0.95rem;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .gauge-bar {
        width: 100%;
        height: 10px;
        background: rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 1.2rem;
    }
    .gauge-progress {
        height: 100%;
        border-radius: 999px;
    }
    .gauge-ticks {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.69rem;
        color: #A8B7AE;
        margin-top: 0.38rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------
# 3. INLINE VECTOR SVG GRAPHICS (ZERO EMOJIS)
# ------------------------------------------------------------------
ICONS = {
    "brand": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#49C6A5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L3 7l9 5 9-5-9-5z"/><path d="M3 17l9 5 9-5"/><path d="M3 12l9 5 9-5"/></svg>',
    "hero_viz": '<svg width="210" height="115" viewBox="0 0 210 115" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="1" y="1" width="208" height="113" rx="12" fill="rgba(16,26,27,0.55)" stroke="rgba(213,180,119,0.28)"/><line x1="16" y1="88" x2="194" y2="88" stroke="#30423C" stroke-width="1"/><line x1="16" y1="58" x2="194" y2="58" stroke="#30423C" stroke-width="1" stroke-dasharray="3 3"/><line x1="16" y1="28" x2="194" y2="28" stroke="rgba(213,180,119,0.35)" stroke-width="1" stroke-dasharray="3 3"/><path d="M18 82 C42 76, 56 42, 80 48 C104 54, 118 24, 142 32 C164 39, 176 20, 192 24" stroke="#49C6A5" stroke-width="2.4" fill="none"/><path d="M18 85 C46 80, 62 62, 86 64 C110 66, 126 46, 150 50 C170 53, 180 40, 192 42" stroke="#D5B477" stroke-width="1.6" stroke-dasharray="4 3" fill="none"/><circle cx="80" cy="48" r="3.5" fill="#49C6A5"/><circle cx="142" cy="32" r="3.5" fill="#49C6A5"/><circle cx="192" cy="24" r="4" fill="#D5B477"/><text x="16" y="17" fill="#A8B7AE" font-family="monospace" font-size="8.5">MOX ARRAY // CALIBRATED FIT</text><text x="16" y="103" fill="#D5B477" font-family="monospace" font-size="8">R² = 0.6117  |  RMSE = 32.63 µg/m³</text></svg>',
    "objective": '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#49C6A5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "integrity": '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#D5B477" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/></svg>',
    "check": '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#49C6A5" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "limit": '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#D5B477" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    "db": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#49C6A5" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "chip": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#D5B477" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "pulse": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#49C6A5" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "precision": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#D5B477" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>'
}

# ------------------------------------------------------------------
# 4. FILE PATHS (ROOT-FOLDER DEPLOYMENT)
# ------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "best_no2_model.joblib")
META_PATH = os.path.join(APP_DIR, "model_metadata.json")
DATA_PATH = os.path.join(APP_DIR, "cleaned_air_quality.csv")
STATS_PATH = os.path.join(APP_DIR, "statistical_summary.csv")
COMP_PATH = os.path.join(APP_DIR, "model_comparison_final.csv")
FIG_DIR = APP_DIR

# ------------------------------------------------------------------
# 5. CACHED ARTIFACT LOADERS
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact missing: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata artifact missing: {META_PATH}")
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_csv(path, parse_dates=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset artifact missing: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)

try:
    model = load_model()
    meta = load_metadata()
    df = load_csv(DATA_PATH, parse_dates=["Datetime"])
    stats_df = load_csv(STATS_PATH)
    comp_df = load_csv(COMP_PATH)
except Exception as e:
    st.error(f"System initialization failed: {e}")
    st.stop()

FEATURE_COLS = meta["feature_columns"]
TARGET_COL = meta["target_column"]
METRICS = meta["metrics"]
FEAT_STATS = meta["feature_stats"]

def classify_no2_band(val):
    """Reference NO2 hourly concentration tier (NOT an official multi-pollutant AQI)."""
    if val < 40:
        return (
            "LOW CONCENTRATION (< 40 µg/m³)",
            "#49C6A5",
            "rgba(73, 198, 165, 0.16)",
            "Background urban ambient level well below international 1-hour exposure thresholds."
        )
    elif val < 100:
        return (
            "MODERATE CONCENTRATION (40–99 µg/m³)",
            "#73BCA8",
            "rgba(115, 188, 168, 0.16)",
            "Typical daytime roadside NO₂ concentration under normal vehicular flow."
        )
    elif val < 200:
        return (
            "ELEVATED CONCENTRATION (100–199 µg/m³)",
            "#D5B477",
            "rgba(213, 180, 119, 0.18)",
            "Elevated combustion buildup approaching the WHO 1-hour guideline of 200 µg/m³."
        )
    else:
        return (
            "HIGH — EXCEEDS WHO 1-HR 200 µg/m³ GUIDELINE",
            "#E06C75",
            "rgba(224, 108, 117, 0.20)",
            "Exceeds the World Health Organization 1-hour NO₂ reference threshold (200 µg/m³)."
        )

def render_meta_strip():
    st.markdown(
        """
        <div class="meta-bar">
            <span>SOURCE: <b>UCI-360 (ENEA ROADWAY)</b></span>
            <span>CALIBRATOR: <b>RANDOM FOREST (200 TREES)</b></span>
            <span>SPLIT: <b>CHRONOLOGICAL 80/20</b></span>
            <span class="gold-tag">QA SUITE: 6/6 VERIFIED</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_hero(kicker, title, subtitle, show_art=False):
    render_meta_strip()
    art_html = f'<div class="hero-art">{ICONS["hero_viz"]}</div>' if show_art else ""
    st.markdown(
        f"""
        <div class="lux-hero">
            <div class="hero-grid">
                <div class="hero-copy">
                    <div class="hero-kicker">{kicker}</div>
                    <div class="hero-h1">{title}</div>
                    <p class="hero-desc">{subtitle}</p>
                </div>
                {art_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_kpi(col, label, value_html, foot, icon_key="pulse", gold=False):
    accent_cls = "kpi-box gold-accent" if gold else "kpi-box"
    col.markdown(
        f"""
        <div class="{accent_cls}">
            <div class="kpi-top">
                <span class="kpi-lbl">{label}</span>
                {ICONS.get(icon_key, "")}
            </div>
            <div class="kpi-num">{value_html}</div>
            <div class="kpi-sub">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------
# 6. LUXURY SIDEBAR NAVIGATION (3 ORGANIZED TIERS, VECTOR ICONS)
# ------------------------------------------------------------------
NAV_SECTIONS = {
    "EXPLORE": [
        ("overview", "Project Overview", ":material/space_dashboard:"),
        ("dataset", "Dataset Information", ":material/database:"),
        ("analytics", "Air Quality Analytics", ":material/monitoring:"),
        ("stats", "Statistical Analysis", ":material/table_chart:"),
        ("interactive", "Interactive Explorer", ":material/scatter_plot:"),
    ],
    "MACHINE LEARNING": [
        ("performance", "Model Performance", ":material/leaderboard:"),
        ("estimator", "NO₂ Live Estimator", ":material/tune:"),
    ],
    "PROJECT": [
        ("conclusion", "Conclusion & Limitations", ":material/verified:"),
    ]
}

if "active_page" not in st.session_state:
    st.session_state.active_page = "overview"

st.sidebar.markdown(
    f"""
    <div class="brand-header">
        <div class="brand-title-row">
            {ICONS["brand"]}
            <div class="brand-name">AIRSENSE AI</div>
        </div>
        <div class="brand-sub">MOX Sensor Calibration &amp; NO₂ Estimation</div>
        <div class="brand-status">
            <span class="status-badge"><span class="status-dot"></span>CALIBRATED ENGINE ONLINE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

for section_title, nav_items in NAV_SECTIONS.items():
    st.sidebar.markdown(f'<div class="nav-group-label">{section_title}</div>', unsafe_allow_html=True)
    for page_key, label, icon_name in nav_items:
        selected = (st.session_state.active_page == page_key)
        if st.sidebar.button(
            label,
            key=f"nav_{page_key}",
            icon=icon_name,
            type="primary" if selected else "secondary",
            use_container_width=True
        ):
            st.session_state.active_page = page_key
            st.rerun()

st.sidebar.markdown(
    f"""
    <div class="sidebar-telemetry">
        <div class="tel-head">Model Telemetry</div>
        <div class="tel-row"><span>Selected Model</span><span class="tel-val">Random Forest</span></div>
        <div class="tel-row"><span>Test RMSE</span><span class="tel-val">{METRICS['Test_RMSE']} µg/m³</span></div>
        <div class="tel-row"><span>Test MAE</span><span class="tel-val">{METRICS['Test_MAE']} µg/m³</span></div>
        <div class="tel-row"><span>Test R²</span><span class="tel-val">{METRICS['Test_R2']}</span></div>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.caption("UCI Air Quality (ID: 360) • Theme-Adaptive UI")

page = st.session_state.active_page

# ------------------------------------------------------------------
# PAGE 1: PROJECT OVERVIEW
# ------------------------------------------------------------------
if page == "overview":
    render_hero(
        "ENVIRONMENTAL INTELLIGENCE // CAPSTONE",
        "AIR QUALITY ANALYSIS & NO₂ CONCENTRATION ESTIMATION SYSTEM",
        "Calibrating low-cost Metal-Oxide (MOX) chemical sensor arrays against certified reference analyzers using leakage-free Machine Learning.",
        show_art=True
    )

    c1, c2, c3, c4 = st.columns(4)
    render_kpi(c1, "Clean Hourly Records", f"{len(df):,}", "Audited from 9,357 raw UCI rows", "db", gold=False)
    render_kpi(c2, "Input Features", f"{len(FEATURE_COLS)}", "5 MOX sensors + 3 weather channels", "chip", gold=True)
    render_kpi(c3, "Test RMSE", f"{METRICS['Test_RMSE']} <span class='unit'>µg/m³</span>", "49.7% error reduction vs. baseline", "pulse", gold=False)
    render_kpi(c4, "Test R² Score", f"{METRICS['Test_R2']:.4f}", "Chronological winter test split", "precision", gold=True)

    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            f"""
            <div class="content-card">
                <div class="card-head">{ICONS["objective"]}<span>Engineering Objective</span></div>
                <p>Certified reference air-quality analyzers provide regulatory accuracy but require high capital expenditure and daily calibration. Low-cost Metal-Oxide (MOX) sensor arrays scale across urban corridors, yet exhibit strong cross-sensitivity to ozone, carbon monoxide, hydrocarbons, temperature, and humidity.</p>
                <p>This platform establishes a <b>non-linear virtual sensor calibration model</b> mapping simultaneous MOX sensor responses and meteorological readings directly to reference <b>NO₂(GT) concentration (µg/m³)</b>.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_right:
        st.markdown(
            f"""
            <div class="content-card gold">
                <div class="card-head">{ICONS["integrity"]}<span>Scientific Integrity &amp; Leakage Prevention</span></div>
                <p>All co-located certified gas analyzer columns (<code>CO(GT)</code>, <code>NMHC(GT)</code>, <code>C6H6(GT)</code>, and <code>NOx(GT)</code>) were strictly excluded from model inputs to eliminate target leakage and reflect true field deployment.</p>
                <p>Validation follows a <b>chronological 80/20 split</b> (unseen winter 2005 test partition) with 5-fold expanding <code>TimeSeriesSplit</code>, reporting physical <b>NO₂ concentration (µg/m³)</b> rather than mislabeling a single gas as an official AQI.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="arch-grid">
            <div class="arch-step">
                <div class="arch-tag">PHASE 01 // DATA AUDIT</div>
                <div class="arch-title">Deterministic Hygiene</div>
                <div class="arch-desc">9,357 raw records audited; -200 sentinel codes replaced with NaN; 7,393 complete hourly rows verified.</div>
            </div>
            <div class="arch-step">
                <div class="arch-tag">PHASE 02 // SCHEMA</div>
                <div class="arch-title">Zero-Leakage Inputs</div>
                <div class="arch-desc">5 MOX chemical sensors (PT08.S1–S5) paired with Temperature (T), Relative (RH) &amp; Absolute Humidity (AH).</div>
            </div>
            <div class="arch-step">
                <div class="arch-tag">PHASE 03 // CALIBRATION</div>
                <div class="arch-title">Chronological Validation</div>
                <div class="arch-desc">5 regression architectures benchmarked via expanding TimeSeriesSplit CV and unseen winter testing.</div>
            </div>
            <div class="arch-step">
                <div class="arch-tag">PHASE 04 // PRODUCTION</div>
                <div class="arch-title">Verified Cloud Inference</div>
                <div class="arch-desc">6/6 automated verification tests passed; deployed as a standalone interactive intelligence system.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------
# PAGE 2: DATASET INFORMATION
# ------------------------------------------------------------------
elif page == "dataset":
    render_hero(
        "DATA PROVENANCE // UCI-360",
        "DATASET ARCHITECTURE & CLEANING AUDIT",
        "Hourly roadside chemical multisensor observations collected from March 2004 to April 2005 by De Vito et al. (ENEA, Italy) under CC BY 4.0."
    )

    c1, c2, c3 = st.columns(3)
    render_kpi(c1, "Raw Observations", "9,357", "15 original columns, 0 duplicates", "db", gold=False)
    render_kpi(c2, "Missing Sentinel", "-200", "Converted to NaN & audited", "pulse", gold=True)
    render_kpi(c3, "Clean Verified Rows", f"{len(df):,}", "0 missing values, 0 sentinels", "precision", gold=False)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["Feature & Target Schema", "Cleaned Dataset Preview"])
    with t1:
        schema_df = pd.DataFrame([
            {"Column": "PT08.S1(CO)", "Role": "Input Feature", "Unit / Domain": "Sensor Response", "Description": "Tin oxide MOX sensor (nominally CO targeted)"},
            {"Column": "PT08.S2(NMHC)", "Role": "Input Feature", "Unit / Domain": "Sensor Response", "Description": "Titania MOX sensor (nominally NMHC targeted)"},
            {"Column": "PT08.S3(NOx)", "Role": "Input Feature", "Unit / Domain": "Sensor Response", "Description": "Tungsten oxide MOX sensor (nominally NOx targeted)"},
            {"Column": "PT08.S4(NO2)", "Role": "Input Feature", "Unit / Domain": "Sensor Response", "Description": "Tungsten oxide MOX sensor (nominally NO2 targeted)"},
            {"Column": "PT08.S5(O3)", "Role": "Input Feature", "Unit / Domain": "Sensor Response", "Description": "Indium oxide MOX sensor (nominally O3 targeted)"},
            {"Column": "T", "Role": "Input Feature", "Unit / Domain": "°C", "Description": "Ambient air temperature"},
            {"Column": "RH", "Role": "Input Feature", "Unit / Domain": "%", "Description": "Relative humidity"},
            {"Column": "AH", "Role": "Input Feature", "Unit / Domain": "Absolute Humidity", "Description": "Water vapor concentration"},
            {"Column": "NO2(GT)", "Role": "Target (y)", "Unit / Domain": "µg/m³", "Description": "Certified reference analyzer hourly NO₂ concentration"}
        ])
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(df.head(30), use_container_width=True)

# ------------------------------------------------------------------
# PAGE 3: AIR QUALITY ANALYTICS
# ------------------------------------------------------------------
elif page == "analytics":
    render_hero(
        "EXPLORATORY DATA ANALYSIS",
        "SCIENTIFIC AIR QUALITY VISUALIZATIONS",
        "High-DPI diagnostic figures profiling NO₂ concentration skewness, diurnal rush-hour cycles, seasonal winter elevation, and sensor cross-sensitivity."
    )

    fig_catalog = {
        "Figure 1: Reference NO₂(GT) Concentration Distribution": (
            "fig1_no2_distribution.png",
            "Moderately right-skewed distribution (mean = 112.15 µg/m³, median = 109.0 µg/m³, P95 = 198.0 µg/m³, P99 = 241.0 µg/m³)."
        ),
        "Figure 2: Multi-Season Time-Series Trend (Mar 2004 – Apr 2005)": (
            "fig2_no2_timeseries.png",
            "Daily mean and 7-day rolling trajectory highlighting pronounced winter elevation during late 2004 and early 2005."
        ),
        "Figure 3: Diurnal (Hourly) Variation of NO₂(GT)": (
            "fig3_hourly_no2_variation.png",
            "Bimodal urban traffic cycle peaking at 19:00 (149.05 µg/m³) and reaching a minimum at 04:00 (57.78 µg/m³)."
        ),
        "Figure 4: Monthly Average NO₂(GT) Concentration": (
            "fig4_monthly_no2_variation.png",
            "Seasonal contrast between February peak (157.90 µg/m³) and August trough (68.94 µg/m³)."
        ),
        "Figure 5: Multi-Sensor Pearson Correlation Heatmap": (
            "fig5_correlation_heatmap.png",
            "Demonstrates strong cross-sensitivity: PT08.S5(O3) (r = +0.71) and PT08.S3(NOx) (r = -0.65) correlate far more strongly with NO₂(GT) than PT08.S4(NO2) (r = +0.16)."
        ),
        "Figure 6: Standardized Boxplots of Input Features & Target": (
            "fig6_feature_boxplots.png",
            "Scale comparison across sensor resistance responses, meteorological channels, and physical pollution spikes."
        ),
        "Figure 7: Input Features vs. Reference NO₂(GT) Scatterplots": (
            "fig7_feature_vs_target_scatter.png",
            "Non-linear response curves (notably PT08.S3(NOx)) justifying non-linear tree ensemble calibration."
        )
    }

    selected_title = st.selectbox("Select Analytical Figure", list(fig_catalog.keys()))
    fname, caption = fig_catalog[selected_title]
    img_path = os.path.join(FIG_DIR, fname)

    if os.path.exists(img_path):
        st.markdown('<div class="figure-frame">', unsafe_allow_html=True)
        st.image(img_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(caption)
    else:
        st.warning(f"Figure file not found: {fname}")

# ------------------------------------------------------------------
# PAGE 4: STATISTICAL ANALYSIS
# ------------------------------------------------------------------
elif page == "stats":
    render_hero(
        "QUANTITATIVE PROFILING",
        "DESCRIPTIVE STATISTICAL SUMMARY",
        "Comprehensive parametric and non-parametric statistics computed across all 7,393 verified hourly observations."
    )

    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Statistical Summary (CSV)",
        data=stats_df.to_csv(index=False).encode("utf-8"),
        file_name="statistical_summary.csv",
        mime="text/csv",
        icon=":material/download:",
        type="primary"
    )

# ------------------------------------------------------------------
# PAGE 5: INTERACTIVE EXPLORER (THEME-ADAPTIVE CHARTS)
# ------------------------------------------------------------------
elif page == "interactive":
    render_hero(
        "DYNAMIC TELEMETRY EXPLORER",
        "INTERACTIVE SENSOR & TEMPORAL EXPLORER",
        "Filter observations by calendar month and inspect live correlation structure, regression trendlines, and diurnal profiles."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_feat = st.selectbox("Select Sensor / Weather Feature (X-axis)", FEATURE_COLS, index=4)
    with col2:
        month_filter = st.multiselect(
            "Filter by Calendar Month(s)",
            options=sorted(df["Month"].unique().tolist()),
            default=sorted(df["Month"].unique().tolist())
        )

    filtered_df = df[df["Month"].isin(month_filter)]
    if filtered_df.empty:
        st.warning("No records match the current month filter. Please select at least one month.")
    else:
        r_val = filtered_df[selected_feat].corr(filtered_df[TARGET_COL])
        m1, m2, m3 = st.columns(3)
        render_kpi(m1, "Filtered Observations", f"{len(filtered_df):,}", "Hourly records active", "db", gold=False)
        render_kpi(m2, "Pearson Correlation (r)", f"{r_val:+.4f}", f"{selected_feat} vs. NO₂(GT)", "pulse", gold=True)
        render_kpi(m3, "Mean NO₂(GT)", f"{filtered_df[TARGET_COL].mean():.1f} <span class='unit'>µg/m³</span>", "Subset average concentration", "precision", gold=False)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        tab_scatter, tab_diurnal = st.tabs(["Sensor vs. NO₂(GT) Regression Scatter", "Diurnal Hourly Profile (Filtered Subset)"])

        with tab_scatter:
            fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=150)
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)

            ax.scatter(
                filtered_df[selected_feat], filtered_df[TARGET_COL],
                alpha=0.35, s=16, color="#267F70", edgecolors="none", label="Hourly Observations"
            )
            if len(filtered_df) > 2:
                x_vals = filtered_df[selected_feat].to_numpy()
                y_vals = filtered_df[TARGET_COL].to_numpy()
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, color="#D5B477", linewidth=2.0, label=f"Linear Trend (r = {r_val:+.3f})")

            ax.axhline(200, color="#E06C75", linestyle="--", linewidth=1.3, label="WHO 1-hr Guideline (200 µg/m³)")
            ax.set_title(f"{selected_feat} vs. Reference NO₂(GT) Concentration (n = {len(filtered_df):,})", fontsize=11, fontweight="bold", color="#73BCA8", pad=10)
            ax.set_xlabel(f"{selected_feat} (Input Feature)", fontsize=9.5, color="#8A9B93")
            ax.set_ylabel("NO₂(GT) Concentration (µg/m³)", fontsize=9.5, color="#8A9B93")
            ax.tick_params(colors="#8A9B93", labelsize=8.5)
            ax.grid(True, linestyle=":", alpha=0.25, color="#73BCA8")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_color("#4A5D56")
            leg = ax.legend(frameon=True, facecolor="#142322", edgecolor="#30423C", fontsize=8.5)
            for text in leg.get_texts():
                text.set_color("#EDF3EF")
            st.pyplot(fig)
            plt.close(fig)

        with tab_diurnal:
            hourly_grp = filtered_df.groupby("Hour")[TARGET_COL].agg(["mean", "median"]).reset_index()
            fig2, ax2 = plt.subplots(figsize=(10.5, 4.2), dpi=150)
            fig2.patch.set_alpha(0.0)
            ax2.patch.set_alpha(0.0)

            ax2.plot(hourly_grp["Hour"], hourly_grp["mean"], marker="o", linewidth=2.2, color="#49C6A5", label="Hourly Mean NO₂(GT)")
            ax2.plot(hourly_grp["Hour"], hourly_grp["median"], marker="s", linewidth=1.6, linestyle="--", color="#D5B477", label="Hourly Median NO₂(GT)")
            ax2.axhline(200, color="#E06C75", linestyle=":", linewidth=1.2, label="WHO 1-hr Limit (200 µg/m³)")
            ax2.set_xticks(range(0, 24))
            ax2.set_title("Diurnal Traffic Cycle Across Selected Month(s)", fontsize=11, fontweight="bold", color="#73BCA8", pad=10)
            ax2.set_xlabel("Hour of Day (00:00 – 23:00)", fontsize=9.5, color="#8A9B93")
            ax2.set_ylabel("NO₂(GT) (µg/m³)", fontsize=9.5, color="#8A9B93")
            ax2.tick_params(colors="#8A9B93", labelsize=8.5)
            ax2.grid(True, linestyle=":", alpha=0.25, color="#73BCA8")
            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax2.spines[spine].set_color("#4A5D56")
            leg2 = ax2.legend(frameon=True, facecolor="#142322", edgecolor="#30423C", fontsize=8.5)
            for text in leg2.get_texts():
                text.set_color("#EDF3EF")
            st.pyplot(fig2)
            plt.close(fig2)

# ------------------------------------------------------------------
# PAGE 6: MODEL PERFORMANCE
# ------------------------------------------------------------------
elif page == "performance":
    render_hero(
        "MACHINE LEARNING BENCHMARKING",
        "CHRONOLOGICAL MODEL EVALUATION & DIAGNOSTICS",
        f"Evaluated via {meta['split_strategy']} ({meta['train_samples']:,} training samples / {meta['test_samples']:,} unseen winter test samples) with 5-Fold TimeSeriesSplit CV."
    )

    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    t_bench, t_pred, t_imp, t_res = st.tabs([
        "Benchmark RMSE Comparison",
        "Actual vs. Predicted",
        "Feature Importance",
        "Residual Diagnostics"
    ])

    with t_bench:
        fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=150)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        models_list = comp_df["Model"].tolist()[::-1]
        rmse_vals = comp_df["Test_RMSE"].tolist()[::-1]
        colors = ["#D5B477" if "Selected" in m else ("#E06C75" if "Dummy" in m else "#267F70") for m in models_list]

        bars = ax.barh(models_list, rmse_vals, color=colors, height=0.58, edgecolor="none")
        for bar, val in zip(bars, rmse_vals):
            ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2, f"{val:.3f} µg/m³", va="center", fontsize=8.5, fontweight="bold", color="#8A9B93")

        ax.set_xlim(0, max(rmse_vals) * 1.18)
        ax.set_title("Chronological Test RMSE Comparison (Lower is Better)", fontsize=11, fontweight="bold", color="#73BCA8", pad=10)
        ax.set_xlabel("Test RMSE (µg/m³)", fontsize=9.5, color="#8A9B93")
        ax.tick_params(colors="#8A9B93", labelsize=8.5)
        ax.grid(True, axis="x", linestyle=":", alpha=0.25, color="#73BCA8")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#4A5D56")
        st.pyplot(fig)
        plt.close(fig)

    with t_pred:
        p8 = os.path.join(FIG_DIR, "fig8_actual_vs_predicted.png")
        if os.path.exists(p8):
            st.markdown('<div class="figure-frame">', unsafe_allow_html=True)
            st.image(p8, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with t_imp:
        p10 = os.path.join(FIG_DIR, "fig10_feature_importance.png")
        if os.path.exists(p10):
            st.markdown('<div class="figure-frame">', unsafe_allow_html=True)
            st.image(p10, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with t_res:
        p9 = os.path.join(FIG_DIR, "fig9_residual_diagnostics.png")
        if os.path.exists(p9):
            st.markdown('<div class="figure-frame">', unsafe_allow_html=True)
            st.image(p9, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# PAGE 7: NO2 LIVE ESTIMATOR
# ------------------------------------------------------------------
elif page == "estimator":
    render_hero(
        "VIRTUAL SENSOR CALIBRATION ENGINE",
        "REAL-TIME NO₂ CONCENTRATION ESTIMATOR",
        "Select a calibrated operating scenario or input custom MOX sensor and meteorological readings to estimate reference NO₂ concentration (µg/m³)."
    )

    presets = {
        "Dataset Median (Baseline Operating Point)": {col: float(FEAT_STATS[col]["median"]) for col in FEATURE_COLS},
        "Quiet Night — Low Traffic Conditions": {
            "PT08.S1(CO)": 840.0, "PT08.S2(NMHC)": 600.0, "PT08.S3(NOx)": 1250.0,
            "PT08.S4(NO2)": 1180.0, "PT08.S5(O3)": 500.0, "T": 14.0, "RH": 62.0, "AH": 0.98
        },
        "Evening Rush Hour — Peak Traffic": {
            "PT08.S1(CO)": 1480.0, "PT08.S2(NMHC)": 1380.0, "PT08.S3(NOx)": 520.0,
            "PT08.S4(NO2)": 1850.0, "PT08.S5(O3)": 1680.0, "T": 16.5, "RH": 52.0, "AH": 0.95
        },
        "Cold Winter Peak — Inversion Episode": {
            "PT08.S1(CO)": 1550.0, "PT08.S2(NMHC)": 1420.0, "PT08.S3(NOx)": 480.0,
            "PT08.S4(NO2)": 1450.0, "PT08.S5(O3)": 1800.0, "T": 5.0, "RH": 70.0, "AH": 0.60
        }
    }

    selected_preset = st.selectbox("Calibrated Operating Scenario", list(presets.keys()))
    defaults = presets[selected_preset]

    mox_cols = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
    weather_cols = ["T", "RH", "AH"]
    unit_labels = {
        "PT08.S1(CO)": "PT08.S1(CO) — Tin Oxide Sensor Response",
        "PT08.S2(NMHC)": "PT08.S2(NMHC) — Titania Sensor Response",
        "PT08.S3(NOx)": "PT08.S3(NOx) — Tungsten Oxide Sensor Response",
        "PT08.S4(NO2)": "PT08.S4(NO2) — Tungsten Oxide Sensor Response",
        "PT08.S5(O3)": "PT08.S5(O3) — Indium Oxide Sensor Response",
        "T": "T — Ambient Temperature (°C)",
        "RH": "RH — Relative Humidity (%)",
        "AH": "AH — Absolute Humidity"
    }

    with st.form("prediction_form"):
        col_mox, col_wx = st.columns([1.25, 1.0])
        inputs = {}

        with col_mox:
            st.markdown('<div class="form-group-header">Group A // MOX Chemical Sensor Array (5 Channels)</div>', unsafe_allow_html=True)
            for col in mox_cols:
                f_min = float(FEAT_STATS[col]["min"])
                f_max = float(FEAT_STATS[col]["max"])
                f_val = float(np.clip(defaults[col], f_min, f_max))
                inputs[col] = st.number_input(
                    label=f"{unit_labels[col]} [{f_min} – {f_max}]",
                    min_value=f_min,
                    max_value=f_max,
                    value=f_val,
                    step=1.0
                )

        with col_wx:
            st.markdown('<div class="form-group-header">Group B // Meteorological Compensation (3 Channels)</div>', unsafe_allow_html=True)
            for col in weather_cols:
                f_min = float(FEAT_STATS[col]["min"])
                f_max = float(FEAT_STATS[col]["max"])
                f_val = float(np.clip(defaults[col], f_min, f_max))
                step = 0.01 if col == "AH" else 0.1
                inputs[col] = st.number_input(
                    label=f"{unit_labels[col]} [{f_min} – {f_max}]",
                    min_value=f_min,
                    max_value=f_max,
                    value=f_val,
                    step=step
                )

        submitted = st.form_submit_button(
            "Estimate Reference NO₂ Concentration",
            icon=":material/calculate:",
            type="primary",
            use_container_width=True
        )

    if submitted:
        input_df = pd.DataFrame([inputs])[FEATURE_COLS]
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            pred_no2 = float(model.named_steps["model"].predict(input_df)[0])
        else:
            pred_no2 = float(model.predict(input_df)[0])

        band_label, band_color, band_bg, band_desc = classify_no2_band(pred_no2)
        gauge_pct = min(100.0, max(2.0, (pred_no2 / 333.0) * 100.0))

        st.markdown(
            f"""
            <div class="est-card">
                <div class="est-kicker">Calibrated Random Forest Output // Reference NO₂(GT)</div>
                <div class="est-reading" style="color:{band_color};">{pred_no2:.2f} µg/m³</div>
                <span class="est-pill" style="color:{band_color}; background:{band_bg}; border: 1px solid {band_color};">
                    {band_label}
                </span>
                <p style="color:#C4D4CC; font-size:0.9rem; margin-top:0.75rem; margin-bottom:0;">
                    {band_desc}
                </p>
                <div class="gauge-bar">
                    <div class="gauge-progress" style="width:{gauge_pct:.1f}%; background:{band_color};"></div>
                </div>
                <div class="gauge-ticks">
                    <span>0 µg/m³</span>
                    <span>40 (Low)</span>
                    <span>100 (Moderate)</span>
                    <span>200 (WHO 1-hr Limit)</span>
                    <span>333 (Dataset Max)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "**Technical Disclaimer:** This output represents an estimated single-pollutant **NO₂ concentration (µg/m³)** "
            "derived from multi-sensor MOX calibration. It is **not** an official multi-pollutant Air Quality Index (AQI), "
            "which requires regulatory averaging across PM₂.₅, PM₁₀, O₃, CO, SO₂, and NO₂.",
            icon=":material/info:"
        )

        report_df = input_df.copy()
        report_df["Scenario_Preset"] = selected_preset
        report_df["Estimated_NO2_ug_m3"] = round(pred_no2, 2)
        report_df["Reference_Band"] = band_label
        st.download_button(
            "Download Estimation Report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="no2_estimation_result.csv",
            mime="text/csv",
            icon=":material/download:"
        )

# ------------------------------------------------------------------
# PAGE 8: CONCLUSION & LIMITATIONS
# ------------------------------------------------------------------
elif page == "conclusion":
    render_hero(
        "PROJECT SUMMARY & ROADMAP",
        "CONCLUSION, LIMITATIONS & FUTURE PERSPECTIVE",
        "Transparent engineering assessment of model performance, physical sensor drift, and IoT hardware deployment."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="content-card">
                <div class="card-head">{ICONS["check"]}<span>Engineering Conclusions</span></div>
                <ul>
                    <li><b>Non-Linear Calibration Efficacy:</b> The calibrated <b>{meta['selected_model']}</b> reduced chronological test RMSE by <b>49.7%</b> relative to the Dummy baseline (<b>{METRICS['Test_RMSE']}</b> vs. <b>64.879 µg/m³</b>) and outperformed Linear Regression (<b>36.133 µg/m³</b>).</li>
                    <li><b>Cross-Sensitivity Resolution:</b> Combining all 5 MOX channels with Temperature, Relative Humidity, and Absolute Humidity resolves cross-gas interference that defeats single-sensor linear calibration.</li>
                    <li><b>Verified Reproducibility:</b> Validated via <b>6/6 automated tests</b> and deployed live with zero target leakage.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="content-card gold">
                <div class="card-head">{ICONS["limit"]}<span>Scientific Limitations &amp; Future Scope</span></div>
                <ul>
                    <li><b>Sensor Drift &amp; Seasonal Shift:</b> Chronological testing on unseen winter 2005 data highlights gradual MOX aging and winter inversion peaks, motivating periodic online recalibration.</li>
                    <li><b>Geographic Calibration Scope:</b> Trained on 2004–2005 Italian roadside data; deployment in new climates requires local reference co-location.</li>
                    <li><b>Single-Pollutant Output:</b> Estimates NO₂ concentration (µg/m³) only, not a composite regulatory AQI.</li>
                    <li><b>Hardware IoT Roadmap:</b> Interfacing an ESP32 microcontroller with physical MOX gas sensors and a DHT22 module to stream live telemetry via REST API.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
