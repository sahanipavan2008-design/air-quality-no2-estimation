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
    page_title="AirSense AI | NO₂ Estimation & Analytics",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# 2. MODERN CUSTOM CSS DESIGN SYSTEM
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean typography & page padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    /* Gradient Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.2);
        color: #7dd3fc;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(125, 211, 252, 0.35);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
        color: #ffffff;
    }
    .hero-sub {
        font-size: 1rem;
        color: #cbd5e1;
        margin: 0;
        max-width: 820px;
    }

    /* Modern KPI Metric Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 0;
    }
    .kpi-foot {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    /* Info / Feature Cards */
    .info-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1rem;
    }
    .info-card h4 {
        margin-top: 0;
        margin-bottom: 0.4rem;
        font-weight: 700;
    }

    /* Result Highlight Card */
    .result-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
        color: #ffffff;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 0.2rem 0;
    }
    .band-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }

    /* Sidebar branding box */
    .sidebar-brand {
        background: linear-gradient(135deg, #0f2027, #2c5364);
        padding: 1.1rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .sidebar-brand h3 {
        margin: 0;
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff;
    }
    .sidebar-brand p {
        margin: 0.25rem 0 0 0;
        font-size: 0.78rem;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------
# 3. FILE PATHS (ROOT-FOLDER DEPLOYMENT)
# ------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "best_no2_model.joblib")
META_PATH = os.path.join(APP_DIR, "model_metadata.json")
DATA_PATH = os.path.join(APP_DIR, "cleaned_air_quality.csv")
STATS_PATH = os.path.join(APP_DIR, "statistical_summary.csv")
COMP_PATH = os.path.join(APP_DIR, "model_comparison_final.csv")
FIG_DIR = APP_DIR

# ------------------------------------------------------------------
# 4. CACHED LOADERS WITH ERROR HANDLING
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_csv(path, parse_dates=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)

try:
    model = load_model()
    meta = load_metadata()
    df = load_csv(DATA_PATH, parse_dates=["Datetime"])
    stats_df = load_csv(STATS_PATH)
    comp_df = load_csv(COMP_PATH)
except Exception as e:
    st.error(f"Application failed to load required artifacts: {e}")
    st.stop()

FEATURE_COLS = meta["feature_columns"]
TARGET_COL = meta["target_column"]
METRICS = meta["metrics"]
FEAT_STATS = meta["feature_stats"]

def classify_no2_band(val):
    """Reference NO2 hourly concentration band (NOT an official multi-pollutant AQI)."""
    if val < 40:
        return "Low (< 40 µg/m³)", "#22c55e", "rgba(34, 197, 94, 0.2)"
    elif val < 100:
        return "Moderate (40–99 µg/m³)", "#38bdf8", "rgba(56, 189, 248, 0.2)"
    elif val < 200:
        return "Elevated (100–199 µg/m³)", "#f59e0b", "rgba(245, 158, 11, 0.2)"
    else:
        return "High / Exceeds 1-hr WHO 200 µg/m³ Guideline", "#ef4444", "rgba(239, 68, 68, 0.2)"

def render_hero(badge, title, subtitle):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-badge">{badge}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_kpi(col, label, value, foot):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <h3>🌫️ AirSense AI</h3>
        <p>MOX Sensor Calibration & NO₂ Estimator</p>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Project Overview",
        "2. Dataset Information",
        "3. Air Quality Analytics",
        "4. Statistical Analysis",
        "5. Interactive Visualizations",
        "6. Model Performance",
        "7. NO2 Concentration Estimator",
        "8. Conclusion & Limitations"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Snapshot**")
st.sidebar.info(
    f"**Model:** {meta['selected_model']}\n\n"
    f"**Test RMSE:** {METRICS['Test_RMSE']} µg/m³\n\n"
    f"**Test MAE:** {METRICS['Test_MAE']} µg/m³\n\n"
    f"**Test R²:** {METRICS['Test_R2']}"
)
st.sidebar.caption("UCI Air Quality Dataset (ID: 360) • Chronological 80/20 Split")

# ------------------------------------------------------------------
# SECTION 1: PROJECT OVERVIEW
# ------------------------------------------------------------------
if page == "1. Project Overview":
    render_hero(
        "AI/ML & Data Analytics Capstone",
        "Air Quality Analysis & NO₂ Concentration Estimation System",
        "Calibrating low-cost Metal-Oxide (MOX) chemical sensor arrays against certified reference analyzers using leakage-free Machine Learning on the official UCI Air Quality Dataset."
    )

    c1, c2, c3, c4 = st.columns(4)
    render_kpi(c1, "Clean Hourly Records", f"{len(df):,}", "From 9,357 raw UCI rows")
    render_kpi(c2, "Input Features", f"{len(FEATURE_COLS)}", "5 MOX Sensors + 3 Weather")
    render_kpi(c3, "Test RMSE", f"{METRICS['Test_RMSE']} µg/m³", "49.7% better than baseline")
    render_kpi(c4, "Test R² Score", f"{METRICS['Test_R2']:.4f}", "Chronological winter test")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="info-card">
                <h4>🎯 Engineering Objective</h4>
                <p>Estimate hourly reference <b>NO₂(GT) concentration (µg/m³)</b> from simultaneous low-cost MOX sensor responses and ambient weather (T, RH, AH), solving hardware cross-sensitivity without expensive reference stations.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            """
            <div class="info-card">
                <h4>🛡️ Scientific Integrity & Leakage Prevention</h4>
                <p>All non-target reference columns (<code>CO(GT)</code>, <code>NMHC(GT)</code>, <code>C6H6(GT)</code>, <code>NOx(GT)</code>) were strictly excluded. Evaluation uses a <b>chronological 80/20 split</b> rather than random shuffling.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------------------------------------------
# SECTION 2: DATASET INFORMATION
# ------------------------------------------------------------------
elif page == "2. Dataset Information":
    render_hero(
        "Data Provenance & Schema",
        "UCI Air Quality Dataset (ID: 360)",
        "Collected hourly at a polluted Italian roadside (March 2004 – April 2005) by De Vito et al. (ENEA, Italy) under CC BY 4.0."
    )

    c1, c2, c3 = st.columns(3)
    render_kpi(c1, "Raw Observations", "9,357", "0 duplicate rows")
    render_kpi(c2, "Missing Sentinel", "-200", "Converted to NaN & audited")
    render_kpi(c3, "Final Clean Rows", f"{len(df):,}", "0 NaNs, 0 sentinels")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📋 Feature & Target Schema", "🔍 Cleaned Dataset Preview"])

    with tab1:
        schema_df = pd.DataFrame([
            {"Column": "PT08.S1(CO)", "Role": "Input Feature", "Description": "Tin oxide MOX sensor (nominally CO targeted)"},
            {"Column": "PT08.S2(NMHC)", "Role": "Input Feature", "Description": "Titania MOX sensor (nominally NMHC targeted)"},
            {"Column": "PT08.S3(NOx)", "Role": "Input Feature", "Description": "Tungsten oxide MOX sensor (nominally NOx targeted)"},
            {"Column": "PT08.S4(NO2)", "Role": "Input Feature", "Description": "Tungsten oxide MOX sensor (nominally NO2 targeted)"},
            {"Column": "PT08.S5(O3)", "Role": "Input Feature", "Description": "Indium oxide MOX sensor (nominally O3 targeted)"},
            {"Column": "T", "Role": "Input Feature", "Description": "Ambient Temperature (°C)"},
            {"Column": "RH", "Role": "Input Feature", "Description": "Relative Humidity (%)"},
            {"Column": "AH", "Role": "Input Feature", "Description": "Absolute Humidity"},
            {"Column": "NO2(GT)", "Role": "Target (y)", "Description": "Certified reference analyzer NO₂ concentration (µg/m³)"}
        ])
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

    with tab2:
        st.dataframe(df.head(25), use_container_width=True)

# ------------------------------------------------------------------
# SECTION 3: AIR QUALITY ANALYTICS
# ------------------------------------------------------------------
elif page == "3. Air Quality Analytics":
    render_hero(
        "Exploratory Data Analysis",
        "High-DPI Air Quality Visualizations",
        "Inspect pollutant distributions, diurnal traffic cycles, seasonal winter peaks, and multi-sensor cross-sensitivity."
    )

    fig_map = {
        "Figure 1: NO₂(GT) Distribution": "fig1_no2_distribution.png",
        "Figure 2: Time-Series Trend (Daily + 7-Day Rolling)": "fig2_no2_timeseries.png",
        "Figure 3: Diurnal (Hourly) Variation": "fig3_hourly_no2_variation.png",
        "Figure 4: Monthly Average NO₂(GT)": "fig4_monthly_no2_variation.png",
        "Figure 5: Pearson Correlation Heatmap": "fig5_correlation_heatmap.png",
        "Figure 6: Feature & Target Boxplots": "fig6_feature_boxplots.png",
        "Figure 7: Features vs. NO₂(GT) Scatterplots": "fig7_feature_vs_target_scatter.png"
    }

    selected_fig = st.selectbox("Select an EDA Figure to Inspect", list(fig_map.keys()))
    img_path = os.path.join(FIG_DIR, fig_map[selected_fig])
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning(f"Figure not found: {fig_map[selected_fig]}")

# ------------------------------------------------------------------
# SECTION 4: STATISTICAL ANALYSIS
# ------------------------------------------------------------------
elif page == "4. Statistical Analysis":
    render_hero(
        "Quantitative Profiling",
        "Descriptive Statistical Analysis",
        "Verified central tendency, dispersion, quartiles, percentiles, and skewness across all 7,393 cleaned hourly records."
    )

    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Statistical Summary (CSV)",
        data=stats_df.to_csv(index=False).encode("utf-8"),
        file_name="statistical_summary.csv",
        mime="text/csv",
        type="primary"
    )

# ------------------------------------------------------------------
# SECTION 5: INTERACTIVE VISUALIZATIONS
# ------------------------------------------------------------------
elif page == "5. Interactive Visualizations":
    render_hero(
        "Dynamic Data Explorer",
        "Interactive Sensor vs. NO₂(GT) Explorer",
        "Filter by calendar month and compare any MOX sensor or meteorological variable against reference NO₂ concentration."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_feat = st.selectbox("Select Input Feature (X-axis)", FEATURE_COLS, index=4)
    with col2:
        month_filter = st.multiselect(
            "Filter by Month(s)",
            options=sorted(df["Month"].unique().tolist()),
            default=sorted(df["Month"].unique().tolist())
        )

    filtered_df = df[df["Month"].isin(month_filter)]
    if filtered_df.empty:
        st.warning("No data matches the selected month filter. Please select at least one month.")
    else:
        r_val = filtered_df[selected_feat].corr(filtered_df[TARGET_COL])
        m1, m2, m3 = st.columns(3)
        render_kpi(m1, "Filtered Observations", f"{len(filtered_df):,}", "Hourly records shown")
        render_kpi(m2, "Pearson Correlation (r)", f"{r_val:+.4f}", f"{selected_feat} vs NO₂(GT)")
        render_kpi(m3, "Mean NO₂(GT)", f"{filtered_df[TARGET_COL].mean():.1f} µg/m³", "Across selected months")

        st.markdown("<br>", unsafe_allow_html=True)
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.scatterplot(
            data=filtered_df, x=selected_feat, y=TARGET_COL,
            alpha=0.35, s=18, color="#0284c7", edgecolor=None, ax=ax
        )
        ax.set_title(f"{selected_feat} vs. NO₂(GT) ({len(filtered_df):,} observations)", fontsize=12, fontweight="bold")
        ax.set_xlabel(selected_feat, fontsize=10)
        ax.set_ylabel("NO₂(GT) (µg/m³)", fontsize=10)
        st.pyplot(fig)

# ------------------------------------------------------------------
# SECTION 6: MACHINE LEARNING MODEL PERFORMANCE
# ------------------------------------------------------------------
elif page == "6. Model Performance":
    render_hero(
        "Model Benchmarking",
        "Machine Learning Evaluation & Diagnostics",
        f"Evaluated via {meta['split_strategy']} ({meta['train_samples']:,} training rows / {meta['test_samples']:,} chronological test rows)."
    )

    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["📈 Actual vs. Predicted", "📊 Feature Importance", "📉 Residual Diagnostics"])
    with t1:
        p8 = os.path.join(FIG_DIR, "fig8_actual_vs_predicted.png")
        if os.path.exists(p8):
            st.image(p8, use_container_width=True)
    with t2:
        p10 = os.path.join(FIG_DIR, "fig10_feature_importance.png")
        if os.path.exists(p10):
            st.image(p10, use_container_width=True)
    with t3:
        p9 = os.path.join(FIG_DIR, "fig9_residual_diagnostics.png")
        if os.path.exists(p9):
            st.image(p9, use_container_width=True)

# ------------------------------------------------------------------
# SECTION 7: NO2 CONCENTRATION ESTIMATOR (WITH PRESET TEMPLATES)
# ------------------------------------------------------------------
elif page == "7. NO2 Concentration Estimator":
    render_hero(
        "Live Virtual Sensor",
        "Real-Time NO₂ Concentration Estimator",
        "Choose a realistic scenario template or enter custom MOX sensor & weather readings to estimate reference NO₂ concentration (µg/m³)."
    )

    # Preset Scenario Templates (based on verified dataset percentiles)
    presets = {
        "Custom / Dataset Median": {col: float(FEAT_STATS[col]["median"]) for col in FEATURE_COLS},
        "🌙 Quiet Night (Low Traffic)": {
            "PT08.S1(CO)": 840.0, "PT08.S2(NMHC)": 600.0, "PT08.S3(NOx)": 1250.0,
            "PT08.S4(NO2)": 1180.0, "PT08.S5(O3)": 500.0, "T": 14.0, "RH": 62.0, "AH": 0.98
        },
        "🚗 Evening Rush Hour (High Traffic)": {
            "PT08.S1(CO)": 1480.0, "PT08.S2(NMHC)": 1380.0, "PT08.S3(NOx)": 520.0,
            "PT08.S4(NO2)": 1850.0, "PT08.S5(O3)": 1680.0, "T
