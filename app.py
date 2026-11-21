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
    page_title="Air Quality Analysis & NO2 Estimation",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Relative paths (matching files in the GitHub repository root folder)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "best_no2_model.joblib")
META_PATH = os.path.join(APP_DIR, "model_metadata.json")
DATA_PATH = os.path.join(APP_DIR, "cleaned_air_quality.csv")
STATS_PATH = os.path.join(APP_DIR, "statistical_summary.csv")
COMP_PATH = os.path.join(APP_DIR, "model_comparison_final.csv")
FIG_DIR = APP_DIR

# ------------------------------------------------------------------
# 2. CACHED LOADERS WITH ERROR HANDLING
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
        return "Low (< 40 µg/m³)", "green"
    elif val < 100:
        return "Moderate (40–99 µg/m³)", "blue"
    elif val < 200:
        return "Elevated (100–199 µg/m³)", "orange"
    else:
        return "High / Exceeds 1-hr WHO 200 µg/m³ Guideline", "red"

# ------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.title("🌫️ Air Quality System")
st.sidebar.caption("MOX Sensor Calibration & NO₂ Estimation")

page = st.sidebar.radio(
    "Navigate",
    [
        "1. Project Overview",
        "2. Dataset Information",
        "3. Air Quality Analytics",
        "4. Statistical Analysis",
        "5. Interactive Visualizations",
        "6. Model Performance",
        "7. NO2 Concentration Estimator",
        "8. Conclusion & Limitations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Selected Model:** {meta['selected_model']}\n\n"
    f"**Test RMSE:** {METRICS['Test_RMSE']} µg/m³\n\n"
    f"**Test R²:** {METRICS['Test_R2']}"
)

# ------------------------------------------------------------------
# SECTION 1: PROJECT OVERVIEW
# ------------------------------------------------------------------
if page == "1. Project Overview":
    st.title("Air Quality Analysis & NO₂ Concentration Estimation System")
    st.markdown(
        "This capstone project applies **Data Analytics** and **Machine Learning** to calibrate low-cost "
        "metal-oxide (MOX) chemical sensors against a certified reference analyzer using the official "
        "**UCI Air Quality Dataset (ID: 360)**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean Hourly Records", f"{len(df):,}")
    c2.metric("Input Features", f"{len(FEATURE_COLS)}")
    c3.metric("Test RMSE", f"{METRICS['Test_RMSE']} µg/m³")
    c4.metric("Test R²", f"{METRICS['Test_R2']:.4f}")

    st.subheader("Core Engineering Objective")
    st.markdown(
        "- **Task:** Simultaneous **NO₂(GT) Concentration Estimation** (µg/m³) from 5 MOX chemical sensors and 3 weather variables.\n"
        "- **Why Not Future Forecasting?** Predicting current reference NO₂ from simultaneous sensor readings matches real-world virtual sensor calibration without requiring users to enter past ground-truth lab history.\n"
        "- **Leakage Prevention:** All other reference analyzer columns (`CO(GT)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`) were strictly excluded, and evaluation uses a **chronological 80/20 split**."
    )

# ------------------------------------------------------------------
# SECTION 2: DATASET INFORMATION
# ------------------------------------------------------------------
elif page == "2. Dataset Information":
    st.title("Dataset Information & Schema")
    st.markdown(
        "- **Source:** UCI Machine Learning Repository — Air Quality Dataset (De Vito, ENEA, Italy, CC BY 4.0).\n"
        "- **Collection Period:** March 2004 – April 2005 (hourly averages at a polluted Italian roadside).\n"
        "- **Cleaning Applied:** Replaced `-200` sentinel codes with `NaN`, dropped rows missing `NO2(GT)` or experiencing full 8-sensor outages, leaving **7,393 complete hourly records**."
    )

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

    st.subheader("Sample Cleaned Records")
    st.dataframe(df.head(20), use_container_width=True)

# ------------------------------------------------------------------
# SECTION 3: AIR QUALITY ANALYTICS
# ------------------------------------------------------------------
elif page == "3. Air Quality Analytics":
    st.title("Air Quality Analytics (Saved High-DPI Figures)")
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
    st.title("Descriptive Statistical Analysis")
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    csv_bytes = stats_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Statistical Summary (CSV)",
        data=csv_bytes,
        file_name="statistical_summary.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------
# SECTION 5: INTERACTIVE VISUALIZATIONS
# ------------------------------------------------------------------
elif page == "5. Interactive Visualizations":
    st.title("Interactive Air Quality Explorer")

    col1, col2 = st.columns(2)
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
        st.metric(f"Filtered Pearson Correlation ({selected_feat} vs NO2(GT))", f"{r_val:.4f}")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.scatterplot(data=filtered_df, x=selected_feat, y=TARGET_COL, alpha=0.3, s=15, color="#1f77b4", ax=ax)
        ax.set_title(f"{selected_feat} vs. NO2(GT) ({len(filtered_df):,} observations)")
        ax.set_xlabel(selected_feat)
        ax.set_ylabel("NO2(GT) (µg/m³)")
        st.pyplot(fig)

# ------------------------------------------------------------------
# SECTION 6: MACHINE LEARNING MODEL PERFORMANCE
# ------------------------------------------------------------------
elif page == "6. Model Performance":
    st.title("Machine Learning Model Evaluation")
    st.markdown(
        f"**Validation Strategy:** {meta['split_strategy']} "
        f"({meta['train_samples']:,} training rows / {meta['test_samples']:,} chronological test rows)."
    )

    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        p8 = os.path.join(FIG_DIR, "fig8_actual_vs_predicted.png")
        if os.path.exists(p8):
            st.image(p8, caption="Actual vs. Predicted NO₂(GT)", use_container_width=True)
    with c2:
        p10 = os.path.join(FIG_DIR, "fig10_feature_importance.png")
        if os.path.exists(p10):
            st.image(p10, caption="Feature Importance", use_container_width=True)

    p9 = os.path.join(FIG_DIR, "fig9_residual_diagnostics.png")
    if os.path.exists(p9):
        st.image(p9, caption="Residual Diagnostics", use_container_width=True)

# ------------------------------------------------------------------
# SECTION 7: NO2 CONCENTRATION ESTIMATOR
# ------------------------------------------------------------------
elif page == "7. NO2 Concentration Estimator":
    st.title("Live NO₂ Concentration Estimator (Sensor Calibration)")
    st.markdown(
        "Enter simultaneous **MOX sensor responses** and **weather readings** to estimate the reference "
        "**NO₂ concentration (µg/m³)**. Default values are set to the dataset medians."
    )

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        inputs = {}
        for idx, col in enumerate(FEATURE_COLS):
            f_min = float(FEAT_STATS[col]["min"])
            f_max = float(FEAT_STATS[col]["max"])
            f_med = float(FEAT_STATS[col]["median"])
            step = 0.01 if col == "AH" else (0.1 if col in ["T", "RH"] else 1.0)
            target_col = c1 if idx < 4 else c2
            with target_col:
                inputs[col] = st.number_input(
                    label=f"{col} (valid range: {f_min} to {f_max})",
                    min_value=f_min,
                    max_value=f_max,
                    value=f_med,
                    step=step
                )
        submitted = st.form_submit_button("Estimate NO₂ Concentration", type="primary")

    if submitted:
        input_df = pd.DataFrame([inputs])[FEATURE_COLS]
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            pred_no2 = float(model.named_steps["model"].predict(input_df.to_numpy())[0])
        else:
            pred_no2 = float(model.predict(input_df)[0])
        band_label, band_color = classify_no2_band(pred_no2)

        st.markdown("---")
        r1, r2 = st.columns(2)
        r1.metric("Estimated NO₂(GT) Concentration", f"{pred_no2:.2f} µg/m³")
        r2.markdown(f"**Reference NO₂ Band:** :{band_color}[**{band_label}**]")

        st.caption(
            "⚠️ **Important Distinction:** This value is an estimated single-pollutant **NO₂ concentration (µg/m³)** "
            "from calibrated MOX sensors—it is **not** an official multi-pollutant Air Quality Index (AQI), which "
            "requires PM₂.₅, PM₁₀, O₃, CO, SO₂, and NO₂ averaged over regulatory windows."
        )

        report_df = input_df.copy()
        report_df["Estimated_NO2_ug_m3"] = round(pred_no2, 2)
        report_df["Reference_Band"] = band_label
        st.download_button(
            "Download Prediction Report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="no2_estimation_result.csv",
            mime="text/csv"
        )

# ------------------------------------------------------------------
# SECTION 8: CONCLUSION & LIMITATIONS
# ------------------------------------------------------------------
elif page == "8. Conclusion & Limitations":
    st.title("Conclusion & Scientific Limitations")
    st.subheader("Key Conclusions")
    st.markdown(
        f"- Multi-sensor machine learning (**{meta['selected_model']}**) achieved a chronological test RMSE of "
        f"**{METRICS['Test_RMSE']} µg/m³** (MAE = **{METRICS['Test_MAE']} µg/m³**, R² = **{METRICS['Test_R2']}**), "
        f"reducing error by ~50% compared to the Dummy baseline (RMSE = 64.88 µg/m³).\n"
        "- Cross-sensitivity among MOX sensors (`PT08.S5(O3)`, `PT08.S3(NOx)`, `PT08.S1(CO)`) and humidity (`AH`) "
        "plays a critical role in accurately estimating true NO₂ concentration."
    )
    st.subheader("Honest Limitations")
    st.markdown(
        "1. **Sensor Drift & Seasonal Shift:** Testing on the final 20% (winter 2005) revealed higher error than training CV, reflecting real-world MOX sensor aging and seasonal shifts.\n"
        "2. **Single-Location Historical Data:** Collected in 2004–2005 at one Italian roadside; recalibration is required before deploying in another city.\n"
        "3. **Not a Full AQI:** Estimates NO₂ concentration only, not official multi-pollutant AQI."
    )
