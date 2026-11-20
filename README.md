# Air Quality Analysis & NO2 Concentration Estimation System

Capstone project applying Data Analytics and Machine Learning to calibrate low-cost Metal-Oxide (MOX) chemical sensors against a certified reference analyzer using the official UCI Air Quality Dataset (ID: 360).

## Key Highlights
- **Approach:** Simultaneous NO2(GT) Concentration Estimation (ug/m3) from 5 MOX sensors + 3 weather variables.
- **Validation:** Chronological 80/20 Train-Test split (zero target leakage; all other GT columns excluded).
- **Selected Model:** Random Forest Regressor (Test RMSE = 32.629 ug/m3, Test MAE = 23.864 ug/m3, Test R2 = 0.6117).

## Run Locally / Deploy on Streamlit Community Cloud

    pip install -r requirements.txt
    streamlit run app.py
