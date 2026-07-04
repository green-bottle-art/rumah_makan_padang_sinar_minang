import streamlit as  st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Hahahahaha",
    page_icon=":smiley:",
    layout="wide",
)

st.title("Hahahahaha")

# LOAD MODEL
rf_model = joblib.load("models/rf_farmer_model.pkl")
scaler = joblib.load("models/scaler_rf.pkl")

st.sidebar.header("Input Data")
region = st.sidebar.selectbox(
    "Region",
    [
        "Garut",
        "Indramayu",
        "Karawang",
        "Subang",
        "Tasikmalaya"
    ]
)

land_area = st.sidebar.number_input(
    "Land Area (in square meters)",
    min_value=100.0,
)
seed_cost = st.sidebar.number_input(
    "Seed Cost (Rupiah)",
)
fertilizer_cost = st.sidebar.number_input(
    "Fertilizer Cost (Rupiah)",
)

labor_cost = st.sidebar.number_input(
    "Labor Cost (Rupiah)",
)

REGION_MAP = {
    "Garut": 0,
    "Indramayu": 1,
    "Karawang": 2,
    "Subang": 3,
    "Tasikmalaya": 4
}

#### Prediksi
# x = np.array([[
#     land_area,
#     seed_cost,
#     fertilizer_cost,
#     labor_cost,
#     REGION_MAP[region]
# ]])

import pandas as pd

df_input = pd.DataFrame({
    "Land area (m2)": [land_area],
    "Land lease value (IDR)": [land_lease],
    "Labor cost (IDR)": [labor],
    "Seed purchase value (IDR)": [seed],
    "Fertilizer purchase value (IDR)": [fertilizer],
    "Pesticide purchase value (IDR)": [pesticide],
    "Equipment rent value (IDR)": [equipment],
    "region_encoded": [REGION_MAP[region]],
    "cost_per_m2": [cost_per_m2],
    "fertilizer_ratio": [fertilizer_ratio],
    "labor_ratio": [labor_ratio],
})

df_scaled = scaler.transform(df_input)

prediction = rf_model.predict(df_scaled)


# x = scaler.transform(x)
# prediction = rf_model.predict(x)


if st.button("Predict"):
    st.metric(
        "Predicted Rice Production",
        f"{prediction[0]:,.2f} kg"
    )

