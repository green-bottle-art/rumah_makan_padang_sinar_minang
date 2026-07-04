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
land_lease = st.sidebar.number_input(
    "Land Lease Value (IDR)",
    min_value=0.0
)

pesticide_cost = st.sidebar.number_input(
    "Pesticide Cost (IDR)",
    min_value=0.0
)

equipment_cost = st.sidebar.number_input(
    "Equipment Rent Value (IDR)",
    min_value=0.0
)

REGION_MAP = {
    "Garut": 0,
    "Indramayu": 1,
    "Karawang": 2,
    "Subang": 3,
    "Tasikmalaya": 4
}




total_cost = (
    land_lease
    + labor_cost
    + seed_cost
    + fertilizer_cost
    + pesticide_cost
    + equipment_cost
)

cost_per_m2 = total_cost / land_area

fertilizer_ratio = fertilizer_cost / land_area

labor_ratio = labor_cost / land_area

df_input = pd.DataFrame({
    "Land area (m2)": [land_area],
    "Land lease value (IDR)": [land_lease],
    "Labor cost (IDR)": [labor_cost],
    "Seed purchase value (IDR)": [seed_cost],
    "Fertilizer purchase value (IDR)": [fertilizer_cost],
    "Pesticide purchase value (IDR)": [pesticide_cost],
    "Equipment rent value (IDR)": [equipment_cost],
    "region_encoded": [REGION_MAP[region]],
    "cost_per_m2": [cost_per_m2],
    "fertilizer_ratio": [fertilizer_ratio],
    "labor_ratio": [labor_ratio],
})

if st.button("Predict"):

    df_scaled = scaler.transform(df_input)

    prediction = rf_model.predict(df_scaled)

    st.metric(
        "Predicted Rice Production",
        f"Rp {prediction[0]:,.0f}"
    )
