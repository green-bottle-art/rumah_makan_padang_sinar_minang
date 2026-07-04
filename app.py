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
rf_model = joblib.load("models/rf_model.pkl")
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





