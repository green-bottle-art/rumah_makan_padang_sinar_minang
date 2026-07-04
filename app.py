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


rf_model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler_rf.pkl")







