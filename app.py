import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Konfigurasi
st.set_page_config(page_title="Prediksi Produksi", layout="wide")
st.title("🌾 Dashboard Analisis Produksi Pertanian")

# 1. LOAD MODEL
try:
    rf_model = joblib.load("models/rf_farmer_model.pkl")
    scaler = joblib.load("models/scaler_rf.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# 2. INPUT SIDEBAR
st.sidebar.header("Input Data Petani")
region = st.sidebar.selectbox("Region", ["Garut", "Indramayu", "Karawang", "Subang", "Tasikmalaya"])
land_area = st.sidebar.number_input("Land Area (m²)", 100.0, 50000.0, 1000.0)
seed_cost = st.sidebar.number_input("Seed Cost", 0.0, 1000000.0, 200000.0)
fertilizer_cost = st.sidebar.number_input("Fertilizer Cost", 0.0, 5000000.0, 500000.0)
labor_cost = st.sidebar.number_input("Labor Cost", 0.0, 5000000.0, 300000.0)
land_lease = st.sidebar.number_input("Land Lease", 0.0, 10000000.0, 1000000.0)
pesticide_cost = st.sidebar.number_input("Pesticide Cost", 0.0, 1000000.0, 100000.0)
equipment_cost = st.sidebar.number_input("Equipment Rent", 0.0, 2000000.0, 200000.0)

# 3. PERHITUNGAN FITUR
total_cost = land_lease + labor_cost + seed_cost + fertilizer_cost + pesticide_cost + equipment_cost
cost_per_m2 = total_cost / land_area
fertilizer_ratio = fertilizer_cost / land_area
labor_ratio = labor_cost / land_area
region_val = {"Garut": 0, "Indramayu": 1, "Karawang": 2, "Subang": 3, "Tasikmalaya": 4}[region]

# 4. PREDIKSI (TANPA NAMA KOLOM)
if st.button("🚀 Jalankan Prediksi"):
    # Urutan array harus SAMA PERSIS dengan urutan fitur saat training di Colab
    # Sesuaikan urutan di bawah ini jika hasil prediksi terasa aneh
    input_data = np.array([[
        land_area, land_lease, labor_cost, seed_cost, 
        fertilizer_cost, pesticide_cost, equipment_cost, 
        cost_per_m2, fertilizer_ratio, labor_ratio, region_val
    ]])
    
    try:
        # Transformasi
        X_scaled = scaler.transform(input_data)
        # Prediksi
        prediksi = rf_model.predict(X_scaled)[0]
        st.success(f"Hasil Prediksi Produksi: Rp {prediksi:,.2f}")
    except Exception as e:
        st.error(f"Error Inferensi: {e}")