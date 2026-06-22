import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load model yang udah dilatih di Langkah 1
@st.cache_resource
def load_model():
    return joblib.load('carprice_xgb.pkl')

try:
    model = load_model()
except:
    st.error("Model 'carprice_xgb.pkl' belum ditemukan. Pastikan sudah di-export dari notebook!")
    st.stop()

# 2. Judul Aplikasi Web
st.title('🚗 CarPrice AI - Prediksi Harga Mobil Bekas')
st.write("Masukkan spesifikasi mobil untuk mendapatkan estimasi harga terbaik.")

# 3. Form Input Komponen UI dari User
st.sidebar.header("Spesifikasi Mobil")

brand = st.sidebar.selectbox('Merek Kendaraan', ['toyota', 'honda', 'suzuki', 'nissan', 'ford'])
year = st.sidebar.slider('Tahun Produksi', 2000, 2026, 2018)
km = st.sidebar.number_input('Jarak Tempuh / Odometer (KM)', min_value=0, max_value=500000, value=75000)
cond = st.sidebar.selectbox('Kondisi Kendaraan', ['excellent', 'good', 'fair', 'poor'])

# 4. Tombol Aksi Prediksi
if st.sidebar.button('Prediksi Harga'):
    # Susun input menjadi DataFrame sesuai format fitur saat training model
    # (Sesuaikan nama kolom dengan fitur yang lo pakai di notebook!)
    input_data = pd.DataFrame([{
        'brand': brand,
        'year': year,
        'odometer': km,
        'condition': cond
    }])
    
    # Lakukan prediksi (output biasanya dalam USD berdasarkan dataset Kaggle lo)
    pred_usd = model.predict(input_data)[0]
    
    # Konversi ke Rupiah sesuai rencana (misal 1 USD = Rp 15.000)
    pred_idr = pred_usd * 15000
    
    # Tampilkan Hasil Ke Layar
    st.success("### Hasil Estimasi")
    st.metric(label="Estimasi Harga Pasar", value=f"Rp {pred_idr:,.0f}")