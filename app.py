import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Konfigurasi halaman biar dapet layout wide dan icon keren
st.set_page_config(
    page_title="CarPrice AI - Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide"
)

# 1. Load model dengan aman
@st.cache_resource
def load_model():
    return joblib.load('carprice_xgb.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ Model 'carprice_xgb.pkl' belum siap atau masih kosong. Selesaikan training di notebook dulu ya bro!")
    st.stop()

# 2. Header Aplikasi
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🚗 CarPrice AI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Sistem Estimasi Harga Mobil Bekas Akurat Berbasis Machine Learning (XGBoost)</p>", unsafe_allow_html=True)
st.write("---")

# 3. Pembagian Layout Input (Bukan di sidebar semua, biar luas)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Spesifikasi Utama")
    brand = st.selectbox('Merek Kendaraan', ['toyota', 'honda', 'suzuki', 'nissan', 'ford'])
    year = st.slider('Tahun Produksi', 2000, 2026, 2018)

with col2:
    st.markdown("### 🔧 Kondisi & Penggunaan")
    km = st.number_input('Jarak Tempuh / Odometer (KM)', min_value=0, max_value=500000, value=75000, step=5000)
    cond = st.selectbox('Kondisi Fisik Kendaraan', ['excellent', 'good', 'fair', 'poor'])

st.write("")
st.write("")

# 4. Tombol Aksi di Tengah
col_btn, col_res = st.columns([1, 2])

with col_btn:
    st.write("")
    st.write("")
    pred_button = st.button('🚀 Hitung Estimasi Harga', use_container_width=True)

with col_res:
    if pred_button:
        # PENTING: Proses konversi teks pilihan user menjadi angka (Mapping)
        # Harus sama persis dengan urutan angka saat training di Jupyter Notebook!
        brand_map = {'toyota': 0, 'honda': 1, 'suzuki': 2, 'nissan': 3, 'ford': 4}
        cond_map = {'excellent': 0, 'good': 1, 'fair': 2, 'poor': 3}
        
        brand_encoded = brand_map[brand]
        cond_encoded = cond_map[cond]
        
        # Susun DataFrame dengan nama kolom dan urutan yang SAMA PERSIS dengan di notebook
        input_data = pd.DataFrame([{
            'brand_encoded': brand_encoded,
            'year': year,
            'odometer': km,
            'cond_encoded': cond_encoded
        }])
        
        # Jalankan prediksi asli lewat model XGBoost
        try:
            pred_usd = model.predict(input_data)[0]
            pred_idr = pred_usd * 15200 # Kurs rupiah saat ini
            
            # Jika hasil prediksi bernilai minus (karena depresiasi ekstrem), kunci di batas bawah rasional
            if pred_idr < 0:
                pred_idr = 10000000 # Minimal Rp 10 Juta
            
            # Tampilan Box Hasil Mewah
            st.markdown("""
                <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B;'>
                    <p style='margin: 0; font-size: 14px; color: #AAA;'>ESTIMASI HARGA PASARAN</p>
                    <h2 style='margin: 0; color: #FFF;'>Rp {:,}</h2>
                    <small style='color: #777;'>*Harga dapat berubah tergantung kelengkapan surat dan pajak kendaraan.</small>
                </div>
            """.format(int(pred_idr)), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Gagal memprediksi. Eror: {e}")
            st.warning("Pastikan file 'carprice_xgb.pkl' yang lo push ke GitHub adalah file asli hasil export notebook terbaru!")