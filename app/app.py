import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import os

# ============================================================
# Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title="CarPrice AI - Enterprise Dashboard",
    page_icon="🏎️",
    layout="wide"
)

OUTPUT_DIR = "outputs"


# ============================================================
# 1. Load model & semua hasil training dari notebook
#    (TIDAK ADA angka hardcode -- semua dibaca dari file hasil
#    01-Preprocessing.ipynb)
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(OUTPUT_DIR, "model_final.pkl"))


@st.cache_data
def load_feature_columns():
    with open(os.path.join(OUTPUT_DIR, "feature_columns.json")) as f:
        return json.load(f)


@st.cache_data
def load_kategori_dropdown():
    with open(os.path.join(OUTPUT_DIR, "kategori_dropdown.json")) as f:
        return json.load(f)


@st.cache_data
def load_model_info():
    with open(os.path.join(OUTPUT_DIR, "model_info.json")) as f:
        return json.load(f)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))


@st.cache_data
def load_feature_importance():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"))


@st.cache_data
def load_eda_summary():
    with open(os.path.join(OUTPUT_DIR, "eda_summary.json")) as f:
        return json.load(f)


@st.cache_data
def load_harga_per_merek():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "harga_per_merek.csv"), index_col=0)


@st.cache_data
def load_harga_per_kondisi():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "harga_per_kondisi.csv"), index_col=0)


@st.cache_data
def load_actual_vs_predicted():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "actual_vs_predicted.csv"))


REQUIRED_FILES = [
    "model_final.pkl", "feature_columns.json", "kategori_dropdown.json",
    "model_info.json", "model_comparison.csv", "feature_importance.csv",
    "eda_summary.json", "harga_per_merek.csv", "harga_per_kondisi.csv",
    "actual_vs_predicted.csv",
]
missing_files = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(OUTPUT_DIR, f))]

if missing_files:
    st.error(
        "⚠️ File hasil training belum lengkap di folder `outputs/`. "
        "Pastikan kamu sudah menjalankan **seluruh cell** di "
        "`01-Preprocessing.ipynb` sampai akhir ya bro!\n\n"
        f"File yang belum ada: {missing_files}"
    )
    st.stop()

try:
    model = load_model()
    feature_columns = load_feature_columns()
    kategori_dropdown = load_kategori_dropdown()
    model_info = load_model_info()
    df_komparasi = load_model_comparison()
    df_importance = load_feature_importance()
    eda_summary = load_eda_summary()
    harga_per_merek = load_harga_per_merek()
    harga_per_kondisi = load_harga_per_kondisi()
    df_avp = load_actual_vs_predicted()
except Exception as e:
    st.error(f"⚠️ Gagal memuat hasil training: {e}")
    st.stop()


# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
    <style>
    .academic-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 8px;
        border-top: 4px solid #FF4B4B;
        text-align: center;
        margin-bottom: 15px;
    }
    .badge-regresi {
        background-color: #00FFA2;
        color: #000;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B; margin-bottom: 0;'>🏎️ Sistem Prediksi Harga Mobil Bekas</h1>
""", unsafe_allow_html=True)
st.markdown(f"""
    <p style='text-align: center; font-size: 15px; color: #AAA;'>
        Implementasi Supervised Learning untuk Estimasi Nilai Kontinu Kendaraan
        <span class='badge-regresi'>REGRESI</span> &nbsp;|&nbsp; Model Final:
        <b style='color:#FF4B4B;'>{model_info['nama_model_terbaik']}</b>
    </p>
""", unsafe_allow_html=True)
st.write("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Kalkulator Prediksi AI",
    "📊 Exploratory Data Analysis (EDA)",
    "📈 Validasi Kinerja Model",
    "⚔️ Analisis Komparatif Algoritma"
])

# Mapping yang harus identik dengan notebook preprocessing
brand_map = {'toyota': 0, 'honda': 1, 'suzuki': 2, 'nissan': 3, 'ford': 4}
cond_map = {'excellent': 0, 'good': 1, 'fair': 2, 'poor': 3}
TAHUN_SEKARANG = 2026

# ============================================================
# TAB 1: KALKULATOR PREDIKSI AI
# ============================================================
with tab1:
    st.markdown("### 🎛️ Input Fitur Kendaraan")
    st.write("Silakan sesuaikan spesifikasi mobil di bawah untuk memprediksi perkiraan harga pasar.")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox(
            "Merek Kendaraan (Manufacturer)",
            list(brand_map.keys()), key="sb_brand"
        )
        year = st.slider(
            "Tahun Produksi (Year)", 2000, 2026, 2018, key="sl_year"
        )
        cylinders = st.selectbox(
            "Jumlah Silinder (Cylinders)",
            kategori_dropdown.get("cylinders", []), key="sb_cyl"
        )
    with col2:
        km = st.number_input(
            "Jarak Tempuh / Odometer (Mil)", min_value=0, max_value=500000,
            value=75000, step=5000, key="ni_km",
            help="Dataset asli menggunakan satuan mil (data Craigslist US)."
        )
        cond = st.selectbox(
            "Kondisi Fisik (Condition)", list(cond_map.keys()), key="sb_cond"
        )
        fuel = st.selectbox(
            "Jenis Bahan Bakar (Fuel)",
            kategori_dropdown.get("fuel", []), key="sb_fuel"
        )

    transmission = st.selectbox(
        "Jenis Transmisi (Transmission)",
        kategori_dropdown.get("transmission", []), key="sb_trans"
    )

    st.caption(
        "ℹ️ Odometer menggunakan satuan **mil**, mengikuti satuan asli pada dataset "
        "training (Craigslist, Amerika Serikat). Jika kamu punya angka km, "
        "kalikan dengan 0.621 untuk mengonversinya ke mil."
    )

    st.write("")
    col_btn, col_res = st.columns([1, 2])

    with col_btn:
        st.write("")
        pred_button = st.button("🚀 Hitung Estimasi Harga", use_container_width=True)

    with col_res:
        if pred_button:
            car_age = max(TAHUN_SEKARANG - year, 0)

            input_dict = {
                "brand_encoded": brand_map[brand],
                "year": year,
                "odometer": km,
                "cond_encoded": cond_map[cond],
                "car_age": car_age,
            }

            # Inisialisasi semua kolom one-hot ke 0, lalu set kolom yang sesuai input ke 1
            for col in feature_columns:
                if col not in input_dict:
                    input_dict[col] = 0

            cyl_col = f"cylinders_{cylinders}"
            fuel_col = f"fuel_{fuel}"
            trans_col = f"transmission_{transmission}"
            for col in [cyl_col, fuel_col, trans_col]:
                if col in input_dict:
                    input_dict[col] = 1

            input_data = pd.DataFrame([input_dict])[feature_columns]

            try:
                pred_usd = model.predict(input_data)[0]
                pred_idr = max(pred_usd * 15200, 12000000)

                st.markdown(f"""
                    <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B;'>
                        <p style='margin: 0; font-size: 14px; color: #AAA;'>HASIL PREDIKSI MODEL {model_info['nama_model_terbaik'].upper()}:</p>
                        <h2 style='margin: 0; color: #FF4B4B;'>Rp {int(pred_idr):,}</h2>
                        <small style='color: #777;'>*Estimasi harga (≈ ${pred_usd:,.0f} USD) berdasarkan model hasil training pada data historis.</small>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Gagal memprediksi: {e}")
        else:
            st.info("👈 Atur spesifikasi mobil, lalu klik tombol untuk melihat estimasi harga.")

# ============================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
with tab2:
    st.markdown("### 📊 Ringkasan Eksplorasi Data (EDA)")
    st.write("Wawasan yang diekstrak dari data hasil cleaning (lihat `01-Preprocessing.ipynb`), bukan data mentah.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Total Observasi Data (setelah cleaning)", value=f"{eda_summary['total_observasi']:,}")
    with m2:
        st.metric(label="Rentang Tahun Diolah", value=f"{eda_summary['tahun_min']} - {eda_summary['tahun_max']}")
    with m3:
        st.metric(label="Rata-rata Odometer", value=f"{eda_summary['rata_odometer']:,.0f} mil")
    with m4:
        st.metric(label="Merek Dominan", value=eda_summary["merek_dominan"].title())

    st.write("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### 🏷️ Rata-rata Harga per Merek (USD)")
        st.bar_chart(harga_per_merek, color="#FF4B4B")

    with col_g2:
        st.markdown("#### 📉 Rata-rata Harga Berdasarkan Kondisi Fisik (USD)")
        st.line_chart(harga_per_kondisi, color="#00FFA2")

    st.caption("Grafik di atas dihitung langsung dari hasil `groupby()` pada data bersih, bukan angka contoh.")

# ============================================================
# TAB 3: VALIDASI KINERJA MODEL
# ============================================================
with tab3:
    st.markdown("### 📈 Metrik Evaluasi Kinerja Model")
    st.write(f"Hasil evaluasi model final (**{model_info['nama_model_terbaik']}**) pada data testing (20% data, tidak pernah dilihat saat training).")

    baris_terbaik = df_komparasi[df_komparasi["model"] == model_info["nama_model_terbaik"]].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class='academic-card'>
                <h3 style='color:#AAA;margin:0;'>MAE</h3>
                <h2 style='color:#FF4B4B;'>{baris_terbaik['MAE']:,.0f}</h2>
                <p style='color:#777;margin:0;'>Rata-rata selisih error absolut (USD)</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class='academic-card'>
                <h3 style='color:#AAA;margin:0;'>RMSE</h3>
                <h2 style='color:#FFAA00;'>{baris_terbaik['RMSE']:,.0f}</h2>
                <p style='color:#777;margin:0;'>Sensitivitas error terhadap outlier (USD)</p>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class='academic-card'>
                <h3 style='color:#AAA;margin:0;'>R² Score</h3>
                <h2 style='color:#00FFA2;'>{baris_terbaik['R2']*100:.1f}%</h2>
                <p style='color:#777;margin:0;'>Variansi data yang dijelaskan model</p>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class='academic-card'>
                <h3 style='color:#AAA;margin:0;'>MAPE</h3>
                <h2 style='color:#00AEEF;'>{baris_terbaik['MAPE']:.1f}%</h2>
                <p style='color:#777;margin:0;'>Rata-rata error dalam persentase</p>
            </div>
        """, unsafe_allow_html=True)

    st.caption(
        f"Validasi tambahan: rata-rata R² 5-Fold Cross Validation pada data training = "
        f"**{model_info['cv_r2_mean']*100:.1f}%** (± {model_info['cv_r2_std']*100:.1f}%)."
    )

    st.write("---")
    st.markdown("### 📊 Analisis Grafis Sisaan (Residual & Calibration Plot)")
    st.caption("Plot di bawah memakai prediksi ASLI dari data test (`actual_vs_predicted.csv`), bukan data simulasi.")

    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.markdown("#### 1. Predicted vs Actual Plot (Kurva Kalibrasi)")
        calib_df = df_avp.rename(columns={"actual": "Nilai Asli (Actual)", "predicted": "Tebakan AI (Predicted)"})
        st.scatter_chart(calib_df, x="Nilai Asli (Actual)", y="Tebakan AI (Predicted)", color="#FF4B4B")
        st.caption("Titik yang mengikuti garis diagonal menandakan kalibrasi prediksi yang baik.")

    with col_plot2:
        st.markdown("#### 2. Residual Plot (Analisis Distribusi Error)")
        resid_df = df_avp.rename(columns={"predicted": "Predicted Value", "residual": "Residuals (Error)"})
        st.scatter_chart(resid_df, x="Predicted Value", y="Residuals (Error)", color="#FFAA00")
        st.caption("Error yang menyebar acak di sekitar nol menandakan tidak ada bias sistematis pada model.")

    st.write("---")
    st.markdown("### 🌟 Feature Importance (XGBoost)")
    fi_chart = df_importance.set_index("fitur")
    st.bar_chart(fi_chart, color="#FF4B4B")
    st.caption("Fitur dengan importance tertinggi memiliki kontribusi paling besar terhadap prediksi harga.")

# ============================================================
# TAB 4: ANALISIS KOMPARATIF ALGORITMA
# ============================================================
with tab4:
    st.markdown("### ⚔️ Tabel Perbandingan Performa Lintas Algoritma")
    st.write("Hasil komparasi nyata dari training 4 model pada data yang sama (`model_comparison.csv`), bukan angka contoh.")

    tampilan_df = df_komparasi.copy()
    tampilan_df["R² Score (%)"] = (tampilan_df["R2"] * 100).round(1).astype(str) + "%"
    tampilan_df = tampilan_df[["Rank", "model", "MAE", "RMSE", "R² Score (%)", "MAPE", "training_time_sec"]]
    tampilan_df.columns = ["Rank", "Algoritma / Model", "MAE", "RMSE", "R² Score (%)", "MAPE (%)", "Waktu Training (s)"]

    st.dataframe(tampilan_df, use_container_width=True, hide_index=True)
    st.write("")

    st.markdown("#### 📊 Grafik Perbandingan Perolehan R² Score (%)")
    chart_r2 = df_komparasi.set_index("model")[["R2"]] * 100
    chart_r2.columns = ["R2 Score (%)"]
    st.bar_chart(chart_r2, color="#00FFA2")

    model_terbaik = df_komparasi.iloc[0]
    st.success(
        f"💡 Kesimpulan Analisis: Berdasarkan uji komparasi di atas, algoritma "
        f"**{model_terbaik['model']}** dipilih sebagai model final karena menghasilkan "
        f"R² tertinggi ({model_terbaik['R2']*100:.1f}%) dengan MAE {model_terbaik['MAE']:,.0f} "
        f"dan RMSE {model_terbaik['RMSE']:,.0f} pada data testing."
    )