"""
Dashboard Prediksi Tren Penjualan Pakan Ternak - Toko Sabrina
================================================================
CATATAN PENTING (dari versi sebelumnya):
Model GLM dan XGBoost sudah dilatih SEKALI dari data asli Toko Sabrina dan
dikunci (random_state=42), sesuai rancangan Bab IV. Ini memastikan angka yang
ditampilkan di aplikasi SELALU SAMA dengan yang tertulis di naskah skripsi
(Bab V) - tidak ada lagi perbedaan hasil antar-run.

Retrain (melatih ulang model dengan data terbaru) hanya dilakukan jika
pengguna secara eksplisit menekan tombol "Latih Ulang Model" di menu
Pengaturan Lanjutan - bukan otomatis setiap kali dibuka.
"""

import streamlit as st

# =============================================================================
# WAJIB PALING ATAS: set_page_config HANYA BOLEH DIPANGGIL SEKALI DI SELURUH FILE
# =============================================================================
st.set_page_config(
    page_title="Toko Sabrina",
    page_icon="app_icon_512.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from pipeline import load_final_model, run_full_pipeline, recursive_forecast, FEATURES# =============================================================================

DATA_FILE = "data_penjualan_toko.xlsx"
MODEL_DIR = "model_final"
JENIS_PAKAN_LIST = ["Pakan Ayam", "Pakan Babi", "Pakan Bebek"]


# =============================================================================
# CUSTOM CSS - tampilan lebih profesional
# =============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        /* Sembunyikan menu & footer bawaan Streamlit Cloud */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        html, body, [class*="css"] {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Banner header utama */
        .app-banner {
            background: linear-gradient(135deg, #0F2A5C 0%, #1E4C9A 60%, #2E6BC9 100%);
            padding: 28px 32px;
            border-radius: 18px;
            margin-bottom: 28px;
            box-shadow: 0 8px 24px rgba(15, 42, 92, 0.25);
        }
        .app-banner h1 {
            color: white;
            font-size: 26px;
            font-weight: 800;
            margin: 0 0 4px 0;
            line-height: 1.3;
        }
        .app-banner p {
            color: #BFD4F5;
            font-size: 15px;
            margin: 0;
        }

        /* Kartu metrik */
        .metric-card {
            background: white;
            border-radius: 14px;
            padding: 20px 22px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            border: 1px solid #EEF1F6;
        }
        .metric-card .label {
            font-size: 13px;
            color: #6B7280;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        .metric-card .value {
            font-size: 30px;
            font-weight: 800;
            color: #0F2A5C;
            margin-top: 4px;
        }
        .metric-card .sub {
            font-size: 12.5px;
            color: #9AA5B1;
            margin-top: 2px;
        }

        /* Tombol utama */
        div.stButton > button {
            background: linear-gradient(135deg, #1E4C9A, #2E6BC9);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.55em 1.4em;
            font-weight: 600;
            transition: all 0.15s ease;
        }
        div.stButton > button:hover {
            box-shadow: 0 4px 14px rgba(30, 76, 154, 0.35);
            transform: translateY(-1px);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0F2A5C;
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            font-weight: 600;
        }

        /* Sembunyikan badge "Built with Streamlit" di pojok bawah */
        [data-testid="stStatusWidget"] { visibility: hidden; }
        div[class*="viewerBadge"] { display: none !important; }
        .stAppDeployButton { display: none !important; }
        a[href*="streamlit.io"] { display: none !important; }
        iframe[title="streamlitApp"] + div { display: none !important; }
        
    </style>
    """, unsafe_allow_html=True)


inject_custom_css()

# =============================================================================
# LOGIN - streamlit-authenticator (username + password, multi-user)
# =============================================================================
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login(location="main")

auth_status = st.session_state.get("authentication_status")

if auth_status is False:
    st.error("Username atau password salah. Silakan coba lagi.")
    st.stop()
elif auth_status is None:
    st.markdown("""
    <div class="app-banner">
        <h1>🌾 Dashboard Penjualan Pakan Ternak Toko Sabrina</h1>
        <p>Silakan login untuk mengakses prediksi tren penjualan &amp; input transaksi.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("Masukkan username dan password pada form di atas untuk melanjutkan.")
    st.stop()

# =============================================================================
# Dari titik ini, pengguna SUDAH LOGIN (auth_status is True)
# =============================================================================
nama_user = st.session_state.get("name", "Pengguna")

with st.sidebar:
    st.markdown(f"### 👋 Halo, {nama_user}")
    st.caption("Toko Sabrina - Pakan Ternak")
    st.divider()
    authenticator.logout("Logout", location="sidebar")

st.markdown("""
<div class="app-banner">
    <h1>Prediksi Tren Penjualan Pakan Ternak Toko Sabrina</h1>
    <p>Dashboard digital pengganti pencatatan manual - akses kapan saja lewat HP Android.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📝 Input Transaksi", "📈 Hasil Prediksi", "⚙️ Pengaturan Lanjutan"])
            
# ---------------------------------------------------------------------
# TAB 1 - INPUT TRANSAKSI (kolom rapi: Tanggal Transaksi, Jenis Pakan,
# Nama Produk, Jumlah, Harga, Total)
# Konversi satuan mengikuti hasil konfirmasi pemilik toko:
# 1 sak = 50 kg, 1 zak = 50 kg, 1 karung = 50 kg, 1 bungkus = 1 kg
# ---------------------------------------------------------------------
KONVERSI_KG = {"kg": 1, "sak": 50, "zak": 50, "karung": 50, "bungkus": 1}

with tab1:
    st.subheader("Tambah Transaksi Baru")
    st.caption("Isi setiap kali ada penjualan pakan - menggantikan catatan nota manual.")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            tanggal = st.date_input("Tanggal Transaksi")
            jenis_pakan = st.selectbox("Jenis Pakan", JENIS_PAKAN_LIST)
            nama_produk = st.text_input("Nama Produk", placeholder="contoh: BR1, Gold Coin, 511")
        with col2:
            jumlah = st.number_input("Jumlah Terjual", min_value=0.0, step=1.0)
            satuan = st.selectbox("Satuan", ["kg", "sak", "zak", "karung", "bungkus"])
            harga = st.number_input("Harga per Satuan (Rp)", min_value=0, step=1000)

        jumlah_kg = jumlah * KONVERSI_KG[satuan]
        total_rp = jumlah * harga

        st.markdown(f"""
        <div class="metric-card" style="margin-top:10px;">
            <div class="label">Total Transaksi</div>
            <div class="value">Rp {total_rp:,.0f}</div>
            <div class="sub">Setara {jumlah_kg:,.1f} kg</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("💾 Simpan Transaksi", use_container_width=True):
            new_row = pd.DataFrame([{
                "Tanggal Transaksi": tanggal,
                "Jenis Pakan": jenis_pakan,
                "Nama Produk": nama_produk,
                "Jumlah": f"{jumlah:g} {satuan}",
                "Harga": harga,
                "Total": total_rp,
            }])
            if os.path.exists(DATA_FILE):
                existing = pd.read_excel(DATA_FILE)
                combined = pd.concat([existing, new_row], ignore_index=True)
            else:
                combined = new_row
            # Pastikan urutan & nama kolom selalu konsisten
            combined = combined.reindex(columns=[
                "Tanggal Transaksi", "Jenis Pakan", "Nama Produk",
                "Jumlah", "Harga", "Total",
            ])
            combined.to_excel(DATA_FILE, index=False)
            st.success("Transaksi berhasil disimpan!")
            st.rerun()

    st.divider()
    st.subheader("Riwayat Transaksi Terbaru")
    if os.path.exists(DATA_FILE):
        df_hist = pd.read_excel(DATA_FILE)
        df_hist = df_hist.reindex(columns=[
            "Tanggal Transaksi", "Jenis Pakan", "Nama Produk",
            "Harga", "Jumlah", "Total",
        ])
        st.dataframe(df_hist.tail(15), use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada data transaksi tersimpan.")
            
    # ---------------------------------------------------------------------
    # TAB 2 - HASIL PREDIKSI
    # ---------------------------------------------------------------------
    with tab2:
        st.subheader("Prediksi Kebutuhan Pakan Minggu Depan")

        try:
            loaded = load_final_model(MODEL_DIR)
            df_feat = loaded["df_feat"]
            best_model = loaded["best_model"]
            best_model_name = loaded["best_model_name"]
            metrics = loaded["metrics"][best_model_name]

            preds_df = recursive_forecast(
                best_model, df_feat, n_weeks=1, model_type=best_model_name
            )
            pred_next = preds_df["prediksi_kg"].iloc[0]

            colA, colB, colC = st.columns(3)
            with colA:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Prediksi Minggu Depan</div>
                    <div class="value">{pred_next:,.0f} kg</div>
                    <div class="sub">Model: {best_model_name}</div>
                </div>""", unsafe_allow_html=True)
            with colB:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">MAPE (Akurasi Model)</div>
                    <div class="value">{metrics.get('MAPE', 0):.1f}%</div>
                    <div class="sub">Semakin rendah semakin akurat</div>
                </div>""", unsafe_allow_html=True)
            with colC:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">RMSE</div>
                    <div class="value">{metrics.get('RMSE', 0):,.0f} kg</div>
                    <div class="sub">Rata-rata kesalahan prediksi</div>
                </div>""", unsafe_allow_html=True)

            st.divider()
            st.subheader("Tren Penjualan Mingguan")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_feat["minggu"], y=df_feat["total_kg"],
                mode="lines+markers", name="Aktual",
                line=dict(color="#1E4C9A", width=3),
            ))
            fig.update_layout(
                height=380, margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(
                "Model atau data belum siap. Pastikan folder "
                f"'{MODEL_DIR}' berisi glm_coef.pkl, xgb_model.pkl, "
                "metrics.pkl, dan df_feat.csv."
            )
            st.exception(e)
        
# -----------------------------------------------------------------------
# TAB 3 - PENGATURAN LANJUTAN
# -----------------------------------------------------------------------
with tab3:
    st.subheader("Pengaturan Lanjutan")
    st.caption("Fitur ini untuk admin/peneliti - gunakan dengan hati-hati.")

    with st.expander("🔄 Latih Ulang Model (Retrain)"):
        st.write(
            "Melatih ulang model GLM & XGBoost menggunakan seluruh data transaksi "
            "terbaru. Proses ini akan MENGUBAH angka yang ditampilkan di aplikasi."
        )
        confirm = st.checkbox("Saya paham dan ingin tetap melanjutkan")
        if st.button("Latih Ulang Model Sekarang", disabled=not confirm):
            with st.spinner("Melatih ulang model..."):
                st.info("Fungsi retrain perlu dihubungkan ke pipeline.py milikmu.")

    with st.expander("👤 Kelola Pengguna"):
        st.write("Edit langsung file `config.yaml` di GitHub untuk menambah/menghapus pengguna.")
        st.code(
            "usernames:\n"
            "  username_baru:\n"
            "    email: contoh@email.com\n"
            "    first_name: Nama\n"
            "    last_name: Belakang\n"
            "    password: passwordbaru123",
            language="yaml",
        )
