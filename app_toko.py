"""
Dashboard Prediksi Tren Penjualan Pakan Ternak - Toko Sabrina
================================================================
Model GLM dan XGBoost sudah dilatih SEKALI dari data asli Toko Sabrina dan
dikunci (random_state=42), sesuai rancangan Bab IV. Ini memastikan angka yang
ditampilkan di aplikasi SELALU SAMA dengan yang tertulis di naskah skripsi
(Bab V) - tidak ada lagi perbedaan hasil antar-run.

Retrain (melatih ulang model dengan data terbaru) hanya dilakukan jika
pengguna secara eksplisit menekan tombol "Latih Ulang Model" di menu
Pengaturan Lanjutan - bukan otomatis setiap kali dibuka.

KONVERSI SATUAN (final, berdasarkan konfirmasi pemilik toko):
- 1 sak / karung / zak = 50 kg
- 1 bungkus            = 1 kg
- kg                   = kg (langsung, tanpa konversi)
"""

import streamlit as st

# =============================================================================
# WAJIB PALING ATAS: set_page_config HANYA BOLEH DIPANGGIL SEKALI DI SELURUH FILE
# =============================================================================
st.set_page_config(
    page_title="Dashboard Pakan Sabrina",
    page_icon="icon-192.png",
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

from pipeline import load_final_model, run_full_pipeline, recursive_forecast, FEATURES

DATA_FILE = "data_penjualan_toko.xlsx"
MODEL_DIR = "model_final"
JENIS_PAKAN_LIST = ["Pakan Ayam", "Pakan Babi", "Pakan Bebek"]

# Konversi satuan FINAL - satu-satunya sumber kebenaran, dipakai di seluruh file
SATUAN_KG = {"Sak/Karung": 50, "Bungkus": 1, "Kg": 1}


# =============================================================================
# CUSTOM CSS - tampilan lebih profesional
# =============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        html, body, [class*="css"] {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        }

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

        section[data-testid="stSidebar"] {
            background: #0F2A5C;
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            font-weight: 600;
        }
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
        <h1>Dashboard Pakan Sabrina</h1>
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
    st.markdown(f"### Halo, {nama_user}")
    st.caption("Toko Sabrina - Pakan Ternak")
    st.divider()
    authenticator.logout("Logout", location="sidebar")

st.markdown("""
<div class="app-banner">
    <h1>Prediksi Tren Penjualan Pakan Ternak Toko Sabrina</h1>
    <p>Dashboard digital pengganti pencatatan manual - akses kapan saja lewat HP Android.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Input Transaksi", "Hasil Prediksi", "Pengaturan Lanjutan"])


# =============================================================================
# NORMALISASI KOLOM - dipakai setiap kali membaca DATA_FILE, supaya data lama
# yang formatnya beda-beda tetap bisa ditampilkan rapi tanpa kolom duplikat.
# =============================================================================
STANDAR_KOLOM = ["Tanggal", "Jenis Pakan", "Nama Produk", "Jumlah Terjual",
                  "Satuan", "Berat per Satuan (kg)", "Total (kg)", "Harga", "Total (Rp)"]

PEMETAAN_KOLOM_LAMA = {
    "tanggal": "Tanggal", "tanggal_transaksi": "Tanggal",
    "jenis_pakan": "Jenis Pakan",
    "nama_produk": "Nama Produk",
    "jumlah_terjual": "Jumlah Terjual", "jumlah_kg": "Total (kg)",
    "satuan": "Satuan",
    "berat_per_kemasan_kg": "Berat per Satuan (kg)",
    "harga": "Harga", "harga_produk": "Harga",
    "total": "Total (Rp)", "total_penjualan": "Total (Rp)",
}


def normalisasi_data_transaksi(df):
    """Gabungkan kolom yang mungkin punya variasi nama (huruf besar/kecil,
    versi lama vs baru) menjadi satu set kolom standar, supaya tabel tidak
    pernah tampil kolom duplikat/None lagi."""
    df = df.rename(columns=PEMETAAN_KOLOM_LAMA)

    for kolom in STANDAR_KOLOM:
        kolom_terkait = [c for c in df.columns if c == kolom]
        if len(kolom_terkait) > 1:
            gabungan = df[kolom_terkait].bfill(axis=1).iloc[:, 0]
            df = df.drop(columns=kolom_terkait)
            df[kolom] = gabungan

    for kolom in STANDAR_KOLOM:
        if kolom not in df.columns:
            df[kolom] = None

    return df[STANDAR_KOLOM]


def load_data_transaksi():
    if os.path.exists(DATA_FILE):
        df_raw = pd.read_excel(DATA_FILE)
        return normalisasi_data_transaksi(df_raw)
    return pd.DataFrame(columns=STANDAR_KOLOM)


# -----------------------------------------------------------------------
# TAB 1 - INPUT TRANSAKSI
# -----------------------------------------------------------------------
with tab1:
    st.subheader("Tambah Transaksi Baru")
    st.caption("Isi setiap kali ada penjualan pakan - menggantikan catatan nota manual.")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            tanggal = st.date_input("Tanggal")
            jenis_pakan = st.selectbox("Jenis Pakan", JENIS_PAKAN_LIST)
            nama_produk = st.text_input("Nama Produk", placeholder="contoh: BR1, Gold Coin, 511")
        with col2:
            jumlah = st.number_input("Jumlah Terjual", min_value=0.0, step=1.0)
            satuan = st.selectbox("Satuan", list(SATUAN_KG.keys()))
            harga = st.number_input("Harga per Satuan (Rp)", min_value=0, step=1000)

        berat_per_satuan = SATUAN_KG[satuan]
        total_kg = jumlah * berat_per_satuan
        total_rp = jumlah * harga

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card" style="margin-top:10px;">
                <div class="label">Total Transaksi</div>
                <div class="value">Rp {total_rp:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card" style="margin-top:10px;">
                <div class="label">Setara Berat</div>
                <div class="value">{total_kg:,.0f} kg</div>
            </div>""", unsafe_allow_html=True)

        if st.button("Simpan Transaksi", use_container_width=True):
            new_row = pd.DataFrame([{
                "Tanggal": tanggal, "Jenis Pakan": jenis_pakan,
                "Nama Produk": nama_produk, "Jumlah Terjual": jumlah,
                "Satuan": satuan, "Berat per Satuan (kg)": berat_per_satuan,
                "Total (kg)": total_kg, "Harga": harga, "Total (Rp)": total_rp,
            }])
            if os.path.exists(DATA_FILE):
                existing = pd.read_excel(DATA_FILE)
                combined = pd.concat([existing, new_row], ignore_index=True)
            else:
                combined = new_row
            combined.to_excel(DATA_FILE, index=False)
            st.success("Transaksi berhasil disimpan!")
            st.rerun()

    st.divider()
    st.subheader("Riwayat Transaksi Terbaru")
    df_hist = load_data_transaksi()
    if len(df_hist) > 0:
        st.dataframe(df_hist.tail(15), use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada data transaksi tersimpan.")

# -----------------------------------------------------------------------
# TAB 2 - HASIL PREDIKSI
# -----------------------------------------------------------------------
with tab2:
    st.subheader("Prediksi Kebutuhan Pakan")

    try:
        loaded = load_final_model(MODEL_DIR)
        df_feat = loaded["df_feat"]
        best_model = loaded["best_model"]
        best_model_name = loaded["best_model_name"]
        metrics = loaded["metrics"][best_model_name]

        preds_df = recursive_forecast(
            best_model, df_feat, n_weeks=4, model_type=best_model_name
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
        st.subheader("Perkiraan 4 Minggu ke Depan")
        tampil = preds_df.copy()
        tampil["minggu"] = pd.to_datetime(tampil["minggu"]).dt.strftime("%d %b %Y")
        tampil.columns = ["Minggu", "Perkiraan (kg)"]
        st.dataframe(tampil, use_container_width=True, hide_index=True)

        st.subheader("Tren Penjualan Mingguan")
        fig = go.Figure()
        hist_tail = df_feat.tail(12)
        fig.add_trace(go.Scatter(
            x=hist_tail["minggu"], y=hist_tail["total_kg"],
            mode="lines+markers", name="Aktual",
            line=dict(color="#1E4C9A", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(preds_df["minggu"]), y=preds_df["prediksi_kg"],
            mode="lines+markers", name="Perkiraan",
            line=dict(color="#F59E0B", width=3, dash="dash"),
        ))
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Tentang akurasi model"):
            st.write(
                f"Model {best_model_name} dievaluasi terhadap data historis yang tidak "
                f"dipakai saat pelatihan, dengan hasil MAE {metrics.get('MAE', 0):,.2f} kg, "
                f"RMSE {metrics.get('RMSE', 0):,.2f} kg, MAPE {metrics.get('MAPE', 0):.2f}%. "
                f"Karena penjualan mingguan cukup fluktuatif, angka ini adalah perkiraan kasar - "
                f"tetap gunakan pengalaman sebagai pemilik toko sebagai pertimbangan tambahan."
            )

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

    with st.expander("Latih Ulang Model (Retrain)"):
        st.write(
            "Melatih ulang model GLM & XGBoost menggunakan data transaksi. "
            "Proses ini akan MENGUBAH angka yang ditampilkan di aplikasi, "
            "dan TIDAK otomatis mengubah naskah skripsi."
        )
        df_baru = load_data_transaksi()
        df_baru_valid = df_baru.dropna(subset=["Tanggal", "Jenis Pakan"])
        df_baru_valid["Tanggal"] = pd.to_datetime(df_baru_valid["Tanggal"])

        if len(df_baru_valid) > 0:
            tanggal_min = df_baru_valid["Tanggal"].min().date()
            tanggal_max = df_baru_valid["Tanggal"].max().date()
        else:
            tanggal_min = tanggal_max = pd.Timestamp.today().date()

        st.write(f"Total transaksi tersimpan: **{len(df_baru_valid)}** "
                 f"(dari {tanggal_min} sampai {tanggal_max})")

        mode_data = st.radio(
            "Data yang dipakai untuk melatih ulang:",
            ["Gunakan SEMUA data (historis + baru) - direkomendasikan",
             "Gunakan HANYA transaksi mulai tanggal tertentu"],
        )

        if mode_data.startswith("Gunakan HANYA"):
            tanggal_mulai = st.date_input(
                "Latih ulang hanya dengan transaksi mulai tanggal:",
                value=tanggal_max, min_value=tanggal_min, max_value=tanggal_max,
            )
            df_untuk_retrain = df_baru_valid[
                df_baru_valid["Tanggal"] >= pd.Timestamp(tanggal_mulai)
            ]
            st.caption(f"Akan memakai **{len(df_untuk_retrain)} transaksi** "
                       f"(mulai {tanggal_mulai}) untuk melatih ulang.")
        else:
            df_untuk_retrain = df_baru_valid
            st.caption(f"Akan memakai **semua {len(df_untuk_retrain)} transaksi** untuk melatih ulang.")

        confirm = st.checkbox("Saya paham dan ingin tetap melanjutkan")
        if st.button("Latih Ulang Model Sekarang", disabled=not confirm):
            if len(df_untuk_retrain) < 10:
                st.warning("Data yang dipilih terlalu sedikit untuk melatih ulang secara bermakna "
                           "(minimal butuh sekitar 10 minggu data mingguan yang valid).")
            else:
                try:
                    df_for_pipeline = df_untuk_retrain.rename(columns={
                        "Tanggal": "tanggal_transaksi",
                        "Jenis Pakan": "jenis_pakan",
                        "Jumlah Terjual": "jumlah_terjual",
                        "Berat per Satuan (kg)": "berat_per_kemasan_kg",
                    })
                    col_map = {"tanggal_transaksi": "tanggal_transaksi",
                               "jenis_pakan": "jenis_pakan",
                               "jumlah_terjual": "jumlah_terjual"}
                    with st.spinner("Melatih ulang model..."):
                        res = run_full_pipeline(df_for_pipeline, col_map,
                                                 "berat_per_kemasan_kg", False, test_pct=0.2)
                    st.success(f"Model berhasil dilatih ulang. Model terbaik: {res['best_model_name']}")
                    st.json({k: {m: round(v, 2) for m, v in vv.items()} for k, vv in res["metrics"].items()})
                    st.caption("Catatan: hasil ini TIDAK otomatis menggantikan model final di folder "
                               "model_final/. Ganti manual file .pkl kalau ingin memakai hasil ini.")
                except Exception as e:
                    st.error(f"Gagal melatih ulang: {e}")

    with st.expander("Kelola Pengguna"):
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
