"""
MODE TOKO (MOBILE) - VERSI FINAL
Dibuka lewat HP Android (browser Chrome), lalu "Tambahkan ke Layar Utama".

PENTING: Aplikasi ini secara DEFAULT memuat model final (model_final/*.pkl)
yang sudah dilatih SEKALI dari data asli Toko Sabrina dan dikunci
(random_state=42), sesuai rancangan Bab IV. Ini memastikan angka yang
ditampilkan di aplikasi SELALU SAMA dengan yang tertulis di naskah skripsi
(Bab V) - tidak ada lagi perbedaan hasil antar-run.

Retrain (melatih ulang model dengan data terbaru) hanya dilakukan jika
pengguna secara eksplisit menekan tombol "Latih Ulang Model" di menu
Pengaturan Lanjutan - bukan otomatis setiap kali dibuka.
"""
import streamlit as st

st.set_page_config(
    page_title="Dashboard Pakan Ternak Toko Sabrina",
    page_icon="app_icon_512.png",
    layout="wide"


)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from pipeline import load_final_model, run_full_pipeline, recursive_forecast, FEATURES

DATA_FILE = "data_penjualan_toko.xlsx"
MODEL_DIR = "model_final"
JENIS_PAKAN_LIST = ["Pakan Ayam", "Pakan Babi", "Pakan Bebek"]

PRIMARY = "#1E3A8A"      # biru tua - warna utama
ACCENT = "#2563EB"       # biru sedang - aksen/tombol
BG_SOFT = "#F8FAFC"      # abu-abu kebiruan sangat lembut - latar kartu

st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: 'Segoe UI', -apple-system, sans-serif; }}
.header-bar {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
    padding: 22px 24px; border-radius: 12px; margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(30,58,138,0.25);
}}
.header-title {{ color: #fff; font-size: 20px; font-weight: 800; margin: 0; letter-spacing: 0.3px; line-height: 1.3; }}
.header-subtitle {{ color: #cfe0ff; font-size: 12px; margin-top: 6px; font-weight: 500; }}
.big-number {{ font-size: 42px; font-weight: 800; color: {PRIMARY}; text-align:center; }}
.subtle {{ color: #64748b; font-size: 13.5px; text-align:center; }}
.card {{ background-color: {BG_SOFT}; padding: 22px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }}
div.stButton > button {{ height: 3em; font-size: 15px; font-weight: 600; border-radius: 8px; }}
div.stButton > button[kind="primary"] {{ background-color: {ACCENT}; border-color: {ACCENT}; }}
.badge {{ background:#dbeafe; color:{PRIMARY}; font-size:11px; padding:4px 12px; border-radius:20px; display:inline-block; font-weight:700; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
.app-footer {{ text-align:center; color:#94a3b8; font-size:11.5px; margin-top:32px; padding-top:14px; border-top:1px solid #e2e8f0; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header-bar">
    <p class="header-title">PREDIKSI TREN PENJUALAN<br>TOKO PAKAN TERNAK</p>
    <p class="header-subtitle">Toko Sabrina</p>
</div>
""", unsafe_allow_html=True)


def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_excel(DATA_FILE)
    return pd.DataFrame(columns=["tanggal_transaksi", "jenis_pakan", "nama_produk",
                                  "harga_produk", "jumlah_terjual", "berat_per_kemasan_kg",
                                  "total_penjualan"])


def save_transaction(tanggal, jenis, nama_produk, harga, jumlah, berat):
    df = load_data()
    new_row = pd.DataFrame([{
        "tanggal_transaksi": pd.Timestamp(tanggal), "jenis_pakan": jenis,
        "nama_produk": nama_produk, "harga_produk": harga,
        "jumlah_terjual": jumlah, "berat_per_kemasan_kg": berat,
        "total_penjualan": harga * jumlah,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(DATA_FILE, index=False)
    return df


tab1, tab2, tab3 = st.tabs(["Input Transaksi", "Hasil Prediksi", "Pengaturan Lanjutan"])

# ---------------- TAB 1: INPUT TRANSAKSI ----------------
with tab1:
    st.markdown("#### Tambah Transaksi Baru")
    st.caption("Isi setiap kali ada penjualan pakan — menggantikan catatan nota manual.")

    SATUAN_KG = {"Sak / Zak / Karung (50 kg)": 50, "Bungkus / Kg (1 kg)": 1}

    with st.form("form_transaksi", clear_on_submit=True):
        tanggal_input = st.date_input("Tanggal", value=pd.Timestamp.today())
        jenis_input = st.selectbox("Jenis Pakan", JENIS_PAKAN_LIST)
        nama_produk_input = st.text_input("Nama Produk", placeholder="contoh: BR1, Gold Coin, 511")
        harga_input = st.number_input("Harga per satuan (Rp)", min_value=0, step=500, value=0)
        jumlah_input = st.number_input("Jumlah terjual", min_value=1, step=1, value=1)
        satuan_pilihan = st.selectbox("Satuan Kemasan", list(SATUAN_KG.keys()))
        berat_input = SATUAN_KG[satuan_pilihan]
        if harga_input and jumlah_input:
            st.caption(f"Total penjualan: **Rp {harga_input * jumlah_input:,.0f}** "
                       f"({jumlah_input * berat_input} kg)")
        simpan = st.form_submit_button("Simpan Transaksi", use_container_width=True, type="primary")

    if simpan:
        save_transaction(tanggal_input, jenis_input, nama_produk_input, harga_input, jumlah_input, berat_input)
        st.success(f"Transaksi tersimpan: {jenis_input} ({nama_produk_input or '-'}), {jumlah_input} {satuan_pilihan.split(' (')[0].lower()} "
                   f"({jumlah_input * berat_input} kg), total Rp {harga_input * jumlah_input:,.0f}, "
                   f"tanggal {tanggal_input.strftime('%d %B %Y')}")

    st.markdown("---")
    df_now = load_data()
    st.markdown(f"#### Riwayat Transaksi Baru ({len(df_now)} catatan)")
    st.caption("Transaksi yang dicatat di sini terpisah dari data historis model final "
               "(2.398 transaksi), dan akan ikut dihitung saat kamu melatih ulang model di tab Pengaturan Lanjutan.")
    if len(df_now) > 0:
        tampil = df_now.sort_values("tanggal_transaksi", ascending=False).head(15).copy()
        tampil["tanggal_transaksi"] = pd.to_datetime(tampil["tanggal_transaksi"]).dt.strftime("%d %b %Y")
        tampil["total_kg"] = tampil["jumlah_terjual"] * tampil["berat_per_kemasan_kg"]
        kolom_tampil = ["tanggal_transaksi", "jenis_pakan", "nama_produk", "jumlah_terjual",
                        "total_kg", "total_penjualan"]
        kolom_tampil = [c for c in kolom_tampil if c in tampil.columns]
        st.dataframe(tampil[kolom_tampil], use_container_width=True, hide_index=True)
        with st.expander("Hapus transaksi terakhir (kalau salah input)"):
            if st.button("Hapus baris terakhir"):
                df_del = load_data().iloc[:-1]
                df_del.to_excel(DATA_FILE, index=False)
                st.success("Baris terakhir dihapus.")
                st.rerun()
    else:
        st.info("Belum ada transaksi baru. Isi form di atas untuk mulai mencatat.")

# ---------------- TAB 2: PERKIRAAN (memakai model FINAL, tidak retrain) ----------------
with tab2:
    if not os.path.exists(MODEL_DIR):
        st.error(f"Folder model final ('{MODEL_DIR}/') tidak ditemukan. "
                 f"Pastikan folder ini ikut di-upload bersama app_toko.py dan pipeline.py.")
        st.stop()

    final = load_final_model(MODEL_DIR)
    df_feat = final["df_feat"]
    best_name = final["best_model_name"]
    best_model = final["best_model"]
    metrics = final["metrics"][best_name]

    st.markdown(
        f'<span class="badge">Model: {best_name} (dilatih dari 2.398 transaksi data asli)</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    forecast = recursive_forecast(best_model, df_feat, 4, best_name)
    minggu_depan = forecast.iloc[0]
    rata2 = df_feat["total_kg"].tail(4).mean()
    selisih_pct = (minggu_depan["prediksi_kg"] - rata2) / rata2 * 100 if rata2 else 0

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="big-number">{minggu_depan["prediksi_kg"]:,.0f} kg</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtle">Perkiraan total pakan terjual minggu depan '
                 f'({pd.Timestamp(minggu_depan["minggu"]).strftime("%d %b %Y")})</div>',
                 unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("  ")
    if selisih_pct > 5:
        st.info(f"Naik sekitar {selisih_pct:.0f}% dari rata-rata 4 minggu terakhir — siapkan stok lebih banyak.")
    elif selisih_pct < -5:
        st.info(f"Turun sekitar {abs(selisih_pct):.0f}% dari rata-rata 4 minggu terakhir — stok bisa dikurangi.")
    else:
        st.info("Perkiraan stabil, mirip rata-rata beberapa minggu terakhir.")

    st.markdown("#### Perkiraan 4 Minggu ke Depan")
    tampil = forecast.copy()
    tampil["minggu"] = pd.to_datetime(tampil["minggu"]).dt.strftime("%d %b %Y")
    tampil.columns = ["Minggu", "Perkiraan (kg)"]
    st.dataframe(tampil, use_container_width=True, hide_index=True)

    fig = go.Figure()
    hist = df_feat.tail(10)
    fig.add_trace(go.Scatter(x=hist["minggu"], y=hist["total_kg"], mode="lines+markers",
                              name="Sebelumnya", line=dict(color=PRIMARY)))
    fig.add_trace(go.Scatter(x=pd.to_datetime(forecast["minggu"]), y=forecast["prediksi_kg"],
                              mode="lines+markers", name="Perkiraan", line=dict(color="#F59E0B", dash="dash")))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    pesan = (f"Perkiraan Kebutuhan Pakan - Toko Sabrina\n"
             f"Minggu {pd.Timestamp(minggu_depan['minggu']).strftime('%d %B %Y')}: "
             f"{minggu_depan['prediksi_kg']:,.0f} kg")
    st.text_area("Salin untuk kirim WhatsApp:", pesan, height=80)

    with st.expander("Tentang akurasi model"):
        st.write(
            f"Model {best_name} ini dievaluasi terhadap 23 minggu data historis yang tidak "
            f"dipakai saat pelatihan (data uji), dengan hasil:\n\n"
            f"- MAE: {metrics['MAE']:,.2f} kg\n"
            f"- RMSE: {metrics['RMSE']:,.2f} kg\n"
            f"- MAPE: {metrics['MAPE']:,.2f}%\n\n"
            f"Karena penjualan mingguan Toko Sabrina cukup fluktuatif, angka ini adalah "
            f"**perkiraan kasar** — tetap gunakan pengalaman kamu sebagai pemilik toko "
            f"sebagai pertimbangan tambahan, bukan satu-satunya acuan."
        )

# ---------------- TAB 3: PENGATURAN LANJUTAN (retrain eksplisit) ----------------
with tab3:
    st.markdown("#### Latih Ulang Model dengan Data Terbaru")
    st.caption(
        "Model yang dipakai di tab 'Hasil Prediksi' adalah model final yang sudah dilatih "
        "dari 2.398 transaksi data historis dan dikunci, supaya hasilnya selalu sama dengan "
        "yang tertulis di skripsi. Gunakan menu ini hanya jika ingin melatih ulang model "
        "dengan menggabungkan transaksi baru yang sudah dicatat di tab 'Input Transaksi' "
        "(misalnya setelah beberapa bulan berjalan dan data baru sudah cukup banyak)."
    )
    df_baru = load_data()
    st.write(f"Transaksi baru yang sudah dicatat: **{len(df_baru)}**")

    if st.button("Latih Ulang Model dengan Data Baru", type="secondary"):
        if len(df_baru) < 10:
            st.warning("Transaksi baru masih terlalu sedikit untuk melatih ulang secara bermakna "
                       "(disarankan menunggu minimal beberapa minggu data baru).")
        else:
            try:
                col_map = {"tanggal_transaksi": "tanggal_transaksi", "jenis_pakan": "jenis_pakan",
                           "jumlah_terjual": "jumlah_terjual"}
                with st.spinner("Melatih ulang model..."):
                    res = run_full_pipeline(df_baru, col_map, "berat_per_kemasan_kg", False, test_pct=0.2)
                st.success(f"Model berhasil dilatih ulang. Model terbaik saat ini: {res['best_model_name']}")
                st.json({k: {m: round(v, 2) for m, v in vv.items()} for k, vv in res["metrics"].items()})
                st.caption("Catatan: hasil retrain ini TIDAK otomatis menggantikan model final "
                           "yang tertulis di skripsi. Update naskah skripsi secara manual jika kamu "
                           "ingin memakai hasil retrain ini sebagai versi final baru.")
            except Exception as e:
                st.error(f"Gagal melatih ulang: {e}")

st.markdown('<div class="app-footer">Sistem Prediksi Tren Penjualan Toko Pakan Ternak &middot; by Elisa</div>',
            unsafe_allow_html=True)
