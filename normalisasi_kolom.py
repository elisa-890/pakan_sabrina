# =============================================================================
# TAMBAHKAN fungsi ini di app_toko.py, taruh SEBELUM baris "with tab1:"
# Fungsi ini menormalkan nama kolom supaya tidak pernah pecah jadi
# kolom duplikat lagi (misal "Nama Produk" vs "nama_produk").
# =============================================================================
STANDAR_KOLOM = ["Tanggal", "Jenis Pakan", "Nama Produk", "Jumlah Terjual", "Harga", "Total"]

# Pemetaan varian nama kolom lama -> nama kolom standar yang dipakai sekarang
PEMETAAN_KOLOM_LAMA = {
    "tanggal": "Tanggal", "tanggal_transaksi": "Tanggal",
    "jenis_pakan": "Jenis Pakan",
    "nama_produk": "Nama Produk",
    "jumlah_terjual": "Jumlah Terjual", "jumlah_kg": "Jumlah Terjual",
    "harga": "Harga",
    "total": "Total",
}

def normalisasi_data_transaksi(df):
    """Gabungkan kolom yang mungkin punya variasi nama (huruf besar/kecil,
    versi lama vs baru) menjadi satu set kolom standar."""
    df = df.rename(columns=PEMETAAN_KOLOM_LAMA)

    # Kalau ada kolom standar yang duplikat setelah rename, gabungkan
    # (ambil nilai yang tidak kosong)
    for kolom in STANDAR_KOLOM:
        kolom_terkait = [c for c in df.columns if c == kolom]
        if len(kolom_terkait) > 1:
            gabungan = df[kolom_terkait].bfill(axis=1).iloc[:, 0]
            df = df.drop(columns=kolom_terkait)
            df[kolom] = gabungan

    # Pastikan semua kolom standar ada, meski kosong
    for kolom in STANDAR_KOLOM:
        if kolom not in df.columns:
            df[kolom] = None

    return df[STANDAR_KOLOM]
