    # ---------------------------------------------------------------------
    # TAB 1 - INPUT TRANSAKSI (versi dengan normalisasi kolom)
    # ---------------------------------------------------------------------
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
                satuan = st.selectbox("Satuan", ["kg", "sak", "zak", "bungkus"])
                harga = st.number_input("Harga per Satuan (Rp)", min_value=0, step=1000)

            total_rp = jumlah * harga
            st.markdown(f"""
            <div class="metric-card" style="margin-top:10px;">
                <div class="label">Total Transaksi</div>
                <div class="value">Rp {total_rp:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("💾 Simpan Transaksi", use_container_width=True):
                new_row = pd.DataFrame([{
                    "Tanggal": tanggal, "Jenis Pakan": jenis_pakan,
                    "Nama Produk": nama_produk, "Jumlah Terjual": f"{jumlah} {satuan}",
                    "Harga": harga, "Total": total_rp,
                }])
                if os.path.exists(DATA_FILE):
                    existing = pd.read_excel(DATA_FILE)
                    existing = normalisasi_data_transaksi(existing)
                    combined = pd.concat([existing, new_row], ignore_index=True)
                else:
                    combined = new_row
                combined = normalisasi_data_transaksi(combined)
                combined.to_excel(DATA_FILE, index=False)
                st.success("Transaksi berhasil disimpan!")
                st.rerun()

        st.divider()
        st.subheader("Riwayat Transaksi Terbaru")
        if os.path.exists(DATA_FILE):
            df_hist = pd.read_excel(DATA_FILE)
            df_hist = normalisasi_data_transaksi(df_hist)
            st.dataframe(df_hist.tail(15), use_container_width=True)
        else:
            st.info("Belum ada data transaksi tersimpan.")
