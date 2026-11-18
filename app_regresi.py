# TUGAS PRAKTIKUM ALJABAR MATRIKS - MODUL 8
# Topik: Proyeksi Ortogonal & Curve Fitting Regresi

import streamlit as st
import pandas as pd
import numpy as np

# Set layout halaman web
st.set_page_config(page_title="Modul 8 - Aljabar Matriks", layout="wide")
st.title("Praktikum Aljabar Matriks (Modul 8)")
st.caption("Eksplorasi Proyeksi Ortogonal, Pembuktian Geometri, & Regresi Linear via Normal Equation")

# Menu navigasi samping
pilihan_menu = st.sidebar.selectbox(
    "Pilih Menu Praktikum:", 
    ["1. Bukti Geometri (Soal No. 3)", "2. Curve Fitting Regresi Mandiri (Soal No. 4 & 5)"]
)

# Fungsi utama buat hitung regresi dan proyeksi
def hitung_regresi_matriks(fitur_x, target_y):
    arr_x = np.asarray(fitur_x, dtype=float)
    arr_y = np.asarray(target_y, dtype=float)
    
    # Bikin matriks X dengan kolom angka 1 di kiri untuk intersep w0
    matriks_X = np.column_stack([np.ones(len(arr_x)), arr_x])   
    
    # Cari bobot w pake normal equation: (X^T * X) * w = X^T * y
    koef_w = np.linalg.solve(matriks_X.T @ matriks_X, matriks_X.T @ arr_y)       
    
    vektor_p = matriks_X @ koef_w
    vektor_galat = arr_y - vektor_p
    
    return matriks_X, koef_w, vektor_p, vektor_galat


# Menu 1: Bukti Geometri (Soal No. 3)
if pilihan_menu == "1. Bukti Geometri (Soal No. 3)":
    st.subheader("Pembuktian Geometri Proyeksi & Vektor Ortogonal")
    st.write("Bagian ini membuktikan perhitungan geometri menggunakan titik data (1, 1), (2, 3), dan (3, 4) sesuai petunjuk modul.")

    if st.button("Mulai Proses Pembuktian", type="primary"):
        x_titik = [1, 2, 3]
        y_titik = [1, 3, 4]

        X, w, P, e = hitung_regresi_matriks(x_titik, y_titik)
        basis_c1, basis_c2 = X[:, 0], X[:, 1]

        kol1, kol2 = st.columns(2)
        with kol1:
            st.write("**Matriks Desain X (Basis c1 & c2)**")
            st.dataframe(pd.DataFrame(X, columns=["c1 (Intersep)", "c2 (Data x)"]))
        with kol2:
            st.write("**Vektor Target y**")
            st.dataframe(pd.DataFrame(y_titik, columns=["y"]))

        st.subheader("Hasil Solusi Analitis")
        kol3, kol4 = st.columns(2)
        with kol3:
            st.write("**Koefisien Bobot (w)**")
            st.dataframe(pd.DataFrame({"Parameter": ["w0", "w1"], "Nilai": np.round(w, 4)}), hide_index=True)
        with kol4:
            st.write("**Titik Proyeksi P & Vektor Galat e**")
            st.dataframe(pd.DataFrame({"Titik P": np.round(P, 4), "Galat e": np.round(e, 4)}), hide_index=True)

        st.subheader("Uji Ortogonalitas (Dot Product)")
        cek_c1_e = basis_c1 @ e
        cek_c2_e = basis_c2 @ e
        
        m_a, m_b = st.columns(2)
        m_a.metric("Hasil c1 . e", f"{cek_c1_e:.2e}")
        m_b.metric("Hasil c2 . e", f"{cek_c2_e:.2e}")

        v_1 = basis_c1.copy()
        v_2 = basis_c2 - (basis_c2 @ v_1) / (v_1 @ v_1) * v_1
        y_arr = np.array(y_titik, dtype=float)
        P_gs = (y_arr @ v_1) / (v_1 @ v_1) * v_1 + (y_arr @ v_2) / (v_2 @ v_2) * v_2

        st.success(f"Apakah hasil Normal Equation konsisten dengan metode Gram-Schmidt? **{np.allclose(P_gs, P)}**")


# Menu 2: Curve Fitting & Regresi (Soal No. 4 & 5)
else:
    st.subheader("Curve Fitting Regresi Linear & Analisis Residual")
    st.write("Gunakan dataset bawaan studi kasus jam belajar atau unggah file CSV/Excel milikmu sendiri.")

    opsi_sumber = st.radio("Pilih Sumber Data:", ["Pakai Data Bawaan (Jam Belajar vs Nilai)", "Upload File (CSV / Excel)"])

    data_frame = None
    if opsi_sumber == "Upload File (CSV / Excel)":
        berkas_user = st.file_uploader("Upload file datamu di sini:", type=["csv", "xlsx", "xls"])
        if berkas_user is not None:
            try:
                if berkas_user.name.lower().endswith(".csv"):
                    data_frame = pd.read_csv(berkas_user, sep=None, engine="python")
                else:
                    data_frame = pd.read_excel(berkas_user)
            except Exception as err:
                st.error(f"Gagal baca file: {err}")
                st.stop()
    else:
        data_frame = pd.DataFrame({
            "jam_belajar": [1, 2, 3, 4, 5, 6, 7, 8],
            "nilai_ujian": [52, 55, 61, 64, 70, 72, 79, 83]
        })

    if data_frame is not None:
        st.subheader("1. Preview Tabel Data")
        st.dataframe(data_frame)

        kolom_angka = data_frame.select_dtypes(include="number").columns.tolist()
        if len(kolom_angka) < 2:
            st.error("Minimal harus ada 2 kolom numerik pada dataset.")
            st.stop()

        st.subheader("2. Pengaturan Variabel")
        kc1, kc2 = st.columns(2)
        with kc1:
            target_kol = st.selectbox("Kolom Target (y)", kolom_angka, index=len(kolom_angka)-1)
        with kc2:
            default_f = [kol for kol in kolom_angka if kol != target_kol]
            fitur_kol = st.multiselect("Kolom Fitur (x)", [kol for kol in kolom_angka if kol != target_kol], default=default_f)

        pakai_w0 = st.checkbox("Sertakan Intersep (w0)", value=True)

        if st.button("Hitung Model Regresi", type="primary"):
            try:
                if not fitur_kol:
                    raise ValueError("Pilih minimal satu kolom fitur!")
                
                sub_data = data_frame[fitur_kol + [target_kol]].apply(pd.to_numeric, errors="raise")
                X_fitur = sub_data[fitur_kol].to_numpy(dtype=float)
                y_target = sub_data[target_kol].to_numpy(dtype=float)

                nama_parameter = [f"w{i+1}" for i in range(len(fitur_kol))]
                if pakai_w0:
                    X_fitur = np.column_stack([np.ones(len(X_fitur)), X_fitur])
                    nama_parameter = ["w0"] + nama_parameter

                XtX_mat = X_fitur.T @ X_fitur
                Xty_mat = X_fitur.T @ y_target
                bobot_w = np.linalg.solve(XtX_mat, Xty_mat)
                bobot_w = bobot_w + 0.0

                st.session_state["cache_hasil"] = {
                    "X": X_fitur, "y": y_target, "w": bobot_w, "nama_w": nama_parameter, 
                    "fitur": fitur_kol, "target": target_kol, "intersep": pakai_w0
                }
            except Exception as ex:
                st.error(f"Terjadi kendala: {ex}")

        cache = st.session_state.get("cache_hasil")
        if cache:
            X, y, w, nama_w = cache["X"], cache["y"], cache["w"], cache["nama_w"]

            st.subheader("3. Nilai Parameter Bobot (w)")
            st.dataframe(pd.DataFrame({"Parameter": nama_w, "Nilai": np.round(w, 4)}), hide_index=True)

            daftar_suku = []
            kumpulan_fitur = ([None] + cache["fitur"]) if cache["intersep"] else cache["fitur"]
            for label, val, ft in zip(nama_w, w, kumpulan_fitur):
                daftar_suku.append(f"{val:.4f}" if ft is None else f"{val:.4f}·{ft}")
            st.info(f"Persamaan Terbentuk: {cache['target']} = " + " + ".join(daftar_suku))

            y_prediksi = X @ w
            sisa_galat = y - y_prediksi
            sum_sq_error = float(np.sum(sisa_galat ** 2))
            sum_sq_total = float(np.sum((y - y.mean()) ** 2))
            r_kuadrat = 1 - sum_sq_error / sum_sq_total if sum_sq_total > 0 else float("nan")

            st.subheader("4. Analisis Residual & Galat (Soal No. 5)")
            r1, r2, r3 = st.columns(3)
            r1.metric("RMSE", f"{np.sqrt(np.mean(sisa_galat ** 2)):.4f}")
            r2.metric("R² Score", f"{r_kuadrat:.4f}")
            r3.metric("Total Penjumlahan Galat", f"{np.sum(sisa_galat):.2e}")

            st.write("Penjelasan Singkat: Kenapa total galat bernilai nol? Karena adanya intersep (w0), vektor galat (e) dipastikan tegak lurus dengan kolom vektor basis pertama c1 (yang isinya angka 1). Akibatnya, hasil dot product c1 . e bernilai 0, yang secara aljabar sama persis dengan menjumlahkan seluruh elemen galat.")

            st.write("**Tabel Perbandingan Aktual vs Prediksi & Galat**")
            tabel_hasil = pd.DataFrame({
                "Aktual": y, 
                "Prediksi": np.round(y_prediksi, 4), 
                "Galat (e)": np.round(sisa_galat, 4)
            })
            st.dataframe(tabel_hasil)

            st.download_button(
                "Unduh Hasil Prediksi (.csv)",
                tabel_hasil.to_csv(index=False).encode("utf-8"),
                file_name="hasil_regresi_modul8.csv", mime="text/csv"
            )

            st.subheader("5. Simulasi Prediksi Data Baru")
            input_nilai = []
            kol_input = st.columns(len(cache["fitur"]))
            for kotak_input, nama_kolom in zip(kol_input, cache["fitur"]):
                input_nilai.append(kotak_input.number_input(f"Masukkan {nama_kolom}", value=float(data_frame[nama_kolom].mean())))
            
            vektor_baru = np.array(([1.0] if cache["intersep"] else []) + input_nilai)
            hasil_akhir = float(vektor_baru @ w)
            st.success(f"Estimasi {cache['target']}: {hasil_akhir:.2f}")
