import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("📊 Sistem Pembentuk Kelompok Belajar Mahasiswa Berbasis Hybrid Fuzzy-KMeans")
st.markdown("""
Sistem ini membantu membentuk kelompok belajar mahasiswa secara **adil dan seimbang** berdasarkan nilai **IPK** dan **minat mata kuliah**.
Metode **Hybrid Fuzzy-KMeans** digunakan untuk memberikan bobot keanggotaan IPK secara fleksibel, sehingga pembagian kelompok mempertimbangkan baik kemampuan akademik maupun preferensi minat masing-masing mahasiswa.
""")

uploaded_file = st.file_uploader("Unggah file CSV (harus mengandung kolom: Nama, IPK, Major, Minor)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Normalisasi nama kolom
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={"nama": "Nama", "ipk": "IPK", "major": "Major", "minor": "Minor"}, inplace=True)

    opsi_perminatan = ["-- Pilih Mata Kuliah --", "Artificial Intelligence", "Jaringan Komputer", "Sistem Informasi"]
    nama_perminatan = st.selectbox("Pilih perminatan Mata Kuliah:", opsi_perminatan, index=0)

    if nama_perminatan == "-- Pilih Mata Kuliah --":
        st.warning("⚠️ Silakan pilih perminatan terlebih dahulu.")
    else:
        jumlah_anggota = st.number_input("Jumlah anggota per kelompok:", min_value=2, max_value=100, value=3)

        if st.button("🔍 Proses Pengelompokan"):

            # Keanggotaan perminatan
            def nilai_perminatan(row):
                if row["Major"] == nama_perminatan:
                    return 1.0
                elif row["Minor"] == nama_perminatan:
                    return 0.5
                return 0.0

            df["Nilai_Perminatan"] = df.apply(nilai_perminatan, axis=1)

            # Keanggotaan IPK dengan KMeans
            def fuzzy_ipk_with_kmeans(data, n_clusters=7):
                ipk_values = data["IPK"].values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                kmeans.fit(ipk_values)

                centroids = kmeans.cluster_centers_.flatten()
                sorted_centroids = sorted((val, idx) for idx, val in enumerate(centroids))
                fuzzy_grades = np.linspace(0.0, 1.0, n_clusters)
                cluster_to_fuzzy = {
                    idx: fuzzy_grades[i]
                    for i, (_, idx) in enumerate(sorted_centroids)
                }

                def get_fuzzy(ipk):
                    cluster = kmeans.predict([[ipk]])[0]
                    return cluster_to_fuzzy[cluster]

                return get_fuzzy

            fuzzy_ipk_func = fuzzy_ipk_with_kmeans(df)
            df["Fuzzy_IPK"] = df["IPK"].apply(fuzzy_ipk_func)

            # Fuzzy Total & Skor Gabungan (Supaya bisa di kelompokkan dengan ngeliat IPK juga)
            df["Fuzzy_Total"] = 0.5 * df["Fuzzy_IPK"] + 0.5 * df["Nilai_Perminatan"]
            df["Gabungan"] = (df["Fuzzy_Total"] + df["IPK"] / 4.0) / 2

            # Sort dan Distribusi Zigzag (Supaya ga nyatu)
            df_sorted = df.sort_values(by="Gabungan", ascending=False).reset_index(drop=True)
            jumlah_kelompok = int(np.ceil(len(df_sorted) / jumlah_anggota))
            kelompok = [[] for _ in range(jumlah_kelompok)]

            zigzag = True
            idx = 0
            while idx < len(df_sorted):
                urutan = range(jumlah_kelompok) if zigzag else reversed(range(jumlah_kelompok))
                for i in urutan:
                    if idx >= len(df_sorted):
                        break
                    kelompok[i].append(df_sorted.loc[idx])
                    idx += 1
                zigzag = not zigzag

            df_all = pd.concat([
                pd.DataFrame(group).assign(Kelompok=f"Kelompok {i+1}")
                for i, group in enumerate(kelompok)
            ], ignore_index=True)

            # Output Hasil
            st.subheader("📋 Hasil Pengelompokan")
            for i in range(1, jumlah_kelompok + 1):
                st.markdown(f"### 📖 Kelompok {i}")
                group_df = df_all[df_all["Kelompok"] == f"Kelompok {i}"].copy().reset_index(drop=True)
                group_df["No"] = range(1, len(group_df)+1)
                group_df["IPK"] = group_df["IPK"].round(2)
                group_df["Fuzzy_Total"] = group_df["Fuzzy_Total"].round(2)
                tampil_df = group_df[["No", "Nama", "IPK", "Fuzzy_Total", "Major", "Minor"]]
                st.dataframe(tampil_df, use_container_width=True, hide_index=True)

            # Visual Distribusi
            st.subheader("📈 Histogram")
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.hist(df["Fuzzy_IPK"], bins=10, color='skyblue', edgecolor='black')
                ax1.set_title("Distribusi Fuzzy IPK")
                ax1.set_xlabel("Fuzzy IPK")
                ax1.set_ylabel("Jumlah Mahasiswa")
                st.pyplot(fig1)
            with col2:
                fig2, ax2 = plt.subplots()
                ax2.hist(df["Fuzzy_Total"], bins=10, color='skyblue', edgecolor='black')
                ax2.set_title("Distribusi Fuzzy Total")
                ax2.set_xlabel("Fuzzy Total")
                ax2.set_ylabel("Jumlah Mahasiswa")
                st.pyplot(fig2)

            # Dsitribusi Pembagian Kelompok
            st.subheader("📊 Persebaran IPK dan Fuzzy Total per Kelompok")
            avg_df = df_all.groupby("Kelompok").agg(
                Rata_IPK=("IPK", "mean"),
                Rata_Fuzzy=("Fuzzy_Total", "mean")
            ).reset_index()
            avg_df["Kelompok_Num"] = avg_df["Kelompok"].str.extract("(\d+)").astype(int)
            avg_df = avg_df.sort_values(by="Kelompok_Num")

            col3, col4 = st.columns(2)
            with col3:
                fig3, ax3 = plt.subplots()
                sns.barplot(x="Kelompok_Num", y="Rata_IPK", data=avg_df, ax=ax3, palette="Blues_d")
                ax3.set_ylim(0, 4)
                ax3.set_title("Rata-rata IPK per Kelompok")
                ax3.set_xlabel("Kelompok")
                ax3.set_ylabel("Rata-rata IPK")
                st.pyplot(fig3)
            with col4:
                fig4, ax4 = plt.subplots()
                sns.barplot(x="Kelompok_Num", y="Rata_Fuzzy", data=avg_df, ax=ax4, palette="Greens_d")
                ax4.set_ylim(0, 1.0)
                ax4.set_title("Rata-rata Fuzzy Total")
                ax3.set_xlabel("Kelompok")
                ax3.set_ylabel("Rata-rata Fuzzy")
                st.pyplot(fig4)

            # Fitur liat hasil di CSV
            st.subheader("⬇️ Unduh Hasil")
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Unduh CSV", csv, "hasil_kelompok.csv", "text/csv")

else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
