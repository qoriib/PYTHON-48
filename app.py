import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix
import random

# Judul Aplikasi
st.title("KMeans vs KMedoids")

# Sidebar untuk navigasi
st.sidebar.markdown("""
    # KMeans vs KMedoids
    Pilih menu dibawah.
""")

# Tentukan tahap yang aktif
if 'stage' not in st.session_state:
    st.session_state.stage = 'Upload Data'

# Tombol untuk berpindah antar bagian
data_upload_button = st.sidebar.button('Upload Data')
silhouette_score_button = st.sidebar.button('Silhouette Score')
clustering_button = st.sidebar.button('Clustering')
visualisasi_button = st.sidebar.button('Visualisasi Hasil')
hasil_akhir_button = st.sidebar.button('Hasil Clustering')

# Mengatur tombol yang bisa dipilih untuk berpindah ke tahap berikutnya
if data_upload_button:
    st.session_state.stage = 'Upload Data'
elif silhouette_score_button:
    st.session_state.stage = 'Silhouette Score'
elif clustering_button:
    st.session_state.stage = 'Clustering'
elif visualisasi_button:
    st.session_state.stage = 'Visualisasi Hasil'
elif hasil_akhir_button:
    st.session_state.stage = 'Hasil Clustering'

# Default Tab
tab = st.session_state.stage

# Upload file dan lakukan perhitungan hanya sekali setelah file diunggah
if tab == "Upload Data":
    uploaded_file = st.file_uploader("Unggah file Excel (format .xlsx)", type=["xlsx"])

    if uploaded_file:
        # Simpan file yang diunggah ke st.session_state untuk akses di seluruh aplikasi
        st.session_state.uploaded_file = uploaded_file
        
        # Baca data
        df = pd.read_excel(uploaded_file)
        
        st.subheader("Data Awal")
        st.write(df)

        # Ekstrak fitur termasuk fitur baru
        features_columns = ['Jumlah Kasus', 'Kepadatan Penduduk', 'Luas Wilayah (Km2)', 'Jumlah Fasilitas Kesehatan']
        features = df[features_columns]

        # Normalisasi
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(features)
        normalized_df = pd.DataFrame(normalized, columns=features_columns)

        st.subheader("Data Setelah Dinormalisasi")
        st.write(normalized_df)

        # Simpan normalized ke dalam session state untuk akses di seluruh aplikasi
        st.session_state.normalized = normalized

        # Simpan k_range untuk digunakan di tahap berikutnya
        k_range = range(2, 11)
        st.session_state.k_range = k_range

        # Lakukan perhitungan silhouette score dan clustering sekali setelah file diunggah
        sil_scores_kmeans = []
        sil_scores_kmedoids = []

        for k in k_range:
            # KMeans
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans_labels = kmeans.fit_predict(normalized)
            sil_scores_kmeans.append(silhouette_score(normalized, kmeans_labels))

            # KMedoids
            random.seed(42)
            initial_medoids = random.sample(range(len(normalized)), k)
            distance_matrix = calculate_distance_matrix(normalized.tolist())
            kmedoids_instance = kmedoids(data=distance_matrix, initial_index_medoids=initial_medoids, data_type='distance_matrix')
            kmedoids_instance.process()
            clusters = kmedoids_instance.get_clusters()

            kmedoids_labels = np.zeros(len(normalized), dtype=int)
            for idx, cluster in enumerate(clusters):
                for i in cluster:
                    kmedoids_labels[i] = idx
            sil_scores_kmedoids.append(silhouette_score(normalized, kmedoids_labels))

        # Simpan hasil perhitungan silhouette score di session_state
        st.session_state.sil_scores_kmeans = sil_scores_kmeans
        st.session_state.sil_scores_kmedoids = sil_scores_kmedoids

# Dalam stage "Silhouette Score"
if tab == "Silhouette Score":
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        # Ambil data dan silhouette score dari session_state
        uploaded_file = st.session_state.uploaded_file
        df = pd.read_excel(uploaded_file)
        
        # Ambil normalized dan k_range dari session_state
        normalized = st.session_state.normalized
        k_range = st.session_state.k_range
        sil_scores_kmeans = st.session_state.sil_scores_kmeans
        sil_scores_kmedoids = st.session_state.sil_scores_kmedoids
        
        # Cari k optimal untuk KMeans dan KMedoids
        optimal_k_kmeans = k_range[np.argmax(sil_scores_kmeans)]
        optimal_k_kmedoids = k_range[np.argmax(sil_scores_kmedoids)]
        
        # Tampilkan silhouette score untuk KMeans
        st.subheader("Silhouette Score: KMeans")
        fig_kmeans, ax_kmeans = plt.subplots(figsize=(10, 5))
        ax_kmeans.plot(k_range, sil_scores_kmeans, 'bo-', label='KMeans')
        ax_kmeans.set_title(f'Silhouette Score: KMeans (Optimal k = {optimal_k_kmeans})')
        ax_kmeans.set_xlabel('Jumlah Klaster (k)')
        ax_kmeans.set_ylabel('Silhouette Score')
        ax_kmeans.legend()
        ax_kmeans.grid(True)
        st.pyplot(fig_kmeans)

        # Tampilkan silhouette score untuk KMedoids
        st.subheader("Silhouette Score: KMedoids")
        fig_kmedoids, ax_kmedoids = plt.subplots(figsize=(10, 5))
        ax_kmedoids.plot(k_range, sil_scores_kmedoids, 'ro-', label='KMedoids')
        ax_kmedoids.set_title(f'Silhouette Score: KMedoids (Optimal k = {optimal_k_kmedoids})')
        ax_kmedoids.set_xlabel('Jumlah Klaster (k)')
        ax_kmedoids.set_ylabel('Silhouette Score')
        ax_kmedoids.legend()
        ax_kmedoids.grid(True)
        st.pyplot(fig_kmedoids)

        # Menampilkan nilai k optimal
        st.subheader("Jumlah cluster optimal")
        st.write(f"Jumlah cluster optimal untuk **KMeans**: {optimal_k_kmeans}")
        st.write(f"Jumlah cluster optimal untuk **KMedoids**: {optimal_k_kmedoids}")
    else:
        st.error("Silakan unggah data terlebih dahulu di tahap 'Upload Data'.")

if tab == "Clustering":
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        uploaded_file = st.session_state.uploaded_file
        df = pd.read_excel(uploaded_file)
        
        # Ambil normalized dan silhouette score dari session_state
        normalized = st.session_state.normalized
        k_range = st.session_state.k_range
        sil_scores_kmeans = st.session_state.sil_scores_kmeans
        sil_scores_kmedoids = st.session_state.sil_scores_kmedoids
        
        k_optimal_kmeans = k_range[np.argmax(sil_scores_kmeans)]
        k_optimal_kmedoids = k_range[np.argmax(sil_scores_kmedoids)]

        # --- Clustering KMeans with optimal k
        kmeans = KMeans(n_clusters=k_optimal_kmeans, random_state=42)
        kmeans_labels = kmeans.fit_predict(normalized)
        kmeans_dbi = davies_bouldin_score(normalized, kmeans_labels)

        # --- Clustering KMedoids with optimal k
        initial_medoids = random.sample(range(len(normalized)), k_optimal_kmedoids)
        distance_matrix = calculate_distance_matrix(normalized.tolist())
        medoids_instance = kmedoids(data=distance_matrix, initial_index_medoids=initial_medoids, data_type='distance_matrix')
        medoids_instance.process()
        kmedoids_clusters = medoids_instance.get_clusters()
        kmedoids_labels = np.zeros(len(normalized), dtype=int)
        for idx, cluster in enumerate(kmedoids_clusters):
            for i in cluster:
                kmedoids_labels[i] = idx
        kmedoids_dbi = davies_bouldin_score(normalized, kmedoids_labels)

        # Tambahkan label ke DataFrame
        df['Cluster_KMeans'] = kmeans_labels
        df['Cluster_KMedoids'] = kmedoids_labels

        # Simpan df ke dalam session_state agar bisa digunakan di tahap berikutnya
        st.session_state.df_with_labels = df

        # Tampilkan skor evaluasi
        st.subheader("Davies-Bouldin Index")
        st.write(f"**KMeans**: {kmeans_dbi:.4f}")
        st.write(f"**KMedoids**: {kmedoids_dbi:.4f}")

        # Setelah tahap clustering selesai, buka tahap berikutnya (Visualisasi Hasil)
        st.session_state.stage = 'Visualisasi Hasil'
    else:
        st.error("Silakan unggah data terlebih dahulu di tahap 'Upload Data'.")

# Dalam stage "Visualisasi Hasil"
if tab == "Visualisasi Hasil":
    if 'df_with_labels' in st.session_state:
        df = st.session_state.df_with_labels  # Ambil DataFrame dengan cluster labels
        
        # Ambil normalized dari session_state
        normalized = st.session_state.normalized

        # Ambil cluster labels dari session_state
        kmeans_labels = df['Cluster_KMeans']
        kmedoids_labels = df['Cluster_KMedoids']
        
        st.subheader("Visualisasi Hasil Clustering")

        # Visualisasi untuk KMeans
        fig_kmeans, ax_kmeans = plt.subplots(figsize=(10, 5))  # Set figure size for horizontal plot
        sns.scatterplot(x=normalized[:, 0], y=normalized[:, 1], hue=kmeans_labels, palette='Set2', ax=ax_kmeans)
        ax_kmeans.set_title("K-Means Clustering")
        ax_kmeans.set_xlabel("Jumlah Kasus")
        ax_kmeans.set_ylabel("Kepadatan")
        ax_kmeans.legend(title="Cluster")
        ax_kmeans.grid(True)
        st.pyplot(fig_kmeans)

        # Visualisasi untuk KMedoids
        fig_kmedoids, ax_kmedoids = plt.subplots(figsize=(10, 5))  # Set figure size for horizontal plot
        sns.scatterplot(x=normalized[:, 0], y=normalized[:, 1], hue=kmedoids_labels, palette='Set1', ax=ax_kmedoids)
        ax_kmedoids.set_title("K-Medoids Clustering")
        ax_kmedoids.set_xlabel("Jumlah Kasus")
        ax_kmedoids.set_ylabel("Kepadatan")
        ax_kmedoids.legend(title="Cluster")
        ax_kmedoids.grid(True)
        st.pyplot(fig_kmedoids)
    else:
        st.error("Silakan lakukan clustering data terlebih dahulu di tahap 'Clustering'.")

# Dalam stage "Hasil Clustering"
if tab == "Hasil Clustering":
    if 'df_with_labels' in st.session_state:
        df = st.session_state.df_with_labels  # Ambil DataFrame dengan cluster labels

        # Tampilkan Hasil Clustering per Kecamatan
        st.subheader("Hasil Clustering per Kecamatan")
        st.dataframe(df[['Kecamatan', 'Cluster_KMeans', 'Cluster_KMedoids']])
    else:
        st.error("Silakan lakukan clustering data terlebih dahulu di tahap 'Clustering'.")