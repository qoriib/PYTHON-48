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
    Pilih bagian untuk melanjutkan.
""")

# Tentukan tahap yang aktif
if 'stage' not in st.session_state:
    st.session_state.stage = 'Data Upload'

# Tombol untuk berpindah antar bagian
data_upload_button = st.sidebar.button('Data Upload', disabled=st.session_state.stage != 'Data Upload')
silhouette_score_button = st.sidebar.button('Perbandingan Silhouette Score', disabled=st.session_state.stage not in ['Perbandingan Silhouette Score', 'Data Upload'])
clustering_button = st.sidebar.button('Clustering', disabled=st.session_state.stage not in ['Clustering', 'Perbandingan Silhouette Score'])
visualisasi_button = st.sidebar.button('Visualisasi Hasil', disabled=st.session_state.stage not in ['Visualisasi Hasil', 'Clustering'])
hasil_akhir_button = st.sidebar.button('Hasil Akhir', disabled=st.session_state.stage not in ['Hasil Akhir', 'Visualisasi Hasil'])

# Mengatur tombol yang bisa dipilih untuk berpindah ke tahap berikutnya
if data_upload_button:
    st.session_state.stage = 'Data Upload'
elif silhouette_score_button:
    st.session_state.stage = 'Perbandingan Silhouette Score'
elif clustering_button:
    st.session_state.stage = 'Clustering'
elif visualisasi_button:
    st.session_state.stage = 'Visualisasi Hasil'
elif hasil_akhir_button:
    st.session_state.stage = 'Hasil Akhir'

# Default Tab
tab = st.session_state.stage

# Upload file
if tab == "Data Upload":
    uploaded_file = st.file_uploader("Unggah file Excel (format .xlsx)", type=["xlsx"])

    if uploaded_file:
        # Simpan file yang diunggah ke st.session_state untuk akses di seluruh aplikasi
        st.session_state.uploaded_file = uploaded_file
        
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

        # Simpan k_range dan sil_scores untuk digunakan di tahap berikutnya
        k_range = range(2, 11)
        st.session_state.k_range = k_range
        st.session_state.sil_scores_kmeans = []
        st.session_state.sil_scores_kmedoids = []

        # Setelah tahap upload selesai, buka tahap berikutnya (Silhouette Score)
        st.session_state.stage = 'Perbandingan Silhouette Score'

# Perbandingan Silhouette Score
if tab == "Perbandingan Silhouette Score":
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        uploaded_file = st.session_state.uploaded_file
        df = pd.read_excel(uploaded_file)
        
        # Ambil normalized dan k_range dari session_state
        normalized = st.session_state.normalized
        k_range = st.session_state.k_range
        sil_scores_kmeans = st.session_state.sil_scores_kmeans
        sil_scores_kmedoids = st.session_state.sil_scores_kmedoids
        
        # Rentang jumlah klaster
        st.subheader("Perbandingan Silhouette Score")
        with st.spinner("Menghitung silhouette scores..."):
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

            # Tampilkan silhouette scores
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(k_range, sil_scores_kmeans, 'bo-', label='KMeans')
            ax.plot(k_range, sil_scores_kmedoids, 'ro-', label='KMedoids')
            ax.set_title('Silhouette Score: KMeans vs KMedoids')
            ax.set_xlabel('Jumlah Klaster (k)')
            ax.set_ylabel('Silhouette Score')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # Setelah tahap perbandingan selesai, buka tahap berikutnya (Clustering)
        st.session_state.stage = 'Clustering'

# Clustering
if tab == "Clustering":
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        uploaded_file = st.session_state.uploaded_file
        df = pd.read_excel(uploaded_file)
        
        # Ambil normalized dan k_range dari session_state
        normalized = st.session_state.normalized
        k_range = st.session_state.k_range
        sil_scores_kmeans = st.session_state.sil_scores_kmeans
        sil_scores_kmedoids = st.session_state.sil_scores_kmedoids
        
        k_optimal_kmeans = k_range[np.argmax(sil_scores_kmeans)]
        k_optimal_kmedoids = k_range[np.argmax(sil_scores_kmedoids)]

        # Tampilkan jumlah cluster optimal
        st.write(f"Jumlah cluster optimal untuk KMeans: {k_optimal_kmeans}")
        st.write(f"Jumlah cluster optimal untuk KMedoids: {k_optimal_kmedoids}")

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

        # Tampilkan skor evaluasi
        st.subheader("Davies-Bouldin Index")
        st.write(f"**KMeans**: {kmeans_dbi:.4f}")
        st.write(f"**KMedoids**: {kmedoids_dbi:.4f}")

        # Setelah tahap clustering selesai, buka tahap berikutnya (Visualisasi Hasil)
        st.session_state.stage = 'Visualisasi Hasil'

# Visualisasi Hasil
if tab == "Visualisasi Hasil":
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        uploaded_file = st.session_state.uploaded_file
        df = pd.read_excel(uploaded_file)
        
        st.subheader("Visualisasi Hasil Clustering")

        fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))

        sns.scatterplot(x=normalized[:, 0], y=normalized[:, 1], hue=kmeans_labels, palette='Set2', ax=ax2[0])
        ax2[0].set_title("K-Means Clustering")
        ax2[0].set_xlabel("Jumlah Kasus")
        ax2[0].set_ylabel("Kepadatan")

        sns.scatterplot(x=normalized[:, 0], y=normalized[:, 1], hue=kmedoids_labels, palette='Set1', ax=ax2[1])
        ax2[1].set_title("K-Medoids Clustering")
        ax2[1].set_xlabel("Jumlah Kasus")
        ax2[1].set_ylabel("Kepadatan")

        st.pyplot(fig2)

        # Setelah tahap visualisasi selesai, buka tahap berikutnya (Hasil Akhir)
        st.session_state.stage = 'Hasil Akhir'

# Hasil Akhir
if tab == "Hasil Akhir":
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        uploaded_file = st.session_state.uploaded_file
        df = pd.read_excel(uploaded_file)
        
        st.subheader("Hasil Akhir per Kecamatan")
        st.dataframe(df[['Kecamatan', 'Cluster_KMeans', 'Cluster_KMedoids']])