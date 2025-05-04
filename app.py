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
st.title("Clustering DBD: KMeans vs KMedoids")

# Upload file
uploaded_file = st.file_uploader("Unggah file Excel (format .xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    st.subheader("Data Awal")
    st.write(df.head())

    # Ekstrak fitur
    features = df[['Jumlah Kasus', 'Kepadatan Penduduk']]

    # Normalisasi
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(features)
    normalized_df = pd.DataFrame(normalized, columns=['Jumlah Kasus', 'Kepadatan Penduduk'])

    st.subheader("Data Setelah Dinormalisasi")
    st.write(normalized_df)

    # Rentang jumlah klaster
    k_range = range(2, 11)
    sil_scores_kmeans = []
    sil_scores_kmedoids = []

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

        # Plot silhouette scores
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(k_range, sil_scores_kmeans, 'bo-', label='KMeans')
        ax.plot(k_range, sil_scores_kmedoids, 'ro-', label='KMedoids')
        ax.set_title('Silhouette Score: KMeans vs KMedoids')
        ax.set_xlabel('Jumlah Klaster (k)')
        ax.set_ylabel('Silhouette Score')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # Find k_optimal based on the maximum silhouette score for KMeans
    k_optimal_kmeans = k_range[np.argmax(sil_scores_kmeans)]
    k_optimal_kmedoids = k_range[np.argmax(sil_scores_kmedoids)]

    # Display the selected optimal k values
    st.write(f"Optimal number of clusters for KMeans: {k_optimal_kmeans}")
    st.write(f"Optimal number of clusters for KMedoids: {k_optimal_kmedoids}")

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

    # Visualisasi hasil clustering
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

    # Tampilkan tabel hasil akhir
    st.subheader("Hasil Akhir per Kecamatan")
    st.dataframe(df[['Kecamatan', 'Cluster_KMeans', 'Cluster_KMedoids']])