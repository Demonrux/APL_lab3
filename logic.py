import pylab as pl
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


def kmeans_clustering(data_file):
    scaler = StandardScaler()
    data_scale = scaler.fit_transform(data_file)

    kmeans = KMeans(n_clusters=15, random_state=42, max_iter=300)

    clusters = kmeans.fit_predict(data_scale)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    plt.scatter(data_file[:, 0], data_file[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:, 0],
                scaler.inverse_transform(kmeans.cluster_centers_)[:, 1],
                s=200, c='red', marker='X', label='Centroids')

    plt.xlabel('X')
    plt.ylabel('Y')
    pl.title('Кластеризация методом К-средних')
    plt.legend()
    plt.tight_layout()
    plt.show()


def hierarchical_clustering(data_file):
    scaler = StandardScaler()
    data_scale = scaler.fit_transform(data_file)

    c_matrix = linkage(data_scale, method='complete')

    plt.figure(figsize=(12, 8))
    dendrogram(c_matrix)
    plt.tick_params(axis='x', labelbottom=False)
    plt.ylabel('Расстояние')
    plt.title('Иерархическая кластеризация')
    plt.show()


def dbscan_clustering(data_file):
    scaler = StandardScaler()
    data_scale = scaler.fit_transform(data_file)

    dbscan = DBSCAN(eps=0.15, min_samples=40)
    clusters = dbscan.fit_predict(data_scale)

    unique_labels = numpy.unique(clusters)

    plt.subplot(1, 1, 1)

    cmap = plt.get_cmap('tab10')
    colors = cmap(numpy.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        if label == -1:
            mask = clusters == -1
            plt.scatter(data_file[mask, 0], data_file[mask, 1],
                        color='black', marker='x', alpha=0.7, label='Шум')
        else:
            mask = clusters == label
            plt.scatter(data_file[mask, 0], data_file[mask, 1],
                        c=colors[i], alpha=0.7, label=f'Кластер{label}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('DBSCAN кластеризация')
    plt.legend()
    plt.tight_layout()
    plt.show()
