import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=0.0001):
        self.n_clusters = n_clusters
        self.centroids = None
        self.max_iter = max_iter
        self.tol = tol

    def predict(self, X):
        # Find the distance between each data points with the k centroids.
        distances = cdist(X, self.centroids, metric='euclidean')
        # assign each data point to the closest centroid based on distance
        clusters = np.array([np.argmin(d) for d in distances])
        return clusters

    def fit(self, X):
        # Randomly pick k data points as our initial Centroids.
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx, :]

        clusters = self.predict(X)

        i = 0
        tol = float('inf')
        while i < self.max_iter and tol > self.tol:
            # Update centroid location by taking the average of the points in each cluster group
            centroids = []
            for k in range(self.n_clusters):
                updated_cluster = X[clusters == k].mean(axis=0)
                centroids.append(updated_cluster)
            centroids = np.vstack(centroids)
            tol = np.linalg.norm(centroids - self.centroids)
            self.centroids = centroids

            clusters = self.predict(X)
            i += 1


if __name__ == '__main__':
    # Load Data
    data = load_digits().data
    pca = PCA(2)

    # Transform the data
    df = pca.fit_transform(data)

    kmeans = KMeans(10)
    kmeans.fit(df)
    # Applying our function
    label = kmeans.predict(df)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.legend()
    plt.show()
