from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()  # load dataset
X = iris['data']  # get actual data
y = iris['target']  # get targets


# model instantiation and training
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)
kmeans.fit(X)


y_kmeans = kmeans.predict(X)  # make predictions / get clusters

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')  # plot of points with colors corresponding to clusters
centers = kmeans.cluster_centers_  # obtaining cluster centroids
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)  # displaying centroids of clusters
plt.show()

def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)  # init the rand number instance
    i = rng.permutation(X.shape[0])[:n_clusters]  # random set of centroids
    current_centers = X[i]  # set selected points as initial centroids
    while True:
        _labels = pairwise_distances_argmin(X, current_centers)  # determine the nearest center for each point
        new_centers = np.array([X[_labels == i].mean(0) for i in range(n_clusters)])  # determination of new centers

        if np.all(current_centers == new_centers):  # Check for a stop if the centers do not change
            break
        current_centers = new_centers  # centres renewal
    return current_centers, _labels


centers, labels = find_clusters(X, 3)  # searching three clusters based on the X-data
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')  # the result of clustering
plt.show()

centers, labels = find_clusters(X, 3, rseed=0)  # Finding cluster with other initial centers
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')  # result of clustering
plt.show()


labels = KMeans(3, random_state=0).fit_predict(X)  # clustering with KMeans to 3 clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')  # dots on the clusters area
plt.show()