import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# data loading
X = np.loadtxt('data_clustering.txt', delimiter=',')

# rate the range width for clustering with the quantile parameter
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Create a MeanShift model with estimated bandwidth and activate bin_seeding to speed up
mean_shift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
mean_shift_model.fit(X)

# Get the coordinates of the cluster centers
cluster_centers = mean_shift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

# Get cluster labels for each point
labels = mean_shift_model.labels_
# Determine the number of unique clusters
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)

#  Displaying clusters on a plot
plt.figure()
markers = 'o*xvs'  # Define a list of markers for different clusters
for i, marker in zip(range(num_clusters), markers):
    # Display the points of each cluster with a unique marker
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='black')

    # Display the center of each cluster
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o', markerfacecolor='black', markeredgecolor='black', markersize=15)

plt.title('Кластери')
plt.show()