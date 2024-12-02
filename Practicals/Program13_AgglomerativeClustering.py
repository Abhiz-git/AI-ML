# Import necessary libraries
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# Generate synthetic dataset
X, y_true = make_blobs(n_samples=20, centers=4, cluster_std=1.0, random_state=42)

# Apply Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=4)
y_pred = model.fit_predict(X)

# Step 3: Plot the dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage 
linkage_matrix = linkage(X, method='ward')  # Ward's method

plt.figure(figsize=(13, 10))
plt.title("Dendrogram for Agglomerative Clustering")
dendrogram(linkage_matrix)
plt.xlabel("Data points")
plt.ylabel("Euclidean distance")
plt.show()
