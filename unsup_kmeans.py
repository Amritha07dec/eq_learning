import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from unsup import csv

# Load your TSFresh-extracted features
features = pd.read_csv(csv, index_col=0)  # Update filename if needed

# Optional: Remove constant features
selector = VarianceThreshold(threshold=0.0)
features_clean = selector.fit_transform(features)

inertias = []
silhouette_scores = []
db_indices = []

cluster_range = range(2, 11)

import time

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    start = time.time()
    preds = kmeans.fit_predict(features_clean)
    print(f"k={k} took {time.time() - start:.2f} seconds")

    # preds = kmeans.fit_predict(features_clean)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features_clean, preds))
    db_indices.append(davies_bouldin_score(features_clean, preds))

# Plot results
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(cluster_range, inertias, marker='o')
plt.title("Inertia vs Number of Clusters")
plt.xlabel("k")
plt.ylabel("Inertia")

plt.subplot(1, 3, 2)
plt.plot(cluster_range, silhouette_scores, marker='o', color='green')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("k")
plt.ylabel("Silhouette Score")

plt.subplot(1, 3, 3)
plt.plot(cluster_range, db_indices, marker='o', color='red')
plt.title("Davies-Bouldin Index vs Number of Clusters")
plt.xlabel("k")
plt.ylabel("DB Index")

plt.tight_layout()
plt.savefig("cluster_metrics_analysis.png")
plt.show()
