from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
import pandas as pd
from data_preprocessing.delete import time_series_list, labels
from train import pad_samples
import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# Prepare data
padded_samples = pad_samples(time_series_list, target_dim=6)

all_series = []
for i, ts in enumerate(padded_samples):
    df = pd.DataFrame(ts, columns=[f'var_{j}' for j in range(ts.shape[1])])
    df['id'] = i
    df['time'] = range(ts.shape[0])
    all_series.append(df)

df_full = pd.concat(all_series)
from tsfresh.feature_extraction import MinimalFCParameters
features = extract_features(df_full, column_id="id", column_sort="time", default_fc_parameters=MinimalFCParameters(), n_jobs=4)#default_fc_parameters=MinimalFCParameters(),

from sklearn.feature_selection import VarianceThreshold

# Remove constant features (likely due to padding)
selector = VarianceThreshold(threshold=0.0)
features_clean = selector.fit_transform(features)



from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ensure no NaNs
#features = features.fillna(0)

kmeans = KMeans(n_clusters=len(set(labels)), random_state=42)##
pred_clusters = kmeans.fit_predict(features_clean)#######problemmms

# Evaluate
ari = adjusted_rand_score(labels, pred_clusters)
nmi = normalized_mutual_info_score(labels, pred_clusters)
print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

# Get unique labels and map them to indices
unique_labels = np.unique(labels)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
indexed_labels = [label_to_index[label] for label in labels]

# Use consistent colormap
colors = plt.cm.tab10.colors
cmap = ListedColormap(colors[:len(unique_labels)])

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
proj = tsne.fit_transform(features_clean)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(proj[:, 0], proj[:, 1], c=indexed_labels, cmap=cmap)

# Legend
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      label=f'Degree {label}',
                      markerfacecolor=colors[idx],
                      markersize=8) for label, idx in label_to_index.items()]
plt.legend(handles=handles, title="Degrees")
plt.title("t-SNE Clusters Colored by Degree")
plt.savefig("tsne_clusters_by_degree_with_correct_legend.png")
plt.show()
print(features.shape)
print(f"Number of time series in original list: {len(time_series_list)}")
print(f"Number of padded samples: {len(padded_samples)}")


csv=features.to_csv("tsfresh_features.csv")
