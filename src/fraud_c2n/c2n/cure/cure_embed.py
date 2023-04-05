
from __future__ import annotations
from typing import List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def cure_embedding(value_feature: np.ndarray, k_list: List[int], pca_dim: int, seed: int=42) -> np.ndarray:
    # value_feature: (n_values, n_features) counts
    X = normalize(value_feature, norm="l2", axis=1)
    parts=[]
    for k in k_list:
        k=int(k)
        km = KMeans(n_clusters=min(k, X.shape[0]), random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        onehot = np.eye(km.n_clusters, dtype=np.float32)[labels]
        parts.append(onehot)
    H = np.concatenate(parts, axis=1) if parts else X.astype(np.float32)
    if H.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        Z = pca.fit_transform(H)
    else:
        Z = H
    return Z.astype(np.float32)
