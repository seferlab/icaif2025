
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import networkx as nx
from sklearn.manifold import MDS

def build_cooccurrence_graph(rows: List[List[int]]) -> nx.Graph:
    G=nx.Graph()
    for idxs in rows:
        for i in idxs:
            G.add_node(i)
        # connect all pairs in row
        for a in range(len(idxs)):
            for b in range(a+1,len(idxs)):
                u,v = idxs[a], idxs[b]
                if G.has_edge(u,v):
                    G[u][v]["weight"] += 1.0
                else:
                    G.add_edge(u,v,weight=1.0)
    return G

def shortest_path_distances(G: nx.Graph, n: int) -> np.ndarray:
    # convert weights to lengths inversely
    H=nx.Graph()
    for u,v,data in G.edges(data=True):
        w=float(data.get("weight",1.0))
        length=1.0/(w)
        H.add_edge(u,v,weight=length)
    # all pairs shortest path (O(n^2 log n)) - okay for a few thousand
    D=np.full((n,n), fill_value=np.inf, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    for u,dist in nx.all_pairs_dijkstra_path_length(H, weight="weight"):
        for v,d in dist.items():
            D[u,v]=float(d)
    # replace inf with max finite * 1.2
    finite=D[np.isfinite(D)]
    maxf=float(finite.max()) if finite.size else 1.0
    D[~np.isfinite(D)] = maxf*1.2
    return D

def mds_embedding(D: np.ndarray, emb_dim: int, seed: int=42) -> np.ndarray:
    mds=MDS(n_components=emb_dim, dissimilarity="precomputed", random_state=seed, normalized_stress="auto")
    X=mds.fit_transform(D)
    return X.astype(np.float32)
