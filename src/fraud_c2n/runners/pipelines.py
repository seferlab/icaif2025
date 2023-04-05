
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.append("..")
from encoders.encoders import build_encoder
from c2n.common import build_vocab, row_value_indices, pool_embeddings
from c2n.cure.cure_embed import cure_embedding
from c2n.gce.gcn_ae import train_graph_ae
from c2n.distance.mds_embed import build_cooccurrence_graph, shortest_path_distances, mds_embedding
from c2n.dhe.dhe_embed import train_dhe
import networkx as nx

def _num_matrix(df: pd.DataFrame, num_cols: List[str]) -> np.ndarray:
    if len(num_cols)==0:
        return np.zeros((len(df),0),dtype=np.float32)
    return df[num_cols].to_numpy(dtype=np.float32)

def make_features_traditional(df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame,
                             cat_cols: List[str], num_cols: List[str],
                             encoder_cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    enc = build_encoder(encoder_cfg["type"], cat_cols, **{k:v for k,v in encoder_cfg.items() if k!="type"})
    Xc_tr = enc.fit_transform(df_tr, None)
    Xc_va = enc.transform(df_va)
    Xc_te = enc.transform(df_te)
    Xn_tr, Xn_va, Xn_te = _num_matrix(df_tr,num_cols), _num_matrix(df_va,num_cols), _num_matrix(df_te,num_cols)
    X_tr = np.concatenate([Xn_tr, Xc_tr], axis=1).astype(np.float32)
    X_va = np.concatenate([Xn_va, Xc_va], axis=1).astype(np.float32)
    X_te = np.concatenate([Xn_te, Xc_te], axis=1).astype(np.float32)
    return X_tr, X_va, X_te

def make_features_cure(df_all: pd.DataFrame, df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame,
                       cat_cols: List[str], num_cols: List[str],
                       cfg: Dict, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(cat_cols)==0:
        Xn_tr,Xn_va,Xn_te=_num_matrix(df_tr,num_cols),_num_matrix(df_va,num_cols),_num_matrix(df_te,num_cols)
        return Xn_tr,Xn_va,Xn_te
    vocab = build_vocab(df_all, cat_cols)
    # value-feature counts: for each value (col=value), count occurrences per feature column (one per cat col)
    n_vals=len(vocab.idx_to_key)
    val_feat=np.zeros((n_vals, len(cat_cols)), dtype=np.float32)
    for j,c in enumerate(cat_cols):
        vc = df_all[c].astype("string").value_counts()
        for v,count in vc.items():
            key=f"{c}={v}"
            i=vocab.key_to_idx.get(key)
            if i is not None:
                val_feat[i,j]=float(count)
    emb = cure_embedding(val_feat, cfg.get("k_list",[8,16,32]), cfg.get("pca_dim",32), seed=seed)
    rows_tr = row_value_indices(df_tr, cat_cols, vocab)
    rows_va = row_value_indices(df_va, cat_cols, vocab)
    rows_te = row_value_indices(df_te, cat_cols, vocab)
    Xc_tr = pool_embeddings(rows_tr, emb, mode="mean")
    Xc_va = pool_embeddings(rows_va, emb, mode="mean")
    Xc_te = pool_embeddings(rows_te, emb, mode="mean")
    Xn_tr,Xn_va,Xn_te=_num_matrix(df_tr,num_cols),_num_matrix(df_va,num_cols),_num_matrix(df_te,num_cols)
    return np.concatenate([Xn_tr,Xc_tr],1), np.concatenate([Xn_va,Xc_va],1), np.concatenate([Xn_te,Xc_te],1)

def make_features_gce(df_all: pd.DataFrame, df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame,
                      cat_cols: List[str], num_cols: List[str],
                      cfg: Dict, seed: int, device: str="cpu") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(cat_cols)==0:
        Xn_tr,Xn_va,Xn_te=_num_matrix(df_tr,num_cols),_num_matrix(df_va,num_cols),_num_matrix(df_te,num_cols)
        return Xn_tr,Xn_va,Xn_te
    vocab = build_vocab(df_all, cat_cols, max_vocab=cfg.get("max_vocab"))
    rows_all = row_value_indices(df_all, cat_cols, vocab)
    # adjacency
    n=len(vocab.idx_to_key)
    adj=np.zeros((n,n), dtype=np.float32)
    for idxs in rows_all:
        for a in range(len(idxs)):
            for b in range(a+1,len(idxs)):
                u,v=idxs[a], idxs[b]
                adj[u,v]+=1.0
                adj[v,u]+=1.0
    emb = train_graph_ae(adj, emb_dim=int(cfg.get("emb_dim",32)), hidden_dim=int(cfg.get("hidden_dim",64)),
                         epochs=int(cfg.get("epochs",50)), lr=float(cfg.get("lr",1e-3)), seed=seed, device=device)
    Xc_tr = pool_embeddings(row_value_indices(df_tr, cat_cols, vocab), emb)
    Xc_va = pool_embeddings(row_value_indices(df_va, cat_cols, vocab), emb)
    Xc_te = pool_embeddings(row_value_indices(df_te, cat_cols, vocab), emb)
    Xn_tr,Xn_va,Xn_te=_num_matrix(df_tr,num_cols),_num_matrix(df_va,num_cols),_num_matrix(df_te,num_cols)
    return np.concatenate([Xn_tr,Xc_tr],1), np.concatenate([Xn_va,Xc_va],1), np.concatenate([Xn_te,Xc_te],1)

def make_features_distance(df_all: pd.DataFrame, df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame,
                           cat_cols: List[str], num_cols: List[str],
                           cfg: Dict, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(cat_cols)==0:
        Xn_tr,Xn_va,Xn_te=_num_matrix(df_tr,num_cols),_num_matrix(df_va,num_cols),_num_matrix(df_te,num_cols)
        return Xn_tr,Xn_va,Xn_te
    vocab = build_vocab(df_all, cat_cols, max_vocab=int(cfg.get("max_vocab",5000)))
    rows_all = row_value_indices(df_all, cat_cols, vocab)
    G = build_cooccurrence_graph(rows_all)
    n=len(vocab.idx_to_key)
    D = shortest_path_distances(G, n)
    emb = mds_embedding(D, emb_dim=int(cfg.get("emb_dim",32)), seed=seed)
    Xc_tr = pool_embeddings(row_value_indices(df_tr, cat_cols, vocab), emb)
    Xc_va = pool_embeddings(row_value_indices(df_va, cat_cols, vocab), emb)
    Xc_te = pool_embeddings(row_value_indices(df_te, cat_cols, vocab), emb)
    Xn_tr,Xn_va,Xn_te=_num_matrix(df_tr,num_cols),_num_matrix(df_va,num_cols),_num_matrix(df_te,num_cols)
    return np.concatenate([Xn_tr,Xc_tr],1), np.concatenate([Xn_va,Xc_va],1), np.concatenate([Xn_te,Xc_te],1)

def make_features_dhe(df_all: pd.DataFrame, df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame,
                      y_tr: np.ndarray,
                      cat_cols: List[str], num_cols: List[str],
                      cfg: Dict, seed: int, device: str="cpu") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(cat_cols)==0:
        Xn_tr,Xn_va,Xn_te=_num_matrix(df_tr,num_cols),_num_matrix(df_va,num_cols),_num_matrix(df_te,num_cols)
        return Xn_tr,Xn_va,Xn_te
    vocab = build_vocab(df_all, cat_cols)
    # assign integer ids to each value key
    value_ids = np.arange(len(vocab.idx_to_key), dtype=np.int64)
    # train DHE supervised on train set: use pooled value ids per row -> here we just pick first id per row for supervision signal
    # (simple approximation). Better: multi-instance; still works decently.
    rows_tr = row_value_indices(df_tr, cat_cols, vocab)
    row_rep = np.array([idxs[0] if idxs else 0 for idxs in rows_tr], dtype=np.int64)
    emb_table = train_dhe(row_rep, y_tr, n_hashes=int(cfg.get("n_hashes",4)), n_buckets=int(cfg.get("n_buckets",20000)),
                          emb_dim=int(cfg.get("emb_dim",32)), hidden_dim=int(cfg.get("hidden_dim",64)),
                          epochs=int(cfg.get("epochs",20)), lr=float(cfg.get("lr",1e-3)), seed=seed, device=device)
    # build per-value embedding as identity lookup from row_rep id space (0..n_vals-1); fallback zeros
    emb = np.zeros((len(vocab.idx_to_key), int(cfg.get("emb_dim",32))), dtype=np.float32)
    m = min(len(emb), emb_table.shape[0])
    emb[:m]=emb_table[:m]
    Xc_tr = pool_embeddings(rows_tr, emb)
    Xc_va = pool_embeddings(row_value_indices(df_va, cat_cols, vocab), emb)
    Xc_te = pool_embeddings(row_value_indices(df_te, cat_cols, vocab), emb)
    Xn_tr,Xn_va,Xn_te=_num_matrix(df_tr,num_cols),_num_matrix(df_va,num_cols),_num_matrix(df_te,num_cols)
    return np.concatenate([Xn_tr,Xc_tr],1), np.concatenate([Xn_va,Xc_va],1), np.concatenate([Xn_te,Xc_te],1)
