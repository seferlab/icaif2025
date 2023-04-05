
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # adj_norm: (n,n)
        return self.lin(adj_norm @ x)

class GCNEncoder(nn.Module):
    def __init__(self, n_nodes: int, hidden_dim: int, emb_dim: int):
        super().__init__()
        # learnable node features (identity alternative)
        self.node_feat = nn.Parameter(torch.randn(n_nodes, hidden_dim) * 0.01)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, emb_dim)

    def forward(self, adj_norm: torch.Tensor) -> torch.Tensor:
        x = self.node_feat
        x = F.relu(self.gcn1(x, adj_norm))
        z = self.gcn2(x, adj_norm)
        return z

class GraphAE(nn.Module):
    def __init__(self, n_nodes: int, hidden_dim: int, emb_dim: int):
        super().__init__()
        self.enc = GCNEncoder(n_nodes, hidden_dim, emb_dim)

    def forward(self, adj_norm: torch.Tensor) -> torch.Tensor:
        z = self.enc(adj_norm)
        # inner-product decoder
        logits = z @ z.t()
        return z, logits

def normalize_adj(adj: np.ndarray) -> torch.Tensor:
    A = torch.tensor(adj, dtype=torch.float32)
    I = torch.eye(A.shape[0], dtype=torch.float32)
    A = A + I
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D = torch.diag(deg_inv_sqrt)
    return D @ A @ D

def train_graph_ae(adj: np.ndarray, emb_dim: int, hidden_dim: int, epochs: int, lr: float, seed: int=42, device: str="cpu") -> np.ndarray:
    torch.manual_seed(seed)
    n = adj.shape[0]
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    adj_norm = normalize_adj(adj).to(device)
    model = GraphAE(n, hidden_dim, emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # BCE with logits; treat adj as {0,1} (binarize)
    target = (adj_t>0).float()
    for _ in range(int(epochs)):
        model.train()
        z, logits = model(adj_norm)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        z,_ = model(adj_norm)
    return z.detach().cpu().numpy().astype(np.float32)
