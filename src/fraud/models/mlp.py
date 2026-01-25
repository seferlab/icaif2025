from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse

import torch
import torch.nn as nn
import torch.optim as optim

def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)

class _MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class MLPClassifier:
    def __init__(self, epochs: int = 10, lr: float = 1e-3, batch_size: int = 512, hidden: int = 256, dropout: float = 0.1):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hidden = hidden
        self.dropout = dropout
        self.model = None
        self.device = "cpu"

    def fit(self, X, y, X_val_num=None, y_val=None, seed: int = 0):
        torch.manual_seed(seed)
        Xd = _to_dense(X)
        yd = np.asarray(y, dtype=np.float32)

        d_in = Xd.shape[1]
        self.model = _MLP(d_in, hidden=self.hidden, dropout=self.dropout).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        ds = torch.utils.data.TensorDataset(torch.from_numpy(Xd), torch.from_numpy(yd))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        Xd = _to_dense(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(Xd).to(self.device)).cpu().numpy()
        p = 1.0/(1.0 + np.exp(-logits))
        p = p.reshape(-1,1)
        return np.concatenate([1-p, p], axis=1)
