"""
PyTorch autoencoder trained on non-fraud rows; anomaly score from reconstruction MSE.
Complements Isolation Forest for proposal 'autoencoder anomaly' objective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import runtime_env  # noqa: F401


class _AE(nn.Module):
    def __init__(self, dim: int, latent: int = 32) -> None:
        super().__init__()
        h = max(64, min(256, dim * 2))
        self.enc = nn.Sequential(
            nn.Linear(dim, h),
            nn.ReLU(),
            nn.Linear(h, latent),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, h),
            nn.ReLU(),
            nn.Linear(h, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        return self.dec(z)


class AutoencoderAnomalyScorer:
    """Sklearn-like .score_samples: higher = more anomalous (reconstruction error)."""

    def __init__(
        self,
        model: _AE,
        numeric_cols: list[str],
        device: torch.device,
        err_mean: float,
        err_std: float,
    ) -> None:
        self.model = model
        self.numeric_cols = numeric_cols
        self.device = device
        self.err_mean = err_mean
        self.err_std = max(err_std, 1e-6)

    @torch.no_grad()
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        Xn = X[[c for c in self.numeric_cols if c in X.columns]].fillna(0.0).to_numpy(dtype=np.float32)
        t = torch.tensor(Xn, device=self.device)
        recon = self.model(t)
        mse = ((recon - t) ** 2).mean(dim=1).cpu().numpy()
        z = (mse - self.err_mean) / self.err_std
        return z.astype(np.float64)


def fit_autoencoder_anomaly(
    X_normal: pd.DataFrame,
    numeric_cols: list[str],
    *,
    epochs: int = 25,
    batch_size: int = 2048,
    lr: float = 1e-3,
    latent: int = 32,
) -> AutoencoderAnomalyScorer:
    cols = [c for c in numeric_cols if c in X_normal.columns]
    Xn = X_normal[cols].fillna(0.0).to_numpy(dtype=np.float32)
    dim = Xn.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _AE(dim, latent=latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xn.shape[0]
    bs = min(batch_size, max(64, n))

    model.train()
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            xb = torch.tensor(Xn[idx], device=device)
            opt.zero_grad()
            recon = model(xb)
            loss = nn.functional.mse_loss(recon, xb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        full = torch.tensor(Xn, device=device)
        recon = model(full)
        err = ((recon - full) ** 2).mean(dim=1).cpu().numpy()
    err_mean = float(err.mean())
    err_std = float(err.std())

    return AutoencoderAnomalyScorer(model, cols, device, err_mean, err_std)
