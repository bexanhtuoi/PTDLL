from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


N_FEATURES = 13
LOOKBACK = 60
N_COINS = 14
EMB_DIM = 4
STOP_MIN = 0.05
STOP_MAX = 0.50
STOP_RANGE = STOP_MAX - STOP_MIN


def asym_mae(pred: torch.Tensor, target: torch.Tensor, over_w: float = 0.5, under_w: float = 2.0) -> torch.Tensor:
    diff = pred - target
    w = torch.where(diff > 0, over_w, under_w)
    return (w * diff.abs()).mean()


def boundary_reg(pred: torch.Tensor, alpha: float = 0.001) -> torch.Tensor:
    margin = 0.01
    near_min = torch.relu(STOP_MIN + margin - pred)
    near_max = torch.relu(pred - (STOP_MAX - margin))
    return alpha * (near_min + near_max).mean()


def auto_label(close: float, future_close: np.ndarray) -> float:
    low = float(future_close.min())
    dd = (close - low) / (close + 1e-12)
    return float(np.clip(dd * 1.2, STOP_MIN, STOP_MAX))


class BaseStopModel(nn.Module, ABC):
    def __init__(self, n_coins: int = N_COINS, emb_dim: int = EMB_DIM):
        super().__init__()
        self.emb = nn.Embedding(n_coins, emb_dim)

    @abstractmethod
    def forward(self, x: torch.Tensor, coin_idx: torch.Tensor) -> torch.Tensor:
        ...

    def predict(self, x: np.ndarray, coin_idx: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32)
            idx_t = torch.tensor(coin_idx, dtype=torch.long)
            return self.forward(x_t, idx_t).squeeze(-1).numpy()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))


class StopANN(BaseStopModel):
    def __init__(self, n_coins: int = N_COINS, emb_dim: int = EMB_DIM):
        super().__init__(n_coins, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(LOOKBACK * N_FEATURES, 64), nn.ReLU(), nn.Dropout(0.3),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + emb_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, coin_idx: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.net(x)
        x = torch.cat([x, self.emb(coin_idx)], dim=1)
        return self.head(x) * STOP_RANGE + STOP_MIN


class StopLSTM(BaseStopModel):
    def __init__(self, n_coins: int = N_COINS, emb_dim: int = EMB_DIM):
        super().__init__(n_coins, emb_dim)
        self.lstm = nn.LSTM(N_FEATURES + emb_dim, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, coin_idx: torch.Tensor) -> torch.Tensor:
        emb = self.emb(coin_idx).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, emb], dim=2)
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]) * STOP_RANGE + STOP_MIN


class StopCNN(BaseStopModel):
    def __init__(self, n_coins: int = N_COINS, emb_dim: int = EMB_DIM):
        super().__init__(n_coins, emb_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(N_FEATURES, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 15 + emb_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, coin_idx: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, self.emb(coin_idx)], dim=1)
        return self.head(x) * STOP_RANGE + STOP_MIN
