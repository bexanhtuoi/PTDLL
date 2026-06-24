from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import nn


N_FEATURES = 17
LOOKBACK = 60
N_COINS = 14
EMB_DIM = 16
STOP_MIN = 0.05
STOP_MAX = 0.50
STOP_RANGE = STOP_MAX - STOP_MIN


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    diff = (pred - target).abs()
    quad = diff.clamp(max=delta)
    lin = diff - quad
    return (0.5 * quad.pow(2) + delta * lin).mean()


def boundary_reg(pred: torch.Tensor, alpha: float = 0.001) -> torch.Tensor:
    margin = 0.01
    near_min = torch.relu(STOP_MIN + margin - pred)
    near_max = torch.relu(pred - (STOP_MAX - margin))
    return alpha * (near_min + near_max).mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(pred, target) + boundary_reg(pred, alpha=0.001)



def auto_label(close: float, future_close: np.ndarray) -> float:
    low = float(future_close.min())
    dd = (close - low) / (close + 1e-12)
    return float(np.clip(dd, STOP_MIN, STOP_MAX))


class FeatureScaler:
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.dead: np.ndarray | None = None

    def fit(self, xs: np.ndarray):
        self.mean = np.nanmean(xs.reshape(-1, xs.shape[-1]), axis=0)
        self.std = np.nanstd(xs.reshape(-1, xs.shape[-1]), axis=0)
        self.dead = self.std < 1e-6
        self.std = np.clip(self.std, 1e-8, None)

    def transform(self, xs: np.ndarray) -> np.ndarray:
        x = (xs - self.mean[None, None, :]) / self.std[None, None, :]
        if self.dead is not None and self.dead.any():
            x[:, :, self.dead] = 0.0
        return x

    def fit_transform(self, xs: np.ndarray) -> np.ndarray:
        self.fit(xs)
        return self.transform(xs)


def init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_head_bias(model: nn.Module, target_mean: float = 0.0):
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        nn.init.constant_(last_linear.bias, target_mean)


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
            return self.forward(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(coin_idx, dtype=torch.long),
            ).squeeze(-1).numpy()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))


class StopANN(BaseStopModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LOOKBACK * N_FEATURES, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + EMB_DIM, 32), nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Tanh(),
        )
        self.apply(init_linear)
        init_head_bias(self)

    def forward(self, x, coin_idx):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        x = torch.cat([x, self.emb(coin_idx)], dim=1)
        return self.head(x) * 3.0


class StopLSTM(BaseStopModel):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES + EMB_DIM, 64, num_layers=1,
                           bidirectional=True, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(
            nn.Linear(64 * 2, 48), nn.LeakyReLU(0.2),
            nn.Linear(48, 1), nn.Tanh(),
        )
        self.apply(init_linear)
        init_head_bias(self)

    def forward(self, x, coin_idx):
        emb = self.emb(coin_idx).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, emb], dim=2)
        _, (h_n, _) = self.lstm(x)
        x = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.head(x) * 3.0


class StopCNN(BaseStopModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(N_FEATURES, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2), nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96), nn.LeakyReLU(0.2), nn.Dropout(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(96 + EMB_DIM, 48), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(48, 24), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(24, 1), nn.Tanh(),
        )
        self.apply(init_linear)
        init_head_bias(self)

    def forward(self, x, coin_idx):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(-1)
        x = torch.cat([x, self.emb(coin_idx)], dim=1)
        return self.head(x) * 3.0


class EnsembleLSTM(BaseStopModel):
    def __init__(self, model_dir: str | Path, n_models: int = 5):
        super().__init__()
        self.models = nn.ModuleList([StopLSTM() for _ in range(n_models)])
        model_dir = Path(model_dir)
        for i, m in enumerate(self.models):
            p = model_dir / f"model_{i}.pt"
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.eval()
        self.eval()

    def forward(self, x: torch.Tensor, coin_idx: torch.Tensor) -> torch.Tensor:
        preds = [m(x, coin_idx) for m in self.models]
        return torch.stack(preds).mean(dim=0)

    def load(self, path: str) -> None:
        pass  # EnsembleLSTM uses model_dir, not single path
