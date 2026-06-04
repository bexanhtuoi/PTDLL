from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np
import torch
from torch import nn


class BaseModel(ABC):
    lookback: int
    n_assets: int
    n_features: int
    device: str

    @abstractmethod
    def get_weights(self, state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def train_episode(
        self, env, start_idx: int | None = None, end_idx: int | None = None
    ) -> tuple[float, dict]:
        ...

    @abstractmethod
    def run_episode(
        self, env, start_idx: int | None = None, end_idx: int | None = None
    ) -> dict:
        ...

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))

    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        ...


class PolicyNet(nn.Module):
    def __init__(self, lookback: int, n_assets: int, n_features: int):
        super().__init__()
        in_channels = n_assets * n_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_assets),
        )

    def forward(self, x):
        b, L, A, F = x.shape
        x = x.view(b, A * F, L)
        return self.fc(self.conv(x))


class ValueNet(nn.Module):
    def __init__(self, lookback: int, n_assets: int, n_features: int):
        super().__init__()
        in_channels = n_assets * n_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        b, L, A, F = x.shape
        x = x.view(b, A * F, L)
        return self.fc(self.conv(x))


class StateEncoder(nn.Module):
    def __init__(self, lookback: int, n_assets: int, n_features: int, hidden: int = 32):
        super().__init__()
        in_channels = n_assets * n_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        b, L, A, F = x.shape
        x = x.view(b, A * F, L)
        return self.conv(x)


class TwinQNet(nn.Module):
    def __init__(self, lookback: int, n_assets: int, n_features: int, hidden: int = 64):
        super().__init__()
        self.encoder = StateEncoder(lookback, n_assets, n_features, hidden=32)
        self.q1 = nn.Sequential(
            nn.Linear(32 + n_assets, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(32 + n_assets, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        feat = self.encoder(state)
        x = torch.cat([feat, action], dim=1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state, action):
        feat = self.encoder(state)
        x = torch.cat([feat, action], dim=1)
        return self.q1(x)


class DeterministicActor(nn.Module):
    def __init__(self, lookback: int, n_assets: int, n_features: int):
        super().__init__()
        in_channels = n_assets * n_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_assets),
        )

    def forward(self, x):
        b, L, A, F = x.shape
        x = x.view(b, A * F, L)
        return torch.softmax(self.fc(self.conv(x)), dim=1)


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: str = "cpu"):
        indices = np.random.randint(0, len(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(np.array(actions), dtype=torch.float32, device=device),
            torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
