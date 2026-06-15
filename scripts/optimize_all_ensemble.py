"""EnsembleAgent for portfolio RL model ensembling."""
from __future__ import annotations

import numpy as np

from portfolio.base import BaseModel


class EnsembleAgent(BaseModel):
    def __init__(self, agents, weights=None):
        self.agents = agents
        self.weights = np.array(weights or [1.0 / len(agents)] * len(agents))
        self.lookback = agents[0].lookback
        self.n_assets = agents[0].n_assets
        self.n_features = agents[0].n_features
        self.device = agents[0].device

    def predict(self, state):
        ws = np.zeros(self.n_assets)
        for w, agent in zip(self.weights, self.agents):
            ws += w * agent.predict(state)
        total = ws.sum()
        if total > 0:
            ws = ws / total
        return ws

    def train_ep(self, env, start_idx=None, end_idx=None):
        return 0.0, {}

    def state_dict(self):
        return {f"agent_{i}": a.state_dict() for i, a in enumerate(self.agents)}

    def load_state_dict(self, state_dict):
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(state_dict[f"agent_{i}"])
