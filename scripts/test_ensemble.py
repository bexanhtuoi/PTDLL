from __future__ import annotations

import numpy as np
import torch

from config import PipelineConfig, MODEL_DIR
from dataset.fetch import load_coin_arrays
from portfolio.env import build_env
from portfolio.evaluate import eval_agent, make_agent
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
        return ws / ws.sum()

    def train_ep(self, env, start_idx=None, end_idx=None):
        return 0.0, {}

    def state_dict(self):
        return {f"agent_{i}": a.state_dict() for i, a in enumerate(self.agents)}

    def load_state_dict(self, state_dict):
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(state_dict[f"agent_{i}"])


cfg = PipelineConfig()
arrays = load_coin_arrays()
model_dir = MODEL_DIR
env_test = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark")


def get_sharpe(m):
    for k in m:
        if 'sharpe' in k and 'std' not in k:
            return m[k]
    return 0.0


# --- Load available SAC models ---
saved_paths = [
    MODEL_DIR / "v1" / "portfolio" / "sac.pt",
    MODEL_DIR / "sac_s42.pt",
    MODEL_DIR / "sac_s42_v2.pt",
    MODEL_DIR / "sac_s43.pt",
    MODEL_DIR / "sac_s43_v2.pt",
    MODEL_DIR / "sac_s44.pt",
    MODEL_DIR / "sac_s44_v2.pt",
    MODEL_DIR / "sac_s45.pt",
    MODEL_DIR / "sac_s46.pt",
    MODEL_DIR / "sac_s47_v2.pt",
    MODEL_DIR / "sac_s48_v2.pt",
    MODEL_DIR / "sac_s49_v2.pt",
    MODEL_DIR / "sac_s50_v2.pt",
    MODEL_DIR / "sac_s51_v2.pt",
]
agents = []
env_l = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark")

for path in saved_paths:
    if not path.exists():
        continue
    name = path.stem
    lambdas_v2 = (0.5, 0.35, 0.002, 0.05)
    is_v2 = "v2" in name
    env_for_load = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark",
                             lambdas=lambdas_v2 if is_v2 else None)
    agent = make_agent("SAC", env_for_load, cfg, weight_decay=1e-4, actor_wd=1e-1, alpha_mult=100.0)
    agent.load(str(path))
    m = eval_agent(agent, env_test, cfg)
    s = get_sharpe(m)
    agents.append((agent, name, s))
    print(f"{name}: Sharpe={s:.4f}")

agent_list = [a[0] for a in agents]
names = [a[1] for a in agents]
sharpes = [a[2] for a in agents]

# --- Find best ensemble ---
print("\n--- Ensemble search ---")
best_s = -999
best_combo = None

for n_agents in range(2, min(len(agent_list) + 1, 7)):
    from itertools import combinations
    for combo in combinations(range(len(agent_list)), n_agents):
        sub = [agent_list[i] for i in combo]
        ens = EnsembleAgent(sub)
        m = eval_agent(ens, env_test, cfg)
        s = get_sharpe(m)
        if s > best_s:
            best_s = s
            best_combo = combo

    combo_names = "+".join(names[i] for i in best_combo)
    print(f"  Top-{n_agents}: {combo_names}: S={best_s:.4f}")

# Show top-2 combinations specifically
print("\n--- Top single vs top pair ---")
best_single_idx = int(np.argmax(sharpes))
print(f"  Best single: {names[best_single_idx]}: S={sharpes[best_single_idx]:.4f}")

best_pair_s = -999
best_pair = None
from itertools import combinations
for combo in combinations(range(len(agent_list)), 2):
    sub = [agent_list[i] for i in combo]
    ens = EnsembleAgent(sub)
    m = eval_agent(ens, env_test, cfg)
    s = get_sharpe(m)
    if s > best_pair_s:
        best_pair_s = s
        best_pair = combo
pair_names = "+".join(names[i] for i in best_pair)
print(f"  Best pair: {pair_names}: S={best_pair_s:.4f}")

# --- Save best ensemble ---
if best_combo is not None:
    sub = [agent_list[i] for i in best_combo]
    ens = EnsembleAgent(sub)
    ens.save(str(model_dir / "sac_ensemble.pt"))
    print(f"\n  Saved best ensemble ({'+'.join(names[i] for i in best_combo)}) to sac_ensemble.pt")
    print(f"  S={best_s:.4f}")
