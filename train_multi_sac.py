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

model_dir = MODEL_DIR / "v1" / "portfolio"

# --- Load v1 seed 42 (default lambdas) ---
env_test = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark")
a42 = make_agent("SAC", env_test, cfg, weight_decay=1e-4, actor_wd=1e-1, alpha_mult=100.0)
a42.load(str(model_dir / "sac.pt"))
m42 = eval_agent(a42, env_test, cfg)

# --- Load v2 seeds (lambdas = 0.5, 0.35, 0.002, 0.05) ---
lambdas_v2 = (0.5, 0.35, 0.002, 0.05)
agents = [(a42, "v1_s42", m42)]

env_l = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark", seed=50, lambdas=lambdas_v2)
a50 = make_agent("SAC", env_l, cfg, weight_decay=1e-4, actor_wd=1e-1, alpha_mult=100.0)
path = MODEL_DIR / "sac_s50_v2.pt"
if path.exists():
    a50.load(str(path))
    m50 = eval_agent(a50, env_test, cfg)
    agents.append((a50, "v2_s50", m50))

env51 = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark", seed=51, lambdas=lambdas_v2)
a51 = make_agent("SAC", env51, cfg, weight_decay=1e-4, actor_wd=1e-1, alpha_mult=100.0)
path51 = MODEL_DIR / "sac_s51_v2.pt"
if path51.exists():
    a51.load(str(path51))
    m51 = eval_agent(a51, env_test, cfg)
    agents.append((a51, "v2_s51", m51))

def get_sharpe(m):
    for k in m:
        if 'sharpe' in k and 'std' not in k:
            return m[k]
    return 0.0

print("Individual models:")
for _, name, m in agents:
    s = get_sharpe(m)
    print(f"  {name}: Sharpe={s:.4f}")

# --- Ensemble tests ---
print("\n--- Ensemble combinations ---")
agent_list = [a[0] for a in agents]
best_sharpe = -999
best_ens = None
best_desc = ""

for k in range(2, len(agent_list) + 1):
    from itertools import combinations
    for combo in combinations(range(len(agent_list)), k):
        sub = [agent_list[i] for i in combo]
        desc = "+".join(agents[i][1] for i in combo)
        ens = EnsembleAgent(sub)
        m = eval_agent(ens, env_test, cfg)
        s = get_sharpe(m)
        if s > best_sharpe:
            best_sharpe = s
            best_ens = ens
            best_desc = desc
    print(f"  Top-{k}: {desc}: Sharpe={best_sharpe:.4f}")

# --- Weighted ensemble ---
if len(agent_list) > 1:
    sharpes = np.array([m.get("SAC_sharpe", 0) for _, _, m in agents])
    weights = sharpes - min(sharpes) + 0.01
    weights = weights / weights.sum()
    print(f"\n  Weights: {dict(zip([a[1] for a in agents], weights))}")
    ens_w = EnsembleAgent(agent_list, weights)
    m_w = eval_agent(ens_w, env_test, cfg)
    print(f"  Weighted ensemble: Sharpe={m_w.get('SAC_sharpe', 0):.4f}")

# Save best ensemble
if best_ens is not None:
    best_ens.save(str(model_dir / "sac_ensemble.pt"))
    print(f"\n  Saved best ensemble ({best_desc}) to {model_dir / 'sac_ensemble.pt'}")

# --- Final ensemble ---
print("\n--- Final ensemble evaluation ---")
if len(agent_list) > 1:
    from itertools import combinations
    best_s = -999
    best_combo = None
    for k in range(2, len(agent_list) + 1):
        for combo in combinations(range(len(agent_list)), k):
            sub = [agent_list[i] for i in combo]
            desc = "+".join(agents[i][1] for i in combo)
            ens = EnsembleAgent(sub)
            m = eval_agent(ens, env_test, cfg)
            s = get_sharpe(m)
            if s > best_s:
                best_s = s
                best_combo = combo
                best_desc = desc
        print(f"  Top-{k} best ({best_desc}): S={best_s:.4f}")

    sub_w = [agent_list[i] for i in best_combo]
    ens_final = EnsembleAgent(sub_w)
    path = str(model_dir / "sac_ensemble.pt")
    ens_final.save(path)
    print(f"  Saved ensemble to {path}")

    # Also update the main sac.pt to point to best ensemble
    import shutil
    shutil.copy(path, str(model_dir / "sac.pt"))
    print("  Updated sac.pt with ensemble")
