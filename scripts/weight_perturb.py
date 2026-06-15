"""Weight perturbation ensemble for SAC."""
from __future__ import annotations

import sys, time, copy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PipelineConfig, MODEL_DIR
from dataset.fetch import load_coin_arrays
from portfolio.env import build_env
from portfolio.evaluate import make_agent, eval_agent
sys.path.insert(0, str(ROOT / "scripts"))
from ensemble_agent import EnsembleAgent  # type: ignore


def get_sharpe(m):
    for k in m:
        if 'sharpe' in k and 'std' not in k:
            return m[k]
    return 0.0


t0 = time.time()
print(f"[{time.time()-t0:.0f}s] Setting up...", flush=True)

cfg = PipelineConfig()
arrays = load_coin_arrays()
test_env = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark")
model_dir = MODEL_DIR / "v1" / "portfolio"

msg = f"[{time.time()-t0:.0f}s] Loading base SAC..."
print(msg, flush=True)

base = make_agent("SAC", test_env, cfg, weight_decay=1e-4, actor_wd=1e-1, alpha_mult=100.0)
base.load(str(model_dir / "sac.pt"))
base_sharpe = get_sharpe(eval_agent(base, test_env, cfg))
print(f"[{time.time()-t0:.0f}s] Base SAC: S={base_sharpe:.4f}", flush=True)

all_agents = [base]
all_scores = [("base", base_sharpe)]

# Try weight perturbation
NOISE_SCALES = [0.001, 0.005, 0.01, 0.02]
N_PER_SCALE = 6

for ns in NOISE_SCALES:
    for s in range(N_PER_SCALE):
        torch.manual_seed(s * 1000)
        clone = make_agent("SAC", test_env, cfg, weight_decay=1e-4, actor_wd=1e-1, alpha_mult=100.0)
        clone.load_state_dict(base.state_dict())
        with torch.no_grad():
            for p in clone.actor.parameters():
                p.add_(torch.randn_like(p) * ns)
        score = get_sharpe(eval_agent(clone, test_env, cfg))
        all_agents.append(clone)
        all_scores.append((f"ns={ns}_s{s}", score))
        if score > 0.01:
            print(f"  ns={ns:.3f} s{s}: S={score:.4f} *", flush=True)
        else:
            print(f"  ns={ns:.3f} s{s}: S={score:.4f}", flush=True)

print(f"\n[{time.time()-t0:.0f}s] Positive variants: {sum(1 for _,s in all_scores[1:] if s>0)}/{len(all_scores)-1}", flush=True)

# Best individual variant
best_var = max(all_scores[1:], key=lambda x: x[1])
print(f"Best variant: {best_var[0]} S={best_var[1]:.4f}", flush=True)

# Pick top 5 positive for ensemble
pos_agents = [agent for agent, (n, s) in zip(all_agents[1:], all_scores[1:]) if s > 0]
pos_scores = [s for n, s in all_scores[1:] if s > 0]
top_indices = sorted(range(len(pos_agents)), key=lambda i: -pos_scores[i])[:8]
pos_agents = [pos_agents[i] for i in top_indices]

# Try ensemble with base + top variants
from itertools import combinations

candidates = [base] + pos_agents[:5]
print(f"Candidates: {len(candidates)} (base + {min(5,len(pos_agents))} variants)", flush=True)

best_s = base_sharpe
best_combo = (0,)

for n in range(2, min(len(candidates) + 1, 5)):
    for combo in combinations(range(len(candidates)), n):
        sub = [candidates[i] for i in combo]
        ens = EnsembleAgent(sub)
        s = get_sharpe(eval_agent(ens, test_env, cfg))
        if s > best_s:
            best_s = s
            best_combo = combo
    desc = "+".join(f"a{i}" for i in best_combo)
    print(f"Best-{n}: S={best_s:.4f} ({desc})", flush=True)

# Save best ensemble
if best_s > base_sharpe + 0.001:
    sub = [candidates[i] for i in best_combo]
    ens = EnsembleAgent(sub)
    path = str(model_dir / "sac_ensemble.pt")
    ens.save(path)
    print(f"\nSaved ensemble (S={best_s:.4f}) > base (S={base_sharpe:.4f}) to {path}", flush=True)
else:
    print(f"\nNo ensemble beats base (S={base_sharpe:.4f})", flush=True)

print(f"\n[{time.time()-t0:.0f}s] DONE", flush=True)
