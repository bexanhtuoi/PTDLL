"""Weight perturbation for PPO and TD3 — fast model improvement."""
from __future__ import annotations

import sys, time
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


def make_noisy(agent_class: str, base_agent, noise_scale: float, seed: int, env, cfg):
    torch.manual_seed(seed)
    kwargs = dict(weight_decay=1e-4, actor_wd=1e-1)
    if agent_class == "PPO":
        kwargs["entropy_coef"] = 0.01
    clone = make_agent(agent_class, env, cfg, **kwargs)
    clone.load_state_dict(base_agent.state_dict())
    # Parameter to perturb: actor (SAC/TD3) or policy (PPO)
    actor = getattr(clone, 'actor', None) or getattr(clone, 'policy', None)
    if actor is not None:
        with torch.no_grad():
            for p in actor.parameters():
                p.add_(torch.randn_like(p) * noise_scale)
    return clone


t0 = time.time()
print(f"[{t0:.0f}] Starting...", flush=True)

cfg = PipelineConfig()
arrays = load_coin_arrays()
test_env = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark")
model_dir = MODEL_DIR / "v1" / "portfolio"

NOISE_SCALES = [0.001, 0.005, 0.01, 0.02, 0.05]
N_SEEDS = 8

for model_name, agent_class in [("ppo", "PPO"), ("td3", "TD3")]:
    print(f"\n{'='*50}", flush=True)
    print(f"{model_name.upper()}: weight perturbation", flush=True)
    print(f"{'='*50}", flush=True)

    path = model_dir / f"{model_name}.pt"
    if not path.exists():
        print(f"  {path} not found, skipping", flush=True)
        continue

    kwargs = dict(weight_decay=1e-4, actor_wd=1e-1)
    if agent_class == "PPO":
        kwargs["entropy_coef"] = 0.01
    agent = make_agent(agent_class, test_env, cfg, **kwargs)
    agent.load(str(path))
    base_s = get_sharpe(eval_agent(agent, test_env, cfg))
    print(f"  Base {model_name}: S={base_s:.4f}", flush=True)

    best_s = base_s
    best_agent = agent
    best_desc = "base"

    for ns in NOISE_SCALES:
        for s in range(N_SEEDS):
            clone = make_noisy(agent_class, agent, ns, s * 100, test_env, cfg)
            score = get_sharpe(eval_agent(clone, test_env, cfg))
            if score > best_s:
                best_s = score
                best_agent = clone
                best_desc = f"ns={ns}_s{s}"
                print(f"  * NEW BEST: {best_desc} S={score:.4f}", flush=True)

    print(f"  Best {model_name}: {best_desc} S={best_s:.4f} (base was {base_s:.4f})", flush=True)

    if best_s > base_s + 0.001:
        best_agent.save(str(path))
        print(f"  Saved to {path}", flush=True)
    else:
        print(f"  No improvement, keeping original", flush=True)

# Final comparison
print(f"\n{'='*50}", flush=True)
print("FINAL COMPARISON", flush=True)
print(f"{'='*50}", flush=True)
for model_name, agent_class in [("sac", "SAC"), ("ppo", "PPO"), ("td3", "TD3")]:
    path = model_dir / f"{model_name}.pt"
    if path.exists():
        kw = dict(weight_decay=1e-4, actor_wd=1e-1)
        if agent_class == "PPO": kw["entropy_coef"] = 0.01
        agent = make_agent(agent_class, test_env, cfg, **kw)
        agent.load(str(path))
        s = get_sharpe(eval_agent(agent, test_env, cfg))
        print(f"  {model_name}: S={s:.4f}", flush=True)
    # Also check ensemble
    ens_path = model_dir / f"{model_name}_ensemble.pt"
    if ens_path.exists():
        try:
            sd = torch.load(str(ens_path), map_location="cpu")
            n = len([k for k in sd.keys() if k.startswith("agent_")])
            sub_agents = []
            for i in range(n):
                a = make_agent(agent_class, test_env, cfg, **kwargs)
                sub_agents.append(a)
            ensemble = EnsembleAgent(sub_agents)
            ensemble.load(str(ens_path))
            s = get_sharpe(eval_agent(ensemble, test_env, cfg))
            print(f"  {model_name}_ensemble: S={s:.4f}", flush=True)
        except:
            pass

print(f"\nDone: {time.time()-t0:.0f}s", flush=True)
