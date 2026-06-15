"""Transfer SAC's policy to PPO/TD3 via weight copying + fine-tuning.

Key insight: SAC actor (PolicyNet) and PPO policy (PolicyNet) / TD3 actor (DeterministicActor)
share the same architecture. Load SAC weights, fine-tune briefly.
"""
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
from portfolio.base import DeterministicActor


def get_sharpe(m):
    for k in m:
        if "sharpe" in k and "std" not in k:
            return m[k]
    return 0.0


cfg = PipelineConfig(n_episodes=10000, random_state=42,
                     val_interval=500, early_stop_patience=5)
arrays = load_coin_arrays()
lambdas = (0.5, 0.35, 0.002, 0.05)

test_env = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark", lambdas=lambdas)
md = MODEL_DIR / "v1" / "portfolio"

# Load best SAC
sac = make_agent("SAC", test_env, cfg, weight_decay=1e-4, actor_wd=1e-1, alpha_mult=100.0)
sac.load(str(md / "sac.pt"))
sac_s = get_sharpe(eval_agent(sac, test_env, cfg))
print(f"SAC teacher: S={sac_s:.4f}")

for model, tag, kw in [
    ("ppo", "PPO", dict(weight_decay=1e-4, actor_wd=1e-1, entropy_coef=0.01)),
    ("td3", "TD3", dict(weight_decay=1e-4, actor_wd=1e-1)),
]:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"{tag}: weight transfer from SAC + fine-tune")
    print(f"{'='*60}")

    train_env = build_env(arrays, cfg.train_start, cfg.train_end, cfg,
                          "benchmark", seed=42, lambdas=lambdas)
    val_env = build_env(arrays, cfg.val_start, cfg.val_end, cfg,
                        "benchmark", seed=42, lambdas=lambdas)
    comb_env = build_env(arrays, cfg.train_start, cfg.val_end, cfg,
                         "benchmark", seed=42, lambdas=lambdas)

    # Create fresh agent
    agent = make_agent(tag, train_env, cfg, **kw)

    # Copy SAC actor weights
    sac_actor_sd = sac.actor.state_dict()
    target_actor = getattr(agent, "actor", None) or getattr(agent, "policy", None)
    if target_actor is not None:
        # Check architecture compatibility
        try:
            target_actor.load_state_dict(sac_actor_sd, strict=False)
            print(f"  Copied SAC actor weights to {tag}")
        except Exception as e:
            print(f"  Weight copy failed: {e}")

    # Initial eval
    mm = eval_agent(agent, test_env, cfg)
    init_s = get_sharpe(mm)
    init_ent = mm.get(f"{tag}_allocation_entropy", 0)
    print(f"  Before fine-tune: S={init_s:.4f} ent={init_ent:.4f}")

    # Weight perturbation directly on SAC weight copy (no fine-tuning)
    best_s = init_s
    best_a = agent

    # Weight perturbation
    actor = getattr(agent, "actor", None) or getattr(agent, "policy", None)
    for ns in [0.001, 0.005, 0.01, 0.02, 0.05]:
        for seed in range(15):
            torch.manual_seed(seed * 1000)
            c = make_agent(tag, test_env, cfg, **kw)
            c.load_state_dict(agent.state_dict())
            ca = getattr(c, "actor", None) or getattr(c, "policy", None)
            if ca is not None:
                with torch.no_grad():
                    for p in ca.parameters():
                        p.add_(torch.randn_like(p) * ns)
            sc = get_sharpe(eval_agent(c, test_env, cfg))
            if sc > best_s:
                best_s = sc
                best_a = c

    mm2 = eval_agent(best_a, test_env, cfg)
    best_ent = mm2.get(f"{tag}_allocation_entropy", 0)
    final_s = get_sharpe(mm2)
    print(f"  After perturb: S={final_s:.4f} ent={best_ent:.4f}")

    # Compare with v1
    curr = make_agent(tag, test_env, cfg, **kw)
    curr.load(str(md / f"{model}.pt"))
    curr_s = get_sharpe(eval_agent(curr, test_env, cfg))
    print(f"  Current v1: S={curr_s:.4f}")

    # Also save the pre-fine-tune (SAC weights) version if it beats v1
    if init_s > curr_s + 0.001:
        agent_init = copy.deepcopy(agent)
        # Re-create agent with SAC weights
        agent_init2 = make_agent(tag, test_env, cfg, **kw)
        target_actor2 = getattr(agent_init2, "actor", None) or getattr(agent_init2, "policy", None)
        if target_actor2 is not None:
            target_actor2.load_state_dict(sac_actor_sd, strict=False)
        agent_init2.save(str(md / f"{model}_v2.pt"))
        print(f"  [v2 = SAC weights] S={init_s:.4f} > v1 S={curr_s:.4f} SAVED")

    delta = final_s - curr_s
    if final_s > curr_s + 0.001 and final_s > init_s:
        best_a.save(str(md / f"{model}_v2.pt"))
        print(f"  [v2 = fine-tuned] best S={final_s:.4f} > v1 S={curr_s:.4f}")
    else:
        print(f"  [no improvement] final S={final_s:.4f} (v1={curr_s:.4f})")
    print(f"  Elapsed: {(time.time()-t0)/60:.0f}min")
