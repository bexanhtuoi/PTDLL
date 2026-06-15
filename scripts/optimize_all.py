"""Train multiple seeds of all 3 models in parallel using multiprocessing.

Strategy:
- SAC 6 seeds with default lambdas (v1 config)
- PPO 6 seeds with v2 lambdas (higher vol penalty)
- TD3 6 seeds with v2 lambdas

Each seed trains all 3 models in parallel, saving to {model}_s{seed}_{v1|v2}.pt
"""
from __future__ import annotations

import multiprocessing, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def train_one(model: str, seed: int, lambdas_v2: bool):
    """Train one model with given seed. Runs in a subprocess."""
    import torch, numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)

    from config import PipelineConfig, MODEL_DIR
    from dataset.fetch import load_coin_arrays
    from portfolio.env import build_env
    from portfolio.evaluate import make_agent, eval_agent

    tag = model.upper()
    label = "v2" if lambdas_v2 else "v1"
    cfg = PipelineConfig(n_episodes=30000, random_state=seed,
                         val_interval=1000, early_stop_patience=10)
    arrays = load_coin_arrays()
    lambdas = (0.5, 0.35, 0.002, 0.05) if lambdas_v2 else None

    train_env = build_env(arrays, cfg.train_start, cfg.train_end, cfg, "benchmark",
                          seed=seed, lambdas=lambdas)
    val_env = build_env(arrays, cfg.val_start, cfg.val_end, cfg, "benchmark",
                        seed=seed, lambdas=lambdas)
    comb_env = build_env(arrays, cfg.train_start, cfg.val_end, cfg, "benchmark",
                         seed=seed, lambdas=lambdas)

    kw = dict(weight_decay=1e-4, actor_wd=1e-1)
    if tag == "SAC": kw["alpha_mult"] = 100.0
    if tag == "PPO": kw["entropy_coef"] = 0.01

    agent = make_agent(tag, train_env, cfg, **kw)
    agent.fit(train_env, val_env, cfg, comb_env=comb_env)

    path = MODEL_DIR / f"{model}_s{seed}_{label}.pt"
    agent.save(str(path))

    test_env = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark",
                         seed=seed, lambdas=lambdas)
    mm = eval_agent(agent, test_env, cfg)
    for k, v in mm.items():
        if 'sharpe' in k and 'std' not in k:
            s = v
            break
    else:
        s = -99

    print(f"[{time.strftime('%H:%M')}] {model} s{seed} {label}: S={s:.4f}")
    return s


def worker(args):
    try:
        return train_one(*args)
    except Exception as e:
        print(f"[FAIL] {args[:2]}: {e}")
        import traceback
        traceback.print_exc()
        return -99


if __name__ == "__main__":
    multiprocessing.freeze_support()

    N_SEEDS = 4
    BASE = 100  # SAC seeds
    t0 = time.time()

    # SAC with default lambdas (v1)
    sac_jobs = [("sac", BASE + i, False) for i in range(N_SEEDS)]
    # PPO with v2 lambdas
    ppo_jobs = [("ppo", 200 + i, True) for i in range(N_SEEDS)]
    # TD3 with v2 lambdas
    td3_jobs = [("td3", 300 + i, True) for i in range(N_SEEDS)]

    # Run sequentially (one per model type to avoid CPU contention)
    for label, jobs in [("SAC v1", sac_jobs), ("PPO v2", ppo_jobs), ("TD3 v2", td3_jobs)]:
        print(f"\n{'='*60}")
        print(f"Training {label} - {len(jobs)} seeds")
        print(f"{'='*60}")
        for job in jobs:
            worker(job)

    print(f"\nTotal time: {(time.time()-t0)/60:.0f}min")
