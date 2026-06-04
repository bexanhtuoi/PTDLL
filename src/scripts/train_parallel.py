from __future__ import annotations

import multiprocessing
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

from log import LOG_DIR, redirect_stdout_to_log
from config import MODEL_DIR, REPORT_DIR, ROOT, PipelineConfig
from dataset.fetch import load_all_coins
from models.env import build_env
from models.train import create_agent, train_model
from models.evaluation import run_test
from utils import ensure_dirs, save_json


def train_one(name: str, overrides: dict) -> None:
    redirect_stdout_to_log(name)

    print(f"Loading data for {name}...")
    all_frames = load_all_coins()
    cfg = PipelineConfig()

    train_env = build_env(all_frames, cfg.train_start, cfg.train_end, cfg)
    val_env = build_env(all_frames, cfg.val_start, cfg.val_end, cfg)
    test_env = build_env(all_frames, cfg.test_start, cfg.test_end, cfg)
    print(f"  Train: {train_env.n_steps}d, {train_env.n_assets}a, {train_env.n_features}f")
    print(f"  Val:   {val_env.n_steps}d")
    print(f"  Test:  {test_env.n_steps}d")

    agent = create_agent(name.upper(), train_env, cfg, **overrides)
    print(f"Training {name.upper()} ({cfg.n_episodes} episodes)...")
    history = train_model(agent, train_env, val_env, cfg)
    agent.save(str(MODEL_DIR / f"{name}.pt"))
    save_json(history, REPORT_DIR / f"rl_{name}_val_history.json")
    print(f"Saved model and val history")

    print(f"Testing {name.upper()}...")
    test_m = run_test(agent, test_env, cfg)
    save_json(test_m, REPORT_DIR / f"rl_{name}_test_metrics.json")
    tag = name.upper()
    print(f"Done! Test Sharpe={test_m.get(f'{tag}_sharpe', 0):.4f} Test Return={test_m.get(f'{tag}_total_return', 0):.4f}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    ensure_dirs(MODEL_DIR, REPORT_DIR, Path(LOG_DIR))

    models = [
        ("ppo", {"gamma": 0.97, "entropy_coef": 0.01}),
        ("sac", {}),
        ("td3", {}),
    ]

    processes = []
    for name, overrides in models:
        p = multiprocessing.Process(target=train_one, args=(name, overrides), daemon=False)
        p.start()
        processes.append((name, p))
        print(f"Started {name} (PID {p.pid})")

    for name, p in processes:
        p.join()
        print(f"{name} completed (exit code {p.exitcode})")

    print("\nAll models trained. Run 'python -m scripts.report' to generate charts.")
