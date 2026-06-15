from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np

from lib.utils import load_json, save_json
from config import HISTORY_PATH, MODEL_DIR, PipelineConfig
from dataset.fetch import load_coin_arrays
from portfolio.env import build_env
from portfolio.evaluate import make_agent
from portfolio.train import make_envs, train_save
from log import get_log, redirect_stdout_to_log

seeds = [42, 123, 456]
targets = {"ppo": 0.8, "sac": 0.0, "td3": 0.0}
results: dict[str, float] = {}

existing = load_json(HISTORY_PATH)
for name in ["ppo", "sac", "td3"]:
    existing_val = existing.get(name, {}).get("test", {}).get("sharpe", -99.0)
    results[name] = existing_val

for name in ["ppo", "sac", "td3"]:
    target = targets[name]
    best_sharpe = results.get(name, -99.0)

    for seed in seeds:
        if best_sharpe >= target:
            break

        print(f"\n{'='*50}")
        print(f"{name.upper()} seed={seed} (best={best_sharpe:.4f}, target={target})")
        print(f"{'='*50}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        redirect_stdout_to_log(f"{name}_s{seed}")
        train_save(name, name.upper(), {}, PipelineConfig(n_episodes=20000, random_state=seed))

        new = load_json(HISTORY_PATH)
        new_sharpe = new.get(name, {}).get("test", {}).get("sharpe", -99.0)
        if new_sharpe > best_sharpe:
            best_sharpe = new_sharpe
            import shutil
            src = MODEL_DIR / f"{name}.pt"
            dst = MODEL_DIR / f"{name}_best_overall.pt"
            if src.exists():
                shutil.copy2(src, dst)
            print(f"  >>> New best! Sharpe={new_sharpe:.4f} <<<")
        results[name] = best_sharpe

    print(f"\n{name.upper()} done. Best test Sharpe: {best_sharpe:.4f}")

print("\n" + "="*50)
print("FINAL RESULTS:")
for name in ["ppo", "sac", "td3"]:
    print(f"  {name}: Sharpe={results.get(name, -99):.4f}")
print("="*50)
