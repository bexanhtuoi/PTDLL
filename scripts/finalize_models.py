"""Finalize portfolio models: update SAC with best variant, retrain PPO/TD3 from scratch."""
from __future__ import annotations

import sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PipelineConfig, MODEL_DIR
from portfolio.train import train_seq

cfg = PipelineConfig(n_episodes=10000, random_state=42, val_interval=500, early_stop_patience=5)

# 1. Update SAC with best variant (0.2850 > 0.2103)
src = MODEL_DIR / "v1" / "portfolio" / "sac_best_variant.pt"
dst = MODEL_DIR / "v1" / "portfolio" / "sac.pt"
if src.exists():
    shutil.copy2(str(src), str(dst))
    print(f"SAC updated: copied sac_best_variant.pt -> sac.pt")

# 2. Retrain PPO from scratch
print("\nRetraining PPO...")
train_seq([("ppo", {})], cfg)

# 3. Retrain TD3 from scratch
print("\nRetraining TD3...")
train_seq([("td3", {})], cfg)

print("\nDONE - All models finalized")
