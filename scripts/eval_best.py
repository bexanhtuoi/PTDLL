import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import numpy as np
from config import MODEL_DIR, PipelineConfig
from dataset.fetch import load_coin_arrays
from portfolio.env import build_env
from portfolio.evaluate import make_agent

cfg = PipelineConfig()
arrays = load_coin_arrays()
test_env = build_env(arrays, cfg.test_start, cfg.test_end, cfg)

for name in ["ppo", "sac", "td3"]:
    tag = name.upper()
    agent = make_agent(tag, test_env, cfg)
    path = MODEL_DIR / f"{name}_best_seed.pt"
    if path.exists():
        agent.load(str(path))
        metrics = agent.score(test_env)
        btc_ret = metrics.get("btc_hold_return", 0)
        rel_ret = metrics.get("btc_hold_relative_return", 0)
        print(f"{name}: Sharpe={metrics['sharpe']:.4f} Ret={metrics['total_return']:.4f} DD={metrics['max_drawdown']:.4f} Vol={metrics['volatility']:.4f} BTC={btc_ret:.4f} Rel={rel_ret:.4f}")
    else:
        print(f"{name}: {path} not found")
