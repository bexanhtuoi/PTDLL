import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from config import MODEL_DIR, HISTORY_PATH, PipelineConfig
from dataset.fetch import load_coin_arrays
from portfolio.env import build_env
from portfolio.evaluate import make_agent
from lib.utils import load_json, save_json
from portfolio.base import val_params

cfg = PipelineConfig()
arrays = load_coin_arrays()
test_env = build_env(arrays, cfg.test_start, cfg.test_end, cfg)
val_env = build_env(arrays, cfg.val_start, cfg.val_end, cfg)

data = load_json(HISTORY_PATH)

seeds = {"ppo": 888, "sac": 777, "td3": 111}

for name in ["ppo", "sac", "td3"]:
    tag = name.upper()
    agent = make_agent(tag, test_env, cfg)
    path = MODEL_DIR / f"{name}_best_seed.pt"
    if not path.exists():
        print(f"{name}: {path} not found")
        continue
    agent.load(str(path))

    test_m = agent.score(test_env)
    val_vp = val_params(val_env, cfg)
    val_start = int(val_vp[2].integers(val_env.lookback, val_vp[1]))
    val_m = agent.score(val_env, start_idx=val_start, end_idx=val_start + val_vp[0])

    data[name] = {
        "train": [],
        "validate": [{"episode": 1, "sharpe": val_m["sharpe"], "total_return": val_m["total_return"]}],
        "test": test_m,
    }
    print(f"{name}: Sharpe={test_m['sharpe']:.4f} Ret={test_m['total_return']:.4f}")

save_json(data, HISTORY_PATH)
print(f"\nSaved to {HISTORY_PATH}")
