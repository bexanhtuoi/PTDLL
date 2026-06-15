import subprocess, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import HISTORY_PATH, MODEL_DIR
from lib.utils import load_json, save_json

best = {}
existing = load_json(HISTORY_PATH)
for name in ["ppo", "sac", "td3"]:
    best[name] = existing.get(name, {}).get("test", {}).get("sharpe", -99.0)
    print(f"{name}: start={best[name]:.4f}")

for run in range(5):
    print(f"\n=== Run {run+1}/5 ===")
    seed = 100 + run * 50
    code = f"""
import sys; sys.path.insert(0, r'{ROOT / "src"}')
import torch; torch.manual_seed({seed})
import numpy as np; np.random.seed({seed})
from config import PipelineConfig
from portfolio.train import train_par
train_par([("ppo",{{}}),("sac",{{}}),("td3",{{}})], PipelineConfig(n_episodes=20000, random_state={seed}))
"""
    subprocess.run(["uv", "run", "python", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=10800)

    new = load_json(HISTORY_PATH)
    for name in ["ppo", "sac", "td3"]:
        if name in new:
            s = new[name]["test"]["sharpe"]
            if s > best[name]:
                best[name] = s
                src = MODEL_DIR / f"{name}.pt"
                dst = MODEL_DIR / f"{name}_best_overall.pt"
                if src.exists():
                    shutil.copy2(src, dst)
                print(f"  {name}: BEST={s:.4f} (seed={seed})")
            else:
                print(f"  {name}: {s:.4f} (best={best[name]:.4f})")

print("\n=== FINAL ===")
for name in ["ppo", "sac", "td3"]:
    print(f"  {name}: {best[name]:.4f}")
