import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lib.utils import load_json
from config import HISTORY_PATH
d = load_json(HISTORY_PATH)
for name in ["ppo", "sac", "td3"]:
    if name in d:
        t = d[name]["test"]
        print(f"{name}: Test Sharpe={t['sharpe']:.4f}, Return={t['total_return']:.4f}")
        if "validate" in d[name] and d[name]["validate"]:
            best = max(d[name]["validate"], key=lambda h: h["sharpe"])
            print(f"  Best Val: Sharpe={best['sharpe']:.4f}, Return={best['total_return']:.4f}")
