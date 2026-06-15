"""Check history.json for original metrics."""
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from lib.utils import load_json
h = load_json(Path("log/history.json"))
if not h:
    print("No history file")
else:
    for k, v in h.items():
        train = v.get("train", {})
        val = v.get("val", {})
        test = v.get("test", {})
        print(f"{k}:")
        print(f"  train: sharpe={train.get('sharpe','N/A')}")
        print(f"  val:   sharpe={val.get('sharpe','N/A')}")
        print(f"  test:  sharpe={test.get('sharpe','N/A')}")
