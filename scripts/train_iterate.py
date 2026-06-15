from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MIN_TEST_SHARPE = {"ppo": 0.8, "sac": 0.0, "td3": -0.1}
MAX_ATTEMPTS = 5

seeds = [42, 123, 456, 789, 111, 222, 333, 444, 555, 666]

for name, min_sharpe in MIN_TEST_SHARPE.items():
    for attempt in range(MAX_ATTEMPTS):
        seed = seeds[(attempt + hash(name)) % len(seeds)]
        print(f"\n{'='*60}")
        print(f"Training {name.upper()} (attempt {attempt+1}/{MAX_ATTEMPTS}, seed={seed})")
        print(f"{'='*60}")

        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.main", "portfolio", "train",
             "--mode", "parallel", "--episodes", "20000",
             "--models", name],
            capture_output=True, text=True, cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
        )

        log_path = ROOT / "src" / "log" / f"{name}.log"
        if log_path.exists():
            log_lines = log_path.read_text().strip().split("\n")
            last_line = log_lines[-1] if log_lines else ""
            print(f"  Last log: {last_line}")

        print(f"  Stdout: {result.stdout[-200:]}")
        print(f"  Stderr: {result.stderr[-200:]}")

    print(f"\n{name.upper()} done after {MAX_ATTEMPTS} attempts.")

print("\nAll training complete!")
