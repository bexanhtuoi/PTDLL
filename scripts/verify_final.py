"""Final verification of all portfolio models."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PipelineConfig, MODEL_DIR
from dataset.fetch import load_coin_arrays
from portfolio.env import build_env
from portfolio.evaluate import make_agent, eval_agent


cfg = PipelineConfig()
arrays = load_coin_arrays()
test_env = build_env(arrays, cfg.test_start, cfg.test_end, cfg, "benchmark")
model_dir = MODEL_DIR / "v1" / "portfolio"


def gs(m):
    for k in m:
        if "sharpe" in k and "std" not in k:
            return m[k]
    return 0


results = {}
for name, cls, kw in [
    ("sac", "SAC", dict(weight_decay=1e-4, actor_wd=1e-1, alpha_mult=100.0)),
    ("ppo", "PPO", dict(weight_decay=1e-4, actor_wd=1e-1, entropy_coef=0.01)),
    ("td3", "TD3", dict(weight_decay=1e-4, actor_wd=1e-1)),
]:
    agent = make_agent(cls, test_env, cfg, **kw)
    agent.load(str(model_dir / f"{name}.pt"))
    m = eval_agent(agent, test_env, cfg)
    s = gs(m)
    results[name] = {
        "sharpe": s,
        "return": m.get(f"{cls}_total_return", 0),
        "win_rate": m.get(f"{cls}_win_rate", 0),
        "pos_sharpe": m.get(f"{cls}_positive_sharpe", 0),
        "entropy": m.get(f"{cls}_allocation_entropy", 0),
    }
    print(f"{name}:")
    print(f"  Sharpe={s:.4f}  Ret={results[name]['return']:.4f}  WinRate={results[name]['win_rate']:.4f}")
    print(f"  PosSharpe={results[name]['pos_sharpe']:.3f}  Entropy={results[name]['entropy']:.4f}")

print("\nComparison vs Baseline:")
eq_sharpe = -0.1532
btc_sharpe = -1.8387
for name, r in results.items():
    vs_eq = r["sharpe"] - eq_sharpe
    vs_btc = r["sharpe"] - btc_sharpe
    print(f"{name}: vs EW {vs_eq:+.4f} vs BTC {vs_btc:+.4f}")
