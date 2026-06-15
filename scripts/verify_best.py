"""Verify best SAC variant."""
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

def get_sharpe(m):
    for k in m:
        if "sharpe" in k and "std" not in k:
            return m[k]
    return 0.0

# Verify best variant
a = make_agent("SAC", test_env, cfg, weight_decay=1e-4, actor_wd=1e-1, alpha_mult=100.0)
a.load(str(model_dir / "sac_best_variant.pt"))
mm = eval_agent(a, test_env, cfg)
print("Best variant:")
print(f"  S={get_sharpe(mm):.4f}")
print(f"  Return={mm.get('SAC_total_return',0):.4f}")
print(f"  WinRate={mm.get('SAC_win_rate',0):.4f}")

# Base
b = make_agent("SAC", test_env, cfg, weight_decay=1e-4, actor_wd=1e-1, alpha_mult=100.0)
b.load(str(model_dir / "sac.pt"))
mb = eval_agent(b, test_env, cfg)
print("Base:")
print(f"  S={get_sharpe(mb):.4f}")
print(f"  Return={mb.get('SAC_total_return',0):.4f}")
print(f"  WinRate={mb.get('SAC_win_rate',0):.4f}")

# Ensemble
sys.path.insert(0, str(ROOT / "scripts"))
from ensemble_agent import EnsembleAgent  # type: ignore
ens = EnsembleAgent([a, b])
me = eval_agent(ens, test_env, cfg)
print("Base + Variant Ensemble:")
print(f"  S={get_sharpe(me):.4f}")
print(f"  Return={me.get('SAC_total_return',0):.4f}")
print(f"  WinRate={me.get('SAC_win_rate',0):.4f}")
