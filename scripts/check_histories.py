from lib.utils import load_json
from config import HISTORY_PATH, RISK_HISTORY_PATH

ph = load_json(HISTORY_PATH)
rh = load_json(RISK_HISTORY_PATH)
print("portfolio_history.json keys:", list(ph.keys()))
print("risk_history.json keys:", list(rh.keys()))

for name in ["ppo", "sac", "td3"]:
    if name in ph:
        t = ph[name]["test"]
        print(f"  {name}: Sharpe={t['sharpe']:.4f}, Return={t['total_return']:.4f}")
