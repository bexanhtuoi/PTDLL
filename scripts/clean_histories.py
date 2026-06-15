from lib.utils import load_json, save_json
from config import HISTORY_PATH, RISK_HISTORY_PATH

ph = load_json(HISTORY_PATH)
for k in list(ph.keys()):
    if k.startswith("risk_"):
        del ph[k]
save_json(ph, HISTORY_PATH)
print(f"Cleaned portfolio_history.json. Keys: {list(ph.keys())}")

rh = load_json(RISK_HISTORY_PATH)
print(f"risk_history.json keys: {list(rh.keys())}")
