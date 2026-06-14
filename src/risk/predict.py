from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import PipelineConfig, MODEL_DIR
from dataset.fetch import load_coin_arrays, COINS_15
from portfolio.base import build_cube
from lib.utils import ffill_grid, shared_dates, ensure_dirs
from config import PREDICTIONS_DIR

from risk.base import LOOKBACK, N_FEATURES, BaseStopModel
from risk.train import LABEL_WINDOW, MODEL_NAMES


def predict_all(
    model: BaseStopModel,
    cfg: PipelineConfig | None = None,
    split: str = "test",
    save_csv: bool = True,
    out_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    cfg = cfg or PipelineConfig()
    out_dir = out_dir or PREDICTIONS_DIR
    ensure_dirs(out_dir)

    coin_arrays = load_coin_arrays()
    usdt_short = COINS_15[7].replace("-USD", "")
    cube, shorts, _, date_index = build_cube(coin_arrays)
    T = cube.shape[0]
    grid, _ = shared_dates(coin_arrays)

    close_grids = []
    for short in shorts:
        ts, close, _ = coin_arrays[short]
        cg = ffill_grid(ts, close.reshape(-1, 1), grid).ravel()
        close_grids.append(cg)
    close_grid = np.column_stack(close_grids)

    usdt_i = shorts.index(usdt_short) if usdt_short in shorts else -1
    risk_indices = [i for i in range(len(shorts)) if i != usdt_i]
    old_to_risk = {old: new for new, old in enumerate(risk_indices)}

    split_map = {
        "train": (cfg.train_start, cfg.train_end),
        "val": (cfg.val_start, cfg.test_start),
        "test": (cfg.test_start, cfg.test_end),
    }
    period_start, period_end = split_map.get(split, split_map["test"])
    ps = np.datetime64(period_start)
    pe = np.datetime64(period_end)

    rows = []
    for t in range(LOOKBACK, T - LABEL_WINDOW):
        if date_index[t] < ps or date_index[t] >= pe:
            continue
        for old_i in risk_indices:
            x = cube[t - LOOKBACK:t, old_i, :N_FEATURES]
            if np.any(np.isnan(x)):
                continue
            x_clean = np.nan_to_num(x, 0).astype(np.float32)
            coin_idx = old_to_risk[old_i]
            pred = float(model.predict(x_clean[np.newaxis, :, :], np.array([coin_idx]))[0])
            close_val = float(close_grid[t, old_i])
            future_slice = close_grid[t:min(t + 90, T), old_i]
            if len(future_slice) < 2:
                continue
            low = float(future_slice.min())
            dd = (close_val - low) / (close_val + 1e-12)
            actual = float(np.clip(dd * 1.2, 0.05, 0.50))
            rows.append({
                "date": str(date_index[t]),
                "coin": shorts[old_i],
                "close": close_val,
                "pred_stop": pred,
                "actual_stop": actual,
                "hit": int(pred >= actual),
            })

    df = pd.DataFrame(rows)
    if save_csv:
        fname = out_dir / f"risk_pred_{split}.csv"
        df.to_csv(fname, index=False)
        print(f"  Saved {len(df)} predictions to {fname}")

    return {"dates": date_index, "df": df}


def make_risk_agent(model_name: str, env=None) -> BaseStopModel | None:
    model_cls = MODEL_NAMES.get(model_name)
    if model_cls is None:
        return None
    path = MODEL_DIR / f"risk_{model_name}.pt"
    if not path.exists():
        print(f"No saved model at {path}")
        return None
    model = model_cls()
    model.load(str(path))
    return model
