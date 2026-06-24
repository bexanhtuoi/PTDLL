from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import PipelineConfig, MODEL_DIR
from dataset.fetch import load_coin_arrays, COINS_15
from portfolio.base import build_cube
from lib.utils import ffill_grid, shared_dates, ensure_dirs
from config import PREDICTIONS_DIR

from risk.base import LOOKBACK, STOP_MIN, STOP_MAX, STOP_RANGE, BaseStopModel, FeatureScaler, N_COINS, EnsembleLSTM
from risk.train import LABEL_WINDOW, MODEL_NAMES, SCALER_PATH


def _load_scaler():
    import os
    scaler = FeatureScaler()
    scaler.z_mean = None
    scaler.z_std = None
    if os.path.exists(SCALER_PATH):
        d = np.load(SCALER_PATH)
        scaler.mean = d["mean"]
        scaler.std = d["std"]
        scaler.dead = d.get("dead") if "dead" in d else None
        scaler.z_mean = d.get("z_mean")
        scaler.z_std = d.get("z_std")
    return scaler


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

    scaler = _load_scaler()

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

    # Precompute extra features for all (t, coin)
    from lib.features import rsi, rolling_skew, rolling_down_ratio, rolling_mean
    T_full = cube.shape[0]
    n_assets = len(shorts)
    extra_cube = np.zeros((T_full, n_assets, 4), dtype=np.float32)
    for i in range(n_assets):
        c = close_grid[:, i]
        dr = np.diff(c, prepend=c[0]) / (c + 1e-12)
        dr[0] = 0.0
        extra_cube[:, i, 0] = rsi(c).ravel()
        extra_cube[:, i, 1] = rolling_skew(dr, 60)
        extra_cube[:, i, 2] = rolling_down_ratio(dr, 60)
        sma200 = rolling_mean(c, 200)
        extra_cube[:, i, 3] = (c - sma200) / (sma200 + 1e-12)
    extra_cube[np.isnan(extra_cube)] = 0.0
    N_CUBE = 13

    rows = []
    for t in range(LOOKBACK, T - LABEL_WINDOW):
        if date_index[t] < ps or date_index[t] >= pe:
            continue
        for old_i in risk_indices:
            x = cube[t - LOOKBACK:t, old_i, :N_CUBE].copy()
            x[np.isnan(x)] = 0.0
            extra = extra_cube[t - LOOKBACK:t, old_i, :].copy()
            x = np.concatenate([x[:, :7], extra, x[:, 7:]], axis=1)
            if scaler.mean is not None:
                x = scaler.transform(x[np.newaxis, :, :])[0]
            coin_idx = old_to_risk[old_i]
            pred_z = float(model.predict(x[np.newaxis, :, :], np.array([coin_idx]))[0])
            if scaler.z_mean is not None and scaler.z_std is not None:
                pred = pred_z * float(scaler.z_std[coin_idx]) + float(scaler.z_mean[coin_idx])
            else:
                pred = pred_z
            pred = float(np.clip(pred, STOP_MIN, STOP_MAX))
            close_val = float(close_grid[t, old_i])
            future_slice = close_grid[t:min(t + LABEL_WINDOW, T), old_i]
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
    # Use ensemble for LSTM
    ens_dir = MODEL_DIR / "v2" / "risk" / "lstm_ensemble"
    if model_name == "lstm" and ens_dir.exists():
        n_models = len(list(ens_dir.glob("model_*.pt")))
        if n_models > 1:
            try:
                model = EnsembleLSTM(ens_dir, n_models)
                return model
            except RuntimeError as e:
                print(f"  lstm ensemble: load failed ({e})")

    model_cls = MODEL_NAMES.get(model_name)
    if model_cls is None:
        return None
    for p in [
        MODEL_DIR / "v2" / "risk" / f"risk_{model_name}.pt",
        MODEL_DIR / f"risk_{model_name}.pt",
        MODEL_DIR / "v1" / "risk" / f"risk_{model_name}.pt",
    ]:
        if p.exists():
            try:
                model = model_cls()
                model.load(str(p))
                return model
            except RuntimeError as e:
                print(f"  {model_name}: checkpoint mismatch ({e})")
                continue
    print(f"No compatible saved model for {model_name}")
    return None
