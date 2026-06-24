from __future__ import annotations

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from config import RISK_HISTORY_PATH, MODEL_DIR, PROCESSED_DIR, PipelineConfig
from dataset.fetch import load_coin_arrays, COINS_15
from portfolio.base import build_cube, COIN_FEATURE_NAMES
from lib.features import pct_change, rsi, rolling_skew, rolling_down_ratio, rolling_mean
from lib.utils import ffill_grid, shared_dates, save_json, load_json, ensure_dirs
from log import get_log

from risk.base import (
    LOOKBACK, N_FEATURES, N_COINS, STOP_MAX, STOP_MIN,
    BaseStopModel, StopANN, StopLSTM, StopCNN,
    combined_loss, auto_label, FeatureScaler,
    init_head_bias,
)

LABEL_WINDOW = 10
BATCH_SIZES = {"ann": 2048, "lstm": 2048, "cnn": 1024}
EPOCHS = 300
LR = 8e-4
PATIENCE = 30

MODEL_NAMES: dict[str, type[BaseStopModel]] = {
    "ann": StopANN, "lstm": StopLSTM, "cnn": StopCNN,
}

SCALER_PATH = PROCESSED_DIR / "risk_scaler.npz"


def to_arrays(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = PROCESSED_DIR / "risk_data.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached["xs"].shape[-1] == N_FEATURES:
            return cached["xs"], cached["idxs"], cached["targets"], cached["dates"]
        cache_path.unlink()

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

    N_CUBE_FEATURES = 13

    # Precompute extra features for all (t, coin) — temporal, varying
    n_assets = len(shorts)
    extra_cube = np.zeros((T, n_assets, 4), dtype=np.float32)
    for i in range(n_assets):
        c = close_grid[:, i]
        dr = pct_change(c, 1)
        dr[0] = 0.0
        extra_cube[:, i, 0] = rsi(c).ravel()
        extra_cube[:, i, 1] = rolling_skew(dr, 60)
        extra_cube[:, i, 2] = rolling_down_ratio(dr, 60)
        sma200 = rolling_mean(c, 200)
        extra_cube[:, i, 3] = (c - sma200) / (sma200 + 1e-12)
    extra_cube[np.isnan(extra_cube)] = 0.0

    row_xs, row_is, row_ts, row_ds = [], [], [], []
    for t in range(LOOKBACK, T - LABEL_WINDOW):
        for ci, old_i in enumerate(risk_indices):
            x = cube[t - LOOKBACK:t, old_i, :N_CUBE_FEATURES].copy()
            x[np.isnan(x)] = 0.0

            # Extra features: [rsi_14, skew_60, dd_consecutive, sma200_dist]
            extra = extra_cube[t - LOOKBACK:t, old_i, :].copy()

            # Combine: [7 coin, 4 extra, 2 cross(cube[7:9]), 4 regime(cube[9:13])] = 17
            x = np.concatenate([x[:, :7], extra, x[:, 7:]], axis=1)

            targ = auto_label(close_grid[t, old_i], close_grid[t:t + LABEL_WINDOW, old_i])

            row_xs.append(x)
            row_is.append(old_to_risk[old_i])
            row_ts.append(targ)
            row_ds.append(date_index[t])

    xs = np.array(row_xs, dtype=np.float32)
    idxs = np.array(row_is, dtype=np.int64)
    targets = np.array(row_ts, dtype=np.float32)
    dates = np.array(row_ds, dtype="datetime64[D]")

    ensure_dirs(PROCESSED_DIR)
    np.savez_compressed(cache_path, xs=xs, idxs=idxs, targets=targets, dates=dates)
    return xs, idxs, targets, dates


def build_data(cfg: PipelineConfig) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    xs, idxs, raw_targets, dates = to_arrays(cfg)

    test_start = np.datetime64(cfg.test_start)
    test_m = dates >= test_start
    test_idx = np.where(test_m)[0]

    train_m = ~test_m
    train_idx = np.where(train_m)[0]

    # Per-coin z-score labels using ALL training data
    z_mean = np.zeros(N_COINS, dtype=np.float32)
    z_std = np.zeros(N_COINS, dtype=np.float32)
    for ci in range(N_COINS):
        mask = (idxs[train_idx] == ci)
        c_train = raw_targets[train_idx][mask]
        z_mean[ci] = float(c_train.mean()) if len(c_train) > 0 else 0.15
        z_std[ci] = float(c_train.std()) if len(c_train) > 1 else 0.08
        if z_std[ci] < 0.03:
            z_std[ci] = 0.08

    targets_z = raw_targets.astype(np.float32)
    for split_idx in [train_idx, test_idx]:
        ci = idxs[split_idx]
        targets_z[split_idx] = (targets_z[split_idx] - z_mean[ci]) / z_std[ci]
    targets_z = np.clip(targets_z, -3.0, 3.0)

    return {
        "z_mean": z_mean, "z_std": z_std,
        "raw": (xs, idxs, raw_targets.astype(np.float32), dates),
        "train": (xs[train_idx], idxs[train_idx], targets_z[train_idx], dates[train_idx]),
        "test": (xs[test_idx], idxs[test_idx], targets_z[test_idx], dates[test_idx]),
        "test_raw": (xs[test_idx], idxs[test_idx], raw_targets[test_idx].astype(np.float32), dates[test_idx]),
    }


def to_loader(xs, idxs, targets, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(xs),
        torch.from_numpy(idxs),
        torch.from_numpy(targets).unsqueeze(1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train(model_name, cfg=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_cls = MODEL_NAMES[model_name]
    cfg = cfg or PipelineConfig()
    batch_size = BATCH_SIZES.get(model_name, 1024)
    ensure_dirs(MODEL_DIR)
    ensure_dirs(MODEL_DIR / "v2" / "risk")

    log = get_log(f"risk_{model_name}")
    data = build_data(cfg)

    scaler = FeatureScaler()
    train_xs = scaler.fit_transform(data["train"][0])
    test_xs = scaler.transform(data["test"][0])
    np.savez_compressed(SCALER_PATH, mean=scaler.mean, std=scaler.std, dead=scaler.dead,
                        z_mean=data["z_mean"], z_std=data["z_std"])

    log.write(f"Data split: train={len(data['train'][0])} test={len(data['test'][0])}")
    if len(data["train"][3]) > 0:
        log.write(f"Train dates: {data['train'][3][0]} ~ {data['train'][3][-1]}")
    if len(data["test"][3]) > 0:
        log.write(f"Test dates:  {data['test'][3][0]} ~ {data['test'][3][-1]}")

    train_loader = to_loader(train_xs, data["train"][1], data["train"][2], batch_size, shuffle=True)
    test_loader = to_loader(test_xs, data["test"][1], data["test"][2], batch_size, shuffle=False)

    # Raw targets for drawdown-space loss
    test_start = np.datetime64(cfg.test_start)
    train_m = data["raw"][3] < test_start
    train_raw = data["raw"][2][train_m].copy()
    train_raw_loader = DataLoader(TensorDataset(
        torch.from_numpy(train_xs),
        torch.from_numpy(data["train"][1]),
        torch.from_numpy(train_raw).unsqueeze(1),
    ), batch_size=batch_size, shuffle=True)
    test_raw_loader = DataLoader(TensorDataset(
        torch.from_numpy(test_xs),
        torch.from_numpy(data["test"][1]),
        torch.from_numpy(data["test_raw"][2]).unsqueeze(1),
    ), batch_size=batch_size, shuffle=False)

    z_mean_t = torch.from_numpy(data["z_mean"]).float()
    z_std_t = torch.from_numpy(data["z_std"]).float()

    model = model_cls()
    init_head_bias(model, 0.0)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_path = MODEL_DIR / "v2" / "risk" / f"risk_{model_name}_s{seed}.pt"

    history = []
    for ep in range(EPOCHS):
        model.train()
        tl = 0.0
        for x, i, t_raw in train_raw_loader:
            opt.zero_grad()
            pred_z = model(x, i)
            pred_dd = pred_z * z_std_t[i, None] + z_mean_t[i, None]
            loss = combined_loss(pred_dd, t_raw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * x.size(0)
        tl /= len(train_raw_loader.dataset)
        sched.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, i, t_raw in test_raw_loader:
                pred_z = model(x, i)
                pred_dd = pred_z * z_std_t[i, None] + z_mean_t[i, None]
                loss = combined_loss(pred_dd, t_raw)
                vl += loss.item() * x.size(0)
        vl /= len(test_raw_loader.dataset)
        history.append({"epoch": ep + 1, "train_loss": tl, "val_loss": vl})
        log.write(f"Epoch {ep+1:3d}: train={tl:.6f} val={vl:.6f} lr={opt.param_groups[0]['lr']:.2e}")

    model.save(str(best_path))

    # Post-hoc calibration: fit per-coin alpha, beta on training predictions
    model.eval()
    all_tr_pred, all_tr_act, all_tr_i = [], [], []
    with torch.no_grad():
        for x, i, t_raw in train_raw_loader:
            pred_z = model(x, i)
            pred_dd = pred_z * z_std_t[i, None] + z_mean_t[i, None]
            all_tr_pred.append(pred_dd.cpu().numpy().ravel())
            all_tr_act.append(t_raw.cpu().numpy().ravel())
            all_tr_i.append(i.cpu().numpy().ravel())
    all_tr_pred = np.concatenate(all_tr_pred)
    all_tr_act = np.concatenate(all_tr_act)
    all_tr_i = np.concatenate(all_tr_i).astype(int)

    alpha = np.ones(N_COINS, dtype=np.float32)
    beta = np.zeros(N_COINS, dtype=np.float32)
    for ci in range(N_COINS):
        m = (all_tr_i == ci)
        p = all_tr_pred[m]; a = all_tr_act[m]
        if len(p) < 20: continue
        def l1(params):
            return np.abs(params[0] * p + params[1] - a).mean()
        from scipy.optimize import minimize
        res = minimize(l1, [1.0, 0.0], method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-6})
        alpha[ci], beta[ci] = res.x

    # Save calibration params
    scaler_path = str(SCALER_PATH).replace(".npz", "_calib.npz")
    np.savez_compressed(scaler_path, alpha=alpha, beta=beta)

    # Final test eval with calibration (in drawdown space)
    model.eval()
    all_pred, all_act, all_idx = [], [], []
    with torch.no_grad():
        for x, i, t in test_loader:
            pred = model(x, i)
            all_pred.append(pred); all_act.append(t); all_idx.append(i)
    all_pred = torch.cat(all_pred).squeeze(-1).numpy()
    all_act = torch.cat(all_act).squeeze(-1).numpy()
    all_idx = torch.cat(all_idx).numpy()
    # Convert z-space to drawdown space + calibrate
    all_pred_dd = all_pred * data["z_std"][all_idx] + data["z_mean"][all_idx]
    all_pred_cal = np.clip(alpha[all_idx] * all_pred_dd + beta[all_idx], STOP_MIN, STOP_MAX)
    all_act_dd = all_act * data["z_std"][all_idx] + data["z_mean"][all_idx]

    test_corr = float(np.corrcoef(all_pred_cal, all_act_dd)[0, 1]) if len(np.unique(all_act_dd)) > 1 else 0.0
    test_mae = float(np.abs(all_pred_cal - all_act_dd).mean())

    per_coin_pred_std = np.array([np.std(all_pred_cal[all_idx == i]) for i in range(N_COINS)])
    per_coin_act_std = np.array([np.std(all_act_dd[all_idx == i]) for i in range(N_COINS)])
    log.write(f"{model_name} done! (calibrated) Test Corr={test_corr:.4f} MAE={test_mae:.4f}")
    log.write(f"  Per-coin std: pred mean={per_coin_pred_std.mean():.4f}  actual mean={per_coin_act_std.mean():.4f}")

    d = load_json(RISK_HISTORY_PATH)
    d[model_name] = {"train": history, "test": {"mae": test_mae, "corr": test_corr, "pred_std": float(all_pred_cal.std()), "act_std": float(all_act_dd.std())}}
    save_json(d, RISK_HISTORY_PATH)
    return model, history
