from __future__ import annotations

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from config import RISK_HISTORY_PATH, MODEL_DIR, PROCESSED_DIR, PipelineConfig
from dataset.fetch import load_coin_arrays, COINS_15
from portfolio.base import build_cube
from lib.utils import ffill_grid, shared_dates, save_json, load_json, ensure_dirs
from log import get_log

from risk.base import (
    LOOKBACK, N_FEATURES,
    BaseStopModel, StopANN, StopLSTM, StopCNN,
    asym_mae, boundary_reg, auto_label,
)

LABEL_WINDOW = 90
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
PATIENCE = 10

MODEL_NAMES: dict[str, type[BaseStopModel]] = {
    "ann": StopANN, "lstm": StopLSTM, "cnn": StopCNN,
}


def to_arrays(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = PROCESSED_DIR / "risk_data.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["xs"], cached["idxs"], cached["targets"], cached["dates"]

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

    row_xs, row_is, row_ts, row_ds = [], [], [], []
    for t in range(LOOKBACK, T - LABEL_WINDOW):
        for old_i in risk_indices:
            x = cube[t - LOOKBACK:t, old_i, :N_FEATURES].copy()
            x[np.isnan(x)] = 0.0
            row_xs.append(x)
            row_is.append(old_to_risk[old_i])
            row_ts.append(auto_label(close_grid[t, old_i], close_grid[t:t + LABEL_WINDOW, old_i]))
            row_ds.append(date_index[t])

    xs = np.array(row_xs, dtype=np.float32)
    idxs = np.array(row_is, dtype=np.int64)
    targets = np.array(row_ts, dtype=np.float32)
    dates = np.array(row_ds, dtype="datetime64[D]")

    ensure_dirs(PROCESSED_DIR)
    np.savez_compressed(cache_path, xs=xs, idxs=idxs, targets=targets, dates=dates)
    return xs, idxs, targets, dates


def build_data(cfg: PipelineConfig) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    xs, idxs, targets, dates = to_arrays(cfg)

    train_end = np.datetime64(cfg.train_end)
    val_start = np.datetime64(cfg.val_start)
    test_start = np.datetime64(cfg.test_start)

    train_m = dates < (train_end - np.timedelta64(LABEL_WINDOW, "D"))
    val_m = (dates >= val_start) & (dates < test_start)
    test_m = dates >= test_start

    return {
        "train": (xs[train_m], idxs[train_m], targets[train_m]),
        "val": (xs[val_m], idxs[val_m], targets[val_m]),
        "test": (xs[test_m], idxs[test_m], targets[test_m]),
    }


def to_loader(
    xs: np.ndarray, idxs: np.ndarray, targets: np.ndarray,
    batch_size: int = BATCH_SIZE, shuffle: bool = True,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(xs),
        torch.from_numpy(idxs),
        torch.from_numpy(targets).unsqueeze(1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train(
    model_name: str, cfg: PipelineConfig | None = None,
) -> tuple[BaseStopModel, list[dict]]:
    model_cls = MODEL_NAMES[model_name]
    cfg = cfg or PipelineConfig()
    ensure_dirs(MODEL_DIR)

    data = build_data(cfg)
    train_loader = to_loader(*data["train"], shuffle=True)
    val_loader = to_loader(*data["val"], shuffle=False)
    test_loader = to_loader(*data["test"], shuffle=False)

    model = model_cls()
    opt = optim.Adam(model.parameters(), lr=LR)
    log = get_log(f"risk_{model_name}")
    best_path = MODEL_DIR / f"risk_{model_name}.pt"

    best_val = float("inf")
    no_improve = 0
    history: list[dict] = []

    for ep in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, idx, tgt in train_loader:
            opt.zero_grad()
            pred = model(x, idx)
            loss = asym_mae(pred, tgt) + boundary_reg(pred)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, idx, tgt in val_loader:
                val_loss += asym_mae(model(x, idx), tgt).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improve = 0
            model.save(str(best_path))
            history.append({"epoch": ep + 1, "train_loss": train_loss, "val_loss": val_loss})
        else:
            no_improve += 1

        log.write(f"Epoch {ep+1:3d}: train={train_loss:.6f} val={val_loss:.6f}{' *' if improved else ''}")

        if no_improve >= PATIENCE:
            log.write(f"Early stop at epoch {ep+1}")
            break

    model.load(str(best_path))

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, idx, tgt in test_loader:
            test_loss += asym_mae(model(x, idx), tgt).item() * x.size(0)
    test_loss /= len(test_loader.dataset)
    log.write(f"{model_name} done! Test MAE={test_loss:.6f}")

    data_dict = load_json(RISK_HISTORY_PATH)
    data_dict[model_name] = {"train": history, "test": {"mae": test_loss}}
    save_json(data_dict, RISK_HISTORY_PATH)

    return model, history
