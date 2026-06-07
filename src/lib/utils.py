from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_csv(data: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(data.keys())
    lines = [",".join(keys)]
    rows = zip(*[np.asarray(v) for v in data.values()])
    for row in rows:
        lines.append(",".join(str(x) for x in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_history(name: str, history: list[dict], test_metrics: dict, history_path: Path) -> None:
    train = []
    validate = []
    for h in history:
        train.append({"episode": h["episode"], "sharpe": h["train_sharpe"]})
        validate.append({k: v for k, v in h.items() if k != "train_sharpe"})
    data = load_json(history_path)
    data[name] = {"train": train, "validate": validate, "test": test_metrics}
    save_json(data, history_path)


def shared_dates(
    coin_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[list, np.ndarray]:
    ts_sets = [set(coin_data[s][0].tolist()) for s in coin_data]
    common = sorted(set.intersection(*ts_sets))
    return common, np.array(common, dtype="datetime64[D]")


def ffill_grid(
    timestamps: np.ndarray, values: np.ndarray, grid: list
) -> np.ndarray:
    ts_to_idx = {str(ts): j for j, ts in enumerate(timestamps)}
    out = np.zeros((len(grid), values.shape[1]), dtype=np.float32)
    last = values[0]
    for j, ts in enumerate(grid):
        key = str(ts)
        if key in ts_to_idx:
            last = values[ts_to_idx[key]]
        out[j] = last
    return out


def btc_grid(
    coin_data: dict, shorts: list, grid: list, T: int
) -> np.ndarray:
    close_btc = coin_data[shorts[0]][1]
    ts_btc = coin_data[shorts[0]][0]
    ts_to_idx = {str(ts): j for j, ts in enumerate(ts_btc)}
    btc = np.zeros(T)
    last = close_btc[0]
    for j, ts in enumerate(grid):
        key = str(ts)
        if key in ts_to_idx:
            last = close_btc[ts_to_idx[key]]
        btc[j] = last
    return btc


def norm_w(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, 1)
    total = w.sum()
    if total < 1e-10:
        return np.ones_like(w) / len(w)
    return w / total