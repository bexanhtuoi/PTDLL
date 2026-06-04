from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def time_train_validation_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(df) * train_ratio)
    validation_end = int(len(df) * (train_ratio + validation_ratio))
    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()
    return train, validation, test


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 365) -> float:
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def max_drawdown(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    peak = np.maximum.accumulate(values)
    dd = np.divide(values, peak, out=np.zeros_like(values), where=peak > 1e-12) - 1
    return float(np.nanmin(dd))


def sharpe_from_equity(equity: pd.Series, periods_per_year: int = 365) -> float:
    returns = equity.pct_change().dropna()
    if returns.std() == 0 or len(returns) == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def max_drawdown_from_equity(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def win_rate_from_trades(trades: pd.DataFrame) -> float:
    sells = trades[trades["side"] == "SELL"]
    if len(sells) == 0 or "pnl" not in sells.columns:
        return 0.0
    return float((sells["pnl"] > 0).mean())
