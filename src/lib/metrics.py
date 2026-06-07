from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 365) -> float:
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 365) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 2:
        return float("inf") if len(downside) == 0 else 0.0
    if np.std(downside) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(downside) * np.sqrt(periods_per_year))


def calmar_ratio(total_return: float, max_drawdown: float) -> float:
    if max_drawdown >= 0:
        return 0.0
    return float(total_return / abs(max_drawdown))


def max_drawdown(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    peak = np.maximum.accumulate(values)
    dd = np.divide(values, peak, out=np.zeros_like(values), where=peak > 1e-12) - 1
    return float(np.nanmin(dd))


def win_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def total_return(pv: np.ndarray) -> float:
    if len(pv) < 2:
        return 0.0
    return float(pv[-1] / pv[0] - 1)


def volatility(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns))


def profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses < 1e-12:
        return float("inf") if gains > 1e-12 else 1.0
    return float(gains / losses)


def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    threshold = value_at_risk(returns, confidence)
    tail = returns[returns <= threshold]
    if len(tail) < 1:
        return threshold
    return float(tail.mean())


def allocation_entropy(weights_history: list[np.ndarray], n_assets: int) -> float:
    entropies = []
    for w in weights_history:
        ws = w[w > 1e-12]
        if len(ws) < 2:
            continue
        entropies.append(-np.sum(ws * np.log(ws)) / np.log(n_assets))
    return float(np.mean(entropies)) if entropies else 0.0


def hit_rate(pred_stop: np.ndarray, actual_max_dd: np.ndarray) -> float:
    if len(pred_stop) == 0:
        return 0.0
    return float(np.mean(pred_stop >= actual_max_dd))


def false_positive_rate(
    pred_stop: np.ndarray, actual_max_dd: np.ndarray,
    rebound: np.ndarray | None = None,
    rebound_threshold: float = 0.10,
) -> float:
    hits = pred_stop <= actual_max_dd
    if not hits.any():
        return 0.0
    if rebound is None:
        return float(np.mean(hits))
    rebounded = rebound > rebound_threshold
    fps = hits & rebounded
    return float(np.mean(fps))


def turnover(weights_history: list[np.ndarray]) -> float:
    if len(weights_history) < 2:
        return 0.0
    return float(sum(
        np.sum(np.abs(weights_history[i] - weights_history[i - 1]))
        for i in range(1, len(weights_history))
    ))
