from __future__ import annotations

import numpy as np


def pct_change(x: np.ndarray, n: int) -> np.ndarray:
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n < 1 or n >= len(x):
        return result
    result[n:] = (x[n:] - x[:-n]) / (x[:-n] + 1e-12)
    return result


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(x, np.nan, dtype=np.float64)
    cum = np.cumsum(x, dtype=np.float64)
    result[window - 1:] = cum[window - 1:] / window
    result[window - 1:] -= np.concatenate([[0], cum[:-window]]) / window
    return result


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    var = rolling_mean(x ** 2, window) - rolling_mean(x, window) ** 2
    var = np.maximum(var, 0)
    return np.sqrt(var)


def ewm_mean(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def candle(close: np.ndarray, high: np.ndarray, low: np.ndarray, open_: np.ndarray) -> np.ndarray:
    n = len(close)
    out = np.empty((n, 3), dtype=np.float64)
    out[:, 0] = (close - open_) / (open_ + 1e-12)
    out[:, 1] = (high - low) / (close + 1e-12)
    out[:, 2] = np.abs(close - open_) / (high - low + 1e-12)
    return out


def returns(close: np.ndarray) -> np.ndarray:
    n = len(close)
    out = np.empty((n, 3), dtype=np.float64)
    out[:, 0] = pct_change(close, 7)
    out[:, 1] = pct_change(close, 14)
    out[:, 2] = pct_change(close, 30)
    return out


def ma(close: np.ndarray) -> np.ndarray:
    n = len(close)
    ma20 = rolling_mean(close, 20)
    ma50 = rolling_mean(close, 50)
    ma200 = rolling_mean(close, 200)
    out = np.empty((n, 6), dtype=np.float64)
    out[:, 0] = ma20
    out[:, 1] = ma50
    out[:, 2] = ma200
    out[:, 3] = (close - ma20) / (ma20 + 1e-12)
    out[:, 4] = (close - ma50) / (ma50 + 1e-12)
    out[:, 5] = (ma50 > ma200).astype(np.float64)
    return out


def volatility(close: np.ndarray) -> np.ndarray:
    n = len(close)
    ret_1d = pct_change(close, 1)
    out = np.empty((n, 3), dtype=np.float64)
    out[:, 0] = ret_1d
    out[:, 1] = rolling_std(ret_1d, 14)
    out[:, 2] = rolling_std(ret_1d, 30)
    return out


def drawdown(close: np.ndarray) -> np.ndarray:
    rolling_max = np.maximum.accumulate(close)
    return (close / rolling_max - 1).reshape(-1, 1)


def volume(volume: np.ndarray) -> np.ndarray:
    n = len(volume)
    change = np.full(n, np.nan, dtype=np.float64)
    change[0] = 0.0
    change[1:] = (volume[1:] - volume[:-1]) / (volume[:-1] + 1e-12)
    vol_ma = rolling_mean(volume, 20)
    out = np.empty((n, 2), dtype=np.float64)
    out[:, 0] = change
    out[:, 1] = volume / (vol_ma + 1e-12)
    return out


def rsi(close: np.ndarray) -> np.ndarray:
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(n, np.nan, dtype=np.float64)
    avg_loss = np.full(n, np.nan, dtype=np.float64)
    avg_gain[13] = float(np.mean(gain[:14]))
    avg_loss[13] = float(np.mean(loss[:14]))
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / 14
    rs_val = np.full(n, np.nan, dtype=np.float64)
    rs_val[13:] = avg_gain[13:] / np.maximum(avg_loss[13:], 1e-12)
    result = np.full(n, 50.0, dtype=np.float64)
    valid = ~np.isnan(rs_val)
    result[valid] = 100 - 100 / (1 + rs_val[valid])
    return np.clip(result, 0, 100).reshape(-1, 1)


def macd(close: np.ndarray) -> np.ndarray:
    n = len(close)
    ema12 = ewm_mean(close, 12)
    ema26 = ewm_mean(close, 26)
    macd_line = ema12 - ema26
    signal = ewm_mean(macd_line, 9)
    out = np.empty((n, 3), dtype=np.float64)
    out[:, 0] = macd_line
    out[:, 1] = signal
    out[:, 2] = macd_line - signal
    return out


def create_features(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    vol: np.ndarray,
) -> np.ndarray:
    features = np.column_stack([
        candle(close, high, low, open_),
        returns(close),
        ma(close),
        volatility(close),
        drawdown(close),
        volume(vol),
        rsi(close),
        macd(close),
    ])
    mask = np.all(np.isfinite(features), axis=1)
    return features[mask]
