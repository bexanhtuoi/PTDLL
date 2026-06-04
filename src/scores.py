from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_BANNED: set[str] = {
    "timestamp", "label", "target", "future_return",
    "open", "high", "low", "close", "volume",
    "sma_short", "sma_long", "sma_diff",
    "ma_20", "ma_50", "ma_200",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in FEATURE_BANNED and pd.api.types.is_numeric_dtype(df[c])
    ]


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    open_ = out["open"].to_numpy(dtype=np.float64)
    volume = out["volume"].to_numpy(dtype=np.float64)

    out["candle_return"] = (close - open_) / open_
    out["high_low_range"] = (high - low) / close
    out["body_ratio"] = np.abs(close - open_) / (high - low + 1e-12)

    out["return_7d"] = out["close"].pct_change(7)
    out["return_14d"] = out["close"].pct_change(14)
    out["return_30d"] = out["close"].pct_change(30)

    out["ma_20"] = out["close"].rolling(20).mean()
    out["ma_50"] = out["close"].rolling(50).mean()
    out["ma_200"] = out["close"].rolling(200).mean()
    out["distance_to_ma20"] = (close - out["ma_20"].to_numpy()) / (out["ma_20"].to_numpy() + 1e-12)
    out["distance_to_ma50"] = (close - out["ma_50"].to_numpy()) / (out["ma_50"].to_numpy() + 1e-12)
    out["trend_regime"] = (out["ma_50"].to_numpy() > out["ma_200"].to_numpy()).astype(int)

    out["return_1"] = out["close"].pct_change()
    out["volatility_14d"] = out["return_1"].rolling(14).std()
    out["volatility_30d"] = out["return_1"].rolling(30).std()

    rolling_max = np.maximum.accumulate(close)
    out["drawdown"] = close / rolling_max - 1

    out["volume_change"] = np.diff(volume, prepend=volume[0]) / (volume + 1e-12)
    vol_ma = pd.Series(volume).rolling(20).mean().to_numpy()
    out["relative_volume"] = volume / (vol_ma + 1e-12)

    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50).clip(0, 100)

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_histogram"] = out["macd"] - out["macd_signal"]

    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
