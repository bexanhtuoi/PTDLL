from __future__ import annotations

import numpy as np
import pandas as pd

from config import RAW_DIR

COINS_15 = [
    "BTC-USD", "LTC-USD", "XRP-USD", "DOGE-USD", "XMR-USD",
    "DASH-USD", "XLM-USD", "USDT-USD", "ETH-USD", "ETC-USD",
    "WAVES-USD", "ZEC-USD", "DCR-USD", "NEO-USD", "BNB-USD",
]

COIN_SHORT = {t: t.replace("-USD", "") for t in COINS_15}


def generate_synthetic_ohlcv(symbol: str, periods: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + sum(ord(c) for c in symbol))
    base = {"BTCUSDT": 65000, "ETHUSDT": 3500, "BNBUSDT": 600}.get(symbol, 1000)
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC")
    drift = 0.00008
    noise = rng.normal(0, 0.004, periods)
    cycle = 0.002 * np.sin(np.linspace(0, 18 * np.pi, periods))
    returns = drift + noise + cycle
    close = base * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0015, 0.0006, periods))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(mean=8, sigma=0.35, size=periods)
    return pd.DataFrame({"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.drop_duplicates("timestamp").sort_values("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)


def load_all_coins() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for ticker in COINS_15:
        short = COIN_SHORT[ticker]
        path = RAW_DIR / f"{short}_yf.csv"
        if path.exists():
            frames[short] = pd.read_csv(path)
    return frames


def aligned_prices(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_idx = None
    for short, df in frames.items():
        ts = set(pd.to_datetime(df["timestamp"]).drop_duplicates())
        if common_idx is None:
            common_idx = ts
        else:
            common_idx &= set(ts)
    common_idx = sorted(common_idx)
    aligned = pd.DataFrame({"timestamp": common_idx})
    for short, df in frames.items():
        lookup = df.set_index(pd.to_datetime(df["timestamp"]))["close"]
        aligned[short] = aligned["timestamp"].map(lookup)
    return aligned.dropna().reset_index(drop=True)
