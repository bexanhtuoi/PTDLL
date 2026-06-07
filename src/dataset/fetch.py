from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from config import RAW_DIR
from lib.utils import ensure_dirs


COINS_15 = [
    "BTC-USD", "LTC-USD", "XRP-USD", "DOGE-USD", "XMR-USD",
    "DASH-USD", "XLM-USD", "USDT-USD", "ETH-USD", "ETC-USD",
    "WAVES-USD", "ZEC-USD", "DCR-USD", "NEO-USD", "BNB-USD",
]

COIN_SHORT = {t: t.replace("-USD", "") for t in COINS_15}


def generate_synthetic_ohlcv(symbol: str, periods: int = 1200, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + sum(ord(c) for c in symbol))
    base = {"BTCUSDT": 65000, "ETHUSDT": 3500, "BNBUSDT": 600}.get(symbol, 1000)
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
    return {"close": close, "high": high, "low": low, "open": open_, "volume": volume}


def crawl_yfinance(start: str = "2017-01-01", end: str = "2026-06-01", interval: str = "1d") -> None:
    ensure_dirs(RAW_DIR)
    for ticker in COINS_15:
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index.name = None
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df["timestamp"] = df.index.astype(str)
        cols = ["open", "high", "low", "close", "volume", "timestamp"]
        df[cols].to_csv(RAW_DIR / f"{COIN_SHORT[ticker]}_yf.csv", index=False)


def load_coin_arrays() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ticker in COINS_15:
        short = COIN_SHORT[ticker]
        path = RAW_DIR / f"{short}_yf.csv"
        if not path.exists():
            continue
        try:
            data = np.genfromtxt(
                path, delimiter=",", dtype=None, encoding="utf-8", names=True,
            )
            if len(data) == 0:
                continue
            ts = data["timestamp"].astype("datetime64[D]")
            close = data["close"].astype(np.float64)
            volume = data["volume"].astype(np.float64)
            valid = ~np.isnan(close) & ~np.isnan(volume)
            if valid.any():
                out[short] = (ts[valid], close[valid], volume[valid])
        except Exception:
            continue
    return out
