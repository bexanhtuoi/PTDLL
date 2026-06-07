from __future__ import annotations

import numpy as np

from config import PipelineConfig
from portfolio.base import CryptoPortfolioEnv, build_cube


def build_env(
    coin_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    start: str, end: str, cfg: PipelineConfig,
) -> CryptoPortfolioEnv:
    filtered: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    start_dt = np.datetime64(start)
    end_dt = np.datetime64(end)
    for short, (timestamps, close, volume) in coin_arrays.items():
        ts = timestamps.astype("datetime64[D]")
        mask = (ts >= start_dt) & (ts < end_dt)
        if mask.any():
            filtered[short] = (timestamps[mask], close[mask], volume[mask])
    cube, asset_names, feature_names, date_index = build_cube(filtered)
    return CryptoPortfolioEnv(
        cube=cube,
        asset_names=asset_names,
        feature_names=feature_names,
        date_index=date_index,
        lookback=cfg.lookback,
        episode_years=cfg.episode_years,
        step_days=cfg.rebalance_days,
        fee_rate=cfg.fee_rate,
        seed=cfg.random_state,
    )
