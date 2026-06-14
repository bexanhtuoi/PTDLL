from __future__ import annotations

import numpy as np

from config import PipelineConfig
from portfolio.base import CryptoPortfolioEnv, build_cube


def build_env(
    coin_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    start: str, end: str, cfg: PipelineConfig,
    reward_style: str | None = None,
    lambdas: tuple[float, float, float, float] | None = None,
    seed: int | None = None,
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
    kwargs = dict(
        cube=cube, asset_names=asset_names, feature_names=feature_names,
        date_index=date_index, lookback=cfg.lookback,
        episode_years=cfg.episode_years, step_days=cfg.rebalance_days,
        fee_rate=cfg.fee_rate, seed=seed or cfg.random_state,
        reward_style=reward_style or cfg.reward_style,
    )
    if lambdas:
        kwargs["lambdas"] = lambdas
    return CryptoPortfolioEnv(**kwargs)
