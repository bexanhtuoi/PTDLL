from __future__ import annotations

import numpy as np
import pandas as pd

from config import PipelineConfig
from utils import max_drawdown, sharpe_ratio


def filter_by_date(frames: dict[str, pd.DataFrame], start: str, end: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for short, df in frames.items():
        ts = pd.to_datetime(df["timestamp"])
        mask = (ts >= start) & (ts < end)
        if mask.any():
            out[short] = df.loc[mask].reset_index(drop=True)
    return out


def compute_coin_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    prev_close = np.r_[close[0:1], close[:-1]]
    daily_ret = (close - prev_close) / (prev_close + 1e-12)
    out["return_1d"] = daily_ret
    for d in [7, 30, 90]:
        prev = np.r_[close[:d], close[:-d]]
        ret = (close - prev) / (prev + 1e-12)
        out[f"return_{d}d"] = ret
    out["volatility"] = pd.Series(daily_ret).rolling(20, min_periods=1).std().to_numpy()
    rolling_max = np.maximum.accumulate(close)
    out["drawdown"] = close / rolling_max - 1
    prev_volume = np.r_[volume[0:1], volume[:-1]]
    volume_change = (volume - prev_volume) / (prev_volume + 1e-12)
    out["volume_change"] = np.clip(volume_change, -5, 5)
    return out


def aligned_features(
    coin_data: dict[str, pd.DataFrame]
) -> tuple[np.ndarray, list[str], list[str], pd.DatetimeIndex]:
    shorts = list(coin_data.keys())
    common_idx = None
    for short, df in coin_data.items():
        ts = set(pd.to_datetime(df["timestamp"]).drop_duplicates())
        if common_idx is None:
            common_idx = ts
        else:
            common_idx &= ts
    common_idx = sorted(common_idx)
    date_index = pd.DatetimeIndex(common_idx)
    T = len(date_index)
    n_assets = len(shorts)
    fps = [compute_coin_features(coin_data[short]) for short in shorts]
    per_coin_names = list(fps[0].columns)
    per_coin = np.zeros((T, n_assets, len(per_coin_names)), dtype=np.float32)
    for i, (short, fp) in enumerate(zip(shorts, fps)):
        df = coin_data[short]
        ts_key = pd.to_datetime(df["timestamp"])
        aligned = fp.set_index(ts_key).reindex(date_index).ffill().fillna(0)
        per_coin[:, i, :] = aligned.to_numpy(dtype=np.float32)
    btc_idx = 0
    return_30d = per_coin[:, :, per_coin_names.index("return_30d")]
    relative_strength = return_30d - return_30d[:, btc_idx:btc_idx + 1]
    ret_1d_idx = per_coin_names.index("return_1d")
    daily_ret = per_coin[:, :, ret_1d_idx]
    corr = np.full((T, n_assets, 1), 0.0, dtype=np.float32)
    for t in range(60, T):
        for a in range(n_assets):
            r = np.corrcoef(daily_ret[t - 60:t, btc_idx], daily_ret[t - 60:t, a])
            corr[t, a, 0] = r[0, 1] if not np.isnan(r[0, 1]) else 0.0
    cross_names = ["relative_strength_vs_BTC", "correlation_vs_BTC"]

    btc_series = pd.Series(
        coin_data[shorts[0]]["close"].values,
        index=pd.to_datetime(coin_data[shorts[0]]["timestamp"]),
    )
    btc_aligned = btc_series.reindex(date_index).ffill().fillna(0).values
    btc_sma200 = pd.Series(btc_aligned).rolling(200, min_periods=1).mean().values
    ma200_pos = (btc_aligned - btc_sma200) / (btc_sma200 + 1e-12)
    vol_idx = per_coin_names.index("volatility")
    mkt_vol = per_coin[:, :, vol_idx].mean(axis=1)
    close_90 = np.r_[btc_aligned[:90], btc_aligned[:-90]]
    mom_90 = (btc_aligned - close_90) / (close_90 + 1e-12)
    breadth = (per_coin[:, :, per_coin_names.index("return_30d")] > 0).mean(axis=1)
    regime_names = ["btc_ma200_position", "market_volatility", "btc_momentum_90d", "market_breadth"]

    n_features = len(per_coin_names) + len(cross_names) + len(regime_names) + 1
    cube = np.zeros((T, n_assets, n_features), dtype=np.float32)
    fi = 0
    cube[:, :, fi:fi + len(per_coin_names)] = per_coin
    fi += len(per_coin_names)
    cube[:, :, fi:fi + 1] = relative_strength[:, :, np.newaxis]
    fi += 1
    cube[:, :, fi:fi + 1] = corr
    fi += 1
    cube[:, :, fi:fi + len(regime_names)] = np.broadcast_to(
        np.stack([ma200_pos, mkt_vol, mom_90, breadth], axis=1)[:, np.newaxis, :],
        (T, n_assets, len(regime_names)),
    )
    fi += len(regime_names)
    feature_names = per_coin_names + cross_names + regime_names + ["weight"]
    return cube, shorts, feature_names, date_index


class CryptoPortfolioEnv:
    def __init__(
        self,
        coin_data: dict[str, pd.DataFrame],
        lookback: int = 60,
        episode_years: int = 2,
        step_days: int = 63,
        fee_rate: float = 0.001,
        seed: int = 42,
        lambdas: tuple[float, float, float] = (0.2, 0.25, 0.002),
    ):
        cube, self.asset_names, self.feature_names, self.date_index = aligned_features(coin_data)
        self.cube = cube.astype(np.float32, copy=False)
        self.n_assets = len(self.asset_names)
        self.n_features = self.cube.shape[2]
        self.lookback = lookback
        self.norm_mean, self.norm_std = self._compute_norm_stats()
        self.episode_len = episode_years * 365
        self.step_days = step_days
        self.fee_rate = fee_rate
        self.lambda_vol, self.lambda_dd, self.lambda_turnover = lambdas
        self.rng = np.random.default_rng(seed)
        self.n_steps = self.cube.shape[0]

    def _compute_norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        n_feat = self.cube.shape[2] - 1
        channels = self.cube[:, :, :n_feat]
        mean = channels.mean(axis=(0, 1), keepdims=True)
        std = channels.std(axis=(0, 1), keepdims=True) + 1e-8
        return mean.astype(np.float32), std.astype(np.float32)

    def _get_raw(self, start: int, end: int) -> np.ndarray:
        return self.cube[start:end]

    def _get_norm(self, start: int, end: int) -> np.ndarray:
        s = self.cube[start:end].copy()
        s[:, :, :-1] = (s[:, :, :-1] - self.norm_mean) / self.norm_std
        return s

    def reset(self, start_idx: int | None = None, end_idx: int | None = None) -> np.ndarray:
        if start_idx is not None and end_idx is not None:
            self.start_idx = max(int(start_idx), self.lookback)
            self.end_idx = min(int(end_idx), self.n_steps - 1)
        elif start_idx is not None:
            self.start_idx = max(int(start_idx), self.lookback)
            self.end_idx = min(self.start_idx + self.episode_len, self.n_steps - 1)
        else:
            max_start = self.n_steps - self.lookback - self.episode_len
            max_start = max(max_start, self.lookback + 1)
            if self.rng.random() < 0.6:
                vol_idx = self.feature_names.index("volatility")
                mkt_vols = self.cube[:, :, vol_idx].mean(axis=1)
                high_vol = np.argsort(mkt_vols)[-len(mkt_vols) // 3:]
                candidates = high_vol[(high_vol >= self.lookback) & (high_vol <= max_start)]
                if len(candidates) > 0:
                    self.start_idx = int(self.rng.choice(candidates))
                else:
                    self.start_idx = int(self.rng.integers(self.lookback, max_start))
            else:
                self.start_idx = int(self.rng.integers(self.lookback, max_start))
            self.end_idx = min(self.start_idx + self.episode_len, self.n_steps - 1)
        self.idx = self.start_idx
        self.weights = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        self.last_weights = self.weights.copy()
        self.portfolio_value = 1.0
        self.portfolio_values: list[float] = [1.0]
        self.action_log: list[dict] = []
        self._period_returns: list[float] = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        start = self.idx - self.lookback
        state = self._get_norm(start, self.idx)
        state[:, :, -1:] = np.broadcast_to(
            self.weights[np.newaxis, :, np.newaxis], (self.lookback, self.n_assets, 1)
        )
        return state

    def step(self, action_weights: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action_weights = np.asarray(action_weights, dtype=np.float64)
        action_weights = np.clip(action_weights, 0, 1)
        total = action_weights.sum()
        if total < 1e-10:
            action_weights = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        else:
            action_weights = action_weights / total
        self.last_weights = self.weights.copy()
        self.weights = action_weights.copy()
        interval = min(self.step_days, self.end_idx - self.idx)
        interval = max(interval, 1)
        next_idx = min(self.idx + interval, self.end_idx)
        ret_idx = self.feature_names.index("return_1d")
        period_returns = self._get_raw(self.idx, next_idx)[:, :, ret_idx]
        portfolio_returns = period_returns @ self.weights
        turnover = float(np.sum(np.abs(self.weights - self.last_weights)))
        portfolio_returns[0] -= turnover * self.fee_rate
        for r in portfolio_returns:
            self.portfolio_value *= 1 + r
            self.portfolio_values.append(self.portfolio_value)
        self._period_returns.extend(portfolio_returns.tolist())
        period_return = float(np.prod(1 + portfolio_returns) - 1)
        period_vol = float(np.std(portfolio_returns)) if len(portfolio_returns) > 1 else 0.0
        dd_90 = abs(max_drawdown(np.array(self.portfolio_values[-90:])))
        btc_period_ret = float(np.prod(1 + period_returns[:, 0]) - 1)
        ew_period_ret = float(np.prod(1 + period_returns.mean(axis=1)) - 1)
        prev_start = max(self.lookback, self.idx - 21)
        prev_rets = self._get_raw(prev_start, self.idx)[:, :, ret_idx]
        if len(prev_rets) == 0:
            prev_rets = period_returns[:1]
        top_mom = period_returns[:, int(prev_rets.mean(axis=0).argmax())]
        mom_period_ret = float(np.prod(1 + top_mom) - 1)
        inv_vol = 1.0 / (np.std(period_returns, axis=0, keepdims=True) + 1e-12)
        rp_w = inv_vol / inv_vol.sum(axis=1, keepdims=True)
        rp_period_ret = float(np.prod(1 + (period_returns * rp_w).sum(axis=1)) - 1)
        wins = sum([period_return > btc_period_ret, period_return > ew_period_ret,
                    period_return > mom_period_ret, period_return > rp_period_ret])
        rank_score = wins / 4.0
        bear_bonus = 0.0
        if btc_period_ret < -0.02 and period_return > 0:
            bear_bonus = min(1.0, -btc_period_ret * 10)
        reward = rank_score - self.lambda_vol * period_vol - self.lambda_dd * dd_90 - self.lambda_turnover * turnover + bear_bonus
        self.action_log.append({
            "day": self.idx, "interval": interval, "weights": self.weights.copy(),
            "reward": reward, "period_return": period_return, "turnover": turnover,
        })
        self.idx = next_idx
        done = self.idx >= self.end_idx
        ns = self._get_state() if not done else np.zeros(
            (self.lookback, self.n_assets, self.n_features), dtype=np.float32
        )
        return ns, reward, done, {
            "weights": self.weights.copy(), "reward": reward, "interval": interval,
            "turnover": turnover, "period_return": period_return,
        }

    def evaluate_episode(self, benchmark_returns: dict[str, np.ndarray] | None = None) -> dict:
        pv = np.array(self.portfolio_values)
        if len(pv) < 2:
            return {"total_return": 0.0, "sharpe": 0.0, "volatility": 0.0, "max_drawdown": 0.0, "n_trades": 0}
        daily_rets = np.diff(pv) / pv[:-1]
        total_return = float(pv[-1] / pv[0] - 1)
        vol = float(np.std(daily_rets)) if len(daily_rets) > 1 else 0.0
        dd = max_drawdown(pv)
        sharpe = float(sharpe_ratio(daily_rets))
        win_rate = float(np.mean(daily_rets > 0)) if len(daily_rets) > 0 else 0.0
        weights_history = [e["weights"] for e in self.action_log]
        entropies = []
        for w in weights_history:
            ws = w[w > 1e-12]
            entropies.append(-np.sum(ws * np.log(ws)) / np.log(self.n_assets))
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        total_turnover = sum(e["turnover"] for e in self.action_log)
        metrics = {
            "total_return": total_return, "sharpe": sharpe, "volatility": vol,
            "max_drawdown": dd, "turnover": total_turnover, "n_trades": len(self.action_log),
            "win_rate": win_rate, "allocation_entropy": avg_entropy,
        }
        if benchmark_returns:
            for name, bench_rets in benchmark_returns.items():
                b_pv = np.cumprod(1 + bench_rets)
                metrics[f"{name}_return"] = float(b_pv[-1] / b_pv[0] - 1)
                metrics[f"{name}_sharpe"] = float(sharpe_ratio(bench_rets))
                metrics[f"{name}_volatility"] = float(np.std(bench_rets)) if len(bench_rets) > 1 else 0.0
                metrics[f"{name}_max_drawdown"] = max_drawdown(b_pv)
                metrics[f"{name}_relative_return"] = total_return - metrics[f"{name}_return"]
        return metrics

    def compute_benchmarks(self) -> dict[str, np.ndarray]:
        ret_idx = self.feature_names.index("return_1d")
        returns = self._get_raw(self.start_idx, self.end_idx)[:, :, ret_idx]
        if len(returns) < 2:
            return {
                "btc_hold": np.array([0.0]), "equal_weight": np.array([0.0]),
                "top_momentum": np.array([0.0]), "risk_parity": np.array([0.0]),
            }
        btc = returns[:, 0]
        equal = np.mean(returns, axis=1)
        mom_lookback = min(21, len(returns))
        momentum = np.mean(returns[-mom_lookback:], axis=0)
        top_mom = returns[:, int(np.argmax(momentum))]
        inv_vol = 1.0 / (np.std(returns, axis=0, keepdims=True) + 1e-12)
        rp_w = inv_vol / np.sum(inv_vol, axis=1, keepdims=True)
        risk_parity = np.sum(returns * rp_w, axis=1)
        return {"btc_hold": btc, "equal_weight": equal, "top_momentum": top_mom, "risk_parity": risk_parity}


def build_env(
    frames: dict[str, pd.DataFrame], start: str, end: str, cfg: PipelineConfig
) -> CryptoPortfolioEnv:
    filtered = filter_by_date(frames, start, end)
    return CryptoPortfolioEnv(
        coin_data=filtered,
        lookback=cfg.lookback,
        episode_years=cfg.episode_years,
        step_days=cfg.rebalance_days,
        fee_rate=cfg.fee_rate,
        seed=cfg.random_state,
    )
