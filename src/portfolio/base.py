from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch
from torch import nn, optim

from config import MODEL_DIR, PipelineConfig
from lib.features import pct_change, rolling_mean, rolling_std, drawdown as dd_series
from lib.metrics import max_drawdown, sharpe_ratio, total_return, volatility, win_rate, allocation_entropy
from lib.utils import shared_dates, ffill_grid, btc_grid
from log import get_log


COIN_FEATURE_NAMES = [
    "return_1d", "return_7d", "return_30d", "return_90d",
    "volatility", "drawdown", "volume_change",
]


def coin_fx(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    T = len(close)
    daily_ret = pct_change(close, 1)
    out = np.zeros((T, 7))
    out[:, 0] = daily_ret
    out[:, 1] = pct_change(close, 7)
    out[:, 2] = pct_change(close, 30)
    out[:, 3] = pct_change(close, 90)
    out[:, 4] = rolling_std(daily_ret, 20)
    out[:, 5] = dd_series(close).ravel()
    out[:, 6] = np.clip(pct_change(volume, 1), -5, 5)
    out[np.isnan(out)] = 0.0
    return out


def asset_cube(
    coin_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    shorts: list[str], grid: list[str], T: int,
) -> np.ndarray:
    names = COIN_FEATURE_NAMES
    n_assets = len(shorts)
    cube = np.zeros((T, n_assets, len(names)), dtype=np.float32)
    for i, short in enumerate(shorts):
        timestamps, close, volume = coin_data[short]
        fp = coin_fx(close, volume)
        cube[:, i, :] = ffill_grid(timestamps, fp, grid)
    return cube


def cross_fx(
    per_coin: np.ndarray, names: list[str], T: int, n_assets: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    btc_idx = 0
    ret_30d = per_coin[:, :, names.index("return_30d")]
    rel_strength = ret_30d - ret_30d[:, btc_idx:btc_idx + 1]

    daily_ret = per_coin[:, :, names.index("return_1d")]
    corr = np.zeros((T, n_assets, 1), dtype=np.float32)
    for t in range(60, T):
        for a in range(n_assets):
            r = np.corrcoef(daily_ret[t - 60:t, btc_idx], daily_ret[t - 60:t, a])
            corr[t, a, 0] = r[0, 1] if not np.isnan(r[0, 1]) else 0.0

    return rel_strength, corr, ["relative_strength_vs_BTC", "correlation_vs_BTC"]


def mkt_regime(
    per_coin: np.ndarray, names: list[str], btc_close: np.ndarray, T: int
) -> tuple[np.ndarray, list[str]]:
    sma200 = np.where(
        np.arange(T) < 200,
        np.cumsum(btc_close) / np.arange(1, T + 1),
        rolling_mean(btc_close, 200),
    )
    ma200_pos = (btc_close - sma200) / (sma200 + 1e-12)

    mkt_vol = per_coin[:, :, names.index("volatility")].mean(axis=1)

    close_90 = np.r_[btc_close[:90], btc_close[:-90]]
    mom_90 = (btc_close - close_90) / (close_90 + 1e-12)

    breadth = (per_coin[:, :, names.index("return_30d")] > 0).mean(axis=1)

    regimes = np.stack([ma200_pos, mkt_vol, mom_90, breadth], axis=1)
    return regimes, ["btc_ma200_position", "market_volatility", "btc_momentum_90d", "market_breadth"]


def stack_cube(
    per_coin: np.ndarray,
    rel_strength: np.ndarray,
    corr: np.ndarray,
    regimes: np.ndarray,
    cross_names: list[str],
    regime_names: list[str],
    T: int, n_assets: int,
) -> tuple[np.ndarray, list[str]]:
    per_names = COIN_FEATURE_NAMES
    n_feat = len(per_names) + len(cross_names) + len(regime_names) + 1
    cube = np.zeros((T, n_assets, n_feat), dtype=np.float32)
    fi = 0
    cube[:, :, fi:fi + len(per_names)] = per_coin
    fi += len(per_names)
    cube[:, :, fi:fi + 1] = rel_strength[:, :, np.newaxis]
    fi += 1
    cube[:, :, fi:fi + 1] = corr
    fi += 1
    cube[:, :, fi:fi + len(regime_names)] = np.broadcast_to(
        regimes[:, np.newaxis, :], (T, n_assets, len(regime_names)),
    )
    fi += len(regime_names)
    return cube, per_names + cross_names + regime_names + ["weight"]


def val_params(val_env, cfg):
    if val_env is None:
        return None
    n = val_env.n_steps
    lb = val_env.lookback
    avail = n - 2 * lb
    val_ep_len = max(val_env.step_days * 2, min(val_env.episode_len, avail - 20))
    val_ep_len = min(val_ep_len, avail - lb)
    val_max_start = max(n - lb - val_ep_len, lb + 1)
    val_rng = np.random.default_rng(cfg.random_state + 999)
    return val_ep_len, val_max_start, val_rng


def build_cube(
    coin_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    shorts = list(coin_data.keys())
    grid, date_index = shared_dates(coin_data)
    T = len(grid)
    n_assets = len(shorts)

    per_coin = asset_cube(coin_data, shorts, grid, T)
    rel_strength, corr, cross_names = cross_fx(per_coin, COIN_FEATURE_NAMES, T, n_assets)
    btc_close_grid = btc_grid(coin_data, shorts, grid, T)
    regimes, regime_names = mkt_regime(per_coin, COIN_FEATURE_NAMES, btc_close_grid, T)
    cube, feature_names = stack_cube(per_coin, rel_strength, corr, regimes, cross_names, regime_names, T, n_assets)

    return cube, shorts, feature_names, date_index


class CryptoPortfolioEnv:
    def __init__(
        self,
        cube: np.ndarray,
        asset_names: list[str],
        feature_names: list[str],
        date_index: np.ndarray,
        lookback: int = 60,
        episode_years: int = 2,
        step_days: int = 63,
        fee_rate: float = 0.001,
        seed: int = 42,
        lambdas: tuple[float, float, float] = (0.2, 0.25, 0.002),
    ):
        self.cube = cube.astype(np.float32, copy=False)
        self.asset_names = asset_names
        self.feature_names = feature_names
        self.date_index = date_index
        self.n_assets = len(asset_names)
        self.n_features = self.cube.shape[2]
        self.lookback = lookback
        self.mean, self.std = self.norm_params()
        self.episode_len = episode_years * 365
        self.step_days = step_days
        self.fee_rate = fee_rate
        self.lambda_vol, self.lambda_dd, self.lambda_turnover = lambdas
        self.rng = np.random.default_rng(seed)
        self.n_steps = self.cube.shape[0]

    def norm_params(self) -> tuple[np.ndarray, np.ndarray]:
        n_feat = self.cube.shape[2] - 1
        channels = self.cube[:, :, :n_feat]
        mean = channels.mean(axis=(0, 1), keepdims=True)
        std = channels.std(axis=(0, 1), keepdims=True) + 1e-8
        return mean.astype(np.float32), std.astype(np.float32)

    def data_slice(self, start: int, end: int) -> np.ndarray:
        return self.cube[start:end]

    def norm_view(self, start: int, end: int) -> np.ndarray:
        s = self.cube[start:end].copy()
        s[:, :, :-1] = (s[:, :, :-1] - self.mean) / self.std
        return s

    def get_state(self) -> np.ndarray:
        start = self.idx - self.lookback
        state = self.norm_view(start, self.idx)
        state[:, :, -1:] = np.broadcast_to(
            self.weights[np.newaxis, :, np.newaxis], (self.lookback, self.n_assets, 1),
        )
        return state

    def sample_start(self) -> int:
        max_start = self.n_steps - self.lookback - self.episode_len
        max_start = max(max_start, self.lookback + 1)
        if self.rng.random() < 0.6:
            vol_idx = self.feature_names.index("volatility")
            mkt_vols = self.cube[:, :, vol_idx].mean(axis=1)
            high_vol = np.argsort(mkt_vols)[-len(mkt_vols) // 3:]
            candidates = high_vol[(high_vol >= self.lookback) & (high_vol <= max_start)]
            if len(candidates) > 0:
                return int(self.rng.choice(candidates))
        return int(self.rng.integers(self.lookback, max_start))

    def reset(self, start_idx: int | None = None, end_idx: int | None = None) -> np.ndarray:
        if start_idx is not None and end_idx is not None:
            self.start_idx = max(int(start_idx), self.lookback)
            self.end_idx = min(int(end_idx), self.n_steps - 1)
        elif start_idx is not None:
            self.start_idx = max(int(start_idx), self.lookback)
            self.end_idx = min(self.start_idx + self.episode_len, self.n_steps - 1)
        else:
            self.start_idx = self.sample_start()
            self.end_idx = min(self.start_idx + self.episode_len, self.n_steps - 1)
        self.idx = self.start_idx
        self.weights = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        self.last_weights = self.weights.copy()
        self.portfolio_value = 1.0
        self.portfolio_values: list[float] = [1.0]
        self.action_log: list[dict] = []
        self._period_returns: list[float] = []
        return self.get_state()

    def bench_rets(self, period_returns: np.ndarray) -> dict[str, float]:
        ret_idx = self.feature_names.index("return_1d")
        btc_ret = float(np.prod(1 + period_returns[:, 0]) - 1)
        ew_ret = float(np.prod(1 + period_returns.mean(axis=1)) - 1)
        prev_start = max(self.lookback, self.idx - 21)
        prev_rets = self.data_slice(prev_start, self.idx)[:, :, ret_idx]
        if len(prev_rets) == 0:
            prev_rets = period_returns[:1]
        top_mom = period_returns[:, int(prev_rets.mean(axis=0).argmax())]
        mom_ret = float(np.prod(1 + top_mom) - 1)
        inv_vol = 1.0 / (np.std(period_returns, axis=0, keepdims=True) + 1e-12)
        rp_w = inv_vol / inv_vol.sum(axis=1, keepdims=True)
        rp_ret = float(np.prod(1 + (period_returns * rp_w).sum(axis=1)) - 1)
        return {"btc": btc_ret, "ew": ew_ret, "mom": mom_ret, "rp": rp_ret}

    def next_step(self) -> tuple[int, int]:
        interval = min(self.step_days, self.end_idx - self.idx)
        interval = max(interval, 1)
        next_idx = min(self.idx + interval, self.end_idx)
        return interval, next_idx

    def record_pv(self, portfolio_returns: np.ndarray) -> None:
        for r in portfolio_returns:
            self.portfolio_value *= 1 + r
            self.portfolio_values.append(self.portfolio_value)
        self._period_returns.extend(portfolio_returns.tolist())

    def step(self, action_weights: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self.last_weights = self.weights.copy()
        self.weights = np.clip(action_weights, 0, 1)
        total = self.weights.sum()
        if total < 1e-10:
            self.weights = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        else:
            self.weights = self.weights / total
        interval, next_idx = self.next_step()
        ret_idx = self.feature_names.index("return_1d")
        period_returns = self.data_slice(self.idx, next_idx)[:, :, ret_idx]
        portfolio_returns = period_returns @ self.weights
        turnover = float(np.sum(np.abs(self.weights - self.last_weights)))
        portfolio_returns[0] -= turnover * self.fee_rate
        self.record_pv(portfolio_returns)
        period_return = float(np.prod(1 + portfolio_returns) - 1)
        period_vol = float(np.std(portfolio_returns)) if len(portfolio_returns) > 1 else 0.0
        dd_90 = abs(max_drawdown(np.array(self.portfolio_values[-90:])))
        bench = self.bench_rets(period_returns)
        reward = self.reward_fn(period_return, period_vol, dd_90, turnover, bench)
        info = {"weights": self.weights.copy(), "reward": reward,
                "interval": interval, "turnover": turnover, "period_return": period_return}
        self.action_log.append({**info, "day": self.idx})
        self.idx = next_idx
        done = self.idx >= self.end_idx
        ns = self.get_state() if not done else np.zeros(
            (self.lookback, self.n_assets, self.n_features), dtype=np.float32,
        )
        return ns, reward, done, info

    def reward_fn(self, period_return: float, period_vol: float, dd_90: float, turnover: float, bench: dict[str, float]) -> float:
        wins = sum([period_return > bench["btc"], period_return > bench["ew"],
                    period_return > bench["mom"], period_return > bench["rp"]])
        rank_score = wins / 4.0
        bear_bonus = min(1.0, -bench["btc"] * 10) if bench["btc"] < -0.02 and period_return > 0 else 0.0
        return rank_score - self.lambda_vol * period_vol - self.lambda_dd * dd_90 - self.lambda_turnover * turnover + bear_bonus

    def pv_metrics(self) -> dict:
        pv = np.array(self.portfolio_values)
        if len(pv) < 2:
            return {"total_return": 0.0, "sharpe": 0.0, "volatility": 0.0, "max_drawdown": 0.0, "n_trades": 0}
        daily_rets = np.diff(pv) / pv[:-1]
        tr = total_return(pv)
        vol_val = volatility(daily_rets)
        dd = max_drawdown(pv)
        sharpe = sharpe_ratio(daily_rets)
        wr = win_rate(daily_rets)
        weights_history = [e["weights"] for e in self.action_log]
        entropy = allocation_entropy(weights_history, self.n_assets)
        total_turnover = sum(e["turnover"] for e in self.action_log)
        return {
            "total_return": tr, "sharpe": sharpe, "volatility": vol_val,
            "max_drawdown": dd, "turnover": total_turnover, "n_trades": len(self.action_log),
            "win_rate": wr, "allocation_entropy": entropy,
        }

    def bench_stats(self, benchmark_returns: dict[str, np.ndarray]) -> dict[str, float]:
        metrics = {}
        tr = total_return(np.array(self.portfolio_values)) if len(self.portfolio_values) >= 2 else 0.0
        for name, bench_rets in benchmark_returns.items():
            b_pv = np.cumprod(1 + bench_rets)
            metrics[f"{name}_return"] = total_return(b_pv)
            metrics[f"{name}_sharpe"] = sharpe_ratio(bench_rets)
            metrics[f"{name}_volatility"] = volatility(bench_rets)
            metrics[f"{name}_max_drawdown"] = max_drawdown(b_pv)
            metrics[f"{name}_relative_return"] = tr - metrics[f"{name}_return"]
        return metrics

    def score_ep(self, benchmark_returns: dict[str, np.ndarray] | None = None) -> dict:
        metrics = self.pv_metrics()
        if benchmark_returns:
            metrics.update(self.bench_stats(benchmark_returns))
        return metrics

    def bench_paths(self) -> dict[str, np.ndarray]:
        ret_idx = self.feature_names.index("return_1d")
        returns = self.data_slice(self.start_idx, self.end_idx)[:, :, ret_idx]
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


class BaseModel(ABC):
    lookback: int
    n_assets: int
    n_features: int
    device: str

    @abstractmethod
    def predict(self, state: np.ndarray) -> np.ndarray:
        ...

    def get_weights(self, state: np.ndarray) -> np.ndarray:
        return self.predict(state)

    def simulate(self, env, start_idx: int | None = None, end_idx: int | None = None) -> np.ndarray:
        state = env.reset(start_idx=start_idx, end_idx=end_idx)
        pv = [1.0]
        done = False
        while not done:
            w = self.predict(state)
            next_state, _, done, _ = env.step(w)
            pv.append(env.portfolio_value)
            state = next_state
        return np.array(pv)

    def score(self, env, start_idx: int | None = None, end_idx: int | None = None) -> dict:
        self.simulate(env, start_idx, end_idx)
        return env.score_ep(env.bench_paths())

    def play_ep(self, env, start_idx: int | None = None, end_idx: int | None = None) -> dict:
        return self.score(env, start_idx, end_idx)

    def eval_ckpt(
        self, val_env, vp, ep: int, sharpe: float,
        history: list, best_val_sharpe: float, no_improve_count: int,
        best_path, name_tag: str, cfg, log,
    ) -> tuple[float, int, bool]:
        val_ep_len, val_max_start, val_rng = vp
        val_start = int(val_rng.integers(val_env.lookback, val_max_start))
        val_metrics = self.score(val_env, start_idx=val_start, end_idx=val_start + val_ep_len)
        val_metrics["episode"] = ep + 1
        val_metrics["train_sharpe"] = sharpe
        history.append(val_metrics)

        improved = False
        if val_metrics["sharpe"] > best_val_sharpe:
            best_val_sharpe = val_metrics["sharpe"]
            self.save(str(best_path))
            no_improve_count = 0
            improved = True
        else:
            no_improve_count += 1

        log.write(
            f"Ep {ep+1:5d}: "
            f"Train S={sharpe:.4f} | "
            f"Val S={val_metrics['sharpe']:.4f} R={val_metrics['total_return']:.4f} "
            f"(Best={best_val_sharpe:.4f})"
            f"{' *' if improved else ''}"
        )

        if cfg.early_stop_patience > 0 and no_improve_count >= cfg.early_stop_patience:
            log.write(f"Early stopping at episode {ep+1} (no improvement for {no_improve_count} evals)")
            return best_val_sharpe, no_improve_count, True

        return best_val_sharpe, no_improve_count, False

    @abstractmethod
    def train_ep(
        self, env, start_idx: int | None = None, end_idx: int | None = None
    ) -> tuple[float, dict]:
        ...

    def fit(
        self,
        train_env,
        val_env = None,
        cfg: PipelineConfig | None = None,
    ) -> list[dict]:
        cfg = cfg or PipelineConfig()

        vp = val_params(val_env, cfg)

        history: list[dict] = []
        best_val_sharpe = -np.inf
        no_improve_count = 0
        max_start = max(
            train_env.n_steps - train_env.lookback - train_env.episode_len,
            train_env.lookback + 1,
        )
        name_tag = type(self).__name__.replace("Agent", "").lower()
        log = get_log(name_tag)
        best_path = MODEL_DIR / f"{name_tag}_best.pt"

        for ep in range(cfg.n_episodes):
            start = int(train_env.rng.integers(train_env.lookback, max_start))
            end = start + train_env.episode_len
            sharpe, _ = self.train_ep(train_env, start_idx=start, end_idx=end)

            if vp and (ep + 1) % cfg.val_interval == 0:
                best_val_sharpe, no_improve_count, stop = \
                    self.eval_ckpt(
                        val_env, vp, ep, sharpe, history,
                        best_val_sharpe, no_improve_count,
                        best_path, name_tag, cfg, log,
                    )
                if stop:
                    break

        if val_env is not None and best_path.exists():
            self.load(str(best_path))

        return history

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))

    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        ...


class PolicyNet(nn.Module):
    def __init__(self, lookback: int, n_assets: int, n_features: int):
        super().__init__()
        in_channels = n_assets * n_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_assets),
        )

    def forward(self, x):
        b, L, A, F = x.shape
        x = x.view(b, A * F, L)
        return self.fc(self.conv(x))


class ValueNet(nn.Module):
    def __init__(self, lookback: int, n_assets: int, n_features: int):
        super().__init__()
        in_channels = n_assets * n_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        b, L, A, F = x.shape
        x = x.view(b, A * F, L)
        return self.fc(self.conv(x))


class StateEncoder(nn.Module):
    def __init__(self, lookback: int, n_assets: int, n_features: int, hidden: int = 32):
        super().__init__()
        in_channels = n_assets * n_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        b, L, A, F = x.shape
        x = x.view(b, A * F, L)
        return self.conv(x)


class TwinQNet(nn.Module):
    def __init__(self, lookback: int, n_assets: int, n_features: int, hidden: int = 64):
        super().__init__()
        self.encoder = StateEncoder(lookback, n_assets, n_features, hidden=32)
        self.q1 = nn.Sequential(
            nn.Linear(32 + n_assets, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(32 + n_assets, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        feat = self.encoder(state)
        x = torch.cat([feat, action], dim=1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state, action):
        feat = self.encoder(state)
        x = torch.cat([feat, action], dim=1)
        return self.q1(x)


class DeterministicActor(nn.Module):
    def __init__(self, lookback: int, n_assets: int, n_features: int):
        super().__init__()
        in_channels = n_assets * n_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_assets),
        )

    def forward(self, x):
        b, L, A, F = x.shape
        x = x.view(b, A * F, L)
        return torch.softmax(self.fc(self.conv(x)), dim=1)


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: str = "cpu"):
        indices = np.random.randint(0, len(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(np.array(actions), dtype=torch.float32, device=device),
            torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
