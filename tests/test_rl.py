import numpy as np

from dataset.fetch import generate_synthetic_ohlcv
from portfolio.ppo import PPOAgent
from portfolio.sac import SACAgent
from portfolio.td3 import TD3Agent
from portfolio.env import build_env
from portfolio.base import CryptoPortfolioEnv, build_cube
from config import PipelineConfig


def _coin_arrays():
    out = {}
    for name in ["BTC", "ETH", "USDT"]:
        d = generate_synthetic_ohlcv(f"{name}USDT", periods=1200)
        out[name] = (
            np.datetime64("2025-01-01") + np.arange(len(d["close"])),
            d["close"].astype(np.float64),
            d["volume"].astype(np.float64),
        )
    return out


def _env():
    cube, names, fnames, didx = build_cube(_coin_arrays())
    return CryptoPortfolioEnv(cube=cube, asset_names=names, feature_names=fnames, date_index=didx,
                              lookback=60, episode_years=2, step_days=63, seed=42)


def test_env_reset_returns_correct_state_shape():
    env = _env()
    state = env.reset()
    lookback = 60
    assert state.shape == (lookback, 3, 14)


def test_env_step_returns_state_reward_done_info():
    env = _env()
    env.reset()
    weights = np.ones(3) / 3
    ns, reward, done, info = env.step(weights)
    assert ns.shape == (60, 3, 14)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "weights" in info


def test_env_score_ep_returns_metrics():
    env = _env()
    env.reset()
    weights = np.ones(3) / 3
    done = False
    while not done:
        _, _, done, _ = env.step(weights)
    metrics = env.score_ep()
    assert "sharpe" in metrics
    assert "total_return" in metrics
    assert "max_drawdown" in metrics


def test_ppo_predict():
    env = _env()
    agent = PPOAgent(lookback=60, n_assets=3, n_features=14)
    state = env.reset()
    w = agent.predict(state)
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-6


def test_ppo_train_and_play_ep():
    env = _env()
    agent = PPOAgent(lookback=60, n_assets=3, n_features=14)
    sharpe, metrics = agent.train_ep(env)
    assert isinstance(sharpe, float)
    assert "total_return" in metrics
    em = agent.play_ep(env)
    assert "btc_hold_return" in em


def test_sac_predict():
    env = _env()
    agent = SACAgent(lookback=60, n_assets=3, n_features=14)
    state = env.reset()
    w = agent.predict(state)
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-6


def test_sac_train_and_play_ep():
    env = _env()
    agent = SACAgent(lookback=60, n_assets=3, n_features=14)
    sharpe, metrics = agent.train_ep(env)
    assert isinstance(sharpe, float)
    assert "total_return" in metrics
    em = agent.play_ep(env)
    assert "btc_hold_return" in em


def test_td3_predict():
    env = _env()
    agent = TD3Agent(lookback=60, n_assets=3, n_features=14)
    state = env.reset()
    w = agent.predict(state)
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-6


def test_td3_train_and_play_ep():
    env = _env()
    agent = TD3Agent(lookback=60, n_assets=3, n_features=14)
    sharpe, metrics = agent.train_ep(env)
    assert isinstance(sharpe, float)
    assert "total_return" in metrics
    em = agent.play_ep(env)
    assert "btc_hold_return" in em


def test_filter_by_date():
    arrays = _coin_arrays()
    cfg = PipelineConfig()
    env = build_env(arrays, "2025-01-01", "2025-01-03", cfg)
    assert len(env.asset_names) > 0
