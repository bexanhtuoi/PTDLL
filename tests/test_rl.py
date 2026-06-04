import numpy as np
import pandas as pd

from dataset.fetch import generate_synthetic_ohlcv, clean_ohlcv
from models import PPOAgent, SACAgent, TD3Agent, CryptoPortfolioEnv
from models.env import filter_by_date


def _env():
    coins = {"BTC": clean_ohlcv(generate_synthetic_ohlcv("BTCUSDT", periods=1200))}
    coins["ETH"] = clean_ohlcv(generate_synthetic_ohlcv("ETHUSDT", periods=1200))
    coins["USDT"] = clean_ohlcv(generate_synthetic_ohlcv("USDT-USD", periods=1200))
    return CryptoPortfolioEnv(coin_data=coins, lookback=60, episode_years=2, step_days=63, seed=42)


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


def test_env_evaluate_episode_returns_metrics():
    env = _env()
    env.reset()
    weights = np.ones(3) / 3
    done = False
    while not done:
        _, _, done, _ = env.step(weights)
    metrics = env.evaluate_episode()
    assert "sharpe" in metrics
    assert "total_return" in metrics
    assert "max_drawdown" in metrics


def test_ppo_get_weights():
    env = _env()
    agent = PPOAgent(lookback=60, n_assets=3, n_features=14)
    state = env.reset()
    w = agent.get_weights(state)
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-6


def test_ppo_train_and_run_episode():
    env = _env()
    agent = PPOAgent(lookback=60, n_assets=3, n_features=14)
    sharpe, metrics = agent.train_episode(env)
    assert isinstance(sharpe, float)
    assert "total_return" in metrics
    em = agent.run_episode(env)
    assert "btc_hold_return" in em


def test_sac_get_weights():
    env = _env()
    agent = SACAgent(lookback=60, n_assets=3, n_features=14)
    state = env.reset()
    w = agent.get_weights(state)
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-6


def test_sac_train_and_run_episode():
    env = _env()
    agent = SACAgent(lookback=60, n_assets=3, n_features=14)
    sharpe, metrics = agent.train_episode(env)
    assert isinstance(sharpe, float)
    assert "total_return" in metrics
    em = agent.run_episode(env)
    assert "btc_hold_return" in em


def test_td3_get_weights():
    env = _env()
    agent = TD3Agent(lookback=60, n_assets=3, n_features=14)
    state = env.reset()
    w = agent.get_weights(state)
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-6


def test_td3_train_and_run_episode():
    env = _env()
    agent = TD3Agent(lookback=60, n_assets=3, n_features=14)
    sharpe, metrics = agent.train_episode(env)
    assert isinstance(sharpe, float)
    assert "total_return" in metrics
    em = agent.run_episode(env)
    assert "btc_hold_return" in em


def test_filter_by_date():
    frames = {"BTC": clean_ohlcv(generate_synthetic_ohlcv("BTCUSDT", periods=500))}
    filtered = filter_by_date(frames, "2025-01-01", "2025-01-03")
    assert "BTC" in filtered
    if len(filtered["BTC"]) > 0:
        ts = pd.to_datetime(filtered["BTC"]["timestamp"])
        assert (ts >= "2025-01-01").all()
        assert (ts < "2025-01-03").all()
