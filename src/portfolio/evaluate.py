from __future__ import annotations

import numpy as np

from config import MODEL_DIR, MODEL_TAGS, PipelineConfig
from portfolio.base import BaseModel
from portfolio.ppo import PPOAgent
from portfolio.sac import SACAgent
from portfolio.td3 import TD3Agent
from log import get_log


def make_agent(name: str, env, cfg: PipelineConfig, **overrides) -> BaseModel:
    kwargs = dict(
        lookback=cfg.lookback,
        n_assets=env.n_assets,
        n_features=env.n_features,
        lr=overrides.get("lr", cfg.lr),
        gamma=overrides.get("gamma", cfg.gamma),
        device="cpu",
    )
    if name == "PPO":
        return PPOAgent(**kwargs, entropy_coef=overrides.get("entropy_coef", cfg.entropy_coef))
    elif name == "SAC":
        return SACAgent(**kwargs)
    elif name == "TD3":
        return TD3Agent(**kwargs)
    raise ValueError(f"Unknown agent: {name}")


def load_agent(name: str, env) -> BaseModel | None:
    path = MODEL_DIR / f"{name}.pt"
    if not path.exists():
        return None
    try:
        agent = make_agent(MODEL_TAGS[name], env, PipelineConfig())
        agent.load(str(path))
        return agent
    except Exception as e:
        get_log("evaluate").write(f"Cannot load {name}: {e}")
        return None


def sim_agent(agent, env) -> np.ndarray:
    return agent.simulate(env)


def eval_config(test_env, cfg) -> tuple[int, int, int, np.random.Generator]:
    n_test = max(20, min(200, cfg.n_episodes // 100))
    available = test_env.n_steps - 2 * test_env.lookback
    ep_len = max(test_env.step_days * 2, min(test_env.episode_len, available - 20))
    ep_len = min(ep_len, available - test_env.lookback)
    max_start = max(test_env.n_steps - test_env.lookback - ep_len, test_env.lookback + 1)
    rng = np.random.default_rng(cfg.random_state + 777)
    return n_test, ep_len, max_start, rng


def avg_metrics(all_metrics: list[dict], agent) -> dict:
    tag = type(agent).__name__.replace("Agent", "")
    avg: dict = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics]
        avg[f"{tag}_{k}"] = float(np.mean(vals))
        avg[f"{tag}_{k}_std"] = float(np.std(vals))
    avg[f"{tag}_positive_sharpe"] = float(np.mean([m["sharpe"] > 0 for m in all_metrics]))
    return avg


def eval_agent(agent: BaseModel, test_env, cfg: PipelineConfig) -> dict:
    n_test, test_episode_len, max_start, test_rng = eval_config(test_env, cfg)
    all_metrics: list[dict] = []
    for _ in range(n_test):
        start = int(test_rng.integers(test_env.lookback, max_start))
        end = start + test_episode_len
        metrics = agent.score(test_env, start_idx=start, end_idx=end)
        all_metrics.append(metrics)
    return avg_metrics(all_metrics, agent)


def log_results(
    agents: list[str],
    train_histories: list[list[dict]],
    test_metrics: list[dict],
    cfg: PipelineConfig,
) -> None:
    log = get_log("evaluate")
    log.write("Results")
    log.write(f"Train: {cfg.train_start} -> {cfg.train_end}")
    log.write(f"Val:   {cfg.val_start} -> {cfg.val_end}")
    log.write(f"Test:  {cfg.test_start} -> {cfg.test_end}")
    log.write(f"Episode: {cfg.episode_years}y, Step: {cfg.rebalance_days}d, Episodes: {cfg.n_episodes}")
    header = f"{'Model':<12} {'Val Sharpe':>12} {'Val Return':>12} {'Test Sharpe':>12} {'Test Return':>12}"
    log.write(header)
    log.write("-" * len(header))
    for name, train_hist, test_m in zip(agents, train_histories, test_metrics):
        val_sharpe = float(np.mean([h["sharpe"] for h in train_hist])) if train_hist else 0.0
        val_ret = float(np.mean([h["total_return"] for h in train_hist])) if train_hist else 0.0
        test_sharpe = test_m.get("sharpe", 0.0)
        test_ret = test_m.get("total_return", 0.0)
        log.write(f"{name:<12} {val_sharpe:>12.4f} {val_ret:>12.4f} {test_sharpe:>12.4f} {test_ret:>12.4f}")
