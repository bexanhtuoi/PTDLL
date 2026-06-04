from __future__ import annotations

import numpy as np

from config import PipelineConfig
from models.base import BaseModel
from utils import save_json


def run_test(agent: BaseModel, test_env, cfg: PipelineConfig) -> dict:
    test_rng = np.random.default_rng(cfg.random_state + 777)
    n_test = max(20, min(200, cfg.n_episodes // 100))

    available = test_env.n_steps - 2 * test_env.lookback
    test_episode_len = max(
        test_env.step_days * 2,
        min(test_env.episode_len, available - 20),
    )
    test_episode_len = min(test_episode_len, available - test_env.lookback)

    max_start = max(
        test_env.n_steps - test_env.lookback - test_episode_len,
        test_env.lookback + 1,
    )

    all_metrics: list[dict] = []
    for _ in range(n_test):
        start = int(test_rng.integers(test_env.lookback, max_start))
        end = start + test_episode_len
        metrics = agent.run_episode(test_env, start_idx=start, end_idx=end)
        all_metrics.append(metrics)
    avg: dict = {}
    tag = type(agent).__name__.replace("Agent", "")
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics]
        avg[f"{tag}_{k}"] = float(np.mean(vals))
        avg[f"{tag}_{k}_std"] = float(np.std(vals))
    avg[f"{tag}_positive_sharpe"] = float(np.mean([m["sharpe"] > 0 for m in all_metrics]))
    return avg


def print_results(
    agents: list[str],
    train_histories: list[list[dict]],
    test_metrics: list[dict],
    cfg: PipelineConfig,
) -> None:
    print(f"\n{'='*60}")
    print("PTDLL RL Pipeline — Results")
    sep = "=" * 60
    print(sep)
    print("PTDLL RL Pipeline - Results")
    print(sep)
    print(f"Train: {cfg.train_start} -> {cfg.train_end}")
    print(f"Val:   {cfg.val_start} -> {cfg.val_end}")
    print(f"Test:  {cfg.test_start} -> {cfg.test_end}")
    print(f"Episode: {cfg.episode_years}y, Step: {cfg.rebalance_days}d, Episodes: {cfg.n_episodes}")
    print(f"\n{'Model':<12} {'Val Sharpe':>12} {'Val Return':>12} {'Test Sharpe':>12} {'Test Return':>12} {'Pos Rate':>10}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    for name, train_hist, test_m in zip(agents, train_histories, test_metrics):
        val_sharpe = float(np.mean([h["sharpe"] for h in train_hist])) if train_hist else 0.0
        val_ret = float(np.mean([h["total_return"] for h in train_hist])) if train_hist else 0.0
        tag = name
        test_sharpe = test_m.get(f"{tag}_sharpe", 0.0)
        test_ret = test_m.get(f"{tag}_total_return", 0.0)
        pos_rate = test_m.get(f"{tag}_positive_sharpe", 0.0)
        print(f"{name:<12} {val_sharpe:>12.4f} {val_ret:>12.4f} {test_sharpe:>12.4f} {test_ret:>12.4f} {pos_rate:>10.2%}")
    print(f"\nBest validation model(s) and detailed metrics saved to reports/")
    print(f"{sep}\n")
