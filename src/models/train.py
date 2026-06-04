from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from config import MODEL_DIR, PROCESSED_DIR, RAW_DIR, REPORT_DIR, ROOT, PipelineConfig
from dataset.fetch import aligned_prices, load_all_coins
from models.base import BaseModel
from models.env import build_env
from models.evaluation import run_test, print_results
from models.ppo import PPOAgent
from models.sac import SACAgent
from models.td3 import TD3Agent
from utils import ensure_dirs, save_csv, save_json


def create_agent(name: str, env, cfg: PipelineConfig, **overrides) -> BaseModel:
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


def train_model(
    agent: BaseModel,
    train_env,
    val_env,
    cfg: PipelineConfig,
) -> list[dict]:
    val_n_steps = val_env.n_steps
    val_lookback = val_env.lookback
    val_available = val_n_steps - 2 * val_lookback
    val_ep_len = max(
        val_env.step_days * 2,
        min(val_env.episode_len, val_available - 20),
    )
    val_ep_len = min(val_ep_len, val_available - val_lookback)
    val_max_start = max(val_n_steps - val_lookback - val_ep_len, val_lookback + 1)
    val_rng = np.random.default_rng(cfg.random_state + 999)

    history: list[dict] = []
    best_val_sharpe = -np.inf
    max_start = max(
        train_env.n_steps - train_env.lookback - train_env.episode_len,
        train_env.lookback + 1,
    )

    for ep in range(cfg.n_episodes):
        start = int(train_env.rng.integers(train_env.lookback, max_start))
        end = start + train_env.episode_len
        sharpe, _ = agent.train_episode(train_env, start_idx=start, end_idx=end)

        if (ep + 1) % cfg.val_interval == 0:
            val_start = int(val_rng.integers(val_lookback, val_max_start))
            val_end = val_start + val_ep_len
            val_metrics = agent.run_episode(val_env, start_idx=val_start, end_idx=val_end)
            val_metrics["episode"] = ep + 1
            val_metrics["train_sharpe"] = sharpe
            history.append(val_metrics)
            if val_metrics["sharpe"] > best_val_sharpe:
                best_val_sharpe = val_metrics["sharpe"]

            print(
                f"  {type(agent).__name__} Ep {ep+1:5d}: "
                f"Train S={sharpe:.4f} | "
                f"Val S={val_metrics['sharpe']:.4f} R={val_metrics['total_return']:.4f} "
                f"(Best={best_val_sharpe:.4f})"
            )

    return history


def clean_generated_outputs() -> None:
    results_dir = ROOT / "results"
    for d in [REPORT_DIR, results_dir / "tables", results_dir / "figures", results_dir / "metrics"]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def publish_report_artifacts() -> None:
    results_dir = ROOT / "results"
    for path in REPORT_DIR.glob("*.csv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, results_dir / "tables" / path.name)
    for path in REPORT_DIR.glob("*.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, results_dir / "metrics" / path.name)
    for path in REPORT_DIR.glob("*.png"):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, results_dir / "figures" / path.name)


def save_aligned_prices(frames: dict[str, pd.DataFrame], cfg: PipelineConfig) -> None:
    train_f = {short: df for short, df in frames.items()
               if (pd.to_datetime(df["timestamp"]) >= cfg.train_start).any()
               and (pd.to_datetime(df["timestamp"]) < cfg.train_end).any()}
    prices = aligned_prices(train_f)
    save_csv(prices, PROCESSED_DIR / "aligned_prices_15_coins.csv")


def run() -> None:
    cfg = PipelineConfig()
    ensure_dirs(RAW_DIR, PROCESSED_DIR, REPORT_DIR, MODEL_DIR)

    print("Loading all coins...")
    all_frames = load_all_coins()
    print(f"  Loaded {len(all_frames)} coins: {list(all_frames)}")

    print("Building environments...")
    train_env = build_env(all_frames, cfg.train_start, cfg.train_end, cfg)
    val_env = build_env(all_frames, cfg.val_start, cfg.val_end, cfg)
    test_env = build_env(all_frames, cfg.test_start, cfg.test_end, cfg)
    print(f"  Train: {train_env.n_steps} days, {train_env.n_assets} assets, {train_env.n_features} features")
    print(f"  Val:   {val_env.n_steps} days")
    print(f"  Test:  {test_env.n_steps} days")

    results: list[tuple[str, BaseModel, list[dict]]] = []

    # --- Retrain PPO with lower gamma + higher entropy for generalization ---
    print(f"\n{'='*40}")
    print("PPO (gamma=0.97, entropy=0.01) retraining...")
    print(f"{'='*40}")
    ppo = create_agent("PPO", train_env, cfg, gamma=0.97, entropy_coef=0.01)
    ppo_hist = train_model(ppo, train_env, val_env, cfg)
    ppo.save(str(MODEL_DIR / "ppo.pt"))
    save_json(ppo_hist, REPORT_DIR / "rl_ppo_val_history.json")
    results.append(("PPO", ppo, ppo_hist))

    # --- Train TD3 ---
    print(f"\n{'='*40}")
    print("Training TD3...")
    print(f"{'='*40}")
    td3 = create_agent("TD3", train_env, cfg)
    td3_hist = train_model(td3, train_env, val_env, cfg)
    td3.save(str(MODEL_DIR / "td3.pt"))
    save_json(td3_hist, REPORT_DIR / "rl_td3_val_history.json")
    results.append(("TD3", td3, td3_hist))

    # --- Load pre-trained SAC ---
    sac = create_agent("SAC", train_env, cfg)
    sac_path = MODEL_DIR / "sac.pt"
    if sac_path.exists():
        sac.load(str(sac_path))
        sac_hist_path = REPORT_DIR / "rl_sac_val_history.json"
        sac_hist: list[dict] = []
        if sac_hist_path.exists():
            with open(sac_hist_path) as f:
                sac_hist = json.load(f)
        results.insert(1, ("SAC", sac, sac_hist))
        print(f"\nLoaded SAC from checkpoint ({len(sac_hist)} val points)")
    else:
        print(f"\n{'='*40}")
        print("Training SAC...")
        print(f"{'='*40}")
        sac = create_agent("SAC", train_env, cfg)
        sac_hist = train_model(sac, train_env, val_env, cfg)
        sac.save(str(MODEL_DIR / "sac.pt"))
        save_json(sac_hist, REPORT_DIR / "rl_sac_val_history.json")
        results.insert(1, ("SAC", sac, sac_hist))

    # --- Test all with new walk-forward evaluation ---
    agents_out: list[str] = []
    train_hists_out: list[list[dict]] = []
    test_metrics_out: list[dict] = []
    for name, agent, hist in results:
        print(f"  Testing {name} on test set...")
        test_m = run_test(agent, test_env, cfg)
        save_json(test_m, REPORT_DIR / f"rl_{name.lower()}_test_metrics.json")
        agents_out.append(name)
        train_hists_out.append(hist)
        test_metrics_out.append(test_m)

    save_aligned_prices(all_frames, cfg)
    publish_report_artifacts()
    print_results(agents_out, train_hists_out, test_metrics_out, cfg)
    print("PTDLL RL Pipeline completed.")
