from __future__ import annotations

import multiprocessing

from config import HISTORY_PATH, MODEL_DIR, PipelineConfig
from dataset.fetch import load_coin_arrays
from portfolio.base import BaseModel
from portfolio.env import build_env
from portfolio.evaluate import make_agent, eval_agent, log_results
from lib.utils import ensure_dirs, save_history
from log import get_log


DEFAULT_MODELS: list[tuple[str, dict]] = [
    ("ppo", {}),
    ("sac", {}),
    ("td3", {}),
]


def make_envs(coin_arrays, cfg: PipelineConfig, reward_style: str | None = None):
    train_env = build_env(coin_arrays, cfg.train_start, cfg.train_end, cfg, reward_style)
    val_env = build_env(coin_arrays, cfg.val_start, cfg.val_end, cfg, reward_style)
    test_env = build_env(coin_arrays, cfg.test_start, cfg.test_end, cfg, reward_style)
    comb_env = build_env(coin_arrays, cfg.train_start, cfg.val_end, cfg, reward_style)
    return train_env, val_env, test_env, comb_env


def train_save(name: str, tag: str, overrides: dict, cfg: PipelineConfig | None = None) -> None:
    all_frames = load_coin_arrays()
    cfg = cfg or PipelineConfig()
    reward_style = overrides.pop("reward_style", None) if overrides else None
    train_env, val_env, test_env, comb_env = make_envs(all_frames, cfg, reward_style)
    agent = make_agent(tag, train_env, cfg, **overrides)
    history = agent.fit(train_env, val_env, cfg, comb_env=comb_env)
    agent.save(str(MODEL_DIR / f"{name}.pt"))
    test_m = agent.score(test_env)
    multi_m = eval_agent(agent, test_env, cfg)
    save_history(name, history, test_m, HISTORY_PATH)
    log = get_log(name)
    log.write(f"{tag} done! Single S={test_m.get('sharpe', 0):.4f} | "
              f"Multi S={multi_m.get(f'{tag}_sharpe', 0):.4f} +/- {multi_m.get(f'{tag}_sharpe_std', 0):.4f}")


def train(
    models: list[tuple[str, dict]] | None = None,
    parallel: bool = False,
    cfg: PipelineConfig | None = None,
) -> None:
    models = models or DEFAULT_MODELS
    ensure_dirs(MODEL_DIR)

    if parallel:
        train_par(models, cfg)
    else:
        train_seq(models, cfg)


def train_seq(models: list[tuple[str, dict]], cfg: PipelineConfig | None = None) -> None:
    from log import tee_stdout

    all_frames = load_coin_arrays()
    cfg = cfg or PipelineConfig()

    results: list[tuple[str, BaseModel, list[dict], dict]] = []
    for name, overrides in models:
        reward_style = overrides.pop("reward_style", None) if overrides else None
        train_env, val_env, test_env, comb_env = make_envs(all_frames, cfg, reward_style)

        tag = name.upper()
        tee_stdout(name)
        log = get_log(name)
        log.write(f"Training {tag}...")
        agent = make_agent(tag, train_env, cfg, **overrides)
        history = agent.fit(train_env, val_env, cfg, comb_env=comb_env)
        agent.save(str(MODEL_DIR / f"{name}.pt"))
        test_m = agent.score(test_env)
        multi_m = eval_agent(agent, test_env, cfg)
        save_history(name, history, test_m, HISTORY_PATH)
        results.append((tag, agent, history, multi_m))
        log.write(f"{tag} done! Single S={test_m.get('sharpe', 0):.4f} | "
                  f"Multi S={multi_m.get(f'{tag}_sharpe', 0):.4f} +/- {multi_m.get(f'{tag}_sharpe_std', 0):.4f}")

    if len(results) > 1:
        log_results([r[0] for r in results], [r[2] for r in results],
                    [r[3] for r in results], cfg)

    get_log("train").write(f"Trained {len(results)} model(s) sequentially.")


def _train_one(name: str, overrides: dict, cfg: PipelineConfig | None = None) -> None:
    from log import tee_stdout
    tee_stdout(name)
    train_save(name, name.upper(), overrides, cfg)


def train_par(models: list[tuple[str, dict]], cfg: PipelineConfig | None = None) -> None:
    multiprocessing.freeze_support()

    processes = []
    for name, overrides in models:
        p = multiprocessing.Process(target=_train_one, args=(name, overrides, cfg), daemon=False)
        p.start()
        processes.append((name, p))

    for name, p in processes:
        p.join()

    get_log("train").write(f"Trained {len(models)} model(s) in parallel.")
