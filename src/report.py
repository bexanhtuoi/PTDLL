from __future__ import annotations

from pathlib import Path

import numpy as np
from config import FIGURES_DIR, HISTORY_PATH, MODEL_TAGS, TABLES_DIR, PipelineConfig
from dataset.fetch import load_coin_arrays
from portfolio.env import build_env
from portfolio.evaluate import load_agent, sim_agent
from lib.metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio,
    max_drawdown, total_return, volatility,
    profit_factor, win_rate,
)
from lib.plot import multi_line, equity_curve
from lib.utils import save_json, ensure_dirs, load_json

METRIC_LABELS: dict[str, str] = {
    "sharpe": "Sharpe Ratio",
    "sortino": "Sortino Ratio",
    "total_return": "Total Return",
    "volatility": "Volatility",
    "max_drawdown": "Max Drawdown",
    "calmar": "Calmar Ratio",
    "profit_factor": "Profit Factor",
    "win_rate": "Win Rate",
}


def val_history(name: str) -> list[dict]:
    data = load_json(HISTORY_PATH)
    entry = data.get(name, {})
    if isinstance(entry, dict):
        return entry.get("validate", [])
    return []


def test_metrics(name: str) -> dict:
    data = load_json(HISTORY_PATH)
    entry = data.get(name, {})
    if isinstance(entry, dict):
        return entry.get("test", {})
    return {}


def eval_metrics(agents: dict[str, object], env) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, agent in agents.items():
        if agent is None:
            continue
        pv = sim_agent(agent, env)
        rets = np.diff(pv) / pv[:-1]
        tr = total_return(pv)
        mdd = max_drawdown(pv)
        results[name] = {
            "sharpe": sharpe_ratio(rets),
            "sortino": sortino_ratio(rets),
            "total_return": tr,
            "volatility": volatility(rets),
            "max_drawdown": mdd,
            "calmar": calmar_ratio(tr, mdd),
            "profit_factor": profit_factor(rets),
            "win_rate": win_rate(rets),
        }
    return results


def metric_row(model: str, metrics: dict, metric_names: list[str]) -> str:
    row = MODEL_TAGS.get(model, model)
    for m in metric_names:
        val = metrics[model].get(m, 0.0)
        row += f" & {val:.4f}" if isinstance(val, float) else f" & {val}"
    return row + r" \\"


def latex_header(metric_names: list[str]) -> list[str]:
    labels = [METRIC_LABELS.get(m, m.replace("_", " ").title()) for m in metric_names]
    n_cols = len(metric_names)
    return [
        r"\begin{tabular}{l" + "c" * n_cols + "}",
        r"\toprule",
        "Model & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]


def make_table(metrics: dict[str, dict[str, float]], save_path: Path) -> None:
    models = list(metrics.keys())
    metric_names = list(metrics[models[0]].keys())

    lines = latex_header(metric_names)
    for model in models:
        lines.append(metric_row(model, metrics, metric_names))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_history(histories: dict[str, list[dict]], figs_dir: Path) -> None:
    sharpe_data: dict[str, tuple] = {}
    return_data: dict[str, tuple] = {}
    for name, hist in histories.items():
        if not hist:
            continue
        eps = [h["episode"] for h in hist]
        sharpes = [h["sharpe"] for h in hist]
        rets = [h["total_return"] for h in hist]
        sharpe_data[name] = (eps, sharpes)
        return_data[name] = (eps, rets)

    if sharpe_data:
        multi_line(sharpe_data, figs_dir / "training_sharpe.png", title="Validation Sharpe")
        multi_line(return_data, figs_dir / "training_return.png", title="Validation Total Return")


def equity_plots(agents: dict[str, object], env, figs_dir: Path) -> None:
    for name, agent in agents.items():
        if agent is None:
            continue
        pv = sim_agent(agent, env)
        equity_curve(pv, figs_dir / f"equity_{name}.png", title=f"{MODEL_TAGS[name]} Test Equity")


def bench_plots(agents: dict[str, object], env, figs_dir: Path) -> None:
    env.reset(start_idx=env.lookback, end_idx=env.n_steps)
    bench_rets = env.data_slice(env.start_idx, env.end_idx)
    ret_idx = env.feature_names.index("return_1d")
    bench_ret = bench_rets[:, :, ret_idx]
    btc_pv = np.cumprod(1 + bench_ret[:, 0])

    for name, agent in agents.items():
        if agent is None:
            continue
        pv = sim_agent(agent, env)
        equity_curve(pv, figs_dir / f"equity_{name}_vs_btc.png",
                     benchmark=btc_pv, title=f"{MODEL_TAGS[name]} vs BTC")


def gen_report() -> None:
    ensure_dirs(FIGURES_DIR, TABLES_DIR)

    cfg = PipelineConfig()
    all_arrays = load_coin_arrays()
    if not all_arrays:
        print("No coin data found.")
        return

    test_env = build_env(all_arrays, cfg.test_start, cfg.test_end, cfg)

    agents = {name: load_agent(name, test_env) for name in ["ppo", "sac", "td3"]}
    histories = {name: val_history(name) for name in ["ppo", "sac", "td3"]}

    plot_history(histories, FIGURES_DIR)
    equity_plots(agents, test_env, FIGURES_DIR)
    bench_plots(agents, test_env, FIGURES_DIR)

    metrics = eval_metrics(agents, test_env)
    if metrics:
        make_table(metrics, TABLES_DIR / "metrics_comparison.tex")
        save_json(metrics, TABLES_DIR / "metrics_comparison.json")

    print(f"Report: figures -> {FIGURES_DIR}/, tables -> {TABLES_DIR}/")
