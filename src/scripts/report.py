from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import MODEL_DIR, REPORT_DIR, ROOT, PipelineConfig
from dataset.fetch import load_all_coins
from models.env import build_env
from models.ppo import PPOAgent
from models.sac import SACAgent
from models.td3 import TD3Agent
from models.train import create_agent

FIGS = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
METRICS = ROOT / "results" / "metrics"

ASSET_NAMES = ["BTC", "LTC", "XRP", "DOGE", "XMR", "DASH", "XLM", "USDT",
               "ETH", "ETC", "WAVES", "ZEC", "DCR", "NEO", "BNB"]

MODEL_NAMES = {"ppo": "PPO", "sac": "SAC", "td3": "TD3"}


def load_val_history(name: str) -> list[dict]:
    path = REPORT_DIR / f"rl_{name}_val_history.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_test_metrics(name: str) -> dict:
    path = REPORT_DIR / f"rl_{name}_test_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_agent(name: str, env) -> PPOAgent | SACAgent | None:
    path = MODEL_DIR / f"{name}.pt"
    if not path.exists():
        return None
    try:
        agent = create_agent(MODEL_NAMES[name], env, PipelineConfig())
        agent.load(str(path))
        return agent
    except Exception as e:
        print(f"  Cannot load {name}: {e}")
        return None


def extract_weights_over_episode(agent, env):
    state = env.reset()
    weights_history: list[np.ndarray] = []
    done = False
    while not done:
        w = agent.get_weights(state)
        weights_history.append(w)
        next_state, _, done, _ = env.step(w)
        state = next_state
    return np.array(weights_history)


def plot_training_history(histories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for name, hist in histories.items():
        if not hist:
            continue
        eps = [h["episode"] for h in hist]
        sharpe = [h["sharpe"] for h in hist]
        ret = [h["total_return"] for h in hist]
        dd = [h["max_drawdown"] for h in hist]
        train_s = [h.get("train_sharpe", 0) for h in hist]

        ax = axes[0, 0]
        ax.plot(eps, sharpe, label=name, alpha=0.8)
        ax.set_title("Validation Sharpe")
        ax.legend()
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

        ax = axes[0, 1]
        ax.plot(eps, ret, label=name, alpha=0.8)
        ax.set_title("Validation Total Return")
        ax.legend()
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

        ax = axes[1, 0]
        ax.plot(eps, dd, label=name, alpha=0.8)
        ax.set_title("Validation Max Drawdown")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(eps, train_s, label=name, alpha=0.8)
        ax.set_title("Train Sharpe (per episode)")
        ax.legend()
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    for ax in axes.flat:
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "training_history.png", dpi=150)
    plt.close()
    print(f"  Saved training_history.png")


def plot_equity_curves(agents: dict[str, object], test_env) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, agent in agents.items():
        if agent is None:
            continue
        state = test_env.reset()
        pv = [1.0]
        done = False
        while not done:
            w = agent.get_weights(state)
            next_state, _, done, info = test_env.step(w)
            pv.append(test_env.portfolio_value)
            state = next_state
        ax.plot(pv, label=f"{MODEL_NAMES[name]} Agent", linewidth=2)

    bench_env = test_env
    bench_env.reset()
    _run_benchmark(bench_env, "btc_hold", ax)
    _run_benchmark(bench_env, "equal_weight", ax)
    _run_benchmark(bench_env, "top_momentum", ax)
    _run_benchmark(bench_env, "risk_parity", ax)

    ax.set_title("Test Set Equity Curves")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Portfolio Value (initial=1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "equity_curve_test.png", dpi=150)
    plt.close()
    print(f"  Saved equity_curve_test.png")


def _run_benchmark(env, name: str, ax) -> None:
    env.reset()
    ret_idx = env.feature_names.index("return_1d")
    bench_rets = env._get_raw(env.start_idx, env.end_idx)[:, :, ret_idx]
    if name == "btc_hold":
        r = bench_rets[:, 0]
    elif name == "equal_weight":
        r = np.mean(bench_rets, axis=1)
    elif name == "top_momentum":
        mom = np.mean(bench_rets[:21], axis=0) if len(bench_rets) >= 21 else np.mean(bench_rets, axis=0)
        r = bench_rets[:, int(np.argmax(mom))]
    elif name == "risk_parity":
        inv_vol = 1.0 / (np.std(bench_rets, axis=0, keepdims=True) + 1e-12)
        w = inv_vol / np.sum(inv_vol, axis=1, keepdims=True)
        r = np.sum(bench_rets * w, axis=1)
    pv = np.cumprod(1 + r)
    ax.plot(pv, label=name.replace("_", " ").title(), linestyle="--", alpha=0.7)


def plot_allocation_heatmaps(agents: dict[str, object], test_env, n_assets: int = 15) -> None:
    for name, agent in agents.items():
        if agent is None:
            continue
        weights = extract_weights_over_episode(agent, test_env)
        if len(weights) < 1:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(weights.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_yticks(range(n_assets))
        ax.set_yticklabels(ASSET_NAMES)
        ax.set_xlabel("Rebalance Step")
        ax.set_title(f"{MODEL_NAMES[name]} Allocation Weights on Test Set")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label("Weight")
        plt.tight_layout()
        plt.savefig(FIGS / f"allocation_heatmap_{name}.png", dpi=150)
        plt.close()
        print(f"  Saved allocation_heatmap_{name}.png")


def plot_benchmark_comparison(test_metrics: dict[str, dict]) -> None:
    models = list(test_metrics.keys())
    if not models:
        return

    metrics_list = [("sharpe", "Sharpe Ratio"), ("total_return", "Total Return"),
                    ("max_drawdown", "Max Drawdown"), ("volatility", "Volatility")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for idx, (metric_key, metric_label) in enumerate(metrics_list):
        ax = axes[idx // 2, idx % 2]
        x = np.arange(len(models))
        width = 0.18
        benchmarks = ["agent", "btc_hold", "equal_weight", "top_momentum", "risk_parity"]
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

        for bi, bench in enumerate(benchmarks):
            tag = bench if bench == "agent" else f"{bench.replace('_', '_')}"
            if bench == "agent":
                vals = [test_metrics[m].get(f"{MODEL_NAMES[m]}_{metric_key}", np.nan) for m in models]
            else:
                vals = [test_metrics[m].get(f"{MODEL_NAMES[m]}_{bench}_{metric_key}", np.nan) for m in models]
            ax.bar(x + bi * width - 2 * width, vals, width, label=bench.replace("_", " ").title(), color=colors[bi])

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(metric_label)
        ax.legend(fontsize=7)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIGS / "benchmark_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved benchmark_comparison.png")


def plot_val_sharpe_distribution(histories: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, hist in histories.items():
        if not hist:
            continue
        sharpe = [h["sharpe"] for h in hist]
        ax.hist(sharpe, bins=30, alpha=0.5, label=f"{MODEL_NAMES[name]} (mean={np.mean(sharpe):.2f})")
    ax.set_xlabel("Validation Sharpe")
    ax.set_ylabel("Frequency")
    ax.set_title("Validation Sharpe Distribution")
    ax.legend()
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "val_sharpe_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved val_sharpe_distribution.png")


def generate_latex_tables(test_metrics_raw: dict[str, dict], histories: dict[str, list[dict]]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Model Comparison on Test Set}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Sharpe $\uparrow$ & Return $\uparrow$ & Vol $\downarrow$ & Max DD $\uparrow$ & Win Rate $\uparrow$ & Pos Sharpe \% $\uparrow$ \\",
        r"\midrule",
    ]
    for name, tag in [("ppo", "PPO"), ("sac", "SAC"), ("td3", "TD3")]:
        m = test_metrics_raw.get(name, {})
        if not m:
            continue
        sharpe = m.get(f"{tag}_sharpe", np.nan)
        ret = m.get(f"{tag}_total_return", np.nan)
        vol = m.get(f"{tag}_volatility", np.nan)
        dd = m.get(f"{tag}_max_drawdown", np.nan)
        wr = m.get(f"{tag}_win_rate", np.nan)
        ps = m.get(f"{tag}_positive_sharpe", np.nan)
        val_sharpe = float(np.mean([h["sharpe"] for h in histories.get(name, [])])) if histories.get(name) else np.nan
        lines.append(
            f"  {tag} & {sharpe:+.2f} & {ret:+.1%} & {vol:.1%} & {dd:.1%} & {wr:.1%} & {ps:.0%} \\\\"
        )
    lines += [
        r"\midrule",
    ]
    for name, tag in [("ppo", "PPO"), ("sac", "SAC"), ("td3", "TD3")]:
        m = test_metrics_raw.get(name, {})
        if not m:
            continue
        val_sharpe = float(np.mean([h["sharpe"] for h in histories.get(name, [])])) if histories.get(name) else np.nan
        lines.append(
            f"  {tag} (val) & {val_sharpe:+.2f} & -- & -- & -- & -- & -- \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex = "\n".join(lines)
    (TABLES / "model_comparison.tex").write_text(tex)
    print(f"  Saved model_comparison.tex")

    bm_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Benchmark Comparison on Test Set}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Benchmark & Sharpe & Return & Volatility & Max DD & Relative Return \\",
        r"\midrule",
    ]
    for name, tag in [("ppo", "PPO"), ("sac", "SAC")]:
        m = test_metrics_raw.get(name, {})
        if not m:
            continue
        bm_lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{tag}}}}} \\")
        for bench in ["btc_hold", "equal_weight", "top_momentum", "risk_parity"]:
            s = m.get(f"{tag}_{bench}_sharpe", np.nan)
            r = m.get(f"{tag}_{bench}_return", np.nan)
            v = m.get(f"{tag}_{bench}_volatility", np.nan)
            d = m.get(f"{tag}_{bench}_max_drawdown", np.nan)
            rr = m.get(f"{tag}_{bench}_relative_return", np.nan)
            bm_lines.append(
                f"  {bench.replace('_', ' ').title()} & {s:+.2f} & {r:+.1%} & {v:.1%} & {d:.1%} & {rr:+.1%} \\\\"
            )
        bm_lines.append(r"\midrule")
    bm_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "benchmark_comparison.tex").write_text("\n".join(bm_lines))
    print(f"  Saved benchmark_comparison.tex")


def save_summary_metrics(test_metrics_raw: dict[str, dict], histories: dict[str, list[dict]]) -> None:
    summary = {}
    for name, tag in [("ppo", "PPO"), ("sac", "SAC"), ("td3", "TD3")]:
        m = test_metrics_raw.get(name, {})
        if not m:
            continue
        summary[name] = {
            "test_sharpe": m.get(f"{tag}_sharpe"),
            "test_total_return": m.get(f"{tag}_total_return"),
            "test_volatility": m.get(f"{tag}_volatility"),
            "test_max_drawdown": m.get(f"{tag}_max_drawdown"),
            "test_positive_sharpe": m.get(f"{tag}_positive_sharpe"),
            "test_win_rate": m.get(f"{tag}_win_rate"),
            "vs_btc_return": m.get(f"{tag}_btc_hold_relative_return"),
            "vs_equal_weight_return": m.get(f"{tag}_equal_weight_relative_return"),
            "benchmark_btc_sharpe": m.get(f"{tag}_btc_hold_sharpe"),
            "benchmark_equal_weight_sharpe": m.get(f"{tag}_equal_weight_sharpe"),
        }
        hist = histories.get(name, [])
        if hist:
            val_sharpes = [h["sharpe"] for h in hist]
            summary[name]["val_sharpe_mean"] = float(np.mean(val_sharpes))
            summary[name]["val_sharpe_std"] = float(np.std(val_sharpes))
            summary[name]["val_sharpe_best"] = max(val_sharpes)
            summary[name]["val_positive_sharpe"] = float(np.mean([s > 0 for s in val_sharpes]))

    (METRICS / "summary_statistics.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved summary_statistics.json")


def main() -> None:
    for d in [FIGS, TABLES, METRICS]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig()

    print("Loading data...")
    all_frames = load_all_coins()
    if not all_frames:
        print("No coin data found. Generate synthetic data or run pipeline first.")
        return

    print("Building test environment...")
    test_env = build_env(all_frames, cfg.test_start, cfg.test_end, cfg)
    print(f"  Test env: {test_env.n_steps} days, {test_env.n_assets} assets")

    print("Loading agents...")
    agents = {}
    for name in ["ppo", "sac", "td3"]:
        print(f"  Loading {name}...")
        agents[name] = load_agent(name, test_env)
        if agents[name] is None:
            print(f"    No saved model found for {name}")

    print("Loading metrics...")
    histories = {name: load_val_history(name) for name in ["ppo", "sac", "td3"]}
    test_metrics_raw = {name: load_test_metrics(name) for name in ["ppo", "sac", "td3"]}

    print("Generating charts...")
    plot_training_history(histories)
    plot_equity_curves(agents, test_env)
    plot_allocation_heatmaps(agents, test_env)
    plot_benchmark_comparison(test_metrics_raw)
    plot_val_sharpe_distribution(histories)

    print("Generating LaTeX tables...")
    generate_latex_tables(test_metrics_raw, histories)

    print("Generating summary metrics...")
    save_summary_metrics(test_metrics_raw, histories)

    print(f"\nAll report artifacts saved to {ROOT / 'results'}/")
    print(f"  Figures:  {FIGS}/")
    print(f"  Tables:   {TABLES}/")
    print(f"  Metrics:  {METRICS}/")


if __name__ == "__main__":
    main()
