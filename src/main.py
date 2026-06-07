from __future__ import annotations

import argparse

import numpy as np

from config import MODEL_DIR, MODEL_TAGS, PREDICTIONS_DIR, PipelineConfig
from dataset.fetch import load_coin_arrays
from portfolio.env import build_env
from portfolio.evaluate import load_agent
from portfolio.train import train
from lib.utils import save_csv, ensure_dirs
from report import gen_report


def cmd_train(args: argparse.Namespace) -> None:
    if args.episodes:
        cfg = PipelineConfig(n_episodes=args.episodes)
    else:
        cfg = PipelineConfig()
    model_list = [(m, {}) for m in args.models]
    parallel = args.mode == "parallel"
    train(models=model_list, parallel=parallel, cfg=cfg)


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = PipelineConfig()
    arrays = load_coin_arrays()
    env = build_env(arrays, cfg.test_start, cfg.test_end, cfg)
    agent = load_agent(args.model, env)
    if agent is None:
        print(f"Model {args.model} not found at {MODEL_DIR / args.model}.pt")
        return

    state = env.reset()
    pv = [1.0]
    weights_history: list[np.ndarray] = []
    done = False
    while not done:
        w = agent.predict(state)
        weights_history.append(w)
        next_state, _, done, _ = env.step(w)
        pv.append(env.portfolio_value)
        state = next_state

    dates = env.date_index[env.start_idx:env.end_idx + 1]
    metrics = env.score_ep(env.bench_paths())

    out_dir = PREDICTIONS_DIR
    ensure_dirs(out_dir)
    save_csv({"date": dates.astype(str), "portfolio_value": np.array(pv[1:])},
             out_dir / f"{args.model}_pv.csv")
    for i, name in enumerate(env.asset_names):
        save_csv({"date": dates.astype(str), "weight": np.array([w[i] for w in weights_history])},
                 out_dir / f"{args.model}_weight_{name}.csv")

    print(f"{MODEL_TAGS[args.model]} test Sharpe={metrics.get('sharpe', 0):.4f}, "
          f"Return={metrics.get('total_return', 0):.4f}")
    print(f"Predictions saved to {out_dir}/")


def portfolio_parser(sub) -> None:
    p = sub.add_parser("portfolio", help="Portfolio management layer")
    p_sub = p.add_subparsers(dest="command", required=True)

    p_train = p_sub.add_parser("train")
    p_train.add_argument("--mode", choices=["seq", "parallel", "resume"], default="seq")
    p_train.add_argument("--models", nargs="+", default=["ppo", "sac", "td3"],
                         choices=["ppo", "sac", "td3"])
    p_train.add_argument("--episodes", "-e", type=int, default=1000,
                         help="Number of training episodes (default: 1000)")
    p_train.set_defaults(func=cmd_train)

    p_predict = p_sub.add_parser("predict")
    p_predict.add_argument("--model", choices=["ppo", "sac", "td3"], default="ppo")
    p_predict.set_defaults(func=cmd_predict)

    p_report = p_sub.add_parser("report")
    p_report.set_defaults(func=lambda a: gen_report())


def cmd_risk_train(args: argparse.Namespace) -> None:
    from risk.train import MODEL_NAMES, train as risk_train

    models = list(MODEL_NAMES.keys()) if "all" in args.models else args.models
    for name in models:
        risk_train(name)


def cmd_risk_predict(args: argparse.Namespace) -> None:
    pass


def cmd_risk_report(args: argparse.Namespace) -> None:
    pass


def risk_parser(sub) -> None:
    p = sub.add_parser("risk", help="Risk management layer")
    p_sub = p.add_subparsers(dest="command", required=True)

    p_train = p_sub.add_parser("train")
    p_train.add_argument("--models", nargs="+", default=["all"],
                         choices=["ann", "lstm", "cnn", "all"])
    p_train.set_defaults(func=cmd_risk_train)

    p_predict = p_sub.add_parser("predict")
    p_predict.add_argument("--model", choices=["ann", "lstm", "cnn"], default="cnn")
    p_predict.set_defaults(func=cmd_risk_predict)

    p_report = p_sub.add_parser("report")
    p_report.set_defaults(func=cmd_risk_report)


def main() -> None:
    parser = argparse.ArgumentParser(description="PTDLL — Portfolio Trading with Deep Reinforcement Learning")
    sub = parser.add_subparsers(dest="layer", required=True)

    portfolio_parser(sub)
    risk_parser(sub)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
