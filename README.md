# PTDLL — Crypto Portfolio with RL + Risk Stop-Loss

Hệ thống 2 tầng: **RL portfolio allocation** (PPO/SAC/TD3) + **ML stop-loss prediction** (ANN/LSTM/CNN) cho 15 crypto coins.

## Architecture

```
portrait.py train --mode parallel --models ppo sac td3    # RL allocate
risk train {ann|lstm|cnn}                                   # Stop-loss predict
```

| Layer | Loại | Models | Input | Output |
|-------|------|--------|-------|--------|
| **Portfolio** | RL | PPO, SAC, TD3 | state cube (60,15,14) | allocation weights (15,) |
| **Risk** | ML | ANN, LSTM, CNN | (60,13) + coin_idx | stop_% ∈ [0.05, 0.50] |

## Quick start

```powershell
uv sync
uv run pytest -v                    # 11 tests
uv run -m python src.main portfolio train --mode seq  # train RL
uv run -m python src.main portfolio report             # charts + tables
```

## 15 coins

BTC, LTC, XRP, DOGE, XMR, DASH, XLM, USDT, ETH, ETC, WAVES, ZEC, DCR, NEO, BNB

## State cube (14 features)

| Feature | Ý nghĩa |
|---------|---------|
| return_1d / 7d / 30d / 90d | Rolling returns |
| volatility | 20-day rolling std |
| drawdown | close / rolling_max - 1 |
| volume_change | volume % change |
| relative_strength_vs_BTC | return_30d vs BTC |
| correlation_vs_BTC | 60d rolling corr |
| btc_ma200_position | BTC vs SMA200 |
| market_volatility | avg vol all coins |
| btc_momentum_90d | BTC 90d return |
| market_breadth | % coins positive |
| weight | injected by env (RL only) |

Risk dùng 13 feature đầu (không weight).

## 3 RL Models

| Model | Paradigm | Key mechanism |
|-------|----------|--------------|
| PPO | On-policy | Clipped surrogate, GAE(λ=0.95), K=4 epochs |
| SAC | Off-policy | Max entropy, auto temperature, twin Q |
| TD3 | Off-policy | Delayed policy, target smoothing, twin Q |

## 3 Risk Models

| Model | Backbone | Temporal |
|-------|----------|----------|
| ANN | MLP | ❌ flatten |
| LSTM | LSTM(64) | ✅ sequence |
| CNN | Conv1D×2 + BN | ✅ conv |

Cả 3 dùng Embedding(14, 4) cho coin_id. Loss: asym_mae(over=0.5, under=2.0).

## Data Split

| Split | Range |
|-------|-------|
| Train | 2017-01-01 → 2024-06-01 |
| Validation | 2024-06-01 → 2025-06-01 |
| Test | 2025-06-01 → 2026-06-01 |

## Project Structure

```
src/
├── config.py              # PipelineConfig (pydantic-settings)
├── main.py                # CLI: portfolio|risk {train|predict|report}
├── lib/                   # features.py, metrics.py, plot.py, utils.py
├── dataset/fetch.py       # 15 coins loader
├── portfolio/             # env.py, base.py, ppo/sac/td3.py, train.py, evaluate.py
├── risk/                  # base.py, train.py, evaluate.py, predict.py
└── report.py              # gen_report (figures, tables, LaTeX)

docs/                      # Architecture docs (Obsidian format)
  overview.md, features.md, stop-loss.md, rl-metrics.md,
  workflow.md, tasks.md, results.md
```

## Results

| Model | Val Sharpe | Test Sharpe | Test Return |
|-------|:----------:|:-----------:|:-----------:|
| PPO | 1.00 | -0.79 | -54% |
| SAC | 1.05 | +0.18 | -5% |
| TD3 | — | — | — |

## License

MIT
