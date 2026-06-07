# PTDLL — RL Portfolio Allocation + Risk Stop-Loss

> **Mục tiêu:** Hệ thống 2 tầng — **RL portfolio allocation** (PPO/SAC/TD3) phân bổ vốn 15 coins + USDT, **risk ML layer** (ANN/LSTM/CNN) dự đoán stop-loss % từng coin.

## Kiến trúc

```
src/
├── config.py                # PipelineConfig: dates, paths, hyperparams
├── main.py                  # CLI: portfolio|risk {train|predict|report}
├── log.py                   # File logger per module
├── lib/
│   ├── features.py          # pct_change, rolling_xx, RSI, MACD, candle, ...
│   ├── metrics.py           # sharpe, sortino, max_drawdown, win_rate, ...
│   ├── plot.py              # multi_line, equity_curve, heatmap, histogram, ...
│   └── utils.py             # save/load JSON/CSV, shared_dates, ffill_grid, norm_w
├── dataset/
│   └── fetch.py             # COINS_15 (15 coins), crawl_yfinance, generate_synthetic_ohlcv, load_coin_arrays
├── portfolio/
│   ├── env.py               # build_env → CryptoPortfolioEnv
│   ├── base.py              # CryptoPortfolioEnv, BaseModel ABC, 5 NN modules, ReplayBuffer
│   ├── ppo.py               # PPOAgent(BaseModel)
│   ├── sac.py               # SACAgent(BaseModel)
│   ├── td3.py               # TD3Agent(BaseModel)
│   ├── train.py             # train_seq, train_par, _train_one, train_save
│   └── evaluate.py          # make_agent, load_agent, sim_agent, eval_agent, log_results
├── risk/
│   ├── base.py              # StopANN, StopLSTM, StopCNN, BaseStopModel, Embedding, asym_mae, boundary_reg, auto_label
│   ├── train.py             # to_arrays (cache), build_data (temporal split), train (fit loop)
│   ├── evaluate.py          # eval_model, compare (dùng lib/metrics hit_rate)
│   └── predict.py           # predict_stop(model, x_60d, coin_idx) → stop_%
└── report.py                # gen_report → figures, tables, history
```

## 2-Tier Flow

```
RL allocate weights (15) ── 60d lookback, 90d rebalance
  │
  ▼
Set stop-loss cho từng coin (trừ USDT):
  predict(x_60d, coin_idx) → stop_% ∈ [0.05, 0.50]
  stop_price = close × (1 - stop_%)
  stop_price CỐ ĐỊNH — không trailing, không thay đổi

Hàng ngày: check daily close ≤ stop_price?
  CÓ    → auto sell → USDT → RL re-allocate
  KHÔNG → hold
```

## 2 Layer

| Layer | Loại | Models | Input | Output |
|-------|------|--------|-------|--------|
| **Portfolio (RL)** | Reinforcement Learning | PPO, SAC, TD3 | state cube (60,15,14) | allocation weights (15,) |
| **Risk (ML)** | Supervised Learning | ANN, LSTM, CNN | x (60,13) + coin_idx | stop_% ∈ [0.05, 0.50] per coin |

## Feature Cube (14 channels)

```
build_cube(coin_data) → cube (T, 15, 14)

  ├── Per-coin (7): return_1d, 7d, 30d, 90d, volatility, drawdown, volume_change
  │     └── từ coin_fx(close, volume)
  ├── Cross-coin (2): relative_strength_vs_BTC, correlation_vs_BTC
  │     └── từ cross_fx(per_coin)
  ├── Market (4): btc_ma200_position, market_volatility, btc_momentum_90d, market_breadth
  │     └── từ mkt_regime(per_coin, btc_close)
  └── Weight (1): injected by env.get_state() tại runtime
```

RL dùng cả 14 channels. Risk dùng 13 channel đầu (bỏ weight).

## Data Split

| Split | Range | Duration | Dùng cho |
|-------|-------|----------|----------|
| Train | 2017-01-01 → 2024-06-01 | ~7 năm | RL training + risk training |
| Validation | 2024-06-01 → 2025-06-01 | ~1 năm | RL val (Sharpe tracking) + risk val |
| Test | 2025-06-01 → 2026-06-01 | ~1 năm | RL test + risk test |

Risk train samples bị giới hạn thêm: chỉ lấy sample có `date[t] + 90 ≤ train_end` (label window không leak vào val), nên train risk thực tế = 2018-01-08 → 2024-03-02.

## Tests

11 unit tests pass — env, PPO/SAC/TD3 agents, indicators.

## Risk Label Distribution

Với real yfinance data (2017→2026), 40,404 samples:

| Split | Samples | Mean | < 10% | 10–30% | > 40% |
|-------|--------:|:----:|:-----:|:------:|:-----:|
| Train | 31,444 | 0.276 | 25% | 30% | 33% |
| Validation | 5,110 | 0.225 | 28% | 44% | 19% |
| Test | 3,850 | 0.291 | 22% | 24% | 35% |

## Results (2026-06-03)

| Model | Val Sharpe | Test Sharpe | Test Return |
|-------|:----------:|:-----------:|:-----------:|
| PPO | 1.00 | -0.79 | -54% |
| SAC | 1.05 | +0.18 | -5% |
| TD3 | — | — | — |
