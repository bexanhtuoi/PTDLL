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
│   ├── base.py              # StopANN, StopLSTM, StopCNN, BaseStopModel, Embedding, combined_loss (MSE + boundary_reg), auto_label
│   ├── train.py             # to_arrays (cache), build_data (temporal split), train (fit loop)
│   ├── evaluate.py          # eval_model, compare (dùng lib/metrics hit_rate)
│   └── predict.py           # predict_stop(model, x_60d, coin_idx) → stop_%
├── gen_report.py            # Full report: 10 charts + LaTeX tables + JSON metadata
└── report.py                # Portfolio report generation (LaTeX tables, figures)
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
| **Risk (ML)** | Supervised Learning | ANN, LSTM, CNN | x (60,17) + coin_idx (0..13) | stop_% ∈ [0.05, 0.50] per coin |

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

RL dùng cả 14 channels. Risk dùng 17 features: 7 per-coin + 4 extra (RSI, skew, down_ratio, SMA200 dist) + 2 cross + 4 market (xem `features.md`).

## Data Split

| Split | Range | Duration | Dùng cho |
|-------|-------|----------|----------|
| Train | 2017-01-01 → 2024-06-01 | ~7 năm | RL training + risk training |
| Validation | 2024-06-01 → 2025-06-01 | ~1 năm | RL val (Sharpe tracking) |
| Test | 2025-06-01 → 2026-06-01 | ~1 năm | RL test + risk test |

Risk label window = 10 ngày (không 90). Target = z-score của forward 10-day max drawdown, tính per-coin từ training data. Không có val split riêng — train trên all pre-test data, test set dùng để validation.

## Tests

11 unit tests pass — env, PPO/SAC/TD3 agents, indicators.

## Portfolio Results (Bear Market 2025-2026)

| Model | Version | Multi Sharpe | Pos. Episodes | Return | Method |
|-------|---------|:-----------:|:-------------:|:------:|--------|
| **PPO** | **v2** | **+0.388** | **100%** | **+19.6%** | SAC transfer + perturb |
| **TD3** | **v2** | **+0.375** | **100%** | **+19.2%** | SAC transfer + perturb |
| SAC | v1 | +0.285 | 95% | +13.5% | Weight perturb (σ=0.03) |

## Risk Results (Test 2025-2026, calibrated)

| Model | Test Corr | Pred Std | Act Std | MAE |
|-------|:--------:|:--------:|:------:|:---:|
| ANN | ~0.08 | ~0.06 | 0.164 | ~0.16 |
| **LSTM Ensemble** | **~0.31** | **~0.10** | **0.164** | **~0.047** |
| CNN | ~0.05 | ~0.05 | 0.164 | ~0.21 |

*Số liệu chính xác xem `results/risk_history.json` sau khi chạy `python src/gen_report.py`.*

Full report: `python src/gen_report.py`
