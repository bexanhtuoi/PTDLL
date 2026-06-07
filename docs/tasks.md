# Danh sách công việc triển khai

> **Hoàn thành core:** RL pipeline (PPO/SAC/TD3) + data pipeline + features. 11/11 tests pass. Risk layer đang xây dựng.

---

## Kiến trúc hiện tại

```
src/
├── config.py                # PipelineConfig: dates, paths, hyperparams
├── main.py                  # CLI: portfolio|risk {train|predict|report}
├── log.py                   # File logger
├── lib/
│   ├── __init__.py
│   ├── features.py          # Technical indicators (numpy, no pandas)
│   ├── metrics.py           # Sharpe, drawdown, win_rate, ...
│   ├── plot.py              # matplotlib charts
│   └── utils.py             # JSON/CSV, date alignment, ffill, norm
├── dataset/
│   └── fetch.py             # 15 coins loader, synthetic, alignment
├── portfolio/               # RL layer
│   ├── base.py              # CryptoPortfolioEnv + BaseModel + 5 NNs
│   ├── ppo.py / sac.py / td3.py
│   ├── env.py / train.py / evaluate.py
└── risk/                    # ML layer (đang xây)
    ├── base.py              # StopANN, StopLSTM, StopNet, Embedding
    ├── train.py             # train_risk()
    ├── evaluate.py          # hit_rate, false_positive, saved_drawdown
    └── predict.py           # predict_stop()
```

## Thứ tự đọc code

1. `config.py` — hiểu tham số pipeline
2. `lib/features.py` — feature engineering (numpy, no pandas)
3. `lib/metrics.py` — evaluation metrics
4. `portfolio/base.py` — CryptoPortfolioEnv + BaseModel + shared NNs
5. `portfolio/ppo.py / sac.py / td3.py` — 3 RL agents
6. `portfolio/train.py` — orchestrator
7. `report.py` — gen_report (figures, tables, LaTeX)
8. `risk/base.py` — StopANN, StopLSTM, StopNet
9. `risk/train.py / evaluate.py / predict.py` — risk pipeline

## Đã hoàn thành

### Data pipeline
- [x] OHLCV loader, synthetic data generator
- [x] shared_dates, ffill_grid, btc_grid alignment
- [x] create_features (RSI, MACD, candle, returns, MA, vol, drawdown)

### RL Portfolio
- [x] CryptoPortfolioEnv (gym-like, 15 assets, state cube)
- [x] BaseModel ABC (predict, fit, save, load, eval_ckpt)
- [x] PolicyNet, ValueNet, StateEncoder, TwinQNet, DeterministicActor
- [x] PPOAgent (on-policy, clipped surrogate, GAE)
- [x] SACAgent (off-policy, max entropy, auto temp)
- [x] TD3Agent (off-policy, deterministic, delayed policy)
- [x] train_seq / train_par (Windows-safe multiprocessing)
- [x] USDT as risk-free asset
- [x] 4 benchmarks (BTC hold, EW, momentum, risk parity)
- [x] gen_report (figures: equity, sharpe history; tables: LaTeX)

### Config & Tests
- [x] PipelineConfig (pydantic-settings)
- [x] 11 tests (env, 3 agents, indicators)
- [x] --mode parallel / --episodes flag

## Hoàn thành

### Risk ML layer
- [x] StopANN, StopLSTM, StopCNN trong risk/base.py + BaseStopModel ABC
- [x] Embedding(14, 4) cho coin_id
- [x] asym_mae loss (over=0.5, under=2.0) + boundary_reg (alpha=0.001)
- [x] auto_label (window 90 ngày, buffer 20%)
- [x] to_arrays (cache NPZ) + build_data (temporal split, label-safe) + train (fit loop)
- [x] evaluate: eval_model (MAE, hit_rate) + compare
- [x] predict: predict_stop(model, x_60d, coin_idx) → stop_%
- [x] main.py risk CLI: train (--models ann|lstm|cnn|all)

### Risk evaluation metrics
- [x] hit_rate, false_positive_rate trong lib/metrics.py

## Tương lai

- [ ] Live trading + risk guardrails
- [ ] Walk-forward validation (thay vì fixed test set)
- [ ] ONNX deployment
- [ ] Streamlit dashboard
