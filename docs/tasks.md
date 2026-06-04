# Danh sách công việc triển khai

> **Refactor hoàn thành:** Chuyển từ classification + REINFORCE pipeline → 3 RL models (PPO/SAC/TD3) + USDT cash. 11/11 tests pass.

---

## Kiến trúc cuối

```
src/
├── config.py              # PipelineConfig: dates, params, hyperparams
├── main.py                # Entry → models.train.run()
├── utils.py               # I/O, sharpe_ratio, max_drawdown, win_rate, time split
├── scores.py              # build_feature_table (RSI/MACD/returns/vol/dd)
├── dataset/
│   └── fetch.py           # load_all_coins, aligned_prices, synthetic OHLCV
├── models/
│   ├── __init__.py        # Exports all 13 symbols
│   ├── base.py            # BaseModel ABC + PolicyNet/ValueNet/StateEncoder/TwinQNet/DeterministicActor/ReplayBuffer
│   ├── ppo.py             # PPOAgent(BaseModel)
│   ├── sac.py             # SACAgent(BaseModel)
│   ├── td3.py             # TD3Agent(BaseModel)
│   ├── env.py             # CryptoPortfolioEnv + filter_by_date + aligned_features + build_env
│   ├── train.py           # create_agent + train_model + run() orchestrator
│   ├── evaluation.py      # run_test + print_results
│   └── predict.py         # predict_weights + predict_portfolio_returns + export_to_onnx
└── scripts/
    └── visualize.py       # Charts: equity curve, allocation heatmap
```

## Thứ tự đọc code

1. `config.py` — hiểu tham số pipeline
2. `scores.py` — feature engineering
3. `models/base.py` — BaseModel contract + shared networks
4. `models/env.py` — CryptoPortfolioEnv + state cube
5. `models/ppo.py` — PPOAgent (ví dụ on-policy)
6. `models/sac.py` — SACAgent (off-policy stochastic)
7. `models/td3.py` — TD3Agent (off-policy deterministic)
8. `models/train.py` — orchestrator ghép pipeline
9. `models/evaluation.py` — test + results table

## Đã xoá

- `pipeline.py` → thay bằng `models/train.py:run()`
- `models/rl.py` → thay bằng `models/ppo.py` + `sac.py` + `td3.py`
- `dataset/labeling.py` — không cần buy/sell labels
- `models/training.py` — classification train loop
- `models/evaluation.py` — classification metrics
- `models/export.py` — sklearn/LSTM ONNX export
- `models/predict.py` — classification signal generation
- `scripts/reports.py` — grid search

## Lưu ý kiến trúc

- **BaseModel ABC**: `forward()` → weights, `train_episode(env)` → loss, `run_episode(env)` → metrics dict
- **USDT**: coin index 7, features ≈ 0, agent tự học hold cash naturally
- **Train Concurrency**: validation chạy mỗi 50 episodes trong cùng vòng loop chính
- **Episode = 2 years**: đủ dài để học multi-regime patterns
- **State normalization**: z-score per channel (exclude weight channel)
- **pyproject.toml**: `pythonpath = ["src"]` cho pytest
- **Stop-Loss Layer**: [stop-loss.md](stop-loss.md) — 2 tầng (RL → Stop → Execute), 3 models đang so sánh
- **Stop-loss cứng**: Set 1 lần sau RL allocate, check daily close, không trailing
