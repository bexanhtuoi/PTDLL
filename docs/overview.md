# PTDLL — RL Portfolio Allocation

> **Mục tiêu:** Hệ thống **RL portfolio allocation** với USDT làm risk-free asset. 3 models (PPO, SAC, TD3) so sánh trên cùng environment. **Không dùng classification** — agent tự học phân bổ vốn + hold cash.

## Kiến trúc

```
src/
├── config.py              # PipelineConfig: train/val/test dates, episode params
├── main.py                # Entry → models.train.run()
├── utils.py               # sharpe, max_drawdown, save helpers
├── scores.py              # build_feature_table (RSI, MACD, returns, ...)
├── dataset/
│   └── fetch.py           # 15 coins OHLCV loader, synthetic data, align
├── models/
│   ├── base.py            # BaseModel (ABC) + PolicyNet, ValueNet, TwinQNet, ...
│   ├── ppo.py             # PPOAgent(BaseModel): on-policy, clipped surrogate
│   ├── sac.py             # SACAgent(BaseModel): off-policy, max entropy
│   ├── td3.py             # TD3Agent(BaseModel): off-policy, deterministic
│   ├── env.py             # CryptoPortfolioEnv + aligned_features
│   ├── train.py           # create_agent, train_model, run()
│   ├── evaluation.py      # run_test, print_results
│   └── predict.py         # predict_weights, export_to_onnx
├── scripts/
│   └── visualize.py       # equity curve + allocation heatmap
└── (planning) stop_loss/
    ├── label.py           # Auto-labeling target stop_%
    ├── baseline.py        # ATR Baseline (rule-based)
    ├── xgb_model.py       # XGBoost Regressor
    ├── stopnet.py         # Conv1D StopNet
    └── evaluate.py        # Evaluation & comparison
```

> **2-tier architecture**: RL allocate → Stop-loss predict stop_% per coin → Daily check → Sell on hit → USDT → RL re-allocate

## 3 RL Models

| Model | Paradigm | Policy | Critic | Update Mechanism |
|-------|----------|--------|--------|-----------------|
| **PPO** | On-policy stochastic | `PolicyNet` (Conv1D+FC) | `ValueNet` (Conv1D+FC) | Clipped surrogate + GAE(λ=0.95) + K=4 epochs |
| **SAC** | Off-policy stochastic | `PolicyNet` | `TwinQNet` (twin Q) | Min double Q + auto entropy α + soft update |
| **TD3** | Off-policy deterministic | `DeterministicActor` (softmax) | `TwinQNet` | Clipped double Q + delayed policy + target smoothing |

### Network architecture
```
Input:  (batch, L=60, A=15, F=14)
            ↓ reshape (batch, A*F=210, L=60)
     Conv1d(210→64, k=5) + ReLU + AdaptiveAvgPool → (batch, 64)
            ↓
     FC(64→128) → ReLU → FC(128→64) → ReLU → FC(64→n_assets)
            ↓
     Dirichlet(softmax×15 + 1) / softmax → allocation weights
```

## Data Split

| Split | Range | Duration |
|-------|-------|----------|
| Train | 2017-01-01 → 2024-06-01 | ~7 năm (2396 days) |
| Validation | 2024-06-01 → 2025-06-01 | ~1 năm |
| Test | 2025-06-01 → 2026-06-01 | ~1 năm |

## Episode Design

- **Episode**: 2 năm (730 days) — random window trong train
- **Step**: 90 days → ~8 steps/episode (730/90)
- **Lookback**: 60 days
- **Fee**: 0.1% per rebalance
- **Reward**: `excess_return - 0.1×vol - 0.05×dd_90 - 0.001×turnover`

## USDT Integration

USDT (coin thứ 8/15) có features ≈ 0. Agent allocate weight vào USDT = hold cash. Không cần classification buy/sell riêng — agent tự học khi nào nên ở ngoài thị trường.

## Pipeline (`models/train.py:run`)

```
1. Clean outputs → xoá results/
2. Load 15 coins → filter_by_date (train/val/test)
3. Build envs: CryptoPortfolioEnv × 3 splits
4. For each model (PPO → SAC → TD3):
   a. create_agent
   b. train_model: 5000 episodes, val every 50 eps
   c. save checkpoint → models/{name}.pt
   d. run_test on test set
5. Print comparison table
6. Publish artifacts
```

## Kết quả hiện tại

| Model | Val Sharpe (avg) | Test Sharpe | Test Return | Pos Rate |
|-------|:-:|:-:|:-:|:-:|
| PPO | 1.00 | -0.79 | -54% | 0% |
| SAC | 1.05 | +0.18 | -5% | 100% |

> SAC outperform PPO trên test set (Sharpe +0.18 vs -0.79). TD3 đang chạy.
> Val Sharpe của cả 2 đều >1.0 — models học được policy có kỹ năng trên validation.

## State cube

```
(T, 15 assets, 14 features)
├── 0: return_1d
├── 1: return_7d
├── 2: return_30d
├── 3: return_90d
├── 4: volatility (20d)
├── 5: drawdown
├── 6: volume_change (clipped [-5, 5])
├── 7: relative_strength_vs_BTC
├── 8: correlation_vs_BTC (rolling 60d)
├── 9: btc_ma200_position
├── 10: market_volatility
├── 11: btc_momentum_90d
├── 12: market_breadth
└── 13: weight (injected by env)
```

## Tests

11 tests — env, PPO/SAC/TD3 agents, indicators, filter_by_date. Pass all.
