# PTDLL — Crypto Portfolio Allocation with RL

A **Reinforcement Learning portfolio optimization system** for 15 crypto coins with USDT as a risk-free asset, featuring 3 RL algorithms (PPO, SAC, TD3) for mid-term investment allocation, plus a planned **stop-loss prediction layer** for risk management.

## Techstack

| Layer | Technology | Purpose |
|---|---|---|
| **RL Framework** | PyTorch 2.3+ | Neural networks, optimizers, gradient computation |
| **Models** | PPO, SAC, TD3 | On-policy & off-policy portfolio optimization |
| **Data** | yfinance, pandas, numpy | OHLCV data loading, feature engineering |
| **Features** | RSI, MACD, volatility, drawdown, MA, returns | Technical indicators & cross-sectional features |
| **State Env** | Custom CryptoPortfolioEnv | Gym-like RL environment with 15 assets, 12 features |
| **Action Space** | Dirichlet distribution | Continuous portfolio weights (sum-to-1) |
| **Export** | ONNX, onnxruntime | Model deployment (prototype) |
| **Config** | pydantic-settings | Typed pipeline configuration |
| **Testing** | pytest | 11 unit tests |
| **Visualization** | matplotlib, seaborn | Equity curves, allocation heatmaps |
| **Package Mgmt** | uv | Python dependency management |

## Project Structure

```
ptdll/
├── src/
│   ├── config.py              # PipelineConfig (pydantic-settings)
│   ├── main.py                # Entry → models.train.run()
│   ├── utils.py               # Sharpe, max_drawdown, I/O helpers
│   ├── scores.py              # Feature engineering (RSI, MACD, returns, vol, ...)
│   ├── dataset/
│   │   └── fetch.py           # 15 coins OHLCV loader, synthetic data, align
│   ├── models/
│   │   ├── base.py            # BaseModel ABC + PolicyNet, ValueNet, TwinQNet, ReplayBuffer
│   │   ├── ppo.py             # PPOAgent(BaseModel) — on-policy, clipped surrogate
│   │   ├── sac.py             # SACAgent(BaseModel) — off-policy, max entropy
│   │   ├── td3.py             # TD3Agent(BaseModel) — off-policy, deterministic
│   │   ├── env.py             # CryptoPortfolioEnv + aligned_features + build_env
│   │   ├── train.py           # create_agent, train_model, run() orchestrator
│   │   ├── evaluation.py      # run_test, print_results
│   │   └── predict.py         # predict_weights, export_to_onnx
│   └── scripts/
│       └── visualize.py       # Equity curve + allocation heatmap
│
├── tests/
│   ├── test_rl.py             # 10 tests (env, agents, filter)
│   └── test_indicators.py     # 1 test (RSI bounds)
│
├── data/
│   ├── raw/                   # Individual coin CSVs from yfinance
│   └── processed/             # Aligned prices, feature tables
│
├── models/                    # Trained checkpoints (.pt)
├── results/                   # Reports, metrics, figures
├── pyproject.toml             # Project config & dependencies
└── .claude/                   # Planning, research, TODO docs
```

## Key Features

- **3 RL Algorithms**: PPO (on-policy), SAC (off-policy stochastic), TD3 (off-policy deterministic) — same environment, same metrics, apples-to-apples comparison.
- **Multi-Asset State Cube**: `(time, 15 assets, 14 features)` tensor including returns, volatility, drawdown, volume, correlation, BTC regime metrics, and weight injection.
- **Conv1D Architecture**: Temporal convolution over asset×feature channels, extracting market patterns from 60-day lookback windows.
- **Dirichlet Action Space**: Continuous allocation weights that naturally sum to 1 — no softmax hacking, proper probability distribution.
- **USDT as Risk-Free Asset**: Cash (USDT) has zero-return features — agent self-learns when to hold cash without buy/sell classification.
- **Concurrent Validation**: Evaluates on 1-year validation window every 50 training episodes, tracks best Sharpe checkpoint.
- **4 Benchmarks**: BTC hold, equal-weight, top momentum, risk parity — relative performance comparison.
- **Stop-Loss Layer (planning)**: 3 predicted stop-loss models (ATR / XGBoost / Conv1D) for dynamic per-coin risk management.

## 3 RL Models

| Model | Paradigm | Policy | Critic | Key Mechanism |
|---|---|---|---|---|
| **PPO** | On-policy | PolicyNet (Conv1D) | ValueNet (Conv1D) | Clipped surrogate, GAE(λ=0.95), K=4 epochs |
| **SAC** | Off-policy | PolicyNet | TwinQNet (twin Q) | Max entropy, auto temperature, soft target update |
| **TD3** | Off-policy | DeterministicActor | TwinQNet | Clipped double Q, delayed policy smoothing |

### Network architecture

```
Input: (batch, L=60, A=15, F=14)
           ↓ reshape
(batch, channels=210, length=60)
           ↓
Conv1d(210→64, kernel=5, padding=2) + ReLU
           ↓
AdaptiveAvgPool1d(1) → flatten → (batch, 64)
           ↓
FC(64→128) → ReLU → FC(128→64) → FC(64→n_assets=15)
           ↓
Dirichlet / softmax → allocation weights
```

## Data Split

| Split | Range | Duration | Data Points |
|-------|-------|----------|:-----------:|
| Train | 2017-01-01 → 2024-06-01 | ~7 năm | 2,396 |
| Validation | 2024-06-01 → 2025-06-01 | ~1 năm | 365 |
| Test | 2025-06-01 → 2026-06-01 | ~1 năm | 365 |

15 coins: BTC, LTC, XRP, DOGE, XMR, DASH, XLM, USDT, ETH, ETC, WAVES, ZEC, DCR, NEO, BNB.

## Episode Design

| Parameter | Value | Description |
|---|---|---|
| Episode length | 2 years (730 days) | Random window in train set |
| Step (rebalance) | 90 days (~3 months) | Quarterly rebalancing |
| Steps/episode | ~8 | 730/90 |
| Lookback | 60 days | Historical window in state |
| Fee | 0.1% | Per rebalance transaction cost |

### Reward function

```
reward = rank_score - 0.2×vol - 0.25×dd_90 - 0.002×turnover + bear_bonus
```

- **rank_score** — wins vs 4 benchmarks (BTC hold, equal-weight, momentum, risk parity)
- **bear_bonus** — extra reward if profitable during bear markets (BTC return < -3%)
- **volume penalty** — discourages excessive trading
- **drawdown penalty** — discourages large portfolio declines

## State Cube (14 channels)

| Index | Feature | Description |
|:-----:|---------|-------------|
| 0 | return_1d | Daily close return |
| 1 | return_7d | 7-day rolling return |
| 2 | return_30d | 30-day rolling return |
| 3 | return_90d | 90-day rolling return |
| 4 | volatility | 20-day rolling std(return_1d) |
| 5 | drawdown | close / max(close) - 1 |
| 6 | volume_change | (vol - vol_prev) / vol_prev, clipped [-5, 5] |
| 7 | relative_strength_vs_BTC | return_30d[coin] vs BTC |
| 8 | correlation_vs_BTC | rolling 60d corr |
| 9 | btc_ma200_position | (BTC - SMA200) / SMA200 |
| 10 | market_volatility | Mean volatility across coins |
| 11 | btc_momentum_90d | (BTC - BTC_90d_ago) / BTC_90d_ago |
| 12 | market_breadth | Fraction of coins with return_30d > 0 |
| 13 | weight | Injected by env (current allocation) |

## Problems

- **Classification-based approaches fail** — Buy/sell signal classification ignores portfolio context and position sizing.
- **Fixed stop-loss is naive** — A single stop-loss percentage doesn't adapt to each coin's volatility or market regime.
- **On-policy models overfit validation** — PPO's test Sharpe (-0.79) drops sharply from validation (1.00), suggesting overfitting to market regimes seen during training.
- **Turnover penalty over-regularizes** — Agents learn buy-and-hold policies (turnover ~0) when penalty is too high, defeating rebalancing purpose.
- **Episode length mismatch** — 2-year episodes (730 days, 365-based) must match crypto's 365-day calendar. Code already uses 365.
- **No risk management during holding period** — Between 90-day rebalances, the portfolio is fully exposed with zero protection.
- **Offline training only** — No mechanism to retrain or adapt as new market data arrives.

## Solutions

- **RL portfolio allocation** — End-to-end learning: agent outputs weights directly, no separate classification or sizing step.
- **3-model comparison** — PPO, SAC, TD3 share identical environment and evaluation, isolating algorithm effects.
- **Dirichlet action distribution** — Proper probability distribution over simplex, unlike Gaussian or softmax-hack approaches.
- **Conv1D temporal encoder** — Learns market patterns from 60-day sliding windows, same architecture reused across all algorithms.
- **Concurrent validation** — Every 50 episodes, evaluates on 1-year hold-out set; tracks best Sharpe, not best training loss.
- **USDT as learnable cash** — Zero-feature asset allows model to self-learn cash holding periods.
- **On-chain + macro context** — Cross-sectional features (correlation, relative strength) give model situational awareness beyond single-coin TA.

## Current Results (2026-06-03)

| Model | Val Sharpe (avg) | Test Sharpe | Test Return | Positive Rate |
|-------|:----------------:|:-----------:|:-----------:|:-------------:|
| PPO | 1.00 | -0.79 | -54.4% | 0% |
| SAC | 1.05 | +0.18 | -4.9% | 100% |
| TD3 | — | — | — | Training |

> SAC generalizes better than PPO — off-policy experience replay + entropy regularization reduces overfitting. Both models achieve Val Sharpe > 1.0, indicating genuine skill on validation.

## Stop-Loss Layer (Planning)

> Full spec: [[PTDLL/stop-loss]] trong Obsidian vault.

### Architecture

**2-tier**: RL allocates capital → Stop-loss model predicts `stop_%` per coin → Daily check → Sell on hit → USDT → RL re-allocates.

```
USDT > 0? ──Có──→ RL allocate ──→ Set stop-loss cứng (1 lần, gọi model)
                                         │
                                    Hàng ngày (chỉ check, không gọi model):
                                    close ≤ stop_price?
                                         │
                                    Có → auto sell → USDT → quay lại RL
                                    Không → hold, chờ ngày mai
```

### Stop-loss properties

| Property | Detail |
|---|---|
| **Cứng** | `stop_price = close × (1 - stop_%)`, set 1 lần, không thay đổi |
| **Không trailing** | Không kéo lên theo peak |
| **Tự động** | Chạm là bán ngay |
| **Check daily** | Dùng daily close, không intraday |
| **USDT** | Không set stop |

### Auto-labeling

Sinh target từ historical data:

```python
Với mỗi ngày T, mỗi coin:
  future_close = close[T : T + 60]
  max_dd = (close[T] - min(future_close)) / close[T]
  target = clip(max_dd * 1.2, 0.05, 0.50)   # buffer 20% chống bán đáy
```

### 3 models comparison

| Model | Type | Pros | Cons |
|-------|------|------|------|
| **ATR Baseline** | Rule-based | Zero training, interpretable | No pattern learning |
| **XGBoost** | Tree | Feature importance, fast | Input must flatten |
| **Conv1D (StopNet)** | Deep | Temporal patterns, shares RL encoder | Needs tuning |

StopNet reuses `StateEncoder` from `models/base.py`:

```
Input: (state, weights)
  → StateEncoder(state) → (batch, 32)
  → concat(32, weights) → FC → Sigmoid → scale [0.05, 0.50]
```

### Loss function

```python
diff = predicted - target
penalty = where(diff > 0, diff * 0.5,     # stop rộng quá → phạt nhẹ
                where(diff < 0, -diff * 2.0))  # stop hẹp quá → phạt gấp 4
```

### Success criteria

1. **Hit Rate ≥ 80%** — stop không bị xuyên thủng
2. **False Positive ≤ 20%** — bán non
3. **Saved Drawdown ≥ -10%** so với không stop
4. **Sharpe cải thiện** so với RL baseline

## Installation

```powershell
uv sync
```

## Usage

```powershell
# Run tests
uv run pytest -v

# Full pipeline (3 models × 5000 episodes)
uv run python -m src.main
```

### Expected outputs

| Artifact | Path |
|---|---|
| PPO checkpoint | `models/ppo.pt` |
| SAC checkpoint | `models/sac.pt` |
| TD3 checkpoint | `models/td3.pt` |
| PPO val history | `results/reports/rl_ppo_val_history.json` |
| SAC val history | `results/reports/rl_sac_val_history.json` |
| Test metrics | `results/reports/rl_*_test_metrics.json` |
| Aligned prices | `data/processed/aligned_prices_15_coins.csv` |

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Sharpe** | Mean daily return / std × √365 |
| **Total Return** | Final portfolio / initial - 1 |
| **Max Drawdown** | Worst peak-to-trough decline |
| **Win Rate** | Fraction of days with positive return |
| **Turnover** | Sum of absolute weight changes across all rebalances |
| **Allocation Entropy** | Normalized Shannon entropy of weight distribution |
| **Positive Sharpe** | Fraction of episodes with Sharpe > 0 |

### Benchmarks

| Benchmark | Description |
|---|---|
| **BTC hold** | Buy and hold Bitcoin |
| **Equal-weight** | Equal weight all 15 coins, rebalanced daily |
| **Top momentum** | All-in the coin with highest 21-day return |
| **Risk parity** | Weight inversely proportional to volatility |

## License

MIT License.
