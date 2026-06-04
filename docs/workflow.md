# Luồng hoạt động

> **End-to-end:** OHLCV 15 coins → Feature cube `(T, 15, 14)` → RL allocate → Stop-loss cứng → Daily close check → Sell on hit → USDT → RL re-allocate

---

## 1. Luồng dữ liệu

```mermaid
graph TD
    A[OHLCV 15 coins] --> B[build_feature_table]
    B --> C[aligned_features → cube T×15×12]
    C --> D[filter_by_date train/val/test]
    D --> E[TRAIN: 2017→2024]
    D --> F[VAL: 2024→2025]
    D --> G[TEST: 2025→2026]

    E --> H[CryptoPortfolioEnv]
    F --> I[CryptoPortfolioEnv]
    G --> J[CryptoPortfolioEnv]

    H --> K[PPOAgent]
    H --> L[SACAgent]
    H --> M[TD3Agent]

    K --> N[Val every 50 eps]
    L --> N
    M --> N

    N --> O[Best Val Sharpe → save checkpoint]
    O --> P[run_test on TEST env]
    P --> Q[print_results comparison]

    Q --> R[Stop-Loss Layer]
    R --> S[ATR Baseline / XGBoost / Conv1D StopNet]
    S --> T[Predict stop_% per coin]
    T --> U[Set stop_price = close × (1 - stop_%)]
    U --> V[Daily: close ≤ stop_price?]
    V -->|Yes| W[Auto sell → USDT]
    V -->|No| X[Hold]
    W --> Y[USDT > 0 → RL re-allocate]
```

---

## 2. Episode lifecycle

```
reset(seed)
  ─ random 2-year window trong train set (730 days)
  ─ state = cube[t-lookback : t]  →  (60, 15, 14)
  ↓
step(action_weights)
  ─ clip(0,1) + normalize sum=1
  ─ hold 90 days (~3 months, quarterly rebalance)
  ─ portfolio_return = Σ(asset_returns × weights)
  ─ reward = excess_return - 0.1×vol - 0.05×dd_90 - 0.001×turnover
  ↓
... repeat ~8 steps → done=True
  ↓
evaluate vs 4 benchmarks:
  BTC hold, Equal-weight, Top momentum, Risk parity
```

---

## 3. Training pipeline

```python
# models/train.py:run()
def run():
    coins = load_all_coins(symbols=COINS_15)
    train_cube, _ = aligned_features(coins, "2017", "2024")
    val_cube,   _ = aligned_features(coins, "2024", "2025")
    test_cube,  _ = aligned_features(coins, "2025", "2026")

    train_env = CryptoPortfolioEnv(train_cube, ...)
    val_env   = CryptoPortfolioEnv(val_cube, ...)
    test_env  = CryptoPortfolioEnv(test_cube, ...)

    for name in ["ppo", "sac", "td3"]:
        agent = create_agent(name, cfg)
        agent = train_model(agent, train_env, val_env, n_episodes=5000)
        agent.save(f"models/{name}.pt")
        metrics = run_test(agent, test_env, n_test=50)
        save_json(metrics, f"results/reports/rl_{name}_test_metrics.json")

    print_results(all_metrics)
```

---

## 4. State cube (14 channels)

Features từ `env.py:aligned_features()` + `compute_coin_features()`:

| Index | Feature | Source |
|:-----:|---------|--------|
| 0 | return_1d | Daily close return |
| 1 | return_7d | 7-day rolling return |
| 2 | return_30d | 30-day rolling return |
| 3 | return_90d | 90-day rolling return |
| 4 | volatility | 20-day rolling std(return_1d) |
| 5 | drawdown | close / rolling_max(close) - 1 |
| 6 | volume_change | (vol - vol_prev) / vol_prev, clipped [-5, 5] |
| 7 | relative_strength_vs_BTC | return_30d[coin] - return_30d[BTC] |
| 8 | correlation_vs_BTC | rolling_60d corr(return, BTC_return) |
| 9 | btc_ma200_position | (BTC - SMA200) / SMA200 |
| 10 | market_volatility | Mean volatility across all coins |
| 11 | btc_momentum_90d | (BTC - BTC_90d_ago) / BTC_90d_ago |
| 12 | market_breadth | Fraction of coins with return_30d > 0 |
| 13 | weight | Injected by env (current allocation) |

---

## 5. Action space

**Dirichlet distribution** (PPO, SAC) hoặc **softmax** (TD3):

```
α = softmax(logits) × 10 + 1          # concentration params
action ~ Dirichlet(α)                  # sample continuous weights (PPO/SAC)
action = softmax(logits)                # deterministic (TD3)
Σ(action) = 1, 0 ≤ actionᵢ ≤ 1         # valid allocation
```

---

## 6. Network architecture

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

> StopNet reuses StateEncoder: Input (state, weights) → StateEncoder → concat(32, weights) → FC → Sigmoid → scale [0.05, 0.50]

---

## 7. Tests (11 tests)

| File | Tests |
|------|:-----:|
| `tests/test_rl.py` | 10 (env ×2, agent weights, 3 agents × train+run, filter) |
| `tests/test_indicators.py` | 1 (RSI bounds) |

All pass, ~7s.
