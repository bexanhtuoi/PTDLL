# Luồng hoạt động

> **End-to-end:** OHLCV 15 coins → Feature cube → RL allocate → 3 risk models predict stop → Daily close check → Sell on hit → USDT → RL re-allocate

---

## 1. Luồng dữ liệu

```
OHLCV 15 coins (BTC, ETH, ..., USDT)
  │
  ▼
feature engineering: coin_fx(), cross_fx(), mkt_regime()
  │
  ▼
stack_cube() → (T, 15, 14) feature_names, asset_names, date_index
  │
  ▼
build_env() → CryptoPortfolioEnv(train/val/test)
  │
  ├── RL: state = env.get_state() → (60, 15, 14) bị normalize + weight channel
  └── Risk: lấy (60, 13) per coin (13 market features, no weight)
```

## 2. Training pipeline

### Portfolio (RL)

```
load_coin_arrays() → build_env(train) + build_env(val) + build_env(test)

For each model (PPO → SAC → TD3):
  make_agent(name, train_env, cfg)
  agent.fit(train_env, val_env)      # n_episodes=1000+
    └── each episode: 2 years, random start
    └── val every 50 eps: score on val_env
    └── early stop if no improvement for 20 val checks
  agent.save("models/{name}.pt")
  test = agent.score(test_env)
  save_history(name, history, test, history_path)
```

**Sequential**: train 3 model nối tiếp. **Parallel**: spawn process per model (multi-process, Windows safe).

### Risk (ML)

```
to_arrays() → cache tại data/processed/risk_data.npz (xs, idxs, targets, dates)

build_data(cfg) → temporal split (train/val/test)
  Train: dates < train_end - 90 ngày (label-safe)
  Val:   dates in [val_start, test_start)
  Test:  dates >= test_start

For each model (ANN → LSTM → CNN):
  model = StopANN/LSTM/CNN(emb=Embedding(14, 4))
  train(model, train_loader, val_loader)
  └── epochs=100, batch=256, Adam(lr=1e-3)
  └── loss: asym_mae(over=0.5, under=2.0) + boundary_reg(alpha=0.001)
  └── early stop patience=10
  └── save best model → models/risk_{name}.pt
```

## 3. Inference flow

```
main.py predict

  portfolio predict --model ppo
    load_agent("ppo") + build_env()
    agent.simulate(test_env) → weights history + PV
    → PREDICTIONS_DIR/{model}_pv.csv, weight_{coin}.csv

  risk predict --model cnn   (chưa implement CLI)
    load model + data
    For each coin: predict(x_60d, coin_idx) → stop_%
    → PREDICTIONS_DIR/risk_predictions.csv
```

## 4. Report flow

```
main.py report
  │
  ├── Portfolio report
  │   load agents for all 3 models
  │   sim_agent(agent, test_env) → PV
  │   eval_metrics → latex table + JSON
  │   plot: equity curves, training sharpes, vs BTC
  │   → FIGURES_DIR/*.png, TABLES_DIR/*.tex
  │
  └── Risk report
      load 3 risk models
      evaluate: MAE, hit_rate, false_positive, saved_drawdown
      → FIGURES_DIR/risk_*.png, TABLES_DIR/risk_*.tex
```

## 5. Actions space (RL)

**Dirichlet distribution** (PPO, SAC) hoặc **softmax** (TD3):

- `α = softmax(logits) × 10 + 1` — concentration params
- `action ~ Dirichlet(α)` — sample (PPO/SAC)
- `action = softmax(logits)` — deterministic (TD3)
- `Σ(action) = 1`, `0 ≤ actionᵢ ≤ 1`

## 6. Network architecture (RL)

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

## 7. Network architecture (Risk)

Cả 3 đều kế thừa `BaseStopModel(ABC)` + `Embedding(14, 4)` cho coin_id:

### ANN
```
Input: (60, 13) → flatten → Linear(780→64) → Dropout(0.3) → concat emb(4)
→ Linear(68→32) → Linear(32→1) → Sigmoid × 0.45 + 0.05 → stop_%
```

### LSTM
```
Input: (60, 13) → concat tile emb(60, 4) → LSTM(17→64) → h_n[-1]
→ Linear(64→1) → Sigmoid × 0.45 + 0.05 → stop_%
```

### CNN
```
Input: (60, 13) → Conv1d(13→32, k3) → BN → ReLU → Pool
→ Conv1d(32→64, k3) → BN → ReLU → Pool → flatten → concat emb(4)
→ Linear(...→64) → Linear(64→1) → Sigmoid × 0.45 + 0.05 → stop_%
```

## 8. Tests (11 tests)

| File | Tests | Nội dung |
|------|:-----:|----------|
| `tests/test_rl.py` | 10 | env reset/step, agent weights, PPO/SAC/TD3 train+run, filter |
| `tests/test_indicators.py` | 1 | RSI bounds |

All pass, ~7s.
