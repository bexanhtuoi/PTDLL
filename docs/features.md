# Đặc tả tính năng

## ✅ Đã hoàn thành — Data Pipeline

### F-DATA — OHLCV Loader
- [x] Load 15 coins từ CSV (hoặc generate synthetic cho test)
- [x] `generate_synthetic_ohlcv`: random walk + trend + volume
- [x] `clean_ohlcv`: sort, drop duplicates, ép numeric
- [x] `aligned_prices`: align multi-coin close prices theo timestamp chung

### F-FEAT — Feature Engineering
- [x] `build_feature_table()`: RSI, MACD, rolling returns 1/7/30/90d, volatility, drawdown, volume_change, relative_volume, distance_to_ma, trend_regime
- [x] `feature_columns` list — 9 technical features
- [x] `aligned_features()`: multi-coin feature cube `(T, 15, 10)`
- [x] Per-coin features (7): return_1d/7d/30d/90d, volatility, drawdown, volume_change
- [x] Cross-sectional features (2): relative_strength_vs_BTC, correlation_vs_BTC
- [x] Weight injection (1): current allocation weights (env injects)
- [x] State normalization: z-score per channel (exclude weight channel)

---

## ✅ Đã hoàn thành — RL Portfolio

### F-BASE — BaseModel ABC + Shared Networks
- [x] `BaseModel` (ABC): `forward`, `get_weights`, `train_episode`, `run_episode`, `save`, `load`
- [x] `PolicyNet`: Conv1d(150→64, k=5) + FC(64→128→64→n_assets), Dirichlet policy
- [x] `ValueNet`: Conv1d(150→32, k=5) + FC(32→64→1)
- [x] `StateEncoder`: Conv1d(150→64, k=5) + ReLU + AdaptiveAvgPool1d → FC(64→128)
- [x] `TwinQNet`: StateEncoder × 2 output heads
- [x] `DeterministicActor`: StateEncoder + FC(128→64→n_assets) + softmax
- [x] `ReplayBuffer`: deque(maxlen=100_000), `sample(batch)` returns stacked tensors

### F-PPO — PPO Agent
- [x] `PPOAgent(BaseModel)`: on-policy, clipped surrogate objective
- [x] GAE(λ=0.95) advantage estimation
- [x] K=4 epochs per update, minibatch
- [x] Dirichlet policy (concentration = logits×10 + 1)
- [x] Policy loss = -min(ratio×A, clip(ratio, 1-ε, 1+ε)×A)
- [x] Value loss = smooth L1
- [x] Entropy bonus (coef=0.01)
- [x] Gradient clip norm = 1.0
- [x] Adam(lr=3e-4) for both policy and value nets
- [x] LR scheduler: StepLR ×0.5/2000eps

### F-SAC — SAC Agent
- [x] `SACAgent(BaseModel)`: off-policy, maximum entropy
- [x] Twin Q-net (min double Q to reduce overestimation)
- [x] Soft target update (τ=0.005)
- [x] Auto entropy tuning (learned log_alpha, target_entropy = -n_assets)
- [x] Dirichlet policy (rsample for reparameterization)
- [x] Replay buffer experience replay
- [x] Adam(lr=3e-4) × 3 optimizers (policy, Q1, Q2)

### F-TD3 — TD3 Agent
- [x] `TD3Agent(BaseModel)`: off-policy, deterministic actor
- [x] DeterministicActor + softmax allocation
- [x] Twin critics (clipped double Q-learning)
- [x] Delayed policy update (every 2 steps)
- [x] Target policy smoothing (Gaussian noise σ=0.1, clipped [-0.5, 0.5])
- [x] Soft target update
- [x] Replay buffer

### F-ENV — CryptoPortfolioEnv
- [x] `CryptoPortfolioEnv`: gym.Env-like interface
- [x] Reset: random 2-year window → state cube `(lookback=60, 15, 10)`
- [x] Step: allocate weights, hold 90 days, compute portfolio return
- [x] Reward: `excess_return - 0.1×vol - 0.05×dd_90 - 0.001×turnover`
- [x] Rebalance fee: 0.1%
- [x] Weight normalization: `w = clip(w, 0, 1)`, then `w /= sum(w)`
- [x] 4 benchmarks: BTC hold, equal-weight, top momentum, risk parity
- [x] Episode evaluation: Sharpe, return, drawdown, turnover, win_rate

### F-TRAIN — Training Orchestrator
- [x] `create_agent(name, config)`: returns PPOAgent | SACAgent | TD3Agent
- [x] `train_model(agent, train_env, val_env, n_episodes)`: loop with concurrent validation
- [x] Validation every 50 episodes: run 1 val episode → record metrics
- [x] Track best Val Sharpe → save checkpoint
- [x] `run()` orchestrator: load coins → 3 envs → train 3 agents → test → publish

### F-EVAL — Evaluation
- [x] `run_test(agent, env, n_test)`: multi-episode test (avg ± std across episodes)
- [x] `print_results(results_dict)`: model comparison table
- [x] Test metrics: Sharpe, return, drawdown, turnover, positive Sharpe %

### F-PREDICT — Prediction / Export
- [x] `predict_weights(agent, env)`: run deterministic policy → final allocation vector
- [x] `predict_portfolio_returns(agent, env)`: cumulative returns over episode
- [x] `export_to_onnx(agent, n_assets, lookback, n_features, filepath)`: ONNX export

---

## ✅ Đã hoàn thành — USDT Integration

### F-USDT — Cash Decision
- [x] USDT (coin index 7) là 1 trong 15 assets
- [x] Features đều ≈ 0 (return=0, vol=0, dd=0)
- [x] Agent có thể allocate weight vào USDT = hold cash
- [x] **Không cần classification buy/sell riêng**

---

## ✅ Đã hoàn thành — Config & Tests

### F-CONFIG — PipelineConfig
- [x] Pydantic BaseSettings (`config.py`): train/val/test dates, episode params, RL hyperparams
- [x] All params overrideable via env vars

### F-TESTS — Unit Tests (11 tests)
- [x] `test_env_reset_shape`: env reset → state shape = (60, 15, 10)
- [x] `test_env_step`: env step → action_shape, reward_scalar, done_bool
- [x] `test_agent_weights`: agent.get_weights() returns valid distribution
- [x] `test_ppo_train_episode`: PPO train → loss ~0 no NaN
- [x] `test_ppo_run_episode`: PPO run → valid returns
- [x] `test_sac_train_episode`: SAC train → loss ~0 no NaN
- [x] `test_sac_run_episode`: SAC run → valid returns
- [x] `test_td3_train_episode`: TD3 train → loss ~0 no NaN
- [x] `test_td3_run_episode`: TD3 run → valid returns
- [x] `test_filter_by_date`: time split works correctly

---

## ✅ Đã hoàn thành — Deleted (Classification pipeline)

- [x] **Xoá** `models/rl.py` — REINFORCE PGAgent → thay bằng PPO/SAC/TD3
- [x] **Xoá** `pipeline.py` — pipeline orchestration → thay bằng `models/train.py:run()`
- [x] **Xoá** `dataset/labeling.py` — SMA crossover / threshold labels
- [x] **Xoá** `models/training.py` — classification train/eval loops
- [x] **Xoá** `models/evaluation.py` — classification metrics / confusion matrix
- [x] **Xoá** `models/export.py` — ONNX export cho sklearn/LSTM
- [x] **Xoá** `models/predict.py` — classification signal generation
- [x] **Xoá** `scripts/reports.py` — grid search
- [x] **Xoá** SVD pipeline, t-SNE khỏi `scores.py`

---

## 📋 Chưa làm / Optional

- [ ] Train TD3 đầy đủ 5000 episodes (đang chạy)
- [ ] Tăng n_episodes cho SAC để hội tụ tốt hơn
- [ ] Early stopping theo Val Sharpe
- [ ] Allocation heatmap visualization
- [ ] ONNX deployment script
- [ ] Streamlit dashboard
- [ ] Paper trading
- [ ] Live trading + risk guardrails

---

## 📋 Đang Planning — Stop-Loss Layer

> Chi tiết: [stop-loss.md](stop-loss.md)

### F-STOP — Auto-labeling
- [ ] Sinh target stop_loss_% từ historical drawdown + buffer 20%
- [ ] Train/val/test split đồng bộ với RL pipeline

### F-STOP — 3 Models
- [ ] ATR Baseline (rule-based, benchmark)
- [ ] XGBoost Regressor (feature importance)
- [ ] Conv1D StopNet (tái sử dụng StateEncoder)

### F-STOP — Integration
- [ ] Daily pipeline: RL → Stop → Execute
- [ ] Stop-loss cứng: set 1 lần, check daily close, không trailing
- [ ] Backtest so sánh with/without stop-loss
