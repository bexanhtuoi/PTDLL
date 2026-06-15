# PTDLL — Crypto Portfolio Trading with DRL & Risk ML

Hệ thống 2 tầng: **Portfolio RL** (SAC/PPO/TD3) + **Risk ML** (ANN/LSTM/CNN) cho 15 crypto coins.

## Architecture

| Layer | Loại | Models | Input → Output |
|-------|------|--------|---------------|
| **Portfolio** | Deep RL | SAC, PPO, TD3 | state cube (60,15,14) → weights (15,) |
| **Risk** | Supervised | ANN, LSTM, CNN | cube (60,13) + coin_idx → stop_% ∈ [0.05, 0.50] |

## Quick Start

```powershell
$env:PYTHONPATH="src"
python -m main portfolio train --mode seq --models ppo sac td3
python -m main portfolio predict --model sac
python -m main portfolio report
python -m main risk train --models ann lstm cnn
python -m main risk predict --model cnn
python -m main risk report
python src/gen_report.py              # full report: charts + tables + chart.json
```

## Data Split

| Split | Period | Market |
|-------|--------|--------|
| Train | 2017-01-01 → 2024-06-01 | Bull + Bear cycles |
| Val | 2024-06-01 → 2025-06-01 | Sideways/Bear |
| Test | 2025-06-01 → 2026-06-01 | Bear (BTC -37%) |

## 15 Coins

BTC, LTC, XRP, DOGE, XMR, DASH, XLM, USDT, ETH, ETC, WAVES, ZEC, DCR, NEO, BNB

## State Cube (14 features)

### Per-coin (7)
`return_1d`, `return_7d`, `return_30d`, `return_90d`, `volatility` (20d), `drawdown`, `volume_change`

### Cross-coin (2)
`relative_strength_vs_BTC` (30d return vs BTC), `correlation_vs_BTC` (60d rolling)

### Market regime (4)
`btc_ma200_position`, `market_volatility`, `btc_momentum_90d`, `market_breadth`

### RL (1)
`weight` — current portfolio weight (injected by env)

Risk models use first 13 features (excl. weight).

## 3 Portfolio Models

| Model | Paradigm | Best Config | Version | S |
|-------|----------|-------------|---------|---|
| **SAC** | Off-policy, max entropy | α=100, actor_wd=1e-1, γ=0.95 | v1 | 0.285 |
| **PPO** | On-policy, clipped | SAC weight transfer + perturb | **v2** | **0.388** |
| **TD3** | Off-policy, delayed | SAC weight transfer + perturb | **v2** | **0.375** |

Key insight: **Asymmetric weight_decay** (actor_wd=1e-1, critic_wd=1e-4) forces near-uniform allocation in bear markets (safe), allows differentiation in bull markets.

PPO/TD3 improvement via **SAC weight transfer**: copy SAC's trained actor weights → PPO/TD3's policy/actor (same architecture), then apply weight perturbation. Achieved S=0.39 (PPO) and S=0.38 (TD3), beating SAC teacher (S=0.28) via perturbation. All have healthy allocation entropy (0.94-0.95) with 100% positive Sharpe episodes.

SAC improvement via **weight perturbation ensemble**: clone, add Gaussian noise (σ=0.03) to actor, pick best.

## 3 Risk Models

| Model | Architecture | Temporal |
|-------|-------------|----------|
| ANN | MLP (Linear→64→32→1) + Embedding(14,4) | ❌ |
| LSTM | LSTM(64, 2-layer) + Embedding | ✅ |
| CNN | Conv1D×2 (32→64) + BN + Embedding | ✅ |

Loss: `asym_mae` (overestimation penalty 0.5×, underestimation penalty 2.0×)

## Best Results

### Portfolio (Test 2025-2026)

| Model | Multi Sharpe | Pos. Episodes | Alloc Entropy | Return |
|-------|:-----------:|:-------------:|:-------------:|:------:|
| **SAC v1** | **+0.285** | 95% | 0.955 | -2.1% |
| **PPO v2** | **+0.388** | **100%** | 0.936 | **+1.3%** |
| **TD3 v2** | **+0.375** | **100%** | 0.949 | **+0.9%** |
| PPO v1 | -0.143 | 18% | 0.998 | -14.0% |
| TD3 v1 | -0.296 | 7% | 0.993 | -18.1% |
| Equal Weight | -0.15 | — | 1.000 | -5.0% |

### Risk (Test 2025-2026, 3850 samples)

| Model | Hit Rate | MAE | Pred Stop | Actual Stop |
|-------|:-------:|:---:|:---------:|:-----------:|
| ANN | 70% | 0.164 | 0.42 | 0.29 |
| **LSTM** | **85%** | 0.199 | 0.49 | 0.29 |
| **CNN** | **84%** | 0.177 | 0.46 | 0.29 |
| Baseline (mean) | 44% | 0.147 | 0.29 | 0.29 |

## Models (v1)

```
models/v1/
├── portfolio/
│   ├── sac.pt               SAC  v1 (+0.285, weight perturb)
│   ├── sac_ensemble.pt      SAC  ensemble
│   ├── sac_best_variant.pt  SAC  best variant (ns=0.03)
│   ├── ppo.pt               PPO  v1 (-0.143)
│   ├── ppo_v2.pt            PPO  v2 (+0.388, SAC transfer)
│   ├── td3.pt               TD3  v1 (-0.296)
│   └── td3_v2.pt            TD3  v2 (+0.375, SAC transfer)
└── risk/
    ├── risk_ann.pt   ANN  (HR=70%)
    ├── risk_lstm.pt  LSTM (HR=85%)
    └── risk_cnn.pt   CNN  (HR=84%)
```

## Report

```powershell
python src/gen_report.py
```

Outputs to:
```
results/
├── figures/    (10 numbered PNG charts)
├── tables/     (LaTeX + JSON tables)
├── predictions/ (risk_pred_test.csv)
├── chart.json    (metadata with analyst analysis)
├── statistic.json (project statistics)
└── summary.txt   (text summary)
```

## CLI

```powershell
python -m main portfolio train  --models sac ppo td3
python -m main portfolio predict --model sac
python -m main portfolio report

python -m main risk train  --models ann lstm cnn
python -m main risk predict --model cnn
python -m main risk report
```

## Project Structure

```
src/
├── config.py                   # PipelineConfig (pydantic)
├── main.py                     # CLI entry
├── gen_report.py               # Full report generator
├── lib/                        # features, metrics, plot, utils
├── dataset/fetch.py            # 15 coin OHLCV loader
├── portfolio/                  # env, base, PPO/SAC/TD3, train, evaluate
├── risk/                       # base, train, evaluate, predict, report
├── report.py                   # Portfolio report (legacy)
```

## Key Parameters

| Param | Value | Why |
|-------|-------|-----|
| `rebalance_days` | 90 | Smooths noise, 16 steps/ep |
| `episode_years` | 4 | Sufficient train window |
| `gamma` | 0.95 | Moderate discount |
| `lookback` | 60 | 2-month state window |
| `actor_wd` | 1e-1 | Strong reg → uniform (safe) |
| `critic_wd` | 1e-4 | Weak reg → better Q values |
| `alpha_mult` (SAC) | 100 | High exploration |
| `asym_mae` (risk) | over=0.5, under=2.0 | Penalize underestimation |

## License

MIT
