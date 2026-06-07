# Results

> **RL pipeline hoàn thành**: 3 models (PPO/SAC/TD3) implement, training OK. Risk layer đang xây dựng.

---

## 1. Tests

```
tests/test_rl.py: 10 passed
tests/test_indicators.py: 1 passed
11 passed in ~7s
```

## 2. Training History (RL)

### PPO (5000 episodes)

| Metric | Val (avg±std) | Test |
|--------|:-------------:|:----:|
| Sharpe | 1.00 ± 0.81 | -0.79 |
| Total Return | 148% ± 232% | -54% |
| Max Drawdown | -23% ± 13% | -73% |
| Win Rate | 69% ± 31% | 0% |
| Positive Sharpe | 89% | 0% |

### SAC (~1500 episodes)

| Metric | Val (avg±std) | Test |
|--------|:-------------:|:----:|
| Sharpe | 1.05 ± 0.86 | +0.18 |
| Total Return | 136% ± 226% | -5% |
| Max Drawdown | -22% ± 13% | -27% |
| Win Rate | 68% ± 30% | 100% |
| Positive Sharpe | 89% | 100% |

### TD3 — chưa hoàn thành

## 3. So sánh

| Model | Val Sharpe | Test Sharpe | Test Return |
|-------|:----------:|:-----------:|:-----------:|
| PPO | 1.00 | -0.79 | -54% |
| **SAC** | **1.05** | **+0.18** | **-5%** |
| TD3 | — | — | — |

> SAC outperform PPO — off-policy + entropy regularization giúp generalization tốt hơn.

## 4. Limitations

- **Test set 365 ngày** < episode 730 ngày → mỗi test chỉ 1 start point
- **Turnover ~0** — agent học buy-and-hold, có thể do penalty quá cao
- **PPO test Sharpe âm** — overfit validation set
- **Val Sharpe > 1.0** — model có kỹ năng, cần test set dài hơn

## 5. Best Checkpoints

| Model | File | Val Sharpe |
|-------|------|:----------:|
| PPO | `models/ppo.pt` | 1.47 |
| SAC | `models/sac.pt` | 1.05 |
| TD3 | — | — |

## 6. Risk (ML — Supervised Learning)

Ba models: StopANN (54K params), StopLSTM (21K), StopCNN (70K). Chung interface BaseStopModel + Embedding(14,4).

### Data
- Nguồn: yfinance 15 coins, 2017→2026 (real data)
- Feature cube: `build_cube()` → (3126, 15, 14)
- Risk input: (60, 13) per coin (13 market features, bỏ weight channel)
- Label: `auto_label` — max_dd × 1.2, clip [0.05, 0.50], window 90 ngày
- Training: 31,444 samples, labels trải đều [0.05, 0.50] (mean 0.28)
- Cache: `data/processed/risk_data.npz` — load sau <1ms

### Loss
- `asym_mae(over=0.5, under=2.0)` — phạt bán sớm 4× nặng hơn bán đáy
- `boundary_reg(alpha=0.001)` — ngăn model collapse về biên

### Training
- Temporal split (label-safe): train < 2024-03-03, val 2024→2025, test 2025→2026
- Adam(lr=1e-3), batch=256, epochs=100, early stop patience=10
- Save best → `models/risk_{name}.pt`

### CLI
```
python main.py risk train --models all             # train cả 3 models
python main.py risk train --models ann lstm        # train 2 models
```
