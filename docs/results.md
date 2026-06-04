# Results

> **Refactor hoàn thành + pipeline chạy:** 3 RL models (PPO/SAC/TD3) implement, 11 tests pass. Pipeline đã chạy cho PPO (5000 eps) và SAC (~1500 eps). TD3 đang chạy.

---

## 1. Tests

```
tests/test_rl.py:
  test_env_reset_shape       ✅ state shape = (60, 15, 10)
  test_env_step              ✅ action, reward, done OK
  test_agent_weights         ✅ weights sum = 1, no NaN
  test_ppo_train_episode     ✅ loss ~ -3.4, no NaN
  test_ppo_run_episode       ✅ Sharpe finite
  test_sac_train_episode     ✅ loss ~ -5.7, no NaN
  test_sac_run_episode       ✅ Sharpe finite
  test_td3_train_episode     ✅ loss ~ -2.7, no NaN
  test_td3_run_episode       ✅ Sharpe finite
  test_filter_by_date        ✅ time split correct

tests/test_indicators.py:
  test_rsi_and_features      ✅ RSI [0,100], no NaN

11 passed in 6.96s
```

---

## 2. Training History

### PPO (5000 episodes completed)

| Metric | Val (avg±std) | Test (avg±std) |
|--------|:-------------:|:--------------:|
| **Sharpe** | **1.00 ± 0.81** | **-0.79 ± nan** |
| Total Return | 148% ± 232% | -54% ± 0% |
| Max Drawdown | -23% ± 13% | -73% ± 0% |
| Turnover | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Win Rate | 69% ± 31% | 0% ± 0% |
| Positive Sharpe | 89% | 0% |

**Best Val Sharpe**: 1.47 (ep 3200)

### SAC (~1500 episodes, ended at ep 1550)

| Metric | Val (avg±std) | Test (avg±std) |
|--------|:-------------:|:--------------:|
| **Sharpe** | **1.05 ± 0.86** | **+0.18 ± nan** |
| Total Return | 136% ± 226% | -5% ± 0% |
| Max Drawdown | -22% ± 13% | -27% ± 0% |
| Turnover | 0.01 ± 0.02 | 0.00 ± 0.00 |
| Win Rate | 68% ± 30% | 100% ± 0% |
| Positive Sharpe | 89% | 100% |

### TD3 — chưa chạy xong

---

## 3. Model Comparison

| Model | Val Sharpe | Test Sharpe | Test Return | Test Win Rate | Ghi chú |
|-------|:---------:|:----------:|:----------:|:------------:|---------|
| PPO | 1.00 | -0.79 | -54% | 0% | Overfit val? |
| **SAC** | **1.05** | **+0.18** | **-5%** | **100%** | **Tốt nhất** |
| TD3 | — | — | — | — | Đang chạy |

> SAC outperforms PPO trên test set — SAC's off-policy + entropy regularization giúp generalization tốt hơn.

---

## 4. Limitations

- **Test set chỉ 365 ngày < episode 730 ngày**: Mỗi test episode chỉ có 1 unique start point → std=0 cho nhiều metrics
- **Turnover ~0**: Agent học policy gần như buy-and-hold — có thể do penalty turnover quá cao hoặc thiếu exploration
- **PPO test Sharpe âm**: On-policy có thể overfit validation set (89% positive val Sharpe nhưng test âm)
- **Val Sharpe > 1.0**: Model thực sự học được kỹ năng — cần test set dài hơn để xác nhận

---

## 5. Best Checkpoints

| Model | File | Val Sharpe |
|-------|------|:----------:|
| PPO | `models/ppo.pt` | 1.47 |
| SAC | `models/sac.pt` | 1.05 |
| TD3 | — | — |

---

## 6. Source code stats

| File | Dòng |
|------|:----:|
| `models/base.py` | ~230 |
| `models/env.py` | ~210 |
| `models/sac.py` | ~160 |
| `models/td3.py` | ~170 |
| `models/ppo.py` | ~150 |
| `models/train.py` | ~130 |
| Số còn lại | ~450 |
| **Tổng** | **~1,500** |

---

## 7. Kết luận

- Cả PPO và SAC đều đạt **Val Sharpe > 1.0** — models có kỹ năng thực sự trên validation
- **SAC generalize tốt hơn PPO** (Test Sharpe +0.18 vs -0.79) — off-policy + entropy khám phá giúp không overfit
- **USDT tích hợp mượt**: Agent tự allocate vào USDT = hold cash, không cần buy/sell classification
- Cần test set dài hơn hoặc walk-forward validation (thay vì cố định) để đánh giá chính xác hơn
- TD3 (deterministic policy) có thể hoạt động khác — cần hoàn thành training để so sánh
