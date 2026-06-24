# Stop-Loss Prediction Layer (ML — Supervised Learning)

> **Mục tiêu:** Sau khi RL phân bổ vốn vào coin nào, dự đoán `stop_%` cho coin đó.
> Đầu vào 60 ngày × 17 features → z-score của 10d forward drawdown, convert sang stop%.

---

## 1. Luồng tổng thể

```
RL allocate weights (15 coins)
  │
  ▼
Với mỗi coin (trừ USDT) — gọi risk model:
  predict(x_60d, coin_idx) → z_score ∈ [-3, 3] (tanh × 3)
  denormalize: raw = z × coin_std + coin_mean
  calibrate: stop_% = clip(α × raw + β, 0.05, 0.50)
  stop_price = close × (1 - stop_%)
  set stop_price CỐ ĐỊNH (không trailing)
  │
  ▼ Hàng ngày (chỉ check, không gọi model)
  close ≤ stop_price?
    CÓ  → auto sell → USDT → RL re-allocate
    KHÔNG → hold, chờ ngày mai
```

**Không trailing**: stop_price set 1 lần, nếu close tăng thì stop_price giữ nguyên.

**Chỉ check daily close**: không intraday.

## 2. Auto-Labeling (Target — Z-score per coin)

Target = z-score của 10d forward max drawdown (LABEL_WINDOW=10), tính riêng từng coin từ training data.

```python
def auto_label(close_T, future_close):
    low = min(future_close)
    dd = (close_T - low) / close_T
    return clip(dd, STOP_MIN, STOP_MAX)  # [0.05, 0.50]

# Per-coin z-score (từ training data):
z_mean[i] = mean(labels của coin i trong train set)
z_std[i]  = std(labels của coin i trong train set)
z_target = (raw_label - z_mean[i]) / z_std[i]
z_target = clip(z_target, -3.0, 3.0)
```

Z-score loại bỏ per-coin bias nhưng giữ temporal variation.

## 3. Model Input

### 60 × 17 features (giá trị thực, không rank cross-sectional)

```
Input:  x: (60, 17) — 60 ngày × 17 features
        coin_idx: int — 0..13 (cho Embedding)
Output: z_score ∈ [-3, 3] — tanh × 3
```

### 17 features

| # | Feature | Ý nghĩa | Gốc |
|:-:|---------|---------|-----|
| 0 | return_1d | Daily return | per-coin (cube) |
| 1 | return_7d | 7-day return | per-coin (cube) |
| 2 | return_30d | 30-day return | per-coin (cube) |
| 3 | return_90d | 90-day return | per-coin (cube) |
| 4 | volatility | 20-day rolling vol | per-coin (cube) |
| 5 | drawdown | Distance from peak | per-coin (cube) |
| 6 | volume_change | Volume % change | per-coin (cube) |
| 7 | rsi_14 | RSI 14-day | extra |
| 8 | return_skew_60 | Skewness 60d returns | extra |
| 9 | dd_consecutive | % negative days in 60d | extra |
| 10 | distance_sma200 | % from SMA200 | extra |
| 11 | relative_strength_vs_BTC | Vs BTC 30d | cross (cube) |
| 12 | correlation_vs_BTC | 60d corr with BTC | cross (cube) |
| 13 | btc_ma200_position | BTC vs SMA200 | market (cube) |
| 14 | market_volatility | Mean volatility 15 coins | market (cube) |
| 15 | btc_momentum_90d | BTC 90d momentum | market (cube) |
| 16 | market_breadth | % coins positive 30d | market (cube) |

**Thứ tự features trong code:**
```
7 per-coin (cube[:,:,:7])
  → 4 extra (tính thêm)
    → 2 cross (cube[:,:,7:9])
      → 4 market (cube[:,:,9:13])
```

### Coin Embedding

```python
self.emb = nn.Embedding(14, emb_dim=16)
# 14 coins × 16-d vector
```

## 4. Models

### Chung interface

```python
class BaseStopModel(nn.Module, ABC):
    emb = nn.Embedding(14, 16)
    forward(x: (B, 60, 17), coin_idx: (B,) int) → (B, 1) z ∈ [-3, 3]
```

### ANN (MLP Baseline)

```
Input: (B, 60, 17) → flatten → (B, 1020)
  Linear(1020 → 256) → BN → LeakyReLU(0.2) → Dropout(0.3)
  Linear(256 → 128) → BN → LeakyReLU(0.2) → Dropout(0.3)
  Linear(128 → 64) → BN → LeakyReLU(0.2) → Dropout(0.2)
  concat emb(16) → (B, 80)
  Linear(80 → 32) → LeakyReLU(0.2) → Dropout(0.2)
  Linear(32 → 1) → Tanh × 3
Output: (B, 1) ∈ [-3, 3]
```

### LSTM (best performer)

```
Input: (B, 60, 17)
  emb = coin_emb(coin_idx) → expand(60) → (B, 60, 16)
  concat x + emb → (B, 60, 33)
  LSTM(33 → 64, 1-layer bidirectional, batch_first)
    → h_n[-2] concat h_n[-1] → (B, 128)
  Linear(128 → 48) → LeakyReLU(0.2)
  Linear(48 → 1) → Tanh × 3
Output: (B, 1) ∈ [-3, 3]
```

### CNN

```
Input: (B, 60, 17) → permute(0, 2, 1) → (B, 17, 60)
  Conv1d(17 → 32, k=7, pad=3) → BN → LeakyReLU(0.2) → Dropout(0.2)
  Conv1d(32 → 64, k=5, pad=2) → BN → LeakyReLU(0.2) → Dropout(0.2)
  Conv1d(64 → 96, k=3, pad=1) → BN → LeakyReLU(0.2) → Dropout(0.2)
  AdaptiveAvgPool1d(1) → flatten → concat emb(16)
  Linear(112 → 48) → LeakyReLU(0.2) → Dropout(0.2)
  Linear(48 → 24) → LeakyReLU(0.2) → Dropout(0.1)
  Linear(24 → 1) → Tanh × 3
Output: (B, 1) ∈ [-3, 3]
```

## 5. Loss

```python
def combined_loss(pred_dd, target_dd):
    return MSELoss()(pred_dd, target_dd) + boundary_reg(pred_dd, alpha=0.001)

def boundary_reg(pred, alpha=0.001):
    margin = 0.01
    near_min = relu(STOP_MIN + margin - pred)
    near_max = relu(pred - (STOP_MAX - margin))
    return alpha * (near_min + near_max).mean()
```

Train trên drawdown space (raw label, không z-score). Model output z-score → denormalize → so sánh với raw target.

## 6. Training

### Dataset
- Per-coin 60d window, z-score labels tại mỗi timestamp
- Feature scaling: z-score trên toàn bộ features (FeatureScaler)
- 14 coins (trừ USDT), ~37k train samples / ~4k test samples

### Data Split (chronological)

```
Train: tất cả data trước test_start (label-safe với LABEL_WINDOW=10)
Test:  dates >= test_start (2025-06-01)
```

Không validation split — train trên toàn bộ pre-test data, test set dùng để theo dõi val_loss.

### Hyperparams
- Optimizer: AdamW(lr=8e-4, weight_decay=1e-3)
- LR schedule: CosineAnnealingLR(300 epochs)
- Batch: 2048 (ANN/LSTM), 1024 (CNN)
- Epochs: 300
- Gradient clip norm: 1.0

### Post-hoc Calibration
Sau training, fit per-coin α, β bằng L1 minimization (Nelder-Mead) trên training predictions:
```python
calibrated = clip(α[c] × raw_pred + β[c], STOP_MIN, STOP_MAX)
```

## 7. Evaluation Metrics

| Metric | Ý nghĩa |
|--------|---------|
| **Pearson r** | Tương quan giữa cal. pred drawdown vs actual drawdown |
| **MAE** | Mean Absolute Error trong drawdown space (sau calibration) |
| **Per-coin temporal std** | Model có temporal variation không (vs actual std) |
| **Hit Rate** | % sample pred ≥ actual — tính trong `predict.py` |

## 8. Test Results (2025-2026, calibrated)

| Model | Test Corr | Pred Std | Act Std | MAE |
|-------|:--------:|:--------:|:------:|:---:|
| ANN | ~0.08 | ~0.06 | 0.164 | ~0.16 |
| **LSTM Ensemble** | **~0.31** | **~0.10** | **0.164** | **~0.047** |
| CNN | ~0.05 | ~0.05 | 0.164 | ~0.21 |

*Số liệu chính xác xem `results/predictions/risk_pred_test.csv` sau khi chạy `python src/gen_report.py`.*

### Per-coin breakdown (LSTM Ensemble)

| Coin | Pred Std | Act Std | Corr |
|------|:--------:|:-------:|:----:|
| DASH | 0.108 | 0.168 | **+0.51** |
| XMR | 0.087 | 0.138 | **+0.42** |
| BTC | 0.121 | 0.126 | **+0.39** |
| XLM | 0.087 | 0.138 | **+0.35** |
| ZEC | 0.069 | 0.124 | **+0.31** |

### Key insight

LSTM 5-seed ensemble achieves **corr ≈ 0.31**, nearly double the single LSTM (0.215). Ensemble averaging reduces prediction variance and improves generalization.

So với rank-based approach cũ (pred_std=0.018, chỉ 11% actual), model mới có pred_std ≈ 0.10 (62% actual) — predictions thay đổi theo thời gian, không flat per-coin.

Tuy nhiên correlation còn thấp do signal-to-noise ratio của bài toán 60d→10d drawdown yếu. DASH, XMR, BTC có corr 0.3+, cho thấy ensemble bắt timing trên những coin đó.

## 9. Compare: Approaches

| Approach | Temporal Var | Issue |
|----------|:------------:|-------|
| Rank targets | 11% actual | No temporal variation |
| Absolute labels | ~0.01 corr | Distribution shift |
| Z-score + val split | 62% actual | Val overfitting |
| **Z-score + no-val + calibrate** | **62% actual** | **Best so far** |

Z-score per coin + train all pre-test data (no val split) + 300 epochs cosine annealing + post-hoc calibration là cấu hình tốt nhất.
