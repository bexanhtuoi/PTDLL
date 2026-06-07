# Stop-Loss Prediction Layer (ML — Supervised Learning)

> **Mục tiêu:** Sau khi RL phân bổ vốn vào coin nào, dự đoán `stop_%` cho coin đó. Bài toán supervised learning đầu vào 60 ngày close → `stop_%` trong 90 ngày tới.

---

## 1. Luồng tổng thể

```
RL allocate weights (15 coins)
  │
  ▼
Với mỗi coin (trừ USDT) — gọi risk model:
  predict(x_60d, coin_idx) → stop_% ∈ [0.05, 0.50]
  stop_price = close × (1 - stop_%)
  set stop_price CỐ ĐỊNH (không trailing, không thay đổi)
  │
  ▼ Hàng ngày (chỉ check, không gọi model)
  close ≤ stop_price?
    CÓ  → auto sell → USDT → RL re-allocate
    KHÔNG → hold, chờ ngày mai
```

**Không trailing**: stop_price set 1 lần, nếu close tăng thì stop_price giữ nguyên. Nếu coin sau đó đỏ mạnh thì stop vẫn còn.

**Chỉ check daily close**: không intraday, dùng daily close price.

## 2. Auto-Labeling (Target)

Target được sinh tự động từ historical data — không cần label thủ công.

```python
def auto_label(close_T: float, future_90d: np.ndarray) -> float:
    # Drawdown thực tế trong 90 ngày tới
    max_dd = (close_T - min(future_90d)) / close_T

    # Buffer 20% để không bán đáy
    target = max_dd * 1.2
    return clip(target, 0.05, 0.50)
```

**Ý nghĩa**: Nếu coin thực sự đã đáy ở mức drawdown 20% trong 90 ngày tới, thì stop-loss nên đặt ở 24% (20% × 1.2) — đủ xa đáy 4% để tránh false positive.

**Tại sao buffer 20%?** Vì stop là cứng, không trailing. Nếu đặt đúng đáy thì chỉ cần 1 biến động nhỏ là bị quét.

## 3. Model Input

### Per-coin, độc lập portfolio

```
Input:  x: (60, 13) — 60 ngày × 13 market features
        coin_idx: int — 0..13 (index của coin, cho Embedding)
Output: stop_% ∈ [0.05, 0.50] — scalar
```

### 13 features (không weight channel)

| # | Feature | Ý nghĩa |
|:-:|---------|---------|
| 0 | return_1d | Daily return |
| 1 | return_7d | 7-day return |
| 2 | return_30d | 30-day return |
| 3 | return_90d | 90-day return |
| 4 | volatility | 20-day rolling vol |
| 5 | drawdown | Distance from peak |
| 6 | volume_change | Volume % change |
| 7 | relative_strength_vs_BTC | Vs BTC 30d |
| 8 | correlation_vs_BTC | 60d corr with BTC |
| 9 | btc_ma200_position | BTC vs SMA200 |
| 10 | market_volatility | Market avg vol |
| 11 | btc_momentum_90d | BTC 90d momentum |
| 12 | market_breadth | % coins positive |

### Coin Embedding

```python
self.coin_emb = nn.Embedding(14, emb_dim=4)
# 14 coins (trừ USDT) × 4-d vector học được
# BTC gần ETH (cùng blue chip), DOGE xa DCR (khác họ)
```

## 4. 3 Models

### Chung interface

```python
class BaseStopModel(nn.Module, ABC):
    emb = nn.Embedding(14, 4)
    forward(x: (B, 60, 13), coin_idx: (B,) int) → (B, 1) stop_%
```

### ANN (MLP Baseline)

```
Input: (B, 60, 13)
  flatten(1) → (B, 780)
  Linear(780 → 64) → ReLU → Dropout(0.3)
  concat emb(4) → (B, 68)
  Linear(68 → 32) → ReLU
  Linear(32 → 1) → Sigmoid × 0.45 + 0.05
Output: (B, 1)
```

Đơn giản nhất, ignore temporal structure. Baseline để so LSTM/CNN.

### LSTM

```
Input: (B, 60, 13)
  emb = coin_emb(coin_idx) → (B, 4)
  emb = unsqueeze(1) → expand(60) → (B, 60, 4)
  concat x + emb → (B, 60, 17)
  LSTM(17 → 64, batch_first) → h_n[-1] → (B, 64)
  Linear(64 → 1) → Sigmoid × 0.45 + 0.05
Output: (B, 1)
```

Học temporal pattern — momentum, drawdown tuần tự. Per-coin forward riêng (shared weights).

### StopCNN

```
Input: (B, 60, 13)
  permute(0, 2, 1) → (B, 13, 60)
  Conv1d(13 → 32, k=3) → BatchNorm → ReLU → MaxPool(2)
  Conv1d(32 → 64, k=3) → BatchNorm → ReLU → MaxPool(2)
  flatten(1) → concat emb(4) → (B, ...)
  Linear(... → 64) → ReLU
  Linear(64 → 1) → Sigmoid × 0.45 + 0.05
Output: (B, 1)
```

Temporal feature extraction bằng Conv1D stack + BatchNorm cho ổn định.

## 5. Loss Functions

### Asymmetric MAE

```python
def asym_mae(pred, target, over_w=0.5, under_w=2.0):
    diff = pred - target
    w = torch.where(diff > 0, over_w, under_w)
    # diff > 0: pred > target (stop rộng → an toàn → phạt nhẹ)
    # diff < 0: pred < target (stop hẹp → bán sớm → phạt nặng)
    return (w * abs(diff)).mean()
```

- **Dự đoán thấp hơn target** (stop quá hẹp) → phạt 2× — nguy hiểm, bán đáy
- **Dự đoán cao hơn target** (stop quá rộng) → phạt 0.5× — an toàn, chỉ lỡ lời nhiều hơn chút

### Boundary Regularization

```python
def boundary_reg(pred, alpha=0.001):
    margin = 0.01
    near_min = relu(STOP_MIN + margin - pred)
    near_max = relu(pred - (STOP_MAX - margin))
    return alpha * (near_min + near_max).mean()
```

Phạt nhẹ α=0.001 khi output cách biên [0.05, 0.50] dưới 1%, tránh model collapse về 1 phía. Tổng loss: `asym_mae(pred, tgt) + boundary_reg(pred)`.

## 6. Training

### Dataset
```python
# Gộp tất cả coins, temporal split trước
for coin_idx in range(14):      # 14 coins (trừ USDT)
    for t in range(60, T - 90):
        x = data[t-60:t, :13]   # (60, 13)
        target = auto_label(data[t-60, 0], data[t:t+90, 0])
        # (data[t-60, 0] = close at t-60, data[t:t+90, 0] = future close)

# Temporal split: train 2017-2023, val 2023-2024, test 2024+
# NGĂN LEAK: split TRƯỚC khi shuffle
```

### Data Split (temporal, label-safe)

```
Train:   dates < 2024-06-01 - 90 days  =  before 2024-03-03
Val:     2024-06-01 ≤ dates < 2025-06-01
Test:    2025-06-01 ≤ dates
```

### Hyperparams
- Optimizer: Adam(lr=1e-3)
- Batch: 256
- Epochs: 100 (early stop patience 10)
- Loss: `asym_mae(over=0.5, under=2.0) + boundary_reg(alpha=0.001)`
- Metric: MAE (chính), hit rate (phụ)

## 7. Evaluation Metrics

| Metric | Ý nghĩa | Công thức |
|--------|---------|-----------|
| **MAE** | Sai số tuyệt đối | `mean(|pred - target|)` |
| **Hit Rate** | Stop KHÔNG bị xuyên thủng | `mean(pred >= target)` |
| **False Positive** | Chạm stop rồi hồi >10% sau 30d | `mean(hit & rebound)` |
| **Saved Drawdown** | Giảm drawdown so với không stop | `dd_without - dd_with` |

## 8. So sánh models

| Model | Tham số | Temporal | Speed | Overfit risk |
|-------|---------|----------|-------|-------------|
| ANN | ~54K | ❌ batch | ⚡ nhanh | 🔴 cao (flatten mất structure) |
| LSTM | ~56K | ✅ sequence | 🟡 trung bình | 🟡 vừa |
| CNN | ~70K | ✅ conv | 🟡 trung bình | 🟢 thấp (BatchNorm) |
