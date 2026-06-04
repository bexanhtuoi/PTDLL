# Stop-Loss Prediction Layer

> **Mục tiêu:** Sau khi RL phân bổ vốn vào coin nào, dự đoán `stop_loss_%` cho coin đó và đặt **stop-loss cứng** tại mức giá đó. Nếu giá chạm → tự động bán → USDT tăng → RL phân bổ lại.

---

## Luồng tổng thể

```
Khởi tạo: 100$ USDT
   │
   ▼
┌─────────────────────────────────────┐
│  USDT > 0?                          │
└──────────┬──────────────────────────┘
           │ Có
           ▼
┌─────────────────────────────────────┐
│  RL phân bổ vốn                     │
│  weights = [BTC: 0.3, ETH: 0.2...] │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Set stop-loss cứng cho từng coin   │
│  (gọi model 1 lần dự đoán stop_%)   │
│                                     │
│  stop_price = close × (1 - stop_%)  │
│  CỐ ĐỊNH — không kéo lên            │
│  USDT: không set stop               │
└──────────┬──────────────────────────┘
           │
           ▼  Hàng ngày (chỉ check)
┌─────────────────────────────────────┐
│  close ≤ stop_price?                │
│                                     │
│  Nếu CÓ → auto sell → USDT          │
│  Nếu KHÔNG → hold, chờ ngày mai     │
└──────────┬──────────────────────────┘
           │
           └──→ USDT > 0? ──→ RL phân bổ lại
```

---

## Đặc điểm stop-loss cứng

| Tính chất | Mô tả |
|-----------|-------|
| **Cố định** | `stop_price` set 1 lần, không thay đổi đến khi bán hoặc RL re-allocate lại |
| **Không trailing** | Không kéo lên theo peak, không theo dõi peak |
| **Tự động** | Chạm là bán ngay, không cần can thiệp |
| **Chỉ check daily** | Dùng daily close, không intraday |

Sau khi RL re-allocate lại (vì có USDT từ coin bị stop), model sẽ dự đoán `stop_%` mới cho lần allocate đó — dựa trên market conditions mới nhất.

---

## 1. Auto-Labeling

Sinh target `stop_loss_%` từ dữ liệu lịch sử:

```python
Với mỗi ngày T, mỗi coin:
  close_T = giá hiện tại
  future_close = close[T : T + 60]
  low_future = min(future_close)
  max_dd = (close_T - low_future) / close_T

  # Target: drawdown × 1.2 — buffer 20% để không bán đáy
  target = max_dd * 1.2
  target = clip(target, 0.05, 0.50)
```

Vì stop là **cứng**, buffer 20% quan trọng hơn — không có trailing để kéo stop lên bù, nếu set sai là chết ngay.

---

## 2. Input & Output

### Input
```
Shape: (lookback=60, n_assets=15, n_features=14)
```

| Index | Feature |
|:-----:|---------|
| 0 | return_1d |
| 1 | return_7d |
| 2 | return_30d |
| 3 | return_90d |
| 4 | volatility (20d) |
| 5 | drawdown |
| 6 | volume_change |
| 7 | relative_strength_vs_BTC |
| 8 | correlation_vs_BTC |
| 9 | btc_ma200_position |
| 10 | market_volatility |
| 11 | btc_momentum_90d |
| 12 | market_breadth |

Kèm `weights` từ RL (để model biết coin nào nắm nhiều).

### Output
```
1 giá trị mỗi coin (trừ USDT): predicted_stop_loss_pct ∈ [0.05, 0.50]
```

| Coin | Ví dụ | stop_price (close=100) |
|------|:-----:|:---------------------:|
| BTC | 18% | 82$ |
| ETH | 25% | 75$ |
| DOGE | 40% | 60$ |
| USDT | 0% | Không set |

---

## 3. 3 Models So Sánh

| Model | Loại | Ưu | Nhược |
|-------|------|----|-------|
| **ATR Baseline** | Rule | 0 train, interpretable | Không học pattern |
| **XGBoost** | Tree | Feature importance, nhanh | Input flatten |
| **Conv1D** | Deep | Temporal, share arch RL | Cần tuning |

### ATR Baseline
```python
def predict_stop_atr(volatility_30d: np.ndarray) -> np.ndarray:
    multiplier = np.where(volatility_30d < 0.05, 3.5,
                np.where(volatility_30d < 0.10, 4.0,
                np.where(volatility_30d < 0.20, 4.5, 5.0)))
    return np.clip(volatility_30d * multiplier, 0.05, 0.50)
```

### XGBoost
- Input flatten: `(batch, 12600)`
- `n_estimators=500, max_depth=6, lr=0.05`

### Conv1D (StopNet)
```python
class StopNet(nn.Module):
    def __init__(self, lookback, n_assets, n_features):
        super().__init__()
        self.encoder = StateEncoder(lookback, n_assets, n_features, hidden=32)
        self.head = nn.Sequential(
            nn.Linear(32 + n_assets, 32), nn.ReLU(),
            nn.Linear(32, n_assets),
            nn.Sigmoid(),
        )

    def forward(self, state, weights):
        feat = self.encoder(state)
        x = torch.cat([feat, weights], dim=1)
        return self.head(x) * 0.45 + 0.05
```

---

## 4. Loss Function

```python
def stop_loss_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = predicted - target
    penalty = torch.where(diff > 0, diff * 0.5, -diff * 2.0)
    return penalty.mean()
```

---

## 5. Evaluation Metrics

| Metric | Ý nghĩa |
|--------|---------|
| **MAE** | Sai số tuyệt đối |
| **Hit Rate** | Stop không bị xuyên thủng |
| **False Positive** | Chạm stop rồi hồi >10% sau 30d |
| **Saved Drawdown** | Giảm drawdown so với không stop |

---

## 6. Implementation Order

```
Step 1: Auto-labeling script → sinh target stop_%
Step 2: ATR Baseline → benchmark
Step 3: XGBoost → train + so sánh
Step 4: Conv1D StopNet → train + so sánh
Step 5: Backtest stop-loss riêng (không RL)
Step 6: Tích hợp pipeline: RL → Stop → Check daily → USDT → RL loop
```

---

## 7. Success Criteria

1. **Hit Rate ≥ 80%**
2. **False Positive ≤ 20%**
3. **Saved Drawdown ≥ -10%** so với không stop
4. **Sharpe cải thiện** so với RL baseline
