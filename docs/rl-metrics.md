# RL Evaluation Metrics

> File này liệt kê tất cả chỉ số đánh giá RL agent trong PTDLL, kèm giải thích dễ hiểu.

---

## Agent metrics (9 chỉ số)

### 1. `total_return`

```python
pv[-1] / pv[0] - 1
```

- **Công thức**: Giá trị danh mục cuối kỳ chia đầu kỳ, trừ 1
- **Ví dụ**: `+502%` nghĩa là $100 đầu kỳ → $602 cuối kỳ
- **Lưu ý**: Mean 5000 episodes có thể bị kéo lên bởi outlier (bull run). Nhìn `positive_share` + `sharpe` quan trọng hơn.

### 2. `sharpe`

```python
mean(daily_returns) / std(daily_returns) * sqrt(365)
```

- **Lợi nhuận trên 1 đơn vị rủi ro** — chỉ số quan trọng nhất
- **Giải thích**: Sharpe 0.81 nghĩa là với mỗi 1% biến động, agent kiếm ~0.81% lợi nhuận
- **Thang đánh giá**: <0 (thua), 0.5 (tạm), 1.0 (tốt), 2.0+ (rất tốt — nghi ngờ overfit)
- **Annualized**: `× sqrt(365)` vì crypto trade 365 ngày/năm
- **Risk-free = 0**: không trừ lãi suất phi rủi ro (cho đơn giản)

### 3. `volatility`

```python
std(daily_returns)
```

- **Độ biến động daily của danh mục** — không annualized
- **Giải thích**: Mỗi ngày danh mục lên/xuống trung bình bao nhiêu %
- VD: vol 0.02 = trung bình ±2%/ngày

### 4. `max_drawdown`

```python
peak = max.accumulate(values)
min(values / peak - 1)
```

- **Lỗ nặng nhất từ đỉnh đến đáy**, trong suốt episode
- **Giải thích**: Ví dụ danh mục từ $100 → $150 → $90 → drawdown = 90/150 - 1 = **-40%**
- Đo lường rủi ro thua lỗ tệ nhất — càng gần 0 càng tốt

### 5. `win_rate`

```python
mean(daily_returns > 0)
```

- **Tỷ lệ ngày xanh** = số ngày danh mục dương chia tổng số ngày
- **Giải thích**: 60% nghĩa là 60% số ngày trong episode có lãi
- Không phải win rate của trade — trade là 1 action/episode, còn cái này tính trên daily returns

### 6. `turnover`

```python
sum(|w_new - w_old|)
```

- **Tổng % vốn bị xáo trộn** qua tất cả các lần rebalance
- **Giải thích**: Nếu turnover = 3.0, nghĩa là mỗi 3 năm agent đảo danh mục tương đương 300% vốn
- Càng thấp càng tốt (tiết kiệm phí giao dịch)

### 7. `n_trades`

```python
len(action_log)
```

- **Số lần agent thay đổi danh mục** = số steps trong episode
- Với 3 năm + 90d rebalance: thường ~8-9 lần
- Nếu <5 nghĩa là episode quá ngắn hoặc có lỗi

### 8. `allocation_entropy`

```python
-sum(w * log(w)) / log(n_assets)    # normalized [0, 1]
```

- **Đo độ đa dạng của danh mục**
  - `1.0` → chia đều 15 coin (giống equal weight)
  - `0.0` → dồn hết vốn vào 1 coin
- Tính trung bình qua tất cả các lần rebalance trong episode
- Entropy thấp + Sharpe cao = agent tự tin vào 1-2 coin (rủi ro tập trung)
- Entropy cao + Sharpe cao = agent đa dạng hóa tốt (lý tưởng)

### 9. `best_eval_sharpe`

```python
max(sharpe của tất cả eval episodes)
```

- **Sharpe cao nhất từng đạt được** trong 5000 eval episodes
- Dùng để biết tiềm năng — agent có thể đạt Sharpe 2.63 ở thời kỳ thị trường thuận lợi

---

## Walk‑forward extra (3 chỉ số tổng hợp)

### 10. `sharpe_std`

```python
std(5000 eval Sharpe)
```

- **Độ ổn định của Sharpe qua 5000 thời kỳ**
- VD: Sharpe 0.81 ± 0.61
- `±0.61` lớn → agent nhạy thị trường: có lúc rất tốt (Sharpe >2), có lúc tệ (Sharpe <0)
- Mong muốn: std càng thấp càng tốt (≤0.3 là ổn định)

### 11. `positive_share`

```python
mean(Sharpe > 0 trong 5000 episodes)
```

- **Tỷ lệ số episode có lãi**
- **88%** = trong 5000 lần test ở các thời kỳ khác nhau, 88% cho Sharpe dương
- Đây là chỉ số quan trọng nhất để đánh giá agent có kỹ năng thực sự hay chỉ may mắn

### 12. `return_std`

```python
std(5000 eval Returns)
```

- **Độ biến động của lợi nhuận kỳ vọng**
- VD: Return 502% ± 1295% → std rất lớn vì có episode lời +8400%, có episode lỗ -70%
- Nhìn `positive_share` + `sharpe` thay vì return để đánh giá chính xác hơn

---

## Benchmark metrics (20 chỉ số)

4 benchmarks × 5 chỉ số mỗi benchmark = 20 metrics.

### Các benchmark

| Tên | Cách tính | Ví dụ dễ hiểu |
|-----|-----------|---------------|
| **btc_hold** | Lấy daily return của BTC (cột 0 trong cube) | Mua $100 BTC đầu kỳ, cuối kỳ có bao nhiêu? |
| **equal_weight** | `mean(returns của 15 coin, axis=1)` mỗi ngày | Chia $100 đều 15 coin, rebalance daily |
| **top_momentum** | Chọn coin có return 21d cao nhất, hold coin đó | Nhìn coin nào tăng mạnh nhất 21 ngày, all-in coin đó |
| **risk_parity** | `w = 1/std(ret)`, chuẩn hóa tổng = 1, rebalance daily | Coin nào ít biến động → cho nhiều vốn, coin nào biến động mạnh → cho ít |

### 5 chỉ số mỗi benchmark

| Chỉ số | Công thức | Ý nghĩa |
|--------|-----------|---------|
| `{name}_return` | `cumprod(1+rets)[-1] - 1` | Benchmark lời/lỗ bao nhiêu % |
| `{name}_sharpe` | `mean/std × sqrt(365)` | Sharpe của benchmark |
| `{name}_volatility` | `std(rets)` | Độ biến động của benchmark |
| `{name}_max_drawdown` | `max_drawdown(cumprod(rets))` | Drawdown tệ nhất của benchmark |
| `{name}_relative_return` | `agent_return - bench_return` | Agent hơn/kém benchmark bao nhiêu % |

### Cách đọc relative_return

- `btc_hold_relative_return = +3.68` → agent lời hơn BTC hold trung bình 368%
- `equal_weight_relative_return = +3.71` → agent lời hơn equal weight 371%
- Giá trị dương = agent đánh bại benchmark

---

## File lưu

| File | Dung lượng | Nội dung |
|------|-----------|---------|
| `results/reports/rl_metrics.json` | ~32 fields | Agent (9) + walk-forward (3) + 4 benchmarks × 5 (20) = trung bình |
| `results/reports/rl_eval_history.json` | 5000 objects | Chi tiết từng eval episode — dùng để phân tích phân phối |

### Cấu trúc file rl_metrics.json (ví dụ)

```json
{
  "total_return": 5.025,
  "sharpe": 0.8119,
  "volatility": 0.021,
  "max_drawdown": -0.35,
  "turnover": 2.1,
  "n_trades": 8,
  "win_rate": 0.55,
  "allocation_entropy": 0.82,
  "btc_hold_return": 1.33,
  "btc_hold_sharpe": 0.89,
  "btc_hold_volatility": 0.035,
  "btc_hold_max_drawdown": -0.45,
  "btc_hold_relative_return": 3.68,
  "equal_weight_return": 1.31,
  "equal_weight_relative_return": 3.71,
  "risk_parity_relative_return": 4.79,
  "top_momentum_relative_return": 2.32,
  "sharpe_std": 0.61,
  "total_return_std": 12.95,
  "positive_share": 0.88,
  "best_eval_sharpe": 2.63
}
```

---

## Tổng kết

Khi nhìn vào kết quả RL, xem theo thứ tự:

1. **positive_share** — agent có kỹ năng không? (>60% là có)
2. **sharpe** ± **sharpe_std** — lời trên rủi ro bao nhiêu? Có ổn định?
3. **relative_return vs benchmarks** — có beat được chiến thuật đơn giản không?
4. Còn lại: max_drawdown (rủi ro), turnover (phí), allocation_entropy (đa dạng hóa)
