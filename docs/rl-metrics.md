# RL Evaluation Metrics

> File này liệt kê tất cả chỉ số đánh giá RL agent trong PTDLL.

---

## Agent metrics (9 chỉ số)

### 1. `total_return`

```python
pv[-1] / pv[0] - 1
```

Giá trị danh mục cuối kỳ chia đầu kỳ, trừ 1. VD: +502% = $100 → $602.

### 2. `sharpe`

```python
mean(daily_returns) / std(daily_returns) * sqrt(365)
```

Lợi nhuận trên 1 đơn vị rủi ro — chỉ số quan trọng nhất. Annualized ×√365.

- Sharpe < 0: thua
- Sharpe 0.5: tạm
- Sharpe 1.0: tốt
- Sharpe 2.0+: rất tốt (nghi ngờ overfit)

### 3. `volatility`

`std(daily_returns)` — độ biến động daily, không annualized.

### 4. `max_drawdown`

```python
peak = max.accumulate(values)
min(values / peak - 1)
```

Lỗ nặng nhất từ đỉnh đến đáy. VD: $100→$150→$90 → drawdown = -40%.

### 5. `win_rate`

`mean(daily_returns > 0)` — tỷ lệ ngày xanh.

### 6. `turnover`

`sum(|w_new - w_old|)` — tổng % vốn xáo trộn qua các lần rebalance.

### 7. `n_trades`

Số lần agent thay đổi danh mục. Với 3 năm + 90d rebalance: ~8-9 lần.

### 8. `allocation_entropy`

```python
-sum(w * log(w)) / log(n_assets)    # normalized [0, 1]
```

Đo độ đa dạng danh mục. 1.0 = chia đều, 0.0 = dồn 1 coin.

### 9. `best_eval_sharpe`

Sharpe cao nhất từng đạt được — biết tiềm năng của agent.

## Benchmark metrics (20 chỉ số)

4 benchmarks × 5 metrics:

| Benchmark | Cách tính |
|-----------|-----------|
| **btc_hold** | Buy & hold BTC |
| **equal_weight** | Chia đều 15 coin, rebalance daily |
| **top_momentum** | All-in coin có 21d return cao nhất |
| **risk_parity** | `w = 1/std(ret)`, chuẩn hóa sum=1 |

5 metrics per benchmark: `{name}_return`, `_sharpe`, `_volatility`, `_max_drawdown`, `_relative_return`.

`_relative_return` = agent_return - bench_return. Dương = agent beat benchmark.

## Walk-forward extra (3 chỉ số)

| Chỉ số | Ý nghĩa |
|--------|---------|
| `sharpe_std` | Std của 5000 eval Sharpe — độ ổn định |
| `positive_share` | Fraction of episodes với Sharpe > 0 |
| `return_std` | Std của lợi nhuận kỳ vọng |

## File lưu

| File | Nội dung |
|------|----------|
| `results/portfolio_history.json` | Agent (9) + 4 benchmarks × 5 = train/val/test |

## Thứ tự đọc kết quả

1. **positive_share** — agent có kỹ năng không? (>60% là có)
2. **sharpe** ± **sharpe_std** — lời trên rủi ro bao nhiêu? Ổn định?
3. **relative_return vs benchmarks** — beat được chiến thuật đơn giản không?
