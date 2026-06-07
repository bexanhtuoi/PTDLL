# Features

Tất cả features được tính trong `portfolio/base.py` (build_cube → coin_fx, cross_fx, mkt_regime, stack_cube). Không dùng pandas — thuần numpy.

## 1. Per-coin Features (7 features)

Tính từng coin riêng biệt trong hàm `coin_fx()`.

| # | Feature | Hàm | Ý nghĩa | Công thức |
|:-:|---------|-----|---------|-----------|
| 0 | `return_1d` | `pct_change(close, 1)` | Daily close return | `(close[t] - close[t-1]) / close[t-1]` |
| 1 | `return_7d` | `pct_change(close, 7)` | 7-day return | `(close[t] - close[t-7]) / close[t-7]` |
| 2 | `return_30d` | `pct_change(close, 30)` | 30-day return | `(close[t] - close[t-30]) / close[t-30]` |
| 3 | `return_90d` | `pct_change(close, 90)` | 90-day return (momentum) | `(close[t] - close[t-90]) / close[t-90]` |
| 4 | `volatility` | `rolling_std(ret_1d, 20)` | 20-day rolling vol | `std(ret_1d[19:])` |
| 5 | `drawdown` | `dd_series(close)` | Distance from peak | `close[t] / max(close[:t+1]) - 1` |
| 6 | `volume_change` | `pct_change(volume, 1)` | Daily volume change | `(vol[t] - vol[t-1]) / vol[t-1], clip [-5,5]` |

**Volume ratio** (`volume / rolling_mean(volume, 20)`) — dùng cho analysis nhưng không trong state cube.

## 2. Cross-coin Features (2 features)

Tính từ ma trận per-coin trong `cross_fx()`.

| # | Feature | Ý nghĩa | Công thức |
|:-:|---------|---------|-----------|
| 7 | `relative_strength_vs_BTC` | Coin outperform BTC? | `return_30d[coin] - return_30d[BTC]` |
| 8 | `correlation_vs_BTC` | Tương quan với BTC | `corr(return_1d[coin], return_1d[BTC], 60 ngày)` |

`correlation` dùng window 60 ngày, tính `np.corrcoef` manual (không pandas).

## 3. Market Regime Features (4 features)

Tính từ toàn bộ thị trường trong `mkt_regime()`.

| # | Feature | Ý nghĩa | Công thức |
|:-:|---------|---------|-----------|
| 9 | `btc_ma200_position` | BTC cách SMA200 bao xa | `(BTC - SMA200) / SMA200` |
| 10 | `market_volatility` | Vol trung bình toàn thị trường | `mean(volatility của 15 coins)` |
| 11 | `btc_momentum_90d` | BTC momentum 3 tháng | `(BTC[t] - BTC[t-90]) / BTC[t-90]` |
| 12 | `market_breadth` | Tỷ lệ coin xanh | `mean(return_30d > 0)` |

## 4. Weight Feature (chỉ RL, không risk)

| # | Feature | Ý nghĩa | Nguồn |
|:-:|---------|---------|-------|
| 13 | `weight` | Current allocation weight | Injected bởi env tại runtime |

Weight là channel cuối, **không được normalize** (z-score) cùng các feature khác. Giúp RL biết đang nắm bao nhiêu mỗi coin.

## 5. Risk Input (13 features)

Risk model chỉ dùng **13 features đầu** (không weight channel):
```
return_1d, return_7d, return_30d, return_90d,
volatility, drawdown, volume_change,
relative_strength_vs_BTC, correlation_vs_BTC,
btc_ma200_position, market_volatility, btc_momentum_90d,
market_breadth
```

Shape: `(60, 13)` — 60 ngày lookback, 1 coin. Coin index dùng Embedding riêng.

## 6. Volume helpers (extra, analysis)

Từ `lib/features.py:volume()`:
- `volume_change` — daily % change, clipped [-5, 5] (đã là feature 6)
- `volume_ratio` — volume / rolling_mean(volume, 20) — dùng phân tích

## 7. OHLCV derived features

Từ `lib/features.py:candle()` + `returns()` + `ma()` + `volatility()` + `rsi()` + `macd()`:
- `create_features()` tổng hợp tất cả, dùng cho phân tích offline
- **Không dùng** trong state cube — state cube chỉ lấy sub-set từ `coin_fx()`

## 8. 15 Coins

Từ `dataset/fetch.py:COINS_15`:
```
BTC, LTC, XRP, DOGE, XMR, DASH, XLM, USDT,
ETH, ETC, WAVES, ZEC, DCR, NEO, BNB
```

USDT là stablecoin → features ≈ 0. USDT không dùng cho stop-loss prediction.

## 9. Tổng hợp state cube

```
build_cube(coin_data) → (T, 15, 14), feature_names, asset_names, date_index

Luồng:
  per_coin = asset_cube(coin_data, shorts, grid, T)     # (T, 15, 7)
  rel/corr = cross_fx(per_coin, names, T, 15)            # cross-sectional
  regimes  = mkt_regime(per_coin, names, btc_close, T)   # market regime
  cube     = stack_cube(per_coin, rel, corr, regimes...)  # (T, 15, 14)

14 features: 7 per-coin + 2 cross + 4 market + 1 weight
```
