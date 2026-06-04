from dataset.fetch import generate_synthetic_ohlcv, clean_ohlcv
from scores import build_feature_table


def test_rsi_bounds_and_features_have_no_nan():
    df = clean_ohlcv(generate_synthetic_ohlcv("BTCUSDT", periods=300))
    features = build_feature_table(df)
    assert features["rsi_14"].between(0, 100).all()
    check_cols = ["rsi_14", "return_7d", "return_14d", "return_30d", "volatility_14d", "relative_volume"]
    assert all(c in features.columns for c in check_cols)
    assert not features[check_cols].isna().any().any()
