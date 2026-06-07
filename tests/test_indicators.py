import numpy as np
from dataset.fetch import generate_synthetic_ohlcv
from lib.features import create_features


def test_rsi_bounds_and_features_have_no_nan():
    d = generate_synthetic_ohlcv("BTCUSDT", periods=300)
    features = create_features(
        d["close"].astype(np.float64),
        d["high"].astype(np.float64),
        d["low"].astype(np.float64),
        d["open"].astype(np.float64),
        d["volume"].astype(np.float64),
    )
    assert features.shape[1] == 22
    assert np.all(np.isfinite(features))
