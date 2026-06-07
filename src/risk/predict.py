from __future__ import annotations

import numpy as np

from risk.base import BaseStopModel


def predict_stop(model: BaseStopModel, x_60d: np.ndarray, coin_idx: int) -> float:
    return float(model.predict(x_60d[np.newaxis, :, :], np.array([coin_idx]))[0])
