from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from risk.base import BaseStopModel
from lib.metrics import hit_rate


def eval_model(model: BaseStopModel, loader: DataLoader) -> dict[str, float]:
    model.eval()
    all_preds: list[float] = []
    all_targets: list[float] = []
    with torch.no_grad():
        for x, idx, tgt in loader:
            pred = model(x, idx).squeeze(-1).numpy()
            all_preds.extend(pred.tolist())
            all_targets.extend(tgt.squeeze(-1).numpy().tolist())
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    mae_val = float(np.mean(np.abs(preds - targets)))
    hr = hit_rate(preds, targets)
    return {"mae": mae_val, "hit_rate": hr}


def compare(models: dict[str, BaseStopModel], loader: DataLoader) -> dict[str, dict[str, float]]:
    return {name: eval_model(model, loader) for name, model in models.items()}
