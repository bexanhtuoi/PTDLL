from __future__ import annotations

import numpy as np
import torch

from models.base import BaseModel


def predict_weights(model: BaseModel, state: np.ndarray) -> np.ndarray:
    return model.get_weights(state)


def predict_portfolio_returns(
    model: BaseModel,
    env,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    model.run_episode.__self__  # ensure it's a model instance
    state = env.reset(start_idx=start_idx, end_idx=end_idx)
    weights_history: list[np.ndarray] = []
    pv = [env.portfolio_value]
    done = False
    while not done:
        weights = predict_weights(model, state)
        weights_history.append(weights)
        with torch.no_grad():
            net = model.actor if hasattr(model, "actor") else model.policy
            logits = net(torch.from_numpy(state).unsqueeze(0).to(model.device))
            pred_weights = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        state, _, done, _ = env.step(pred_weights)
        pv.append(env.portfolio_value)
    return np.array(pv), np.array(weights_history)


def export_to_onnx(model: BaseModel, input_size: tuple[int, int, int], path: str) -> None:
    net = getattr(model, "actor", None) or getattr(model, "policy", None)
    if net is None:
        raise AttributeError("Model has no policy/actor network to export")
    net.eval()
    dummy = torch.randn(1, *input_size)
    torch.onnx.export(
        net, dummy, path,
        input_names=["state"], output_names=["weights"],
        opset_version=17, do_constant_folding=True,
    )
