from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_equity_curve(bot: pd.DataFrame, benchmark: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(bot["timestamp"]), bot["equity"], label="Agent")
    plt.plot(pd.to_datetime(benchmark["timestamp"]), benchmark["equity"], label="Buy & Hold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_allocation_heatmap(weights_history: list[np.ndarray], asset_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = np.array(weights_history)
    plt.figure(figsize=(10, max(4, len(asset_names) * 0.4)))
    plt.imshow(w.T, aspect="auto", cmap="YlGn")
    plt.yticks(range(len(asset_names)), asset_names)
    plt.xlabel("Rebalance step")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
