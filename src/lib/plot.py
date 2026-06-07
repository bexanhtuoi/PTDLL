from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def decorate_axes(ax, title=None, xlabel=None, ylabel=None, grid=True):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)


def line(x, y, save_path: Path, title=None, xlabel=None, ylabel=None,
         figsize=(10, 6), dpi=150, grid=True, color=None, linestyle="-",
         linewidth=2, alpha=0.8):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    decorate_axes(ax, title, xlabel, ylabel, grid)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def multi_line(data: dict, save_path: Path, title=None, xlabel=None, ylabel=None,
               figsize=(10, 6), dpi=150, grid=True):
    fig, ax = plt.subplots(figsize=figsize)
    for label, val in data.items():
        if isinstance(val, tuple):
            x, y = val
        else:
            x, y = range(len(val)), val
        ax.plot(x, y, label=label, linewidth=2, alpha=0.8)
    ax.legend()
    decorate_axes(ax, title, xlabel, ylabel, grid)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def equity_curve(pv: np.ndarray, save_path: Path, benchmark: np.ndarray | None = None,
                 title="Equity Curve", xlabel="Step", ylabel="Portfolio Value",
                 figsize=(12, 6), dpi=150, grid=True):
    data = {"Agent": pv}
    if benchmark is not None:
        data["Benchmark"] = benchmark
    multi_line(data, save_path, title=title, xlabel=xlabel, ylabel=ylabel,
               figsize=figsize, dpi=dpi, grid=grid)


def bar(categories, values, save_path: Path, title=None, xlabel=None, ylabel=None,
        figsize=(10, 6), dpi=150, grid=True, color="#2E86AB"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(categories, values, color=color)
    decorate_axes(ax, title, xlabel, ylabel, grid)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def grouped_bar(groups: dict, save_path: Path, title=None, xlabel=None, ylabel=None,
                figsize=(12, 8), dpi=150, grid=True):
    fig, ax = plt.subplots(figsize=figsize)
    n = len(next(iter(groups.values())))
    x = np.arange(n)
    width = 0.8 / len(groups)
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
    for i, (label, vals) in enumerate(groups.items()):
        offset = (i - len(groups) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=colors[i % len(colors)])
    ax.legend(fontsize=7)
    decorate_axes(ax, title, xlabel, ylabel, grid)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def pie(values, labels, save_path: Path, title=None,
        figsize=(8, 8), dpi=150):
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(values, labels=labels, autopct="%1.1f%%")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def histogram(data, save_path: Path, title=None, xlabel=None,
              bins=30, figsize=(10, 5), dpi=150, grid=True, alpha=0.5):
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(data, dict):
        for label, vals in data.items():
            ax.hist(vals, bins=bins, alpha=alpha, label=label)
        ax.legend()
    else:
        ax.hist(data, bins=bins, alpha=alpha)
    decorate_axes(ax, title, xlabel, grid=grid)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def heatmap(matrix: np.ndarray, save_path: Path, xticklabels=None, yticklabels=None,
            title=None, figsize=(10, 6), dpi=150, cmap="YlOrRd"):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    if title:
        ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("Value")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()
