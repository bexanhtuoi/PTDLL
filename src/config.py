from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"
PREDICTIONS_DIR = ROOT / "results" / "predictions"
HISTORY_PATH = ROOT / "results" / "portfolio_history.json"
RISK_HISTORY_PATH = ROOT / "results" / "risk_history.json"

MODEL_TAGS: dict[str, str] = {"ppo": "PPO", "sac": "SAC", "td3": "TD3"}


class PipelineConfig(BaseSettings, frozen=True):
    train_start: str = "2017-01-01"
    train_end: str = "2024-06-01"
    val_start: str = "2024-06-01"
    val_end: str = "2025-06-01"
    test_start: str = "2025-06-01"
    test_end: str = "2026-06-01"

    episode_years: int = Field(default=4, ge=1, le=5)
    rebalance_days: int = Field(default=90, ge=21, le=365)
    lookback: int = Field(default=60, ge=10, le=120)
    fee_rate: float = Field(default=0.001, ge=0, le=0.01)
    random_state: int = Field(default=42, ge=0)

    n_episodes: int = Field(default=50000, ge=100, le=100000)
    val_interval: int = Field(default=50, ge=1, le=1000)
    val_n_episodes: int = Field(default=15, ge=1, le=50)
    lr: float = Field(default=3e-4, ge=1e-6, le=1e-2)
    gamma: float = Field(default=0.99, ge=0.9, le=0.999)
    entropy_coef: float = Field(default=0.05, ge=0, le=0.1)
    reward_style: str = Field(default="direct", pattern="^(direct|benchmark)$")

    checkpoint_interval: int = Field(default=0, ge=0, le=10000)
    early_stop_patience: int = Field(default=20, ge=0, le=200)

    model_config = {"frozen": True}
