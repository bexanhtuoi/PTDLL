from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import PipelineConfig, MODEL_DIR, FIGURES_DIR, TABLES_DIR, RISK_HISTORY_PATH
from lib.utils import ensure_dirs, save_json, load_json

from risk.base import BaseStopModel
from risk.predict import predict_all, make_risk_agent
from risk.train import MODEL_NAMES


def gen_risk_report() -> None:
    ensure_dirs(FIGURES_DIR, TABLES_DIR)
    cfg = PipelineConfig()

    print("Generating risk report...")

    results = {}
    for name in ["ann", "lstm", "cnn"]:
        model = make_risk_agent(name)
        if model is None:
            continue
        print(f"  Evaluating {name}...")
        out = predict_all(model, cfg, split="test", save_csv=True)
        df = out["df"]
        hit_rate = float(df["hit"].mean())
        avg_pred = float(df["pred_stop"].mean())
        avg_actual = float(df["actual_stop"].mean())
        mae = float(np.abs(df["pred_stop"] - df["actual_stop"]).mean())

        results[name] = {
            "hit_rate": hit_rate,
            "avg_pred_stop": avg_pred,
            "avg_actual_stop": avg_actual,
            "mae": mae,
            "n_samples": len(df),
        }
        print(f"    HR={hit_rate:.4f} pred={avg_pred:.4f} actual={avg_actual:.4f} MAE={mae:.4f}")

    # Summary table
    if results:
        table_path = TABLES_DIR / "risk_metrics_comparison.json"
        save_json(results, table_path)
        print(f"  Saved metrics to {table_path}")

        # Simple text summary
        tex_lines = [
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            "Model & HR & Pred Stop & Actual Stop & MAE \\\\",
            r"\midrule",
        ]
        for name, r in results.items():
            tex_lines.append(
                f"{name.upper()} & {r['hit_rate']:.3f} & {r['avg_pred_stop']:.3f} "
                f"& {r['avg_actual_stop']:.3f} & {r['mae']:.3f} \\\\"
            )
        tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
        tex_path = TABLES_DIR / "risk_metrics_comparison.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
        print(f"  Saved table to {tex_path}")

    print("Risk report done.")
