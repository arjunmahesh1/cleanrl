#!/usr/bin/env python3
"""Plot target-perturbation evaluation return across training checkpoints."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_checkpoint_step(row: pd.Series) -> float:
    if "checkpoint_step" in row and pd.notna(row["checkpoint_step"]):
        return float(row["checkpoint_step"])

    text = " ".join(
        str(row.get(key, ""))
        for key in ("model_path", "scenario_label", "run_name")
    )
    patterns = (
        r"(?:^|[_/-])step[_-]?(\d+)",
        r"(?:^|[_/-])checkpoint[_-]?(\d+)",
        r"global[_-]?step[_-]?(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return np.nan


def ci95(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return 0.0
    return 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True, help="CSV produced by evaluate_ppo_robust.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--y-metric", default="mean_return")
    parser.add_argument("--model-order", nargs="*", default=[])
    parser.add_argument("--display-label", action="append", default=[], help="Mapping like a2p95=cap 2.95")
    parser.add_argument("--scenario-label", default="", help="Optional exact scenario_label filter")
    parser.add_argument("--axis", default="", help="Optional xml/action axis filter")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    if args.scenario_label:
        df = df[df["scenario_label"] == args.scenario_label]
    if args.axis:
        axis_cols = [
            "axis",
            "xml_axis",
            "xml_perturb_axis",
            "xml_body_name_selector",
            "xml_geom_name_selector",
            "xml_joint_name_selector",
            "xml_actuator_joint_selector",
        ]
        mask = np.zeros(len(df), dtype=bool)
        for col in axis_cols:
            if col in df.columns:
                mask |= df[col].astype(str).str.contains(args.axis, case=False, na=False).to_numpy()
        df = df[mask]

    if df.empty:
        raise SystemExit("No rows remain after filtering.")
    if args.y_metric not in df.columns:
        raise SystemExit(f"Missing y metric column: {args.y_metric}")

    df = df.copy()
    df["checkpoint_step"] = df.apply(infer_checkpoint_step, axis=1)
    df = df.dropna(subset=["checkpoint_step"])
    if df.empty:
        raise SystemExit(
            "Could not infer checkpoint_step. Add a checkpoint_step column or use model paths containing step_<N>."
        )
    df["checkpoint_step"] = df["checkpoint_step"].astype(int)

    label_map = {}
    for item in args.display_label:
        if "=" not in item:
            raise SystemExit(f"--display-label must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        label_map[key] = value

    group_cols = ["model_label", "checkpoint_step"]
    summary = (
        df.groupby(group_cols)[args.y_metric]
        .agg(["mean", "std", "count", ci95])
        .reset_index()
        .rename(columns={"ci95": "ci95_return"})
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "checkpoint_learning_dynamics_summary.csv", index=False)

    model_order = args.model_order or list(summary["model_label"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for model in model_order:
        sub = summary[summary["model_label"] == model].sort_values("checkpoint_step")
        if sub.empty:
            continue
        x = sub["checkpoint_step"].to_numpy()
        y = sub["mean"].to_numpy()
        err = sub["ci95_return"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, label=label_map.get(model, model))
        ax.fill_between(x, y - err, y + err, alpha=0.18)

    ax.set_xlabel("Training environment steps")
    ax.set_ylabel(args.y_metric.replace("_", " ").title())
    ax.set_title(args.title or "Checkpoint Learning Dynamics")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "checkpoint_learning_dynamics.pdf")
    fig.savefig(out_dir / "checkpoint_learning_dynamics.png", dpi=250)


if __name__ == "__main__":
    main()
