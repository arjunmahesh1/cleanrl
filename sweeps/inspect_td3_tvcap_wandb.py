#!/usr/bin/env python3
"""Match a TD3 TV-cap TensorBoard sweep to synchronized W&B run summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import wandb


SUMMARY_KEYS = [
    "eval/episodic_return",
    "charts/SPS",
    "losses/qf1_values",
    "losses/qf_loss",
    "targets/min_q_next_p95_pre_clip",
    "robust/td3_q_target_clip_fraction",
    "robust/td3_q_target_excess_mean",
    "_runtime",
    "_step",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--project", default="arjun-mahesh-duke-university/td3-experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local = pd.read_csv(args.run_summary)
    local["run_name"] = local["event_path"].map(lambda value: Path(str(value)).parent.name)
    wanted = set(local["run_name"])

    rows: list[dict[str, object]] = []
    for run in wandb.Api(timeout=90).runs(args.project, per_page=500):
        if run.name not in wanted:
            continue
        row: dict[str, object] = {
            "run_name": run.name,
            "wandb_id": run.id,
            "wandb_group": run.group,
            "wandb_state": run.state,
        }
        for key in SUMMARY_KEYS:
            row[key] = run.summary.get(key)
        rows.append(row)

    wandb_rows = pd.DataFrame(rows)
    matched = local[["run_name", "model_label", "seed"]].merge(wandb_rows, on="run_name", how="left")
    matched.to_csv(out_dir / "wandb_run_summary.csv", index=False)

    matched_count = int(matched["wandb_id"].notna().sum())
    eval_count = int(pd.to_numeric(matched["eval/episodic_return"], errors="coerce").notna().sum())
    print(f"matched W&B runs: {matched_count}/{len(matched)}")
    print(f"runs with final eval summary: {eval_count}/{len(matched)}")

    numeric = [key for key in SUMMARY_KEYS if key in matched]
    for key in numeric:
        matched[key] = pd.to_numeric(matched[key], errors="coerce")
    aggregate = matched.groupby("model_label")[numeric].agg(["median", "mean", "count"])
    aggregate.to_csv(out_dir / "wandb_model_summary.csv")
    selected = [key for key in ["eval/episodic_return", "_runtime", "charts/SPS"] if key in matched]
    print(matched.groupby("model_label")[selected].agg(["median", "mean", "count"]).round(3))


if __name__ == "__main__":
    main()
