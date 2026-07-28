#!/usr/bin/env python3
"""Inspect completed TD3 KL-moment runs without loading full event histories."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_RE = re.compile(r"__td3_kl_(vanilla|klb\d+(?:p\d+)?)__(\d+)__")
TAGS = [
    "eval/episodic_return",
    "losses/qf_loss",
    "losses/actor_loss",
    "kl/log_moment1_mean",
    "kl/log_moment_target_mean_raw",
    "kl/log_moment_target_min_for_exp",
    "kl/log_moment_target_max_for_exp",
    "kl/moment1_mean",
    "kl/moment_target_mean",
    "kl/moment_target_gt_1e_minus_6_frac",
    "kl/moment_loss_scale",
    "kl/qf1_moment_mse_unscaled",
    "kl/value_g_loss",
    "kl/log_moment1_clamp_fraction",
    "kl/log_moment_target_clamp_fraction",
    "kl/implied_q1_mean",
    "kl/implied_q_target_mean",
    "kl/implied_q1_target_mse",
    "kl/implied_q1_p95",
    "kl/implied_q1_p99",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scalar-reservoir", type=int, default=2_000)
    parser.add_argument("--final-window", type=int, default=250_000)
    return parser.parse_args()


def beta_value(model: str) -> float:
    if model == "vanilla":
        return float("nan")
    return float(model.removeprefix("klb").replace("p", "."))


def parse_run(path: Path) -> tuple[str, int]:
    match = RUN_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse model and seed from {path}")
    return match.group(1), int(match.group(2))


def final_median(events, window: int) -> float:
    steps = np.asarray([event.step for event in events], dtype=np.int64)
    values = np.asarray([event.value for event in events], dtype=np.float64)
    keep = steps >= max(0, int(steps.max()) - window)
    return float(np.median(values[keep]))


def inspect_run(run_dir: Path, scalar_reservoir: int, final_window: int) -> dict[str, object]:
    model, seed = parse_run(run_dir)
    event_files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda path: (path.stat().st_size, path.stat().st_mtime))
    if not event_files:
        raise FileNotFoundError(f"No event file in {run_dir}")
    event_path = event_files[-1]
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": scalar_reservoir})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))

    row: dict[str, object] = {
        "model_label": model,
        "seed": seed,
        "beta": beta_value(model),
        "run_dir": str(run_dir),
        "event_path": str(event_path),
    }
    max_step = 0
    for tag in TAGS:
        if tag not in available:
            continue
        events = accumulator.Scalars(tag)
        if not events:
            continue
        max_step = max(max_step, max(event.step for event in events))
        row[f"final__{tag}"] = final_median(events, final_window)
        values = np.asarray([event.value for event in events], dtype=np.float64)
        row[f"loaded_mean__{tag}"] = float(np.mean(values))
        row[f"loaded_min__{tag}"] = float(np.min(values))
        row[f"loaded_max__{tag}"] = float(np.max(values))
    row["max_step"] = max_step
    return row


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in runs if column.startswith("final__")]
    rows: list[dict[str, object]] = []
    for model, sub in runs.groupby("model_label"):
        row: dict[str, object] = {
            "model_label": model,
            "beta": beta_value(model),
            "n_seeds": int(sub["seed"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}__median"] = float(values.median()) if not values.empty else float("nan")
            row[f"{metric}__q25"] = float(values.quantile(0.25)) if not values.empty else float("nan")
            row[f"{metric}__q75"] = float(values.quantile(0.75)) if not values.empty else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["sort_beta"] = summary["beta"].fillna(-1)
    return summary.sort_values("sort_beta").drop(columns="sort_beta")


def main() -> None:
    args = parse_args()
    run_dirs = sorted({model.parent for model in args.run_dir.rglob("*.cleanrl_model")})
    if not run_dirs:
        raise FileNotFoundError(f"No completed model directories below {args.run_dir}")

    rows = []
    for index, run_dir in enumerate(run_dirs, start=1):
        rows.append(inspect_run(run_dir, args.scalar_reservoir, args.final_window))
        print(f"loaded {index}/{len(run_dirs)}: {run_dir.name}", flush=True)

    runs = pd.DataFrame(rows).sort_values(["model_label", "seed"])
    summary = aggregate(runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out_dir / "run_summary.csv", index=False)
    summary.to_csv(args.out_dir / "model_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
