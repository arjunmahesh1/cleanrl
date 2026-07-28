#!/usr/bin/env python3
"""Inspect final TD3 TV-cap training diagnostics with bounded memory use."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_RE = re.compile(r"__td3_tvcap_(vanilla|tvc(?:1e9|\d+(?:p\d+)?))__(\d+)__")
TAGS = [
    "eval/episodic_return",
    "losses/qf1_values",
    "losses/qf_loss",
    "targets/min_q_next_p95_pre_clip",
    "targets/td_target_mean",
    "robust/td3_q_target_clip_fraction",
    "robust/td3_q_target_excess_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scalar-reservoir", type=int, default=5_000)
    parser.add_argument("--final-window", type=int, default=250_000)
    return parser.parse_args()


def cap_value(model: str) -> float:
    if model == "vanilla":
        return float("nan")
    return float(model.removeprefix("tvc").replace("p", "."))


def completed_runs(root: Path) -> list[Path]:
    return sorted({model.parent for model in root.rglob("*.cleanrl_model")})


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
        "cap": cap_value(model),
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
        row[f"tail_mean__{tag}"] = float(np.mean(values))
        row[f"tail_max__{tag}"] = float(np.max(values))
    row["max_step"] = max_step
    return row


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    p95 = "final__targets/min_q_next_p95_pre_clip"
    clip = "final__robust/td3_q_target_clip_fraction"
    if p95 in runs:
        runs["final_q_p95_over_cap"] = runs[p95] / runs["cap"]
    if "tail_max__robust/td3_q_target_clip_fraction" in runs:
        runs["active_in_loaded_tail"] = np.where(
            runs["cap"].notna(),
            runs["tail_max__robust/td3_q_target_clip_fraction"] >= 0.01,
            np.nan,
        )

    metrics = [
        "final__eval/episodic_return",
        "final__losses/qf1_values",
        "final__losses/qf_loss",
        p95,
        "final__targets/td_target_mean",
        clip,
        "tail_mean__robust/td3_q_target_clip_fraction",
        "final__robust/td3_q_target_excess_mean",
        "final_q_p95_over_cap",
        "active_in_loaded_tail",
    ]
    rows: list[dict[str, object]] = []
    for model, sub in runs.groupby("model_label"):
        row: dict[str, object] = {"model_label": model, "n_seeds": int(sub["seed"].nunique())}
        for metric in metrics:
            if metric not in sub:
                continue
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}__median"] = float(values.median()) if not values.empty else float("nan")
            row[f"{metric}__q25"] = float(values.quantile(0.25)) if not values.empty else float("nan")
            row[f"{metric}__q75"] = float(values.quantile(0.75)) if not values.empty else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["sort_cap"] = summary["model_label"].map(cap_value).fillna(-1)
    return summary.sort_values("sort_cap").drop(columns="sort_cap")


def main() -> None:
    args = parse_args()
    run_dirs = completed_runs(args.run_dir)
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
