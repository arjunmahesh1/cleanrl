#!/usr/bin/env python3
"""Validate and package a focused physical-support TD3-KL evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--env-id", default="Walker2d-v5")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "vanilla",
            "klprho0",
            "klprho0p01",
            "klprho0p05",
            "klprho0p1",
            "klprho0p2",
            "klprho0p5",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--expected-scenarios", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser()
    shards = sorted((result_dir / "raw_metrics" / "shards").glob("shard_*.csv"))
    if not shards:
        raise SystemExit("No evaluation shards found")
    frames = [pd.read_csv(path, low_memory=False) for path in shards if path.stat().st_size]
    df = pd.concat(frames, ignore_index=True)
    expected_rows = len(args.models) * len(args.seeds) * args.expected_scenarios
    if len(df) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} rows, found {len(df)}")
    if set(df["model_label"]) != set(args.models):
        raise SystemExit(f"Unexpected models: {sorted(df['model_label'].unique())}")
    if set(df["seed"].astype(int)) != set(args.seeds):
        raise SystemExit(f"Unexpected seeds: {sorted(df['seed'].astype(int).unique())}")
    if set(df["env_id"]) != {args.env_id}:
        raise SystemExit(f"Unexpected environments: {sorted(df['env_id'].unique())}")
    duplicate_count = int(
        df.duplicated(["env_id", "model_label", "seed", "scenario_label"]).sum()
    )
    if duplicate_count:
        raise SystemExit(f"Found {duplicate_count} duplicate evaluation keys")

    outputs = result_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    combined = outputs / "combined_metrics.csv"
    df.to_csv(combined, index=False)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-td3-kl-physical")

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("analyze_td3_kl_physical_diagnostics.py")),
            "--event-dir",
            str(Path(args.event_dir).expanduser()),
            "--out-dir",
            str(result_dir / "analysis_plots" / "training_diagnostics"),
            "--formats",
            "png",
            "pdf",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("analyze_reliability_dominance.py")),
            "--metrics-csv",
            str(combined),
            "--out-dir",
            str(result_dir / "analysis_plots" / "reliability_dominance"),
            "--formats",
            "png",
            "pdf",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("analyze_td3_robustness_gate.py")),
            "--metrics-csv",
            str(combined),
            "--out-dir",
            str(result_dir / "analysis_plots" / "robustness_gate"),
            "--formats",
            "png",
            "pdf",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("select_td3_kl_candidate.py")),
            "--metrics-csv",
            str(combined),
            "--out-dir",
            str(result_dir / "analysis_plots" / "candidate_selection"),
        ],
        check=True,
        env=env,
    )

    readme = f"""# Physical-support TD3-KL focused diagnostic

- Environment: {args.env_id}
- Models: {", ".join(args.models)}
- Seeds: {", ".join(map(str, args.seeds))}
- Scenarios per model/seed: {args.expected_scenarios}
- Evaluation rows: {len(df)}

This is a staged promotion experiment. Training health and nominal retention
are checked before catastrophe-aware robustness metrics. No radius is promoted
to a longer run unless it passes the predeclared candidate-selection gate
against both vanilla TD3 and the rho=0 physical-ensemble control.
"""
    (result_dir / "README.md").write_text(readme, encoding="utf-8")
    figure_count = sum(
        path.suffix in {".png", ".pdf"} for path in result_dir.rglob("*")
    )
    print(f"shards: {len(shards)}")
    print(f"rows: {len(df)}")
    print(f"figures: {figure_count}")
    print(f"wrote {result_dir}")


if __name__ == "__main__":
    main()
