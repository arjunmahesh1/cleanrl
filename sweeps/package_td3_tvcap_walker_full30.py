#!/usr/bin/env python3
"""Validate and package the full Walker2d TD3 TV-cap evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from build_td3_tvcap_walker_manifest import CAPS


CATEGORY_AXES = {
    "single_axis_perturbations": "friction damping actuator_gain mass".split(),
    "targeted_localized_perturbations": (
        "foot_left_actuator_gain foot_left_damping foot_left_friction foot_left_mass "
        "leg_left_actuator_gain leg_left_damping leg_left_mass "
        "thigh_left_actuator_gain thigh_left_damping thigh_left_mass"
    ).split(),
    "combos": "friction_damping friction_mass friction_mass_damping mass_damping".split(),
    "gaussian_action_noise": ["action_noise"],
    "bernoulli_action_noise": ["action_replace"],
}

EXPECTED_CATEGORY_ROWS = {
    "single_axis_perturbations": 15_120,
    "targeted_localized_perturbations": 37_680,
    "combos": 14_640,
    "gaussian_action_noise": 1_440,
    "bernoulli_action_noise": 2_640,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--caps", nargs="+", default=CAPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 31)))
    parser.add_argument("--fixed-model-label", default="tvc250")
    return parser.parse_args()


def parse_scenario(label: str) -> tuple[str, float]:
    axis, token = str(label).rsplit("_", 1)
    sign = -1.0 if token.startswith("m") else 1.0
    if token.startswith("m"):
        token = token[1:]
    return axis, sign * float(token.replace("p", "."))


def model_label(cap: str) -> str:
    return "vanilla" if cap == "vanilla" else f"tvc{cap.replace('.', 'p')}"


def run_package(result_dir: Path, category: str, models: list[str]) -> None:
    category_dir = result_dir / "Walker2d" / category
    nominal = "0.0" if category in {"gaussian_action_noise", "bernoulli_action_noise"} else "1.0"
    title = f"Walker2d TD3 TV-Cap Full 30-Seed: {category.replace('_', ' ').title()}"
    comparison = [model for model in models if model != "vanilla"]
    command = [
        sys.executable,
        str(Path(__file__).with_name("package_alpha_robust_eval.py")),
        "--raw-metrics-dir",
        str(category_dir / "raw_metrics"),
        "--out-dir",
        str(category_dir),
        "--title",
        title,
        "--baseline-model-label",
        "vanilla",
        "--comparison-model-labels",
        *comparison,
        "--model-order",
        *models,
        "--nominal-factor",
        nominal,
        "--disable-variance-whiskers",
        "--panel-max-cols",
        "4",
    ]
    for model in comparison:
        command.extend(["--display-label", f"{model}=TV c={model.removeprefix('tvc').replace('p', '.')}"])
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser()
    shard_paths = sorted((result_dir / "raw_metrics" / "shards").glob("shard_*.csv"))
    if len(shard_paths) != 716:
        raise SystemExit(f"Expected 716 shards, found {len(shard_paths)}")

    df = pd.concat([pd.read_csv(path) for path in shard_paths if path.stat().st_size], ignore_index=True)
    parsed = df["scenario_label"].map(parse_scenario)
    df["axis"] = parsed.map(lambda item: item[0])
    df["factor"] = parsed.map(lambda item: item[1])

    models = [model_label(cap) for cap in args.caps]
    expected_rows = 298 * len(models) * len(args.seeds)
    if len(df) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} rows, found {len(df)}")
    if set(df["model_label"]) != set(models):
        raise SystemExit(f"Unexpected models: {sorted(set(df['model_label']))}")
    if set(df["seed"].astype(int)) != set(args.seeds):
        raise SystemExit(f"Unexpected seeds: {sorted(set(df['seed'].astype(int)))}")
    duplicate_count = int(df.duplicated(["env_id", "model_label", "seed", "scenario_label"]).sum())
    if duplicate_count:
        raise SystemExit(f"Found {duplicate_count} duplicate evaluation keys")

    outputs = result_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    df.to_csv(outputs / "combined_metrics.csv", index=False)

    for category, axes in CATEGORY_AXES.items():
        category_dir = result_dir / "Walker2d" / category
        raw_dir = category_dir / "raw_metrics"
        raw_dir.mkdir(parents=True, exist_ok=True)
        subset = df[df["axis"].isin(axes)].copy()
        expected = EXPECTED_CATEGORY_ROWS[category] * len(models) * len(args.seeds) // (8 * 30)
        if len(subset) != expected:
            raise SystemExit(f"{category}: expected {expected} rows, found {len(subset)}")
        subset.to_csv(raw_dir / "metrics.csv", index=False)
        print(f"{category}: {len(subset)} rows")
        run_package(result_dir, category, models)

    analysis_command = [
        sys.executable,
        str(Path(__file__).with_name("plot_full30_seed_reliability.py")),
        "--result-dir",
        str(result_dir),
        "--formats",
        "png",
        "pdf",
        "--all-scatter-axes",
        "--fixed-model-label",
        args.fixed_model_label,
    ]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-td3-full30")
    subprocess.run(analysis_command, check=True, env=env)

    readme = f"""# Walker2d TD3 TV-Cap Full 30-Seed Evaluation

- Models: {', '.join(models)}
- Seeds: {min(args.seeds)}--{max(args.seeds)}
- Evaluation episodes per row: 20
- Evaluation rows: {len(df)}
- Fixed deployment-style cap in seed-conditioned analysis: {args.fixed_model_label}
- Variance whiskers are disabled in category return/gain figures.

Layout:
- `raw_metrics/shards/`: chunk-level source CSVs.
- `outputs/combined_metrics.csv`: validated combined evaluation table.
- `Walker2d/`: PPO-style category folders with raw metrics, tables, and plots.
- `analysis_plots/`: seed spaghetti, fixed-seed cap comparisons, seed scatter, reliability/AUC, and seed-conditioned gain analyses.
"""
    (result_dir / "README.md").write_text(readme, encoding="utf-8")

    figure_count = sum(1 for path in result_dir.rglob("*") if path.suffix in {".png", ".pdf"})
    print(f"combined rows: {len(df)}")
    print(f"figures: {figure_count}")
    print(f"wrote {result_dir}")


if __name__ == "__main__":
    main()
