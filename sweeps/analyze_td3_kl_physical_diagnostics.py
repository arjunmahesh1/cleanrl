#!/usr/bin/env python3
"""Analyze TensorBoard diagnostics from physical-support TD3-KL sweeps."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_RE = re.compile(r"__td3_klrho_(vanilla|klprho[\w]+)__(\d+)__")
TAGS = [
    "charts/episodic_return",
    "eval/episodic_return",
    "losses/qf_loss",
    "losses/actor_loss",
    "kl_physical/requested_radius",
    "kl_physical/effective_beta_median",
    "kl_physical/effective_beta_mean",
    "kl_physical/worst_case_saturation_fraction",
    "kl_physical/joint_return_std_across_dynamics",
    "kl_physical/reference_target_mean",
    "kl_physical/robust_target_mean",
    "kl_physical/worst_member_target_mean",
    "kl_physical/pessimism_gap_mean",
    "kl_physical/pessimism_gap_p95",
    "kl_physical/implicit_kl_radius_mean",
    "kl_physical/implicit_kl_radius_p95",
    "kl_physical/worst_member_adversarial_weight_mean",
    "kl_physical/effective_num_dynamics_mean",
    "kl_physical/nominal_obs_max_abs_error",
    "kl_physical/nominal_reward_abs_error",
    "charts/SPS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bin-width", type=int, default=10_000)
    parser.add_argument("--final-window", type=int, default=25_000)
    parser.add_argument("--min-nominal-retention", type=float, default=0.70)
    parser.add_argument("--max-q-loss-ratio", type=float, default=10.0)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def event_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, _, files in os.walk(root, followlinks=True):
        for filename in files:
            if filename.startswith("events.out.tfevents"):
                paths.append(Path(directory) / filename)
    return sorted(paths)


def parse_run(path: Path) -> tuple[str, int, float]:
    match = RUN_RE.search(str(path))
    if not match:
        raise ValueError(f"Could not parse KL model and seed from {path}")
    model = match.group(1)
    seed = int(match.group(2))
    if model == "vanilla":
        radius = float("nan")
    else:
        radius = float(model.removeprefix("klprho").replace("p", ".").replace("m", "-"))
    return model, seed, radius


def model_key(model: str) -> tuple[int, float]:
    if model == "vanilla":
        return 0, -1.0
    return 1, float(model.removeprefix("klprho").replace("p", ".").replace("m", "-"))


def display_model(model: str) -> str:
    if model == "vanilla":
        return "Vanilla"
    return rf"$\rho={model.removeprefix('klprho').replace('p', '.')}$"


def binned(steps: np.ndarray, values: np.ndarray, width: int) -> list[tuple[int, float]]:
    bins = (steps // width) * width
    return [
        (int(step), float(np.median(values[bins == step])))
        for step in np.unique(bins)
    ]


def final_value(steps: np.ndarray, values: np.ndarray, window: int) -> float:
    selected = values[steps >= max(0, int(steps.max()) - window)]
    if selected.size == 0:
        selected = values[-min(20, values.size) :]
    return float(np.median(selected))


def normalized_auc(points: list[tuple[int, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    if x[-1] <= x[0]:
        return float("nan")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def load_events(
    root: Path, bin_width: int, final_window: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    binned_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    paths = event_paths(root)
    if not paths:
        raise FileNotFoundError(f"No TensorBoard event files below {root}")

    for index, path in enumerate(paths, start=1):
        model, seed, radius = parse_run(path)
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        run: dict[str, object] = {
            "model_label": model,
            "seed": seed,
            "radius": radius,
            "event_path": str(path),
        }
        return_points: list[tuple[int, float]] = []
        max_step = 0
        for tag in TAGS:
            present = tag in available
            coverage_rows.append(
                {"model_label": model, "seed": seed, "tag": tag, "present": present}
            )
            if not present:
                continue
            events = accumulator.Scalars(tag)
            if not events:
                continue
            steps = np.asarray([event.step for event in events], dtype=np.int64)
            values = np.asarray([event.value for event in events], dtype=np.float64)
            max_step = max(max_step, int(steps.max()))
            points = binned(steps, values, bin_width)
            if tag == "charts/episodic_return":
                return_points = points
            for step, value in points:
                binned_rows.append(
                    {
                        "model_label": model,
                        "seed": seed,
                        "radius": radius,
                        "tag": tag,
                        "step": step,
                        "value": value,
                    }
                )
            run[f"final__{tag}"] = final_value(steps, values, final_window)
            run[f"max__{tag}"] = float(np.max(values))
            run[f"mean__{tag}"] = float(np.mean(values))
        run["max_step"] = max_step
        run["training_return_auc"] = normalized_auc(return_points)
        run_rows.append(run)
        if index % 10 == 0 or index == len(paths):
            print(f"loaded {index}/{len(paths)} event files")

    runs = pd.DataFrame(run_rows).sort_values(
        ["model_label", "seed"], key=lambda column: column if column.name == "seed" else column.map(model_key)
    )
    trajectories = pd.DataFrame(binned_rows)
    coverage = pd.DataFrame(coverage_rows)
    return runs, trajectories, coverage


def add_paired_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    eval_col = "final__eval/episodic_return"
    auc_col = "training_return_auc"
    vanilla = runs[runs["model_label"] == "vanilla"].set_index("seed")
    if vanilla.empty:
        raise ValueError("No vanilla runs found")
    for column, output in [
        (eval_col, "paired_nominal_retention"),
        (auc_col, "paired_training_auc_retention"),
    ]:
        if column in runs and column in vanilla:
            denominator = runs["seed"].map(vanilla[column])
            runs[output] = runs[column] / denominator
    qloss = "final__losses/qf_loss"
    if qloss in runs and qloss in vanilla:
        denominator = runs["seed"].map(vanilla[qloss]).clip(lower=1e-12)
        runs["paired_q_loss_ratio"] = runs[qloss] / denominator
    return runs


def summarize(runs: pd.DataFrame, min_retention: float, max_q_loss_ratio: float) -> pd.DataFrame:
    metric_columns = [
        "paired_nominal_retention",
        "paired_training_auc_retention",
        "paired_q_loss_ratio",
        "final__eval/episodic_return",
        "training_return_auc",
        "final__losses/qf_loss",
        "final__kl_physical/effective_beta_median",
        "final__kl_physical/worst_case_saturation_fraction",
        "final__kl_physical/joint_return_std_across_dynamics",
        "final__kl_physical/pessimism_gap_mean",
        "final__kl_physical/pessimism_gap_p95",
        "final__kl_physical/implicit_kl_radius_mean",
        "final__kl_physical/worst_member_adversarial_weight_mean",
        "final__kl_physical/effective_num_dynamics_mean",
    ]
    rows: list[dict[str, object]] = []
    for model, sub in runs.groupby("model_label"):
        row: dict[str, object] = {
            "model_label": model,
            "radius": sub["radius"].median(),
            "n_seeds": sub["seed"].nunique(),
        }
        for metric in metric_columns:
            if metric not in sub:
                continue
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}__median"] = values.median() if not values.empty else np.nan
            row[f"{metric}__min"] = values.min() if not values.empty else np.nan
            row[f"{metric}__max"] = values.max() if not values.empty else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        "model_label", key=lambda column: column.map(model_key)
    )
    retention = summary.get("paired_nominal_retention__median", pd.Series(np.nan, index=summary.index))
    q_ratio = summary.get("paired_q_loss_ratio__median", pd.Series(np.nan, index=summary.index))
    realized = summary.get(
        "final__kl_physical/implicit_kl_radius_mean__median",
        pd.Series(np.nan, index=summary.index),
    )
    finite = np.isfinite(
        summary.select_dtypes(include=[np.number]).drop(columns=["radius"], errors="ignore")
    ).all(axis=1)
    summary["numerically_finite"] = finite
    summary["promotion_gate"] = (
        (summary["model_label"] != "vanilla")
        & (retention >= min_retention)
        & (q_ratio <= max_q_loss_ratio)
        & finite
        & realized.notna()
    )
    return summary


def trajectory_summary(trajectories: pd.DataFrame) -> pd.DataFrame:
    return (
        trajectories.groupby(["model_label", "radius", "tag", "step"], dropna=False)["value"]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), n="count")
        .reset_index()
    )


def plot_trajectory(
    axes: plt.Axes,
    trajectories: pd.DataFrame,
    tag: str,
    ylabel: str,
    models: list[str],
) -> None:
    sub = trajectories[trajectories["tag"] == tag]
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, max(1, len(models))))
    for color, model in zip(colors, models):
        model_sub = sub[sub["model_label"] == model].sort_values("step")
        if model_sub.empty:
            continue
        axes.plot(model_sub["step"], model_sub["median"], label=display_model(model), color=color)
        axes.fill_between(
            model_sub["step"],
            model_sub["q25"],
            model_sub["q75"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    axes.set_xlabel("Environment step")
    axes.set_ylabel(ylabel)
    axes.grid(alpha=0.25)


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    for extension in formats:
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def make_plots(
    runs: pd.DataFrame,
    trajectories: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
) -> None:
    models = sorted(runs["model_label"].unique(), key=model_key)
    robust_models = [model for model in models if model != "vanilla"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    plot_trajectory(axes[0], trajectories, "charts/episodic_return", "Training return", models)
    plot_trajectory(axes[1], trajectories, "losses/qf_loss", "Twin-critic loss", models)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Physical-support TD3-KL: learning dynamics")
    fig.tight_layout()
    save_figure(fig, out_dir / "learning_dynamics", formats)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    radius = [runs.loc[runs["model_label"] == model, "radius"].median() for model in robust_models]
    panels = [
        ("paired_nominal_retention", "Paired nominal retention"),
        ("final__kl_physical/pessimism_gap_mean", "Pessimism gap"),
        ("final__kl_physical/effective_num_dynamics_mean", "Effective dynamics"),
    ]
    for axis, (metric, label) in zip(axes, panels):
        medians = [runs.loc[runs["model_label"] == model, metric].median() for model in robust_models]
        lows = [runs.loc[runs["model_label"] == model, metric].min() for model in robust_models]
        highs = [runs.loc[runs["model_label"] == model, metric].max() for model in robust_models]
        axis.errorbar(
            radius,
            medians,
            yerr=[np.asarray(medians) - np.asarray(lows), np.asarray(highs) - np.asarray(medians)],
            marker="o",
            capsize=3,
        )
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlabel(r"Requested KL radius $\rho$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    fig.suptitle("Requested radius, nominal cost, and realized adversary")
    fig.tight_layout()
    save_figure(fig, out_dir / "radius_tradeoff", formats)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_trajectory(
        axes[0],
        trajectories,
        "kl_physical/implicit_kl_radius_mean",
        "Achieved KL radius",
        robust_models,
    )
    plot_trajectory(
        axes[1],
        trajectories,
        "kl_physical/worst_member_adversarial_weight_mean",
        "Weight on worst member",
        robust_models,
    )
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Adversarial reweighting dynamics")
    fig.tight_layout()
    save_figure(fig, out_dir / "adversary_dynamics", formats)


def write_report(
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    out_dir: Path,
    min_retention: float,
    max_q_loss_ratio: float,
) -> None:
    promoted = summary.loc[summary["promotion_gate"], "model_label"].tolist()
    missing = coverage.groupby("tag")["present"].mean().sort_values()
    lines = [
        "# Physical-support TD3-KL diagnostic",
        "",
        "Promotion requires:",
        f"- median same-seed nominal retention >= {min_retention:.2f}",
        f"- median same-seed final critic-loss ratio <= {max_q_loss_ratio:.1f}",
        "- finite diagnostics and a measured realized KL radius",
        "",
        f"Promoted by the automated health gate: {', '.join(promoted) if promoted else 'none'}",
        "",
        "The health gate is necessary but not sufficient. Promoted settings must still",
        "beat vanilla on predeclared perturbation and catastrophe-risk metrics.",
        "",
        "## Scalar coverage",
        "",
    ]
    lines.extend(f"- `{tag}`: {fraction:.1%}" for tag, fraction in missing.items())
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, trajectories, coverage = load_events(
        Path(args.event_dir).expanduser(), args.bin_width, args.final_window
    )
    runs = add_paired_metrics(runs)
    summary = summarize(runs, args.min_nominal_retention, args.max_q_loss_ratio)
    trajectories_summary = trajectory_summary(trajectories)

    runs.to_csv(out_dir / "run_summary.csv", index=False)
    summary.to_csv(out_dir / "model_summary.csv", index=False)
    trajectories_summary.to_csv(out_dir / "training_scalars_binned.csv", index=False)
    coverage.to_csv(out_dir / "tag_coverage.csv", index=False)
    make_plots(runs, trajectories_summary, out_dir, args.formats)
    write_report(
        summary,
        coverage,
        out_dir,
        args.min_nominal_retention,
        args.max_q_loss_ratio,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
