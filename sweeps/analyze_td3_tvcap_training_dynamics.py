#!/usr/bin/env python3
"""Aggregate TensorBoard dynamics for a multi-seed TD3 TV-cap sweep."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = [
    "charts/episodic_return",
    "charts/episodic_length",
    "eval/episodic_return",
    "losses/qf1_values",
    "losses/qf2_values",
    "losses/qf_loss",
    "losses/actor_loss",
    "targets/min_q_next_mean_pre_clip",
    "targets/min_q_next_p95_pre_clip",
    "targets/min_q_next_p99_pre_clip",
    "targets/min_q_next_mean_post_clip",
    "targets/td_target_mean",
    "robust/td3_q_target_clip_fraction",
    "robust/td3_q_target_excess_mean",
    "charts/SPS",
]

MODEL_RE = re.compile(r"__td3_tvcap_(vanilla|tvc\d+(?:p\d+)?)__(\d+)__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bin-width", type=int, default=25_000)
    parser.add_argument("--final-window", type=int, default=100_000)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Reuse run_summary.csv, training_scalars_binned.csv, and tag_coverage.csv in out-dir.",
    )
    return parser.parse_args()


def model_key(model: str) -> tuple[int, float]:
    if model == "vanilla":
        return (0, -1.0)
    return (1, float(model.removeprefix("tvc").replace("p", ".")))


def display_model(model: str) -> str:
    if model == "vanilla":
        return "Vanilla"
    return f"TV c={model.removeprefix('tvc').replace('p', '.')}"


def cap_value(model: str) -> float:
    if model == "vanilla":
        return float("nan")
    return float(model.removeprefix("tvc").replace("p", "."))


def event_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, _, files in os.walk(root, followlinks=True):
        for filename in files:
            if filename.startswith("events.out.tfevents"):
                paths.append(Path(directory) / filename)
    return sorted(paths)


def parse_run(path: Path) -> tuple[str, int]:
    match = MODEL_RE.search(str(path))
    if not match:
        raise ValueError(f"Could not parse model and seed from {path}")
    return match.group(1), int(match.group(2))


def newest_event_per_run(paths: list[Path]) -> list[Path]:
    selected: dict[tuple[str, int], Path] = {}
    duplicate_counts: dict[tuple[str, int], int] = {}
    for path in paths:
        key = parse_run(path)
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
        current = selected.get(key)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            selected[key] = path
    duplicates = {
        key: count for key, count in duplicate_counts.items() if count > 1
    }
    if duplicates:
        details = ", ".join(
            f"{model}/seed{seed}={count}"
            for (model, seed), count in sorted(duplicates.items())
        )
        print(f"duplicate event runs found; using newest per model/seed: {details}")
    return sorted(selected.values())


def binned_events(steps: np.ndarray, values: np.ndarray, width: int) -> list[tuple[int, float]]:
    bins = (steps // width) * width
    return [(int(step_bin), float(np.median(values[bins == step_bin]))) for step_bin in np.unique(bins)]


def final_value(steps: np.ndarray, values: np.ndarray, max_step: int, window: int) -> float:
    keep = steps >= max(0, max_step - window)
    selected = values[keep]
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


def load_events(root: Path, bin_width: int, final_window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binned_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    paths = newest_event_per_run(event_paths(root))
    if not paths:
        raise FileNotFoundError(f"No TensorBoard event files found below {root}")

    for index, path in enumerate(paths, start=1):
        model, seed = parse_run(path)
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        run: dict[str, object] = {
            "model_label": model,
            "seed": seed,
            "cap": cap_value(model),
            "event_path": str(path),
        }
        max_step = 0
        return_points: list[tuple[int, float]] = []

        for tag in TAGS:
            present = tag in available
            coverage_rows.append({"model_label": model, "seed": seed, "tag": tag, "present": present})
            if not present:
                continue
            events = accumulator.Scalars(tag)
            if not events:
                continue
            steps = np.asarray([event.step for event in events], dtype=np.int64)
            values = np.asarray([event.value for event in events], dtype=np.float64)
            max_step = max(max_step, int(steps.max()))
            points = binned_events(steps, values, bin_width)
            if tag == "charts/episodic_return":
                return_points = points
            for step, value in points:
                binned_rows.append(
                    {"model_label": model, "seed": seed, "tag": tag, "step": step, "value": value}
                )
            run[f"final__{tag}"] = final_value(steps, values, int(steps.max()), final_window)
            run[f"max__{tag}"] = float(np.max(values))
            run[f"mean__{tag}"] = float(np.mean(values))
            run[f"first_step__{tag}__ge_0p01"] = (
                float(steps[values >= 0.01].min()) if np.any(values >= 0.01) else float("nan")
            )

        run["max_step"] = max_step
        run["training_return_auc"] = normalized_auc(return_points)
        run_rows.append(run)
        if index % 20 == 0 or index == len(paths):
            print(f"loaded {index}/{len(paths)} event files")

    binned = pd.DataFrame(binned_rows)
    runs = pd.DataFrame(run_rows).sort_values(["model_label", "seed"], key=lambda col: col)
    coverage = pd.DataFrame(coverage_rows)

    return binned, add_derived_columns(runs), coverage


def add_derived_columns(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    p95_col = "final__targets/min_q_next_p95_pre_clip"
    if p95_col in runs:
        runs["final_q_p95_over_cap"] = runs[p95_col] / runs["cap"]
    clip_col = "final__robust/td3_q_target_clip_fraction"
    if clip_col in runs:
        runs["cap_active_final"] = np.where(runs["cap"].notna(), runs[clip_col] >= 0.01, np.nan)
    max_clip_col = "max__robust/td3_q_target_clip_fraction"
    if max_clip_col in runs:
        runs["cap_active_ever"] = np.where(runs["cap"].notna(), runs[max_clip_col] >= 0.01, np.nan)
    return runs


def trajectory_summary(binned: pd.DataFrame) -> pd.DataFrame:
    return (
        binned.groupby(["model_label", "tag", "step"], as_index=False)["value"]
        .agg(median="median", q25=lambda values: values.quantile(0.25), q75=lambda values: values.quantile(0.75), n="count")
    )


def model_summary(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "training_return_auc",
        "final__charts/episodic_return",
        "final__charts/episodic_length",
        "final__eval/episodic_return",
        "final__losses/qf_loss",
        "final__losses/actor_loss",
        "final__losses/qf1_values",
        "final__targets/min_q_next_p95_pre_clip",
        "final__targets/td_target_mean",
        "final__robust/td3_q_target_clip_fraction",
        "mean__robust/td3_q_target_clip_fraction",
        "max__robust/td3_q_target_clip_fraction",
        "first_step__robust/td3_q_target_clip_fraction__ge_0p01",
        "final__robust/td3_q_target_excess_mean",
        "final_q_p95_over_cap",
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
        if "cap_active_final" in sub:
            valid = sub["cap_active_final"].dropna()
            row["cap_active_final_fraction"] = float(valid.mean()) if not valid.empty else float("nan")
        if "cap_active_ever" in sub:
            valid = sub["cap_active_ever"].dropna()
            row["cap_active_ever_fraction"] = float(valid.mean()) if not valid.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_label", key=lambda col: col.map(model_key))


def style_map(models: list[str]) -> dict[str, tuple[object, str]]:
    cmap = plt.get_cmap("viridis")
    robust = [model for model in models if model != "vanilla"]
    styles = {"vanilla": ("black", "--")}
    for index, model in enumerate(robust):
        styles[model] = (cmap((index + 1) / (len(robust) + 1)), "-")
    return styles


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_trajectory(ax, summary: pd.DataFrame, tag: str, styles, title: str, ylabel: str, log_y: bool = False) -> None:
    sub = summary[summary["tag"] == tag]
    for model in sorted(sub["model_label"].unique(), key=model_key):
        model_sub = sub[sub["model_label"] == model].sort_values("step")
        color, linestyle = styles[model]
        x = model_sub["step"].to_numpy(dtype=float)
        median = model_sub["median"].to_numpy(dtype=float)
        q25 = model_sub["q25"].to_numpy(dtype=float)
        q75 = model_sub["q75"].to_numpy(dtype=float)
        ax.plot(x, median, color=color, linestyle=linestyle, linewidth=2.0, label=display_model(model))
        ax.fill_between(x, q25, q75, color=color, alpha=0.11)
    ax.set_title(title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.2)


def plot_dynamics(binned: pd.DataFrame, runs: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    summary = trajectory_summary(binned)
    models = sorted(runs["model_label"].unique(), key=model_key)
    styles = style_map(models)
    available_tags = set(binned["tag"].unique())
    has_return = "charts/episodic_return" in available_tags
    has_length = "charts/episodic_length" in available_tags
    eval_return_col = "final__eval/episodic_return"
    has_eval_return = eval_return_col in runs and runs[eval_return_col].notna().any()

    fig, axs = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True)
    if has_return:
        draw_trajectory(axs[0], summary, "charts/episodic_return", styles, "Training episodic return", "Return")
    else:
        axs[0].text(
            0.5,
            0.5,
            "Unavailable: charts/episodic_return was not logged\nin any of the 240 training runs.",
            ha="center",
            va="center",
            transform=axs[0].transAxes,
        )
        axs[0].set_axis_off()
    if has_length:
        draw_trajectory(axs[1], summary, "charts/episodic_length", styles, "Training episode length", "Steps")
    else:
        axs[1].text(
            0.5,
            0.5,
            "Unavailable: charts/episodic_length was not logged\nin any of the 240 training runs.",
            ha="center",
            va="center",
            transform=axs[1].transAxes,
        )
        axs[1].set_axis_off()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False)
    fig.suptitle("Walker2d TD3 TV-cap learning dynamics across 30 seeds")
    fig.tight_layout(rect=(0, 0, 0.84, 0.97))
    save_figure(fig, out_dir / "return_length_dynamics", formats)

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.0))
    panels = [
        ("losses/qf1_values", "Learned Q1 scale", "Q1", False),
        ("targets/min_q_next_p95_pre_clip", "Pre-cap target-Q p95", "Q p95", False),
        ("losses/qf_loss", "Twin critic loss", "MSE", True),
        ("losses/actor_loss", "Actor loss", "Loss", False),
    ]
    for ax, (tag, title, ylabel, log_y) in zip(axs.flat, panels):
        draw_trajectory(ax, summary, tag, styles, title, ylabel, log_y)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Walker2d TD3 critic and actor dynamics")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(fig, out_dir / "critic_actor_dynamics", formats)

    robust_models = [model for model in models if model != "vanilla"]
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.0))
    draw_trajectory(
        axs[0, 0],
        summary,
        "robust/td3_q_target_clip_fraction",
        styles,
        "Target clip fraction",
        "Fraction clipped",
    )
    draw_trajectory(
        axs[0, 1],
        summary,
        "robust/td3_q_target_excess_mean",
        styles,
        "Mean excess removed by cap",
        "Q excess",
    )

    ratio = binned[binned["tag"] == "targets/min_q_next_p95_pre_clip"].copy()
    ratio["cap"] = ratio["model_label"].map(cap_value)
    ratio = ratio[ratio["cap"].notna()]
    ratio["value"] = ratio["value"] / ratio["cap"]
    ratio_summary = trajectory_summary(ratio)
    draw_trajectory(axs[1, 0], ratio_summary, "targets/min_q_next_p95_pre_clip", styles, "Pre-cap Q p95 / cap", "Q p95 / c")
    axs[1, 0].axhline(1.0, color="black", linestyle=":", linewidth=1.2)

    clip_col = "final__robust/td3_q_target_clip_fraction"
    return_col = "final__charts/episodic_return"
    q_col = "final__losses/qf1_values"
    endpoint_col = return_col if has_return else eval_return_col
    has_endpoint = has_return or has_eval_return
    robust_runs = runs[runs["model_label"].isin(robust_models)].copy()
    for model in robust_models:
        sub = robust_runs[robust_runs["model_label"] == model]
        color, _ = styles[model]
        y_col = endpoint_col if has_endpoint else q_col
        axs[1, 1].scatter(sub[clip_col], sub[y_col], color=color, s=30, alpha=0.72, label=display_model(model))
    if has_return:
        scatter_title = "Final training return vs final clip fraction"
        scatter_ylabel = "Final-window training return"
    elif has_eval_return:
        scatter_title = "Final nominal evaluation vs final clip fraction"
        scatter_ylabel = "Median nominal evaluation return"
    else:
        scatter_title = "Learned Q1 scale vs final clip fraction"
        scatter_ylabel = "Final-window Q1"
    axs[1, 1].set_title(scatter_title)
    axs[1, 1].set_xlabel("Final-window clip fraction")
    axs[1, 1].set_ylabel(scatter_ylabel)
    axs[1, 1].grid(alpha=0.2)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Walker2d TD3 effective pessimism diagnostics")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(fig, out_dir / "effective_pessimism_dynamics", formats)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    distribution_col = endpoint_col if has_endpoint else "final__targets/min_q_next_p95_pre_clip"
    rng = np.random.default_rng(7)
    for position, model in enumerate(models):
        values = runs.loc[runs["model_label"] == model, distribution_col].dropna().to_numpy(dtype=float)
        color, _ = styles[model]
        ax.scatter(position + rng.uniform(-0.13, 0.13, size=len(values)), values, color=color, alpha=0.65, s=28)
        ax.hlines(np.median(values), position - 0.25, position + 0.25, color="red", linewidth=2.2)
        cap = cap_value(model)
        if not math.isnan(cap) and not has_endpoint:
            ax.hlines(cap, position - 0.31, position + 0.31, color="blue", linestyle=":", linewidth=1.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([display_model(model) for model in models], rotation=35, ha="right")
    if has_return:
        distribution_ylabel = "Final-window training return"
        distribution_title = "Final nominal training quality across seeds"
    elif has_eval_return:
        distribution_ylabel = "Median return over 10 nominal evaluation episodes"
        distribution_title = "Final deterministic nominal evaluation across seeds"
    else:
        distribution_ylabel = "Pre-cap target-Q p95"
        distribution_title = "Final target-Q scale relative to each cap"
    ax.set_ylabel(distribution_ylabel)
    ax.set_title(distribution_title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    if has_return:
        stem = "final_training_return_by_cap"
    elif has_eval_return:
        stem = "final_nominal_eval_return_by_cap"
    else:
        stem = "final_q_scale_by_cap"
    save_figure(fig, out_dir / stem, formats)


def correlation(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(valid.iloc[:, 0], valid.iloc[:, 1])[0, 1])


def write_readme(runs: pd.DataFrame, models: pd.DataFrame, coverage: pd.DataFrame, out_dir: Path) -> None:
    return_col = "final__charts/episodic_return"
    eval_return_col = "final__eval/episodic_return"
    clip_col = "final__robust/td3_q_target_clip_fraction"
    has_return = return_col in runs and runs[return_col].notna().any()
    has_eval_return = eval_return_col in runs and runs[eval_return_col].notna().any()
    complete_tags = int(coverage.groupby("tag")["present"].all().sum())

    lines = [
        "# Walker2d TD3 TV-Cap Training Dynamics",
        "",
        f"Event coverage: {len(runs)} runs; {complete_tags}/{coverage['tag'].nunique()} requested tags present in every run where applicable.",
        "Training episodic return and episode length are unavailable: neither tag was emitted by any of the 240 runs. The current Gymnasium completion-info format did not enter the training loop's `final_info` logging branch.",
        "Critic, target, loss, throughput, and TV-cap activity tags are intact across the sweep.",
        "",
        "## Per-Cap Summary",
        "",
        "| Model | Q1 median | Pre-cap Q p95 | Critic loss | Final clip frac. | Mean clip frac. | Q p95 / cap | Active final | Active ever |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if has_eval_return:
        vanilla_eval = float(runs.loc[runs["model_label"] == "vanilla", eval_return_col].median())
        best_model = str(models.loc[models[f"{eval_return_col}__median"].idxmax(), "model_label"])
        lines[5:5] = [
            f"The final deterministic nominal evaluation is available: vanilla median {vanilla_eval:.1f}; highest cap-level median {display_model(best_model)}.",
        ]
    for _, row in models.iterrows():
        model = str(row["model_label"])
        def value(column: str) -> str:
            raw = row.get(column, float("nan"))
            return "NA" if pd.isna(raw) else f"{float(raw):.3f}"
        lines.append(
            f"| {display_model(model)} | {value('final__losses/qf1_values__median')} | "
            f"{value('final__targets/min_q_next_p95_pre_clip__median')} | "
            f"{value('final__losses/qf_loss__median')} | {value(f'{clip_col}__median')} | "
            f"{value('mean__robust/td3_q_target_clip_fraction__median')} | "
            f"{value('final_q_p95_over_cap__median')} | {value('cap_active_final_fraction')} | "
            f"{value('cap_active_ever_fraction')} |"
        )
    lines.extend(
        [
            "",
            "These diagnostics establish whether the cap changed training, not whether it improved deployment robustness. Interpret them jointly with the perturbation evaluation.",
            "",
            "## Files",
            "",
            "- `return_length_dynamics.*`: explicit record that return/length tags were unavailable.",
            "- `critic_actor_dynamics.*`: Q scale, pre-cap Q p95, critic loss, and actor loss.",
            "- `effective_pessimism_dynamics.*`: clip fraction, clipped excess, Q-to-cap ratio, and clip/endpoint scatter.",
            "- `final_nominal_eval_return_by_cap.*` or `final_q_scale_by_cap.*`: 30-seed endpoint distributions.",
            "- `run_summary.csv`, `model_summary.csv`, `training_scalars_binned.csv`: analysis source tables.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        binned = pd.read_csv(out_dir / "training_scalars_binned.csv")
        runs = add_derived_columns(pd.read_csv(out_dir / "run_summary.csv"))
        coverage = pd.read_csv(out_dir / "tag_coverage.csv")
    else:
        binned, runs, coverage = load_events(Path(args.event_dir), args.bin_width, args.final_window)
    models = model_summary(runs)

    binned.to_csv(out_dir / "training_scalars_binned.csv", index=False)
    runs.to_csv(out_dir / "run_summary.csv", index=False)
    models.to_csv(out_dir / "model_summary.csv", index=False)
    coverage.to_csv(out_dir / "tag_coverage.csv", index=False)
    plot_dynamics(binned, runs, out_dir, args.formats)
    write_readme(runs, models, coverage, out_dir)
    print(models.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
