#!/usr/bin/env python3
"""Create the paper-facing TD3 TV-cap activity diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAP_MODELS = ["tvc100", "tvc150", "tvc200", "tvc225", "tvc250", "tvc275", "tvc300"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("sweeps/results/TD3_TVCap_Walker_full30_20260710_td3_tvcap_full30"),
    )
    return parser.parse_args()


def cap_value(model: str) -> float:
    return float(model.removeprefix("tvc"))


def summarize_trajectory(data: pd.DataFrame, tag: str) -> pd.DataFrame:
    sub = data[(data["tag"] == tag) & data["model_label"].isin(CAP_MODELS)].copy()
    return (
        sub.groupby(["model_label", "step"], as_index=False)["value"]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
    )


def draw_trajectory(ax, summary: pd.DataFrame, colors: dict[str, object], ylabel: str) -> None:
    for model in CAP_MODELS:
        sub = summary[summary["model_label"] == model].sort_values("step")
        x = sub["step"].to_numpy(dtype=float) / 1e6
        median = sub["median"].to_numpy(dtype=float)
        q25 = sub["q25"].to_numpy(dtype=float)
        q75 = sub["q75"].to_numpy(dtype=float)
        ax.plot(x, median, color=colors[model], linewidth=2, label=f"$c={int(cap_value(model))}$")
        ax.fill_between(x, q25, q75, color=colors[model], alpha=0.10)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)


def main() -> None:
    args = parse_args()
    dynamics_dir = args.results_dir / "analysis_plots/training_dynamics/Walker2d-v4"
    data = pd.read_csv(dynamics_dir / "training_scalars_binned.csv")
    runs = pd.read_csv(dynamics_dir / "run_summary.csv")

    cmap = plt.get_cmap("viridis")
    colors = {model: cmap((index + 1) / (len(CAP_MODELS) + 1)) for index, model in enumerate(CAP_MODELS)}

    clip = summarize_trajectory(data, "robust/td3_q_target_clip_fraction")
    q95 = summarize_trajectory(data, "targets/min_q_next_p95_pre_clip")
    q95["cap"] = q95["model_label"].map(cap_value)
    for column in ["median", "q25", "q75"]:
        q95[column] = q95[column] / q95["cap"]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    draw_trajectory(axes[0], clip, colors, "ClipFraction")
    axes[0].set_title("How often the TD target is clipped")
    axes[0].set_ylim(-0.015, 0.68)

    draw_trajectory(axes[1], q95, colors, r"Pre-cap $Q^-_{0.95}/c$")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_title("Learned value scale relative to the cap")

    robust_runs = runs[runs["model_label"].isin(CAP_MODELS)].copy()
    robust_runs["cap"] = robust_runs["model_label"].map(cap_value)
    active_counts = robust_runs.groupby("model_label")["cap_active_ever"].sum().to_dict()
    rng = np.random.default_rng(7)
    for model in CAP_MODELS:
        sub = robust_runs[robust_runs["model_label"] == model]
        x = cap_value(model) + rng.uniform(-3.2, 3.2, size=len(sub))
        y = sub["mean__robust/td3_q_target_clip_fraction"].clip(lower=1e-6)
        axes[2].scatter(x, y, color=colors[model], alpha=0.62, s=25, edgecolor="none")
        axes[2].scatter(
            [cap_value(model)],
            [max(float(y.median()), 1e-6)],
            color=colors[model],
            edgecolor="black",
            linewidth=0.7,
            marker="D",
            s=52,
            zorder=4,
        )
    axes[2].set_yscale("log")
    axes[2].set_xticks(
        [cap_value(model) for model in CAP_MODELS],
        [f"{int(cap_value(model))}\n{int(active_counts.get(model, 0))}/30" for model in CAP_MODELS],
    )
    axes[2].set_xlabel("Cap $c$\n(seeds ever exceeding 1% ClipFraction)")
    axes[2].set_ylabel("Run-average ClipFraction")
    axes[2].set_title("The same cap induces seed-dependent activity")
    axes[2].grid(alpha=0.22, which="both")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Walker2d TD3 TV-cap training diagnostics across 30 seeds", y=0.995, fontsize=16)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(CAP_MODELS),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))

    out_dir = args.results_dir / "analysis_plots/paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(out_dir / f"cap_activity_dynamics.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(out_dir / "cap_activity_dynamics.pdf")


if __name__ == "__main__":
    main()
