#!/usr/bin/env python3
"""Plot vanilla PPO training dynamics split by final seed-quality group."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


TAGS = [
    "charts/episodic_return",
    "charts/episodic_length",
    "losses/entropy",
    "losses/explained_variance",
    "losses/clipfrac",
    "losses/approx_kl",
    "values/p99",
    "targets/returns_p99_pre_transform",
]

GROUP_ORDER = ["weak vanilla", "middle", "elite vanilla"]
GROUP_COLORS = {
    "weak vanilla": "#d95f02",
    "middle": "#7570b3",
    "elite vanilla": "#1b9e77",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-dir", required=True, help="Directory containing copied TensorBoard event files.")
    parser.add_argument("--seed-summary-csv", required=True, help="Seed-conditioned summary CSV with seed/group columns.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--env-label", required=True, help="Display label, e.g. Walker2d or HalfCheetah.")
    parser.add_argument("--bin-width", type=int, default=25_000)
    parser.add_argument("--milestones", nargs="+", type=float, default=[1000, 2000, 3000, 4000])
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    text = str(path)
    match = re.search(r"__ppo_alpha_vanilla__(\d+)__", text)
    if not match:
        match = re.search(r"__vanilla__(\d+)__", text)
    if not match:
        raise ValueError(f"Could not infer seed from {path}")
    return int(match.group(1))


def load_events(event_dir: Path, summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    event_paths = sorted(event_dir.rglob("events.out.tfevents*"))
    if not event_paths:
        raise FileNotFoundError(f"No TensorBoard event files found below {event_dir}")

    seed_meta = summary.set_index("seed")[["group", "vanilla_nominal"]].to_dict("index")
    for event_path in event_paths:
        seed = seed_from_path(event_path)
        if seed not in seed_meta:
            continue
        accumulator = event_accumulator.EventAccumulator(
            str(event_path),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        for tag in TAGS:
            if tag not in available:
                continue
            for event in accumulator.Scalars(tag):
                rows.append(
                    {
                        "seed": seed,
                        "group": seed_meta[seed]["group"],
                        "vanilla_nominal": seed_meta[seed]["vanilla_nominal"],
                        "tag": tag,
                        "step": int(event.step),
                        "value": float(event.value),
                    }
                )
    if not rows:
        raise RuntimeError("No scalar rows were loaded from TensorBoard event files.")
    return pd.DataFrame(rows)


def bin_scalars(df: pd.DataFrame, bin_width: int) -> pd.DataFrame:
    out = df.copy()
    out["step_bin"] = (np.floor(out["step"] / bin_width) * bin_width).astype(int)
    return (
        out.groupby(["seed", "group", "vanilla_nominal", "tag", "step_bin"], as_index=False)["value"]
        .median()
        .rename(columns={"step_bin": "step"})
    )


def group_summary(binned: pd.DataFrame, tag: str) -> pd.DataFrame:
    sub = binned[binned["tag"] == tag]
    return (
        sub.groupby(["group", "step"], as_index=False)["value"]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
    )


def plot_grouped_panel(
    binned: pd.DataFrame,
    tags: list[str],
    titles: list[str],
    ylabels: list[str],
    out_stem: Path,
    formats: list[str],
    env_label: str,
    ncols: int,
) -> None:
    nrows = int(np.ceil(len(tags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.4 * nrows), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)

    for ax, tag, title, ylabel in zip(axes_arr, tags, titles, ylabels):
        sub = binned[binned["tag"] == tag]
        summary = group_summary(binned, tag)
        for group in GROUP_ORDER:
            color = GROUP_COLORS[group]
            gseed = sub[sub["group"] == group]
            for _, seed_sub in gseed.groupby("seed"):
                seed_sub = seed_sub.sort_values("step")
                ax.plot(seed_sub["step"], seed_sub["value"], color=color, alpha=0.14, linewidth=1.0)

            gs = summary[summary["group"] == group].sort_values("step")
            if gs.empty:
                continue
            x = gs["step"].to_numpy(dtype=float)
            median = gs["median"].to_numpy(dtype=float)
            q25 = gs["q25"].to_numpy(dtype=float)
            q75 = gs["q75"].to_numpy(dtype=float)
            ax.plot(x, median, color=color, linewidth=2.8, label=group)
            ax.fill_between(x, q25, q75, color=color, alpha=0.14)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22)

    for ax in axes_arr[len(tags) :]:
        ax.axis("off")
    for ax in axes_arr[-ncols:]:
        ax.set_xlabel("Environment steps")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(env_label, y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_stem.with_suffix(f".{fmt}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def first_crossing(sub: pd.DataFrame, threshold: float) -> float:
    hit = sub[sub["value"] >= threshold]
    if hit.empty:
        return np.nan
    return float(hit["step"].min())


def write_milestones(
    binned: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    milestones: list[float],
) -> pd.DataFrame:
    ret = binned[binned["tag"] == "charts/episodic_return"].copy()
    rows: list[dict[str, object]] = []
    for _, seed_row in summary.sort_values("seed").iterrows():
        seed = int(seed_row["seed"])
        seed_ret = ret[ret["seed"] == seed].sort_values("step")
        for threshold in milestones:
            rows.append(
                {
                    "seed": seed,
                    "group": seed_row["group"],
                    "vanilla_nominal": seed_row["vanilla_nominal"],
                    "threshold": int(threshold) if float(threshold).is_integer() else threshold,
                    "first_step": first_crossing(seed_ret, threshold),
                }
            )
        last = seed_ret[seed_ret["step"] >= seed_ret["step"].max() - 100_000]
        rows.append(
            {
                "seed": seed,
                "group": seed_row["group"],
                "vanilla_nominal": seed_row["vanilla_nominal"],
                "threshold": "last100k_median_return",
                "first_step": float(last["value"].median()) if not last.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "vanilla_return_milestones_by_seed.csv", index=False)
    return out


def summarize_milestones(milestones: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        sub = milestones[milestones["group"] == group]
        if sub.empty:
            continue
        for threshold, tsub in sub.groupby("threshold", sort=False):
            values = pd.to_numeric(tsub["first_step"], errors="coerce")
            rows.append(
                {
                    "group": group,
                    "threshold": threshold,
                    "n_seeds": int(tsub["seed"].nunique()),
                    "n_reached": int(values.notna().sum()),
                    "median_value": float(values.median()) if values.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def format_seed_list(summary: pd.DataFrame, group: str) -> str:
    seeds = summary[summary["group"] == group]["seed"].astype(int).sort_values().tolist()
    return ", ".join(str(seed) for seed in seeds)


def write_readme(
    out_dir: Path,
    env_label: str,
    summary: pd.DataFrame,
    milestones: pd.DataFrame,
    milestone_summary: pd.DataFrame,
) -> None:
    group_counts = summary["group"].value_counts().to_dict()
    lines = [
        f"# {env_label} Vanilla Learning Dynamics",
        "",
        "Grouping source: `../seed_conditioned_effect/*/vanilla_nominal_vs_robust_gain_summary.csv`.",
        "The plotted runs are vanilla PPO training scalars, grouped by final vanilla nominal evaluation quality.",
        "",
        "## Groups",
    ]
    for group in GROUP_ORDER:
        sub = summary[summary["group"] == group]
        if sub.empty:
            continue
        lines.extend(
            [
                f"- {group}: {group_counts.get(group, 0)} seeds; seeds {format_seed_list(summary, group)}; "
                f"vanilla nominal range {sub['vanilla_nominal'].min():.1f} to {sub['vanilla_nominal'].max():.1f}.",
            ]
        )

    lines.extend(["", "## Milestone Summary"])
    for _, row in milestone_summary.iterrows():
        threshold = row["threshold"]
        label = (
            "last 100k median return"
            if str(threshold) == "last100k_median_return"
            else f"first step reaching return >= {threshold}"
        )
        median = row["median_value"]
        median_text = "not reached" if pd.isna(median) else f"{median:.0f}"
        lines.append(
            f"- {row['group']}: {label}; {int(row['n_reached'])}/{int(row['n_seeds'])} seeds; median {median_text}."
        )

    lines.extend(
        [
            "",
            "## Files",
            "- `vanilla_return_length_by_final_quality.png/pdf`: training episodic return and length by group.",
            "- `vanilla_optimizer_value_by_final_quality.png/pdf`: optimizer/value diagnostics by group.",
            "- `vanilla_training_scalars_selected.csv`: extracted TensorBoard scalar rows.",
            "- `vanilla_return_milestones_by_seed.csv`: per-seed milestone crossings.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.seed_summary_csv)
    summary["seed"] = summary["seed"].astype(int)
    required = {"seed", "group", "vanilla_nominal"}
    missing = required - set(summary.columns)
    if missing:
        raise SystemExit(f"Seed summary is missing columns: {sorted(missing)}")

    raw = load_events(Path(args.event_dir), summary)
    raw.to_csv(out_dir / "vanilla_training_scalars_selected.csv", index=False)
    binned = bin_scalars(raw, args.bin_width)

    plot_grouped_panel(
        binned,
        tags=["charts/episodic_return", "charts/episodic_length"],
        titles=["Training episodic return", "Episode length"],
        ylabels=["Training episodic return", "Episode length"],
        out_stem=out_dir / "vanilla_return_length_by_final_quality",
        formats=args.formats,
        env_label=f"{args.env_label} vanilla PPO training dynamics split by final vanilla quality",
        ncols=1,
    )

    plot_grouped_panel(
        binned,
        tags=[
            "losses/entropy",
            "losses/explained_variance",
            "losses/clipfrac",
            "losses/approx_kl",
            "values/p99",
            "targets/returns_p99_pre_transform",
        ],
        titles=[
            "Entropy",
            "Explained variance",
            "PPO clip fraction",
            "Approx KL",
            "Value p99",
            "Return-target p99",
        ],
        ylabels=["", "", "", "", "", ""],
        out_stem=out_dir / "vanilla_optimizer_value_by_final_quality",
        formats=args.formats,
        env_label=f"{args.env_label} vanilla PPO optimizer/value dynamics by final vanilla quality",
        ncols=3,
    )

    milestones = write_milestones(binned, summary, out_dir, args.milestones)
    milestone_summary = summarize_milestones(milestones)
    milestone_summary.to_csv(out_dir / "vanilla_return_milestone_summary.csv", index=False)
    write_readme(out_dir, args.env_label, summary, milestones, milestone_summary)


if __name__ == "__main__":
    main()
