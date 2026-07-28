#!/usr/bin/env python3
"""Score a focused TD3 robustness evaluation with catastrophe-aware metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NOMINAL_FACTOR = {
    "mass": 1.0,
    "actuator_gain": 1.0,
    "friction_mass_damping": 1.0,
    "action_replace": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-model-label", default="vanilla")
    parser.add_argument("--min-nominal-retention", type=float, default=0.70)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def parse_scenario(label: str) -> tuple[str, float]:
    axis, factor = str(label).rsplit("_", 1)
    return axis, float(factor.replace("p", ".").replace("m", "-"))


def lower_tail_mean(values: pd.Series, fraction: float = 0.2) -> float:
    ordered = np.sort(values.to_numpy(dtype=float))
    count = max(1, int(np.ceil(fraction * len(ordered))))
    return float(np.mean(ordered[:count]))


def add_reference_columns(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    parsed = df["scenario_label"].map(parse_scenario)
    df = df.copy()
    df["axis"] = parsed.map(lambda value: value[0])
    df["factor"] = parsed.map(lambda value: value[1])
    df["nominal_factor"] = df["axis"].map(NOMINAL_FACTOR)
    if df["nominal_factor"].isna().any():
        missing = sorted(df.loc[df["nominal_factor"].isna(), "axis"].unique())
        raise ValueError(f"Missing nominal factors for axes: {missing}")
    df["is_nominal"] = np.isclose(df["factor"], df["nominal_factor"])

    nominal = (
        df[(df["model_label"] == baseline) & df["is_nominal"]]
        .groupby("axis")["mean_return"]
        .median()
    )
    if nominal.empty:
        raise ValueError(f"No nominal rows for baseline {baseline!r}")
    df["vanilla_nominal_median"] = df["axis"].map(nominal)
    df["normalized_return"] = df["mean_return"] / df["vanilla_nominal_median"]

    paired = df[df["model_label"] == baseline][
        ["seed", "scenario_label", "mean_return"]
    ].rename(columns={"mean_return": "same_seed_vanilla_scenario"})
    df = df.merge(paired, on=["seed", "scenario_label"], how="left", validate="many_to_one")
    df["paired_gain"] = df["mean_return"] - df["same_seed_vanilla_scenario"]
    return df


def reliability_auc(normalized_returns: pd.Series) -> float:
    values = np.clip(normalized_returns.to_numpy(dtype=float), 0.0, 1.0)
    return float(np.mean(values))


def model_summary(
    df: pd.DataFrame,
    min_retention: float,
    baseline_model_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, sub in df.groupby("model_label"):
        nominal = sub[sub["is_nominal"]]["normalized_return"]
        shifted = sub[~sub["is_nominal"]]
        normalized = shifted["normalized_return"]
        row = {
            "model_label": model,
            "n_seeds": sub["seed"].nunique(),
            "nominal_retention_median": nominal.median(),
            "nominal_retention_min": nominal.min(),
            "shifted_return_median": normalized.median(),
            "shifted_return_mean": normalized.mean(),
            "shifted_return_cvar20": lower_tail_mean(normalized),
            "shifted_return_min": normalized.min(),
            "reliability_auc": reliability_auc(normalized),
            "catastrophe_prob_t0p25": float((normalized < 0.25).mean()),
            "catastrophe_prob_t0p50": float((normalized < 0.50).mean()),
            "catastrophe_prob_t0p75": float((normalized < 0.75).mean()),
            "paired_gain_mean": shifted["paired_gain"].mean(),
            "paired_gain_median": shifted["paired_gain"].median(),
            "paired_win_rate": float((shifted["paired_gain"] > 0).mean()),
        }
        row["balanced_score"] = min(
            row["nominal_retention_median"],
            row["reliability_auc"],
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    baseline = summary.loc[
        summary["model_label"] == baseline_model_label
    ].iloc[0]
    summary["delta_reliability_auc_vs_vanilla"] = (
        summary["reliability_auc"] - baseline["reliability_auc"]
    )
    summary["delta_cvar20_vs_vanilla"] = (
        summary["shifted_return_cvar20"] - baseline["shifted_return_cvar20"]
    )
    summary["delta_catastrophe_t0p5_vs_vanilla"] = (
        summary["catastrophe_prob_t0p50"] - baseline["catastrophe_prob_t0p50"]
    )
    summary["robustness_gate"] = (
        (summary["model_label"] != baseline_model_label)
        & (summary["nominal_retention_median"] >= min_retention)
        & (summary["delta_reliability_auc_vs_vanilla"] > 0)
        & (summary["delta_cvar20_vs_vanilla"] > 0)
        & (summary["delta_catastrophe_t0p5_vs_vanilla"] < 0)
    )
    return summary.sort_values(
        ["robustness_gate", "balanced_score"],
        ascending=[False, False],
    )


def seed_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, seed), sub in df.groupby(["model_label", "seed"]):
        nominal = sub[sub["is_nominal"]]["normalized_return"]
        axis_rows = []
        for _, axis_df in sub[~sub["is_nominal"]].groupby("axis"):
            normalized = axis_df["normalized_return"]
            axis_rows.append(
                {
                    "reliability_auc": reliability_auc(normalized),
                    "shifted_return_cvar20": lower_tail_mean(normalized),
                    "catastrophe_prob_t0p50": float((normalized < 0.5).mean()),
                }
            )
        axis_metrics = pd.DataFrame(axis_rows)
        rows.append(
            {
                "model_label": model,
                "seed": int(seed),
                "nominal_retention": nominal.median(),
                **axis_metrics.mean().to_dict(),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_seed_deltas(
    seed_summary: pd.DataFrame,
    baseline_model_label: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = seed_summary[
        seed_summary["model_label"] == baseline_model_label
    ]
    rows: list[dict[str, object]] = []
    for model, model_df in seed_summary.groupby("model_label"):
        if model == baseline_model_label:
            continue
        baseline_idx = rng.integers(
            0, len(baseline), size=(replicates, len(baseline))
        )
        model_idx = rng.integers(
            0, len(model_df), size=(replicates, len(model_df))
        )
        row: dict[str, object] = {
            "model_label": model,
            "n_model_seeds": len(model_df),
            "n_baseline_seeds": len(baseline),
            "bootstrap_replicates": replicates,
        }
        for metric in [
            "nominal_retention",
            "reliability_auc",
            "shifted_return_cvar20",
            "catastrophe_prob_t0p50",
        ]:
            model_values = model_df[metric].to_numpy(dtype=float)
            baseline_values = baseline[metric].to_numpy(dtype=float)
            deltas = (
                model_values[model_idx].mean(axis=1)
                - baseline_values[baseline_idx].mean(axis=1)
            )
            row[f"delta__{metric}"] = model_values.mean() - baseline_values.mean()
            row[f"delta__{metric}__ci95_low"] = np.quantile(deltas, 0.025)
            row[f"delta__{metric}__ci95_high"] = np.quantile(deltas, 0.975)
            if metric.startswith("catastrophe_prob"):
                row[f"probability_metric_improves__{metric}"] = float(
                    (deltas < 0).mean()
                )
            else:
                row[f"probability_metric_improves__{metric}"] = float(
                    (deltas > 0).mean()
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "delta__reliability_auc",
        ascending=False,
    )


def axis_summary(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df[~df["is_nominal"]]
    return (
        shifted.groupby(["axis", "model_label"])
        .agg(
            normalized_return_median=("normalized_return", "median"),
            normalized_return_mean=("normalized_return", "mean"),
            paired_gain_mean=("paired_gain", "mean"),
            paired_win_rate=("paired_gain", lambda values: (values > 0).mean()),
            catastrophe_prob_t0p5=("normalized_return", lambda values: (values < 0.5).mean()),
        )
        .reset_index()
    )


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    for extension in formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
) -> None:
    models = ["vanilla"] + sorted(
        model for model in df["model_label"].unique() if model != "vanilla"
    )
    colors = dict(zip(models, plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(models)))))

    axes_names = list(dict.fromkeys(df["axis"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for axis, axis_name in zip(axes.flat, axes_names):
        axis_df = df[df["axis"] == axis_name]
        curves = (
            axis_df.groupby(["model_label", "factor"])["mean_return"]
            .median()
            .reset_index()
        )
        for model in models:
            model_df = curves[curves["model_label"] == model].sort_values("factor")
            axis.plot(
                model_df["factor"],
                model_df["mean_return"],
                marker="o",
                color=colors[model],
                linestyle="--" if model == "vanilla" else "-",
                label=model,
            )
        axis.set_title(axis_name)
        axis.set_xlabel("Perturbation factor")
        axis.set_ylabel("Evaluation return")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Focused Walker2d TD3 robustness evaluation")
    fig.tight_layout()
    save_figure(fig, out_dir / "focused_return_curves", formats)

    thresholds = np.linspace(0, 1, 201)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    shifted = df[~df["is_nominal"]]
    for model in models:
        values = shifted.loc[
            shifted["model_label"] == model,
            "normalized_return",
        ].to_numpy(dtype=float)
        survival = [(values >= threshold).mean() for threshold in thresholds]
        axis.plot(
            thresholds,
            survival,
            color=colors[model],
            linestyle="--" if model == "vanilla" else "-",
            label=model,
        )
    axis.set_xlabel(r"Catastrophe threshold $t$ relative to vanilla nominal median")
    axis.set_ylabel(r"Reliability $P(R \geq tR_0)$")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir / "reliability_curves", formats)

    fig, axis = plt.subplots(figsize=(6.5, 4.8))
    for _, row in summary.iterrows():
        model = str(row["model_label"])
        axis.scatter(
            row["nominal_retention_median"],
            row["reliability_auc"],
            color=colors[model],
            s=60,
        )
        axis.annotate(model, (row["nominal_retention_median"], row["reliability_auc"]), xytext=(4, 4), textcoords="offset points")
    axis.axvline(1.0, color="black", linestyle=":", linewidth=1)
    axis.axhline(
        summary.loc[summary["model_label"] == "vanilla", "reliability_auc"].iloc[0],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    axis.set_xlabel("Median nominal retention")
    axis.set_ylabel("Reliability AUC")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_dir / "nominal_reliability_pareto", formats)

    ordered = bootstrap.sort_values("delta__reliability_auc")
    fig, axis = plt.subplots(figsize=(7.0, max(3.8, 0.5 * len(ordered))))
    y = np.arange(len(ordered))
    estimate = ordered["delta__reliability_auc"].to_numpy(dtype=float)
    low = ordered["delta__reliability_auc__ci95_low"].to_numpy(dtype=float)
    high = ordered["delta__reliability_auc__ci95_high"].to_numpy(dtype=float)
    axis.errorbar(
        estimate,
        y,
        xerr=[estimate - low, high - estimate],
        fmt="o",
        capsize=3,
    )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(y, ordered["model_label"])
    axis.set_xlabel("Reliability AUC delta vs vanilla (95% seed bootstrap CI)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_dir / "reliability_delta_bootstrap", formats)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.metrics_csv).expanduser())
    df = add_reference_columns(df, args.baseline_model_label)
    summary = model_summary(
        df,
        args.min_nominal_retention,
        args.baseline_model_label,
    )
    axes = axis_summary(df)
    seeds = seed_level_summary(df)
    bootstrap = bootstrap_seed_deltas(
        seeds,
        args.baseline_model_label,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    df.to_csv(out_dir / "metrics_with_robustness_columns.csv", index=False)
    summary.to_csv(out_dir / "model_robustness_summary.csv", index=False)
    axes.to_csv(out_dir / "axis_robustness_summary.csv", index=False)
    seeds.to_csv(out_dir / "seed_level_robustness_summary.csv", index=False)
    bootstrap.to_csv(out_dir / "model_delta_seed_bootstrap.csv", index=False)
    make_plots(df, summary, bootstrap, out_dir, args.formats)

    promoted = summary.loc[summary["robustness_gate"], "model_label"].tolist()
    report = [
        "# Focused TD3 robustness gate",
        "",
        f"Promoted: {', '.join(promoted) if promoted else 'none'}",
        "",
        "A variant passes only if it preserves median nominal return, improves",
        "reliability AUC and lower-tail return, and reduces catastrophic failures",
        "at the predeclared 0.5 threshold relative to vanilla nominal.",
        "Reliability metrics use the vanilla nominal median across seeds.",
        "Same-seed gains are reported only as a common-random-number diagnostic;",
        "they are not used by the promotion gate.",
        "Independent-seed bootstrap intervals are reported separately. With only",
        "three pilot seeds, they diagnose uncertainty rather than establish a claim.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
