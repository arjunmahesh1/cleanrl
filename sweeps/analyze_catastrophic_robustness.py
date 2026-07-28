#!/usr/bin/env python3
"""Quantify axis-balanced reliability, catastrophe risk, and seed fragility."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIGNAL_AXES = {"action_noise", "action_replace", "action_noise_bernoulli", "state_noise"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-model-label", default="vanilla")
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def parse_scenario(label: str) -> tuple[str, float]:
    axis, token = str(label).rsplit("_", 1)
    sign = -1.0 if token.startswith("m") else 1.0
    if token.startswith("m"):
        token = token[1:]
    return axis, sign * float(token.replace("p", "."))


def model_key(label: str) -> tuple[int, float, str]:
    if label == "vanilla":
        return 0, -1.0, label
    match = re.search(r"(-?\d+)(?:p(\d+))?$", label)
    if not match:
        return 2, math.inf, label
    fraction = match.group(2) or "0"
    return 1, float(f"{match.group(1)}.{fraction}"), label


def nominal_factor(axis: str) -> float:
    return 0.0 if axis in SIGNAL_AXES else 1.0


def lower_tail_mean(values: pd.Series, fraction: float) -> float:
    array = np.sort(values.to_numpy(dtype=float))
    count = max(1, int(np.ceil(fraction * len(array))))
    return float(array[:count].mean())


def prepare(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    df = df.copy()
    if "axis" not in df or "factor" not in df:
        parsed = df["scenario_label"].map(parse_scenario)
        df["axis"] = parsed.map(lambda value: value[0])
        df["factor"] = parsed.map(lambda value: value[1])
    df["factor"] = df["factor"].astype(float)
    df["seed"] = df["seed"].astype(int)
    df["mean_return"] = df["mean_return"].astype(float)
    df["nominal_factor"] = df["axis"].map(nominal_factor)
    df["is_nominal"] = np.isclose(df["factor"], df["nominal_factor"])

    references = (
        df[(df["model_label"] == baseline) & df["is_nominal"]]
        .groupby(["env_id", "axis"])["mean_return"]
        .median()
        .rename("vanilla_nominal_median")
    )
    df = df.join(references, on=["env_id", "axis"])
    if df["vanilla_nominal_median"].isna().any():
        raise ValueError("Missing vanilla nominal reference for at least one axis")
    df["normalized_return"] = df["mean_return"] / df["vanilla_nominal_median"]
    df["clipped_normalized_return"] = df["normalized_return"].clip(0.0, 1.0)
    return df


def per_axis_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (env, axis, model), sub in df.groupby(["env_id", "axis", "model_label"]):
        nominal = sub[sub["is_nominal"]]["normalized_return"]
        shifted = sub[~sub["is_nominal"]]
        values = shifted["normalized_return"]
        row = {
            "env_id": env,
            "axis": axis,
            "model_label": model,
            "n_shifted_rows": len(shifted),
            "nominal_retention": nominal.median(),
            "reliability_auc": shifted["clipped_normalized_return"].mean(),
            "return_cvar10": lower_tail_mean(values, 0.10),
            "return_cvar20": lower_tail_mean(values, 0.20),
            "return_q10": values.quantile(0.10),
            "return_min": values.min(),
        }
        for threshold in [0.25, 0.50, 0.75]:
            row[f"catastrophe_prob_t{str(threshold).replace('.', 'p')}"] = float(
                (values < threshold).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_models(axis_summary: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [
        "nominal_retention",
        "reliability_auc",
        "return_cvar10",
        "return_cvar20",
        "return_q10",
        "return_min",
        "catastrophe_prob_t0p25",
        "catastrophe_prob_t0p5",
        "catastrophe_prob_t0p75",
    ]
    for (env, model), sub in axis_summary.groupby(["env_id", "model_label"]):
        row: dict[str, object] = {
            "env_id": env,
            "model_label": model,
            "n_axes": sub["axis"].nunique(),
        }
        for metric in metrics:
            row[f"axis_mean__{metric}"] = sub[metric].mean()
            row[f"axis_median__{metric}"] = sub[metric].median()
            row[f"worst_axis__{metric}"] = (
                sub[metric].max() if metric.startswith("catastrophe") else sub[metric].min()
            )
        row["balanced_reliability_score"] = min(
            row["axis_median__nominal_retention"],
            row["axis_mean__reliability_auc"],
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    for env, sub in summary.groupby("env_id"):
        baseline_row = sub[sub["model_label"] == baseline].iloc[0]
        mask = summary["env_id"] == env
        summary.loc[mask, "delta_reliability_auc_vs_vanilla"] = (
            summary.loc[mask, "axis_mean__reliability_auc"]
            - baseline_row["axis_mean__reliability_auc"]
        )
        summary.loc[mask, "delta_catastrophe_t0p5_vs_vanilla"] = (
            summary.loc[mask, "axis_mean__catastrophe_prob_t0p5"]
            - baseline_row["axis_mean__catastrophe_prob_t0p5"]
        )
        summary.loc[mask, "delta_cvar20_vs_vanilla"] = (
            summary.loc[mask, "axis_mean__return_cvar20"]
            - baseline_row["axis_mean__return_cvar20"]
        )
    return summary.sort_values(
        ["env_id", "balanced_reliability_score"],
        ascending=[True, False],
    )


def variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose robustness variance into seed, perturbation, and interaction terms."""
    rows: list[dict[str, object]] = []
    shifted = df[~df["is_nominal"]]
    for (env, axis, model), sub in shifted.groupby(["env_id", "axis", "model_label"]):
        matrix = sub.pivot_table(
            index="seed",
            columns="factor",
            values="normalized_return",
            aggfunc="mean",
        )
        matrix = matrix.dropna(axis=0, how="any").dropna(axis=1, how="any")
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            continue
        values = matrix.to_numpy(dtype=float)
        grand = values.mean()
        seed_effect = values.mean(axis=1, keepdims=True) - grand
        shift_effect = values.mean(axis=0, keepdims=True) - grand
        interaction = values - grand - seed_effect - shift_effect
        seed_var = float(np.mean(seed_effect**2))
        shift_var = float(np.mean(shift_effect**2))
        interaction_var = float(np.mean(interaction**2))
        total = seed_var + shift_var + interaction_var
        rows.append(
            {
                "env_id": env,
                "axis": axis,
                "model_label": model,
                "n_seeds": matrix.shape[0],
                "n_factors": matrix.shape[1],
                "seed_main_variance": seed_var,
                "perturbation_main_variance": shift_var,
                "seed_perturbation_interaction_variance": interaction_var,
                "total_decomposed_variance": total,
                "seed_fraction": seed_var / total if total > 0 else np.nan,
                "perturbation_fraction": shift_var / total if total > 0 else np.nan,
                "interaction_fraction": interaction_var / total if total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def seed_level_axis_balanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df[~df["is_nominal"]]
    shifted_by_axis = (
        shifted.groupby(["env_id", "model_label", "seed", "axis"])
        .agg(
            reliability_auc=("clipped_normalized_return", "mean"),
            catastrophe_prob_t0p5=(
                "normalized_return",
                lambda values: (values < 0.5).mean(),
            ),
        )
        .reset_index()
    )
    shifted_by_seed = (
        shifted_by_axis.groupby(["env_id", "model_label", "seed"])[
            ["reliability_auc", "catastrophe_prob_t0p5"]
        ]
        .mean()
        .reset_index()
    )
    nominal_by_seed = (
        df[df["is_nominal"]]
        .groupby(["env_id", "model_label", "seed"])["normalized_return"]
        .median()
        .rename("nominal_retention")
        .reset_index()
    )
    return shifted_by_seed.merge(
        nominal_by_seed,
        on=["env_id", "model_label", "seed"],
        validate="one_to_one",
    )


def bootstrap_deltas(
    seed_metrics: pd.DataFrame,
    baseline: str,
    replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, object]] = []
    for env, env_df in seed_metrics.groupby("env_id"):
        baseline_df = env_df[env_df["model_label"] == baseline]
        if baseline_df.empty:
            raise ValueError(f"No baseline seed metrics for {env}")
        for model, model_df in env_df.groupby("model_label"):
            if model == baseline:
                continue
            baseline_indices = rng.integers(
                0,
                len(baseline_df),
                size=(replicates, len(baseline_df)),
            )
            model_indices = rng.integers(
                0,
                len(model_df),
                size=(replicates, len(model_df)),
            )
            row: dict[str, object] = {
                "env_id": env,
                "model_label": model,
                "n_model_seeds": len(model_df),
                "n_baseline_seeds": len(baseline_df),
                "bootstrap_replicates": replicates,
            }
            for metric in [
                "reliability_auc",
                "catastrophe_prob_t0p5",
                "nominal_retention",
            ]:
                baseline_values = baseline_df[metric].to_numpy(dtype=float)
                model_values = model_df[metric].to_numpy(dtype=float)
                pairwise_difference = (
                    model_values[:, np.newaxis] - baseline_values[np.newaxis, :]
                )
                delta = (
                    model_values[model_indices].mean(axis=1)
                    - baseline_values[baseline_indices].mean(axis=1)
                )
                row[f"delta__{metric}"] = model_values.mean() - baseline_values.mean()
                row[f"delta__{metric}__ci95_low"] = np.quantile(delta, 0.025)
                row[f"delta__{metric}__ci95_high"] = np.quantile(delta, 0.975)
                row[f"prob_delta__{metric}__gt_0"] = float((delta > 0).mean())
                if metric.startswith("catastrophe_prob"):
                    row[f"prob_random_seed__{metric}__improves"] = float(
                        (pairwise_difference < 0).mean()
                    )
                else:
                    row[f"prob_random_seed__{metric}__improves"] = float(
                        (pairwise_difference > 0).mean()
                    )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "delta__reliability_auc",
        ascending=False,
    )


def regime_table(axis_summary: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (env, axis), sub in axis_summary.groupby(["env_id", "axis"]):
        vanilla = sub[sub["model_label"] == baseline].iloc[0]
        eligible = sub[
            (sub["model_label"] != baseline)
            & (sub["nominal_retention"] >= 0.8)
        ]
        if eligible.empty:
            best = vanilla
        else:
            best = eligible.sort_values(
                ["catastrophe_prob_t0p5", "reliability_auc"],
                ascending=[True, False],
            ).iloc[0]
        vanilla_failure = float(vanilla["catastrophe_prob_t0p5"])
        best_failure = float(best["catastrophe_prob_t0p5"])
        reduction = vanilla_failure - best_failure
        if vanilla_failure < 0.2:
            regime = "stable"
        elif reduction >= 0.1:
            regime = "recoverable failure"
        else:
            regime = "persistent failure"
        rows.append(
            {
                "env_id": env,
                "axis": axis,
                "selection_mode": "ex_post_axis_oracle_with_nominal_gate",
                "regime": regime,
                "vanilla_catastrophe_prob_t0p5": vanilla_failure,
                "best_eligible_model": best["model_label"],
                "best_catastrophe_prob_t0p5": best_failure,
                "catastrophe_probability_reduction": reduction,
                "best_nominal_retention": best["nominal_retention"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["env_id", "vanilla_catastrophe_prob_t0p5"],
        ascending=[True, False],
    )


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    for extension in formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    df: pd.DataFrame,
    axis_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    variance: pd.DataFrame,
    bootstrap: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
) -> None:
    for env, env_df in df.groupby("env_id"):
        models = sorted(env_df["model_label"].unique(), key=model_key)
        colors = dict(
            zip(models, plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(models))))
        )
        shifted = env_df[~env_df["is_nominal"]]
        thresholds = np.linspace(0.0, 1.0, 201)

        fig, axis = plt.subplots(figsize=(7.6, 5.0))
        for model in models:
            model_df = shifted[shifted["model_label"] == model]
            axis_curves = []
            for _, axis_df in model_df.groupby("axis"):
                values = axis_df["normalized_return"].to_numpy(dtype=float)
                axis_curves.append([(values >= threshold).mean() for threshold in thresholds])
            reliability = np.mean(np.asarray(axis_curves), axis=0)
            axis.plot(
                thresholds,
                reliability,
                color=colors[model],
                linestyle="--" if model == "vanilla" else "-",
                label=model,
            )
        axis.set_xlabel(r"Threshold $t$ relative to vanilla nominal median")
        axis.set_ylabel(r"Axis-balanced reliability $P(R \geq tR_0)$")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        save_figure(fig, out_dir / f"{env}_axis_balanced_reliability", formats)

        catastrophe = axis_summary[axis_summary["env_id"] == env].pivot(
            index="axis",
            columns="model_label",
            values="catastrophe_prob_t0p5",
        )
        catastrophe = catastrophe.reindex(columns=models)
        fig_height = max(4.5, 0.28 * len(catastrophe))
        fig, axis = plt.subplots(figsize=(10, fig_height))
        image = axis.imshow(catastrophe.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="magma_r")
        axis.set_xticks(range(len(models)), models, rotation=45, ha="right")
        axis.set_yticks(range(len(catastrophe)), catastrophe.index)
        axis.set_title("Catastrophic-failure probability at 50% nominal return")
        fig.colorbar(image, ax=axis, label="Failure probability")
        fig.tight_layout()
        save_figure(fig, out_dir / f"{env}_catastrophe_probability_heatmap", formats)

        variance_env = variance[variance["env_id"] == env]
        variance_model = (
            variance_env.groupby("model_label")[
                [
                    "seed_main_variance",
                    "perturbation_main_variance",
                    "seed_perturbation_interaction_variance",
                ]
            ]
            .mean()
            .reindex(models)
        )
        fig, axis = plt.subplots(figsize=(9, 4.8))
        bottom = np.zeros(len(variance_model))
        labels = [
            ("seed_main_variance", "Persistent seed quality"),
            ("perturbation_main_variance", "Perturbation severity"),
            ("seed_perturbation_interaction_variance", "Seed x perturbation fragility"),
        ]
        for column, label in labels:
            values = variance_model[column].to_numpy(dtype=float)
            axis.bar(models, values, bottom=bottom, label=label)
            bottom += values
        axis.set_ylabel("Axis-mean normalized-return variance")
        axis.tick_params(axis="x", rotation=45)
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        save_figure(fig, out_dir / f"{env}_robustness_variance_decomposition", formats)

        score = model_summary[model_summary["env_id"] == env]
        fig, axis = plt.subplots(figsize=(7.0, 4.8))
        for _, row in score.iterrows():
            model = row["model_label"]
            axis.scatter(
                row["axis_median__nominal_retention"],
                row["axis_mean__reliability_auc"],
                color=colors[model],
                s=55,
            )
            axis.annotate(
                model,
                (
                    row["axis_median__nominal_retention"],
                    row["axis_mean__reliability_auc"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("Axis-median nominal retention")
        axis.set_ylabel("Axis-balanced reliability AUC")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        save_figure(fig, out_dir / f"{env}_nominal_catastrophe_pareto", formats)

        boot_env = bootstrap[bootstrap["env_id"] == env].sort_values(
            "delta__reliability_auc"
        )
        fig, axis = plt.subplots(figsize=(7.2, max(4.2, 0.45 * len(boot_env))))
        y = np.arange(len(boot_env))
        estimate = boot_env["delta__reliability_auc"].to_numpy(dtype=float)
        low = boot_env["delta__reliability_auc__ci95_low"].to_numpy(dtype=float)
        high = boot_env["delta__reliability_auc__ci95_high"].to_numpy(dtype=float)
        axis.errorbar(
            estimate,
            y,
            xerr=[estimate - low, high - estimate],
            fmt="o",
            capsize=3,
        )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(y, boot_env["model_label"])
        axis.set_xlabel(
            "Axis-balanced reliability AUC delta vs vanilla "
            "(95% seed bootstrap CI)"
        )
        axis.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        save_figure(fig, out_dir / f"{env}_reliability_delta_bootstrap", formats)

        boot_env = bootstrap[bootstrap["env_id"] == env].sort_values(
            "delta__catastrophe_prob_t0p5",
            ascending=False,
        )
        fig, axis = plt.subplots(figsize=(7.2, max(4.2, 0.45 * len(boot_env))))
        y = np.arange(len(boot_env))
        estimate = boot_env["delta__catastrophe_prob_t0p5"].to_numpy(dtype=float)
        low = boot_env["delta__catastrophe_prob_t0p5__ci95_low"].to_numpy(dtype=float)
        high = boot_env["delta__catastrophe_prob_t0p5__ci95_high"].to_numpy(dtype=float)
        axis.errorbar(
            estimate,
            y,
            xerr=[estimate - low, high - estimate],
            fmt="o",
            capsize=3,
        )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(y, boot_env["model_label"])
        axis.set_xlabel(
            "Catastrophe-probability delta vs vanilla "
            "(95% seed bootstrap CI; lower is better)"
        )
        axis.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        save_figure(fig, out_dir / f"{env}_catastrophe_delta_bootstrap", formats)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare(
        pd.read_csv(Path(args.metrics_csv).expanduser(), low_memory=False),
        args.baseline_model_label,
    )
    axes = per_axis_summary(df)
    models = aggregate_models(axes, args.baseline_model_label)
    variance = variance_decomposition(df)
    seed_metrics = seed_level_axis_balanced_metrics(df)
    bootstrap = bootstrap_deltas(
        seed_metrics,
        args.baseline_model_label,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    regimes = regime_table(axes, args.baseline_model_label)

    df.to_csv(out_dir / "metrics_with_catastrophe_columns.csv", index=False)
    axes.to_csv(out_dir / "axis_catastrophe_summary.csv", index=False)
    models.to_csv(out_dir / "model_catastrophe_summary.csv", index=False)
    variance.to_csv(out_dir / "seed_perturbation_variance_decomposition.csv", index=False)
    seed_metrics.to_csv(out_dir / "seed_level_axis_balanced_metrics.csv", index=False)
    bootstrap.to_csv(out_dir / "model_delta_seed_bootstrap.csv", index=False)
    regimes.to_csv(out_dir / "perturbation_recovery_regimes.csv", index=False)
    make_plots(df, axes, models, variance, bootstrap, out_dir, args.formats)

    readme = """# Catastrophic robustness analysis

The reliability curve uses a fixed threshold relative to the median nominal
return of vanilla TD3. Its AUC equals the expected normalized return clipped
to [0, 1], so unusually large lucky returns cannot hide catastrophic failures.

Metrics are first computed within each perturbation axis and then averaged
equally across axes. This prevents a family with a denser factor grid from
receiving more weight.

The variance decomposition separates:
- persistent seed quality: seed main-effect variance;
- perturbation severity: perturbation main-effect variance;
- seed-specific fragility: seed-by-perturbation interaction variance.

The final term measures whether different training seeds fail under different
deployment shifts, beyond a seed being uniformly good or bad.

`perturbation_recovery_regimes.csv` uses an explicitly labeled ex-post
axis oracle over models that retain at least 80% nominal return. It measures
whether a failure family is recoverable by the trained menu; it is not a
zero-target-budget deployment claim. Fixed-model aggregate rows provide the
corresponding zero-budget comparison.

`model_delta_seed_bootstrap.csv` resamples training seeds independently for
each method and vanilla. It therefore does not assume that equal numeric seed
labels create meaningful policy pairs. It also reports the common-language
probability that a randomly selected robust seed improves over a randomly
selected vanilla seed.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(models.to_string(index=False))


if __name__ == "__main__":
    main()
