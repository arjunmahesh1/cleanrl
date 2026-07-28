#!/usr/bin/env python3
"""Separate nominal training failures from deployment-time failures."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIGNAL_AXES = {
    "action_noise",
    "action_replace",
    "action_noise_bernoulli",
    "state_noise",
}
METRICS = (
    "nominal_failure_probability",
    "deployment_failure_probability",
    "deployment_failure_given_nominal_competence",
    "reliability_auc_given_nominal_competence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-model-label", default="vanilla")
    parser.add_argument("--competence-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def parse_scenario(label: str) -> tuple[str, float]:
    axis, token = str(label).rsplit("_", 1)
    sign = -1.0 if token.startswith("m") else 1.0
    if token.startswith("m"):
        token = token[1:]
    return axis, sign * float(token.replace("p", "."))


def nominal_factor(axis: str) -> float:
    return 0.0 if axis in SIGNAL_AXES else 1.0


def model_key(label: str) -> tuple[int, float, str]:
    if label == "vanilla":
        return 0, -1.0, label
    match = re.search(r"(-?\d+)(?:p(\d+))?$", label)
    if not match:
        return 2, math.inf, label
    fraction = match.group(2) or "0"
    return 1, float(f"{match.group(1)}.{fraction}"), label


def prepare(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    result = df.copy()
    if "axis" not in result or "factor" not in result:
        parsed = result["scenario_label"].map(parse_scenario)
        result["axis"] = parsed.map(lambda item: item[0])
        result["factor"] = parsed.map(lambda item: item[1])
    result["seed"] = result["seed"].astype(int)
    result["factor"] = result["factor"].astype(float)
    result["mean_return"] = result["mean_return"].astype(float)
    result["is_nominal"] = np.isclose(
        result["factor"],
        result["axis"].map(nominal_factor),
    )
    reference = (
        result[(result["model_label"] == baseline) & result["is_nominal"]]
        .groupby(["env_id", "axis"])["mean_return"]
        .median()
        .rename("vanilla_nominal_median")
    )
    result = result.join(reference, on=["env_id", "axis"])
    if result["vanilla_nominal_median"].isna().any():
        raise ValueError("Missing vanilla nominal reference for at least one axis")
    result["normalized_return"] = (
        result["mean_return"] / result["vanilla_nominal_median"]
    )
    return result


def make_seed_axis_metrics(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (env, model, seed, axis), group in df.groupby(
        ["env_id", "model_label", "seed", "axis"],
        sort=False,
    ):
        nominal = group[group["is_nominal"]]["normalized_return"]
        shifted = group[~group["is_nominal"]]["normalized_return"]
        if nominal.empty or shifted.empty:
            raise ValueError(
                f"Incomplete axis for {env}/{model}/seed={seed}/{axis}"
            )
        nominal_return = float(nominal.median())
        competent = nominal_return >= threshold
        rows.append(
            {
                "env_id": env,
                "model_label": model,
                "seed": int(seed),
                "axis": axis,
                "nominal_normalized_return": nominal_return,
                "nominal_competent": competent,
                "nominal_failure": float(not competent),
                "deployment_failure": float((shifted < threshold).mean()),
                "reliability_auc": float(np.clip(shifted, 0.0, 1.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def model_matrices(
    seed_axis: pd.DataFrame,
    env: str,
    model: str,
    axes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    selected = seed_axis[
        (seed_axis["env_id"] == env)
        & (seed_axis["model_label"] == model)
    ]
    seeds = sorted(selected["seed"].unique())

    def matrix(column: str) -> np.ndarray:
        frame = selected.pivot(index="seed", columns="axis", values=column)
        frame = frame.reindex(index=seeds, columns=axes)
        if frame.isna().any().any():
            raise ValueError(f"Incomplete seed-axis matrix for {env}/{model}")
        return frame.to_numpy(dtype=float)

    return (
        matrix("nominal_competent"),
        matrix("deployment_failure"),
        matrix("reliability_auc"),
        seeds,
    )


def evaluate_draws(
    competent: np.ndarray,
    deployment_failure: np.ndarray,
    reliability_auc: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    selected_competent = competent[indices]
    selected_failure = deployment_failure[indices]
    selected_auc = reliability_auc[indices]
    denominators = selected_competent.sum(axis=1)

    conditional_failure_by_axis = np.divide(
        (selected_failure * selected_competent).sum(axis=1),
        denominators,
        out=np.full_like(denominators, np.nan, dtype=float),
        where=denominators > 0,
    )
    conditional_auc_by_axis = np.divide(
        (selected_auc * selected_competent).sum(axis=1),
        denominators,
        out=np.full_like(denominators, np.nan, dtype=float),
        where=denominators > 0,
    )

    def row_nanmean(values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        counts = valid.sum(axis=1)
        return np.divide(
            np.nansum(values, axis=1),
            counts,
            out=np.full(len(values), np.nan, dtype=float),
            where=counts > 0,
        )

    return {
        "nominal_failure_probability": 1.0 - selected_competent.mean(
            axis=(1, 2)
        ),
        "deployment_failure_probability": selected_failure.mean(axis=(1, 2)),
        "deployment_failure_given_nominal_competence": row_nanmean(
            conditional_failure_by_axis
        ),
        "reliability_auc_given_nominal_competence": row_nanmean(
            conditional_auc_by_axis
        ),
    }


def summarize(
    seed_axis: pd.DataFrame,
    baseline: str,
    replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(bootstrap_seed)
    summary_rows: list[dict[str, object]] = []
    bootstrap_rows: list[pd.DataFrame] = []

    def finite_quantile(values: np.ndarray, probability: float) -> float:
        finite = values[np.isfinite(values)]
        return (
            float(np.quantile(finite, probability))
            if finite.size
            else float("nan")
        )

    def probability_positive(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float((finite > 0.0).mean()) if finite.size else float("nan")

    for env, env_df in seed_axis.groupby("env_id"):
        axes = sorted(env_df["axis"].unique())
        model_draws: dict[str, dict[str, np.ndarray]] = {}
        model_points: dict[str, dict[str, float]] = {}
        model_seed_counts: dict[str, int] = {}

        for model in sorted(env_df["model_label"].unique(), key=model_key):
            competent, failure, auc, seeds = model_matrices(
                seed_axis,
                env,
                model,
                axes,
            )
            model_seed_counts[model] = len(seeds)
            point_indices = np.arange(len(seeds))[None, :]
            model_points[model] = {
                key: float(value[0])
                for key, value in evaluate_draws(
                    competent,
                    failure,
                    auc,
                    point_indices,
                ).items()
            }
            bootstrap_indices = rng.integers(
                0,
                len(seeds),
                size=(replicates, len(seeds)),
            )
            model_draws[model] = evaluate_draws(
                competent,
                failure,
                auc,
                bootstrap_indices,
            )

        baseline_draws = model_draws[baseline]
        baseline_points = model_points[baseline]
        for model in sorted(model_draws, key=model_key):
            row: dict[str, object] = {
                "env_id": env,
                "model_label": model,
                "n_axes": len(axes),
                "n_seeds": model_seed_counts[model],
                "competence_threshold": seed_axis.attrs[
                    "competence_threshold"
                ],
                "bootstrap_replicates": replicates,
            }
            long_frame = pd.DataFrame(
                {
                    "replicate": np.arange(replicates),
                    "env_id": env,
                    "model_label": model,
                }
            )
            for metric in METRICS:
                draws = model_draws[model][metric]
                point = model_points[model][metric]
                baseline_point = baseline_points[metric]
                delta = draws - baseline_draws[metric]
                higher_is_better = metric == (
                    "reliability_auc_given_nominal_competence"
                )
                improvement = delta if higher_is_better else -delta
                point_improvement = (
                    point - baseline_point
                    if higher_is_better
                    else baseline_point - point
                )
                row[metric] = point
                row[f"{metric}__ci95_low"] = finite_quantile(draws, 0.025)
                row[f"{metric}__ci95_high"] = finite_quantile(draws, 0.975)
                row[f"improvement_vs_{baseline}__{metric}"] = (
                    point_improvement
                )
                row[
                    f"improvement_vs_{baseline}__{metric}__ci95_low"
                ] = finite_quantile(improvement, 0.025)
                row[
                    f"improvement_vs_{baseline}__{metric}__ci95_high"
                ] = finite_quantile(improvement, 0.975)
                row[
                    f"probability_improves_vs_{baseline}__{metric}"
                ] = probability_positive(improvement)
                long_frame[metric] = draws
                long_frame[f"improvement_vs_{baseline}__{metric}"] = (
                    improvement
                )
            summary_rows.append(row)
            bootstrap_rows.append(long_frame)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(bootstrap_rows, ignore_index=True),
    )


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    for extension in formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    summary: pd.DataFrame,
    baseline: str,
    out_dir: Path,
    formats: list[str],
) -> None:
    labels = {
        "nominal_failure_probability": "Nominal training failure",
        "deployment_failure_probability": "Deployment failure (all seeds)",
        "deployment_failure_given_nominal_competence": (
            "Deployment failure | nominally competent"
        ),
        "reliability_auc_given_nominal_competence": (
            "Reliability AUC | nominally competent"
        ),
    }
    for env, env_df in summary.groupby("env_id"):
        env_df = env_df.sort_values(
            "model_label",
            key=lambda values: values.map(model_key),
        )
        models = env_df["model_label"].tolist()
        y = np.arange(len(models))
        fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.6), sharey=True)
        for axis, metric in zip(axes, METRICS):
            point = env_df[metric].to_numpy(dtype=float)
            low = env_df[f"{metric}__ci95_low"].to_numpy(dtype=float)
            high = env_df[f"{metric}__ci95_high"].to_numpy(dtype=float)
            axis.errorbar(
                point,
                y,
                xerr=[point - low, high - point],
                fmt="o",
                capsize=3,
            )
            axis.set_title(labels[metric], fontsize=10)
            axis.set_xlabel("Probability" if "probability" in metric else "AUC")
            axis.grid(axis="x", alpha=0.25)
        axes[0].set_yticks(y, models)
        fig.suptitle(
            f"{env}: training and deployment failure channels",
            fontsize=13,
        )
        fig.tight_layout()
        save_figure(fig, out_dir / f"{env}_failure_channels", formats)

        comparisons = env_df[env_df["model_label"] != baseline]
        y = np.arange(len(comparisons))
        fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.3), sharey=True)
        for axis, metric in zip(axes, METRICS):
            column = f"improvement_vs_{baseline}__{metric}"
            point = comparisons[column].to_numpy(dtype=float)
            low = comparisons[f"{column}__ci95_low"].to_numpy(dtype=float)
            high = comparisons[f"{column}__ci95_high"].to_numpy(dtype=float)
            axis.errorbar(
                point,
                y,
                xerr=[point - low, high - point],
                fmt="o",
                capsize=3,
            )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
            axis.set_title(labels[metric], fontsize=10)
            axis.set_xlabel(f"Improvement vs {baseline}")
            axis.grid(axis="x", alpha=0.25)
        axes[0].set_yticks(y, comparisons["model_label"])
        fig.suptitle(
            f"{env}: failure-channel effects relative to {baseline}",
            fontsize=13,
        )
        fig.tight_layout()
        save_figure(
            fig,
            out_dir / f"{env}_failure_channel_improvements",
            formats,
        )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.competence_threshold <= 1.0:
        raise ValueError("--competence-threshold must lie in (0, 1]")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = prepare(
        pd.read_csv(Path(args.metrics_csv).expanduser(), low_memory=False),
        args.baseline_model_label,
    )
    seed_axis = make_seed_axis_metrics(metrics, args.competence_threshold)
    seed_axis.attrs["competence_threshold"] = args.competence_threshold
    summary, bootstrap = summarize(
        seed_axis,
        args.baseline_model_label,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    seed_axis.to_csv(out_dir / "seed_axis_failure_channels.csv", index=False)
    summary.to_csv(out_dir / "model_failure_channel_summary.csv", index=False)
    bootstrap.to_csv(out_dir / "failure_channel_seed_bootstrap.csv", index=False)
    make_plots(
        summary,
        args.baseline_model_label,
        out_dir,
        args.formats,
    )

    threshold = args.competence_threshold
    readme = f"""# Training and deployment failure channels

Let `R0,p` be the vanilla cross-seed nominal median for axis `p`, and let
`t={threshold:g}`. A seed-axis policy is nominally competent when its nominal
return is at least `t R0,p`.

The analysis reports:

- nominal training failure: `P(R_nominal < t R0,p)`;
- unconditional deployment failure under non-nominal perturbations;
- deployment failure conditional on nominal competence;
- reliability AUC conditional on nominal competence.

Axes receive equal weight. Bootstrap resampling treats one complete trained
policy seed, including all axes and perturbation levels, as the sampling
cluster. Robust and `{args.baseline_model_label}` seeds are resampled
independently.

Conditional results are diagnostic rather than primary performance claims:
conditioning removes failed training runs and therefore changes the deployed
policy population. The unconditional metric remains the correct
algorithm-level result. The conditional metric asks whether an observed
failure came mainly from optimization reliability or from sensitivity after
a competent policy had been learned.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
