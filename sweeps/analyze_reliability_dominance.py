#!/usr/bin/env python3
"""Measure threshold-free catastrophe recovery relative to vanilla."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-model-label", default="vanilla")
    parser.add_argument("--reference-model-label", default="vanilla")
    parser.add_argument("--threshold-points", type=int, default=201)
    parser.add_argument("--dominance-tolerance", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
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


def prepare(df: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    df = df.copy()
    if "axis" not in df or "factor" not in df:
        parsed = df["scenario_label"].map(parse_scenario)
        df["axis"] = parsed.map(lambda value: value[0])
        df["factor"] = parsed.map(lambda value: value[1])
    df["factor"] = df["factor"].astype(float)
    df["mean_return"] = df["mean_return"].astype(float)
    df["is_nominal"] = np.isclose(
        df["factor"],
        df["axis"].map(nominal_factor),
    )
    reference = (
        df[(df["model_label"] == reference_model) & df["is_nominal"]]
        .groupby(["env_id", "axis"])["mean_return"]
        .median()
        .rename("reference_nominal_median")
    )
    df = df.join(reference, on=["env_id", "axis"])
    if df["reference_nominal_median"].isna().any():
        raise ValueError(
            f"Missing {reference_model} nominal median for at least one axis"
        )
    df["normalized_return"] = df["mean_return"] / df["reference_nominal_median"]
    return df


def survival_curves(
    df: pd.DataFrame,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    shifted = df[~df["is_nominal"]]
    for (env, axis, model), sub in shifted.groupby(
        ["env_id", "axis", "model_label"]
    ):
        values = sub["normalized_return"].to_numpy(dtype=float)
        survival = (values[:, np.newaxis] >= thresholds[np.newaxis, :]).mean(
            axis=0
        )
        rows.extend(
            {
                "env_id": env,
                "axis": axis,
                "model_label": model,
                "threshold": float(threshold),
                "survival": float(value),
            }
            for threshold, value in zip(thresholds, survival)
        )
    return pd.DataFrame(rows)


def seed_axis_survival_curves(
    df: pd.DataFrame,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Keep each training seed as the resampling unit across perturbation levels."""
    rows: list[dict[str, object]] = []
    shifted = df[~df["is_nominal"]]
    for (env, axis, model, seed), sub in shifted.groupby(
        ["env_id", "axis", "model_label", "seed"]
    ):
        values = sub["normalized_return"].to_numpy(dtype=float)
        survival = (values[:, np.newaxis] >= thresholds[np.newaxis, :]).mean(
            axis=0
        )
        rows.extend(
            {
                "env_id": env,
                "axis": axis,
                "model_label": model,
                "seed": int(seed),
                "threshold": float(threshold),
                "survival": float(value),
            }
            for threshold, value in zip(thresholds, survival)
        )
    return pd.DataFrame(rows)


def summarize_dominance(
    curves: pd.DataFrame,
    baseline: str,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = curves.merge(
        curves[curves["model_label"] == baseline][
            ["env_id", "axis", "threshold", "survival"]
        ].rename(columns={"survival": "baseline_survival"}),
        on=["env_id", "axis", "threshold"],
        how="left",
        validate="many_to_one",
    )
    paired["survival_delta"] = paired["survival"] - paired["baseline_survival"]

    model_curves = (
        paired.groupby(["env_id", "model_label", "threshold"])[
            ["survival", "baseline_survival", "survival_delta"]
        ]
        .mean()
        .reset_index()
    )

    def one_summary(sub: pd.DataFrame) -> dict[str, float]:
        sub = sub.sort_values("threshold")
        threshold = sub["threshold"].to_numpy(dtype=float)
        delta = sub["survival_delta"].to_numpy(dtype=float)
        maximum = int(np.argmax(delta))
        minimum = int(np.argmin(delta))
        return {
            "reliability_auc_delta": float(np.trapezoid(delta, threshold)),
            "catastrophe_recovery_area": float(
                np.trapezoid(np.maximum(delta, 0.0), threshold)
            ),
            "catastrophe_harm_area": float(
                np.trapezoid(np.maximum(-delta, 0.0), threshold)
            ),
            "dominance_threshold_fraction": float(
                (delta >= -tolerance).mean()
            ),
            "strict_improvement_threshold_fraction": float(
                (delta > tolerance).mean()
            ),
            "worst_survival_delta": float(delta[minimum]),
            "worst_survival_delta_threshold": float(threshold[minimum]),
            "max_survival_delta": float(delta[maximum]),
            "max_survival_delta_threshold": float(threshold[maximum]),
        }

    model_rows: list[dict[str, object]] = []
    for (env, model), sub in model_curves.groupby(["env_id", "model_label"]):
        model_rows.append(
            {
                "env_id": env,
                "model_label": model,
                "comparison": f"{model}_minus_{baseline}",
                **one_summary(sub),
            }
        )

    axis_rows: list[dict[str, object]] = []
    for (env, axis, model), sub in paired.groupby(
        ["env_id", "axis", "model_label"]
    ):
        axis_rows.append(
            {
                "env_id": env,
                "axis": axis,
                "model_label": model,
                "comparison": f"{model}_minus_{baseline}",
                **one_summary(sub),
            }
        )
    return (
        paired,
        model_curves,
        pd.DataFrame(model_rows).sort_values(
            ["env_id", "reliability_auc_delta"],
            ascending=[True, False],
        ),
        pd.DataFrame(axis_rows),
    )


def bootstrap_dominance(
    seed_curves: pd.DataFrame,
    model_curves: pd.DataFrame,
    summaries: pd.DataFrame,
    baseline: str,
    thresholds: np.ndarray,
    replicates: int,
    seed: int,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independently resample model and baseline training seeds.

    Every selected training seed retains all of its perturbation axes and
    levels. Axes are averaged with equal weight inside each seed before seeds
    are resampled, preserving cross-axis dependence induced by one policy.
    """
    rng = np.random.default_rng(seed)
    curve_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for env, env_df in seed_curves.groupby("env_id"):
        models = sorted(
            model
            for model in env_df["model_label"].unique()
            if model != baseline
        )
        axes = sorted(env_df["axis"].unique())
        for model in models:
            def seed_curves(label: str) -> np.ndarray:
                selected = env_df[env_df["model_label"] == label]
                axis_counts = selected.groupby("seed")["axis"].nunique()
                complete_seeds = axis_counts[axis_counts == len(axes)].index
                selected = selected[selected["seed"].isin(complete_seeds)]
                matrix = (
                    selected.groupby(["seed", "threshold"], as_index=False)[
                        "survival"
                    ]
                    .mean()
                    .pivot(index="seed", columns="threshold", values="survival")
                    .reindex(columns=thresholds)
                    .dropna()
                )
                return matrix.to_numpy(dtype=float)

            baseline_matrix = seed_curves(baseline)
            model_matrix = seed_curves(model)
            if baseline_matrix.size == 0 or model_matrix.size == 0:
                continue

            baseline_indices = rng.integers(
                0,
                len(baseline_matrix),
                size=(replicates, len(baseline_matrix)),
            )
            model_indices = rng.integers(
                0,
                len(model_matrix),
                size=(replicates, len(model_matrix)),
            )
            draws = (
                model_matrix[model_indices].mean(axis=1)
                - baseline_matrix[baseline_indices].mean(axis=1)
            )

            point = (
                model_curves[
                    (model_curves["env_id"] == env)
                    & (model_curves["model_label"] == model)
                ]
                .sort_values("threshold")["survival_delta"]
                .to_numpy(dtype=float)
            )
            if len(point) != len(thresholds):
                raise ValueError(f"Incomplete reliability curve for {env}/{model}")

            curve_rows.append(
                pd.DataFrame(
                    {
                        "env_id": env,
                        "model_label": model,
                        "threshold": thresholds,
                        "survival_delta": point,
                        "ci95_low": np.quantile(draws, 0.025, axis=0),
                        "ci95_high": np.quantile(draws, 0.975, axis=0),
                        "probability_delta_gt_0": (draws > 0.0).mean(axis=0),
                    }
                )
            )

            auc_draws = np.trapezoid(draws, thresholds, axis=1)
            recovery_draws = np.trapezoid(
                np.maximum(draws, 0.0),
                thresholds,
                axis=1,
            )
            harm_draws = np.trapezoid(
                np.maximum(-draws, 0.0),
                thresholds,
                axis=1,
            )
            worst_draws = draws.min(axis=1)
            point_row = summaries[
                (summaries["env_id"] == env)
                & (summaries["model_label"] == model)
            ].iloc[0]
            summary_rows.append(
                {
                    "env_id": env,
                    "model_label": model,
                    "n_axes": len(axes),
                    "n_model_seeds_min": len(model_matrix),
                    "n_baseline_seeds_min": len(baseline_matrix),
                    "bootstrap_replicates": replicates,
                    "reliability_auc_delta": point_row[
                        "reliability_auc_delta"
                    ],
                    "reliability_auc_delta__ci95_low": np.quantile(
                        auc_draws, 0.025
                    ),
                    "reliability_auc_delta__ci95_high": np.quantile(
                        auc_draws, 0.975
                    ),
                    "probability_reliability_auc_improves": (
                        auc_draws > 0.0
                    ).mean(),
                    "catastrophe_recovery_area": point_row[
                        "catastrophe_recovery_area"
                    ],
                    "catastrophe_recovery_area__ci95_low": np.quantile(
                        recovery_draws, 0.025
                    ),
                    "catastrophe_recovery_area__ci95_high": np.quantile(
                        recovery_draws, 0.975
                    ),
                    "catastrophe_harm_area": point_row[
                        "catastrophe_harm_area"
                    ],
                    "catastrophe_harm_area__ci95_low": np.quantile(
                        harm_draws, 0.025
                    ),
                    "catastrophe_harm_area__ci95_high": np.quantile(
                        harm_draws, 0.975
                    ),
                    "worst_survival_delta": point_row["worst_survival_delta"],
                    "worst_survival_delta__ci95_low": np.quantile(
                        worst_draws, 0.025
                    ),
                    "worst_survival_delta__ci95_high": np.quantile(
                        worst_draws, 0.975
                    ),
                    "probability_curve_empirically_dominates": (
                        worst_draws >= -tolerance
                    ).mean(),
                }
            )

    return pd.concat(curve_rows, ignore_index=True), pd.DataFrame(summary_rows)


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    for extension in formats:
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def make_plots(
    model_curves: pd.DataFrame,
    summaries: pd.DataFrame,
    bootstrap_curves: pd.DataFrame,
    bootstrap_summaries: pd.DataFrame,
    baseline: str,
    reference_model: str,
    out_dir: Path,
    formats: list[str],
) -> None:
    for env, env_curves in model_curves.groupby("env_id"):
        models = sorted(
            [
                model
                for model in env_curves["model_label"].unique()
                if model != baseline
            ],
            key=model_key,
        )
        colors = plt.get_cmap("viridis")(
            np.linspace(0.08, 0.92, max(1, len(models)))
        )
        fig, axis = plt.subplots(figsize=(7.5, 4.9))
        for color, model in zip(colors, models):
            sub = env_curves[env_curves["model_label"] == model].sort_values(
                "threshold"
            )
            uncertainty = bootstrap_curves[
                (bootstrap_curves["env_id"] == env)
                & (bootstrap_curves["model_label"] == model)
            ].sort_values("threshold")
            axis.plot(
                sub["threshold"],
                sub["survival_delta"],
                color=color,
                label=model,
            )
            if not uncertainty.empty:
                axis.fill_between(
                    uncertainty["threshold"],
                    uncertainty["ci95_low"],
                    uncertainty["ci95_high"],
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel(
            rf"Threshold $t$ relative to {reference_model} nominal median"
        )
        axis.set_ylabel(
            rf"Reliability difference $S_{{\mathrm{{model}}}}(t)"
            rf"-S_{{\mathrm{{{baseline}}}}}(t)$"
        )
        axis.set_xlim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        save_figure(
            fig,
            out_dir / f"{env}_catastrophe_recovery_profile",
            formats,
        )

        env_summary = summaries[
            (summaries["env_id"] == env)
            & (summaries["model_label"] != baseline)
        ].sort_values("reliability_auc_delta")
        fig, axis = plt.subplots(
            figsize=(7.3, max(3.8, 0.46 * len(env_summary)))
        )
        y = np.arange(len(env_summary))
        recovery = env_summary["catastrophe_recovery_area"].to_numpy(
            dtype=float
        )
        harm = env_summary["catastrophe_harm_area"].to_numpy(dtype=float)
        axis.barh(y, recovery, label="Recovery area")
        axis.barh(y, -harm, label="Harm area")
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_yticks(y, env_summary["model_label"])
        axis.set_xlabel("Area between reliability curves")
        axis.legend(frameon=False)
        axis.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        save_figure(
            fig,
            out_dir / f"{env}_catastrophe_recovery_harm_area",
            formats,
        )

        env_bootstrap = bootstrap_summaries[
            bootstrap_summaries["env_id"] == env
        ].sort_values("reliability_auc_delta")
        fig, axis = plt.subplots(
            figsize=(7.3, max(3.8, 0.46 * len(env_bootstrap)))
        )
        y = np.arange(len(env_bootstrap))
        estimate = env_bootstrap["reliability_auc_delta"].to_numpy(dtype=float)
        low = env_bootstrap[
            "reliability_auc_delta__ci95_low"
        ].to_numpy(dtype=float)
        high = env_bootstrap[
            "reliability_auc_delta__ci95_high"
        ].to_numpy(dtype=float)
        axis.errorbar(
            estimate,
            y,
            xerr=[estimate - low, high - estimate],
            fmt="o",
            capsize=3,
        )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(y, env_bootstrap["model_label"])
        axis.set_xlabel(
            f"Reliability AUC delta vs {baseline} "
            "(95% independent-seed bootstrap CI)"
        )
        axis.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        save_figure(
            fig,
            out_dir / f"{env}_reliability_dominance_bootstrap",
            formats,
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare(
        pd.read_csv(Path(args.metrics_csv).expanduser(), low_memory=False),
        args.reference_model_label,
    )
    thresholds = np.linspace(0.0, 1.0, args.threshold_points)
    curves = survival_curves(df, thresholds)
    seed_curves = seed_axis_survival_curves(df, thresholds)
    axis_curves, model_curves, summaries, axis_summaries = summarize_dominance(
        curves,
        args.baseline_model_label,
        args.dominance_tolerance,
    )
    bootstrap_curves, bootstrap_summaries = bootstrap_dominance(
        seed_curves,
        model_curves,
        summaries,
        args.baseline_model_label,
        thresholds,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        args.dominance_tolerance,
    )
    axis_curves.to_csv(out_dir / "axis_reliability_curves.csv", index=False)
    model_curves.to_csv(
        out_dir / "axis_balanced_reliability_curves.csv",
        index=False,
    )
    summaries.to_csv(out_dir / "model_reliability_dominance.csv", index=False)
    axis_summaries.to_csv(
        out_dir / "axis_reliability_dominance.csv",
        index=False,
    )
    bootstrap_curves.to_csv(
        out_dir / "reliability_delta_seed_bootstrap_curves.csv",
        index=False,
    )
    bootstrap_summaries.to_csv(
        out_dir / "model_reliability_dominance_seed_bootstrap.csv",
        index=False,
    )
    make_plots(
        model_curves,
        summaries,
        bootstrap_curves,
        bootstrap_summaries,
        args.baseline_model_label,
        args.reference_model_label,
        out_dir,
        args.formats,
    )

    readme = r"""# Reliability dominance analysis

For normalized return \(X=R/R_0\), define
\[
S_m(t)=P(X_m\geq t),\qquad 0\leq t\leq 1.
\]
The difference \(S_m(t)-S_0(t)\) is the catastrophe-probability reduction at
threshold \(t\). Positive values favor the robust model.

The signed area equals the clipped normalized-return improvement:
\[
\int_0^1 [S_m(t)-S_0(t)]\,dt
=
\mathbb E[\operatorname{clip}(X_m,0,1)]
-
\mathbb E[\operatorname{clip}(X_0,0,1)].
\]
`catastrophe_recovery_area` integrates the positive part, while
`catastrophe_harm_area` integrates the negative part. A zero harm area and
100% dominance-threshold fraction mean the empirical robust reliability curve
never falls below vanilla on the evaluated threshold grid. This is a
descriptive empirical dominance statement, not a population proof.

Curves are computed within each perturbation axis and then averaged equally
over axes so grid density does not determine the result.

Uncertainty is estimated by independently resampling model and
`__BASELINE_MODEL__`
training seeds while keeping every perturbation level from a sampled seed
together. The shaded bands are pointwise 95% intervals. The reported
`probability_curve_empirically_dominates` is the fraction of bootstrap
replicates whose minimum reliability difference over the full threshold grid
is nonnegative; it is stricter than inspecting pointwise intervals. The
normalizing \(R_0\) is the full-sample `__REFERENCE_MODEL__` nominal
median, so these
intervals are conditional on the benchmark's chosen empirical reference.
"""
    readme = readme.replace(
        "__BASELINE_MODEL__",
        args.baseline_model_label,
    ).replace(
        "__REFERENCE_MODEL__",
        args.reference_model_label,
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
