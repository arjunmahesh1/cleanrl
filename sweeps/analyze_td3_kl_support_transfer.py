#!/usr/bin/env python3
"""Test whether physical TD3-KL gains align with its training support."""

from __future__ import annotations

import argparse
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
DIRECT_SUPPORT_AXES = {"mass", "actuator_gain"}
SUPPORT_CLASS_ORDER = ["direct_support", "related_physics", "out_of_support"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--robust-model-label", required=True)
    parser.add_argument("--baseline-model-label", default="vanilla")
    parser.add_argument("--ensemble-control-label", default="klprho0")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
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


def support_class(axis: str) -> str:
    if axis in DIRECT_SUPPORT_AXES:
        return "direct_support"
    if "mass" in axis or "actuator_gain" in axis:
        return "related_physics"
    return "out_of_support"


def lower_tail_mean(values: pd.Series, fraction: float = 0.2) -> float:
    ordered = np.sort(values.to_numpy(dtype=float))
    count = max(1, int(np.ceil(fraction * len(ordered))))
    return float(ordered[:count].mean())


def prepare(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    result = df.copy()
    parsed = result["scenario_label"].map(parse_scenario)
    result["axis"] = parsed.map(lambda item: item[0])
    result["factor"] = parsed.map(lambda item: item[1])
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
    result["support_class"] = result["axis"].map(support_class)
    return result


def make_seed_axis_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    shifted = df[~df["is_nominal"]]
    for (env, model, seed, axis, axis_class), group in shifted.groupby(
        ["env_id", "model_label", "seed", "axis", "support_class"],
        sort=False,
    ):
        normalized = group["normalized_return"]
        rows.append(
            {
                "env_id": env,
                "model_label": model,
                "seed": int(seed),
                "axis": axis,
                "support_class": axis_class,
                "reliability_auc": float(np.clip(normalized, 0.0, 1.0).mean()),
                "catastrophe_prob_t0p5": float((normalized < 0.5).mean()),
                "cvar20": lower_tail_mean(normalized),
            }
        )
    return pd.DataFrame(rows)


def make_seed_class_metrics(seed_axis: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_axis.groupby(
            ["env_id", "model_label", "seed", "support_class"],
            as_index=False,
        )[["reliability_auc", "catastrophe_prob_t0p5", "cvar20"]]
        .mean()
    )


def bootstrap_delta(
    candidate: np.ndarray,
    reference: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate_indices = rng.integers(
        0,
        len(candidate),
        size=(replicates, len(candidate)),
    )
    reference_indices = rng.integers(
        0,
        len(reference),
        size=(replicates, len(reference)),
    )
    return (
        candidate[candidate_indices].mean(axis=1)
        - reference[reference_indices].mean(axis=1)
    )


def summarize_classes(
    seed_class: pd.DataFrame,
    models: list[str],
    references: list[str],
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for env in sorted(seed_class["env_id"].unique()):
        env_df = seed_class[seed_class["env_id"] == env]
        for model in models:
            for reference in references:
                if model == reference:
                    continue
                for axis_class in SUPPORT_CLASS_ORDER:
                    candidate = env_df[
                        (env_df["model_label"] == model)
                        & (env_df["support_class"] == axis_class)
                    ]
                    baseline = env_df[
                        (env_df["model_label"] == reference)
                        & (env_df["support_class"] == axis_class)
                    ]
                    if candidate.empty or baseline.empty:
                        raise ValueError(
                            f"Missing {axis_class} data for {model} vs {reference}"
                        )
                    row: dict[str, object] = {
                        "env_id": env,
                        "model_label": model,
                        "reference_label": reference,
                        "support_class": axis_class,
                        "n_model_seeds": candidate["seed"].nunique(),
                        "n_reference_seeds": baseline["seed"].nunique(),
                    }
                    for metric in (
                        "reliability_auc",
                        "catastrophe_prob_t0p5",
                        "cvar20",
                    ):
                        candidate_values = candidate[metric].to_numpy(dtype=float)
                        baseline_values = baseline[metric].to_numpy(dtype=float)
                        draws = bootstrap_delta(
                            candidate_values,
                            baseline_values,
                            replicates,
                            rng,
                        )
                        delta = float(candidate_values.mean() - baseline_values.mean())
                        row[f"delta_{metric}"] = delta
                        row[f"delta_{metric}__ci95_low"] = float(
                            np.quantile(draws, 0.025)
                        )
                        row[f"delta_{metric}__ci95_high"] = float(
                            np.quantile(draws, 0.975)
                        )
                        if metric == "catastrophe_prob_t0p5":
                            row[f"probability_{metric}_improves"] = float(
                                (draws < 0.0).mean()
                            )
                        else:
                            row[f"probability_{metric}_improves"] = float(
                                (draws > 0.0).mean()
                            )
                    rows.append(row)
    return pd.DataFrame(rows)


def support_alignment_contrast(
    seed_class: pd.DataFrame,
    models: list[str],
    references: list[str],
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pivot = seed_class.pivot(
        index=["env_id", "model_label", "seed"],
        columns="support_class",
        values="reliability_auc",
    ).reset_index()
    rows: list[dict[str, object]] = []
    for env in sorted(pivot["env_id"].unique()):
        env_df = pivot[pivot["env_id"] == env]
        for model in models:
            for reference in references:
                if model == reference:
                    continue
                candidate = env_df[env_df["model_label"] == model]
                baseline = env_df[env_df["model_label"] == reference]
                candidate_values = (
                    candidate["direct_support"] - candidate["out_of_support"]
                ).to_numpy(dtype=float)
                baseline_values = (
                    baseline["direct_support"] - baseline["out_of_support"]
                ).to_numpy(dtype=float)
                draws = bootstrap_delta(
                    candidate_values,
                    baseline_values,
                    replicates,
                    rng,
                )
                contrast = float(candidate_values.mean() - baseline_values.mean())
                rows.append(
                    {
                        "env_id": env,
                        "model_label": model,
                        "reference_label": reference,
                        "support_alignment_contrast": contrast,
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "probability_contrast_positive": float((draws > 0.0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    for extension in formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    class_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    robust_model: str,
    baseline: str,
    control: str,
    out_dir: Path,
    formats: list[str],
) -> None:
    comparisons = [
        (robust_model, baseline),
        (robust_model, control),
        (control, baseline),
    ]
    labels = {
        "direct_support": "Direct support",
        "related_physics": "Related physics",
        "out_of_support": "Out of support",
    }
    colors = {
        "direct_support": "#2a9d8f",
        "related_physics": "#e9c46a",
        "out_of_support": "#6c757d",
    }

    fig, axes = plt.subplots(1, len(comparisons), figsize=(13.2, 4.2), sharey=True)
    for axis, (model, reference) in zip(axes, comparisons):
        sub = class_summary[
            (class_summary["model_label"] == model)
            & (class_summary["reference_label"] == reference)
        ].set_index("support_class")
        y = np.arange(len(SUPPORT_CLASS_ORDER))
        estimate = sub.loc[SUPPORT_CLASS_ORDER, "delta_reliability_auc"].to_numpy()
        low = sub.loc[
            SUPPORT_CLASS_ORDER,
            "delta_reliability_auc__ci95_low",
        ].to_numpy()
        high = sub.loc[
            SUPPORT_CLASS_ORDER,
            "delta_reliability_auc__ci95_high",
        ].to_numpy()
        axis.errorbar(
            estimate,
            y,
            xerr=[estimate - low, high - estimate],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
        for position, axis_class, value in zip(y, SUPPORT_CLASS_ORDER, estimate):
            axis.scatter(value, position, color=colors[axis_class], s=44, zorder=3)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{model} minus {reference}")
        axis.set_xlabel("Reliability AUC difference")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(
        np.arange(len(SUPPORT_CLASS_ORDER)),
        [labels[value] for value in SUPPORT_CLASS_ORDER],
    )
    fig.tight_layout()
    save_figure(fig, out_dir / "support_class_reliability_deltas", formats)

    contrast_sub = contrasts[
        (contrasts["model_label"] == robust_model)
        & (contrasts["reference_label"].isin([baseline, control]))
    ].copy()
    contrast_sub["comparison"] = (
        contrast_sub["model_label"] + " minus " + contrast_sub["reference_label"]
    )
    y = np.arange(len(contrast_sub))
    estimate = contrast_sub["support_alignment_contrast"].to_numpy(dtype=float)
    low = contrast_sub["ci95_low"].to_numpy(dtype=float)
    high = contrast_sub["ci95_high"].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.errorbar(
        estimate,
        y,
        xerr=[estimate - low, high - estimate],
        fmt="o",
        capsize=3,
    )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(y, contrast_sub["comparison"])
    axis.set_xlabel(
        "Support alignment: direct-support AUC gain minus out-of-support gain"
    )
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_dir / "support_alignment_contrast", formats)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [
        args.robust_model_label,
        args.ensemble_control_label,
    ]
    references = [
        args.baseline_model_label,
        args.ensemble_control_label,
    ]

    metrics = prepare(
        pd.read_csv(Path(args.metrics_csv).expanduser(), low_memory=False),
        args.baseline_model_label,
    )
    present_direct = set(metrics["axis"]) & DIRECT_SUPPORT_AXES
    if present_direct != DIRECT_SUPPORT_AXES:
        raise ValueError(
            f"Expected direct-support axes {sorted(DIRECT_SUPPORT_AXES)}, "
            f"found {sorted(present_direct)}"
        )
    seed_axis = make_seed_axis_metrics(metrics)
    seed_class = make_seed_class_metrics(seed_axis)
    rng = np.random.default_rng(args.bootstrap_seed)
    class_summary = summarize_classes(
        seed_class,
        models,
        references,
        args.bootstrap_replicates,
        rng,
    )
    contrasts = support_alignment_contrast(
        seed_class,
        models,
        references,
        args.bootstrap_replicates,
        rng,
    )

    metrics.to_csv(out_dir / "metrics_with_support_classes.csv", index=False)
    seed_axis.to_csv(out_dir / "seed_axis_metrics.csv", index=False)
    seed_class.to_csv(out_dir / "seed_support_class_metrics.csv", index=False)
    class_summary.to_csv(out_dir / "support_class_summary.csv", index=False)
    contrasts.to_csv(out_dir / "support_alignment_contrast.csv", index=False)
    make_plots(
        class_summary,
        contrasts,
        args.robust_model_label,
        args.baseline_model_label,
        args.ensemble_control_label,
        out_dir,
        args.formats,
    )

    readme = rf"""# Physical-support transfer analysis

This analysis was fixed before the independent confirmation evaluation
completed. Axes are partitioned into:

- `direct_support`: global mass and global actuator gain, the two perturbed
  dynamics families present in the five-member training support;
- `related_physics`: localized actuator-gain or mass axes and global
  combinations involving mass; and
- `out_of_support`: all remaining deployment perturbations.

Within each axis, return is normalized by that axis's vanilla nominal median.
Reliability AUC is the mean clipped normalized return over non-nominal
conditions. Axes receive equal weight within each support class. Confidence
intervals independently bootstrap training seeds and retain every axis-level
summary from a sampled seed.

The support-alignment contrast is
\[
\bigl(\Delta\mathrm{{AUC}}_{{\mathrm{{direct}}}}\bigr)
-
\bigl(\Delta\mathrm{{AUC}}_{{\mathrm{{out}}}}\bigr).
\]
A positive contrast means the algorithm's relative benefit tracks the
geometry represented in its training support. It does not by itself show an
overall robustness improvement; the class-specific AUC deltas must also be
positive.

Robust model: `{args.robust_model_label}`.
Ensemble control: `{args.ensemble_control_label}`.
Baseline: `{args.baseline_model_label}`.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(class_summary.to_string(index=False))
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
