#!/usr/bin/env python3
"""Apply a predeclared promotion rule to focused physical TD3-KL results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NOMINAL_FACTOR = {
    "mass": 1.0,
    "actuator_gain": 1.0,
    "friction_mass_damping": 1.0,
    "action_replace": 0.0,
}

METRICS = (
    "reliability_auc",
    "cvar20",
    "catastrophe_prob_t0p50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-model-label", default="vanilla")
    parser.add_argument("--ensemble-control-label", default="klprho0")
    parser.add_argument("--min-nominal-retention", type=float, default=0.90)
    parser.add_argument("--min-prob-auc-vs-vanilla", type=float, default=0.95)
    parser.add_argument("--min-prob-cvar-vs-vanilla", type=float, default=0.90)
    parser.add_argument("--min-prob-auc-vs-control", type=float, default=0.90)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args()


def parse_scenario(label: str) -> tuple[str, float]:
    axis, token = str(label).rsplit("_", 1)
    sign = -1.0 if token.startswith("m") else 1.0
    if token.startswith("m"):
        token = token[1:]
    return axis, sign * float(token.replace("p", "."))


def lower_tail_mean(values: pd.Series, fraction: float = 0.2) -> float:
    ordered = np.sort(values.to_numpy(dtype=float))
    count = max(1, int(np.ceil(fraction * len(ordered))))
    return float(ordered[:count].mean())


def prepare_metrics(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    parsed = df["scenario_label"].map(parse_scenario)
    result = df.copy()
    result["axis"] = parsed.map(lambda item: item[0])
    result["factor"] = parsed.map(lambda item: item[1])
    result["nominal_factor"] = result["axis"].map(NOMINAL_FACTOR)
    if result["nominal_factor"].isna().any():
        missing = sorted(result.loc[result["nominal_factor"].isna(), "axis"].unique())
        raise ValueError(f"Missing nominal factors for axes: {missing}")
    result["is_nominal"] = np.isclose(
        result["factor"],
        result["nominal_factor"],
    )

    vanilla_nominal = (
        result[(result["model_label"] == baseline) & result["is_nominal"]]
        .groupby("axis")["mean_return"]
        .median()
    )
    if set(vanilla_nominal.index) != set(NOMINAL_FACTOR):
        raise ValueError("Focused evaluation is missing a vanilla nominal axis")
    result["vanilla_nominal_median"] = result["axis"].map(vanilla_nominal)
    result["normalized_return"] = (
        result["mean_return"] / result["vanilla_nominal_median"]
    )
    return result


def seed_axis_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nominal_rows: list[dict[str, object]] = []
    shifted_rows: list[dict[str, object]] = []
    for (model, seed, axis), group in df.groupby(
        ["model_label", "seed", "axis"],
        sort=False,
    ):
        nominal = group[group["is_nominal"]]["normalized_return"]
        shifted = group[~group["is_nominal"]]["normalized_return"]
        if nominal.empty or shifted.empty:
            raise ValueError(f"Incomplete axis for {model}, seed={seed}, axis={axis}")
        nominal_rows.append(
            {
                "model_label": model,
                "seed": int(seed),
                "axis": axis,
                "nominal_retention": float(nominal.median()),
            }
        )
        shifted_rows.append(
            {
                "model_label": model,
                "seed": int(seed),
                "axis": axis,
                "reliability_auc": float(np.clip(shifted, 0.0, 1.0).mean()),
                "cvar20": lower_tail_mean(shifted),
                "catastrophe_prob_t0p50": float((shifted < 0.5).mean()),
            }
        )

    nominal_axis = pd.DataFrame(nominal_rows)
    shifted_axis = pd.DataFrame(shifted_rows)
    seed_metrics = (
        shifted_axis.groupby(["model_label", "seed"], as_index=False)[list(METRICS)]
        .mean()
        .merge(
            nominal_axis.groupby(["model_label", "seed"], as_index=False)[
                "nominal_retention"
            ].median(),
            on=["model_label", "seed"],
            validate="one_to_one",
        )
    )
    return seed_metrics, shifted_axis


def bootstrap_delta(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    metric: str,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate_values = candidate[metric].to_numpy(dtype=float)
    reference_values = reference[metric].to_numpy(dtype=float)
    candidate_indices = rng.integers(
        0,
        len(candidate_values),
        size=(replicates, len(candidate_values)),
    )
    reference_indices = rng.integers(
        0,
        len(reference_values),
        size=(replicates, len(reference_values)),
    )
    return (
        candidate_values[candidate_indices].mean(axis=1)
        - reference_values[reference_indices].mean(axis=1)
    )


def summarize(
    seed_metrics: pd.DataFrame,
    baseline: str,
    control: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    if baseline not in set(seed_metrics["model_label"]):
        raise ValueError(f"Missing baseline model {baseline!r}")
    if control not in set(seed_metrics["model_label"]):
        raise ValueError(f"Missing ensemble control {control!r}")

    rng = np.random.default_rng(seed)
    references = {
        "vanilla": seed_metrics[seed_metrics["model_label"] == baseline],
        "control": seed_metrics[seed_metrics["model_label"] == control],
    }
    rows: list[dict[str, object]] = []
    for model, candidate in seed_metrics.groupby("model_label", sort=False):
        row: dict[str, object] = {
            "model_label": model,
            "n_seeds": len(candidate),
            "nominal_retention_median": float(
                candidate["nominal_retention"].median()
            ),
            **{
                metric: float(candidate[metric].mean())
                for metric in METRICS
            },
        }
        for reference_name, reference in references.items():
            for metric in METRICS:
                deltas = bootstrap_delta(
                    candidate,
                    reference,
                    metric,
                    replicates,
                    rng,
                )
                prefix = f"delta_{metric}_vs_{reference_name}"
                row[prefix] = float(
                    candidate[metric].mean() - reference[metric].mean()
                )
                row[f"{prefix}_ci95_low"] = float(np.quantile(deltas, 0.025))
                row[f"{prefix}_ci95_high"] = float(np.quantile(deltas, 0.975))
                if metric == "catastrophe_prob_t0p50":
                    row[f"prob_improves_{metric}_vs_{reference_name}"] = float(
                        (deltas < 0).mean()
                    )
                else:
                    row[f"prob_improves_{metric}_vs_{reference_name}"] = float(
                        (deltas > 0).mean()
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def apply_gate(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    result = summary.copy()
    eligible = ~result["model_label"].isin(
        [args.baseline_model_label, args.ensemble_control_label]
    )
    result["passes_nominal_retention"] = (
        result["nominal_retention_median"] >= args.min_nominal_retention
    )
    result["passes_auc_vs_vanilla"] = (
        (result["delta_reliability_auc_vs_vanilla"] > 0)
        & (
            result["prob_improves_reliability_auc_vs_vanilla"]
            >= args.min_prob_auc_vs_vanilla
        )
    )
    result["passes_cvar_vs_vanilla"] = (
        (result["delta_cvar20_vs_vanilla"] > 0)
        & (
            result["prob_improves_cvar20_vs_vanilla"]
            >= args.min_prob_cvar_vs_vanilla
        )
    )
    result["passes_catastrophe_vs_vanilla"] = (
        result["delta_catastrophe_prob_t0p50_vs_vanilla"] <= 0
    )
    result["passes_auc_vs_ensemble_control"] = (
        (result["delta_reliability_auc_vs_control"] > 0)
        & (
            result["prob_improves_reliability_auc_vs_control"]
            >= args.min_prob_auc_vs_control
        )
    )
    result["promotion_gate"] = (
        eligible
        & result["passes_nominal_retention"]
        & result["passes_auc_vs_vanilla"]
        & result["passes_cvar_vs_vanilla"]
        & result["passes_catastrophe_vs_vanilla"]
        & result["passes_auc_vs_ensemble_control"]
    )
    return result.sort_values(
        ["promotion_gate", "reliability_auc", "nominal_retention_median"],
        ascending=[False, False, False],
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = prepare_metrics(
        pd.read_csv(Path(args.metrics_csv).expanduser(), low_memory=False),
        args.baseline_model_label,
    )
    seed_metrics, axis_metrics = seed_axis_metrics(metrics)
    summary = apply_gate(
        summarize(
            seed_metrics,
            args.baseline_model_label,
            args.ensemble_control_label,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        ),
        args,
    )

    promoted = summary.loc[summary["promotion_gate"], "model_label"].tolist()
    selected = promoted[0] if promoted else "none"

    metrics.to_csv(out_dir / "metrics_with_selection_columns.csv", index=False)
    axis_metrics.to_csv(out_dir / "seed_axis_metrics.csv", index=False)
    seed_metrics.to_csv(out_dir / "seed_metrics.csv", index=False)
    summary.to_csv(out_dir / "candidate_summary.csv", index=False)
    (out_dir / "selected_candidate.txt").write_text(
        selected + "\n",
        encoding="utf-8",
    )

    report = f"""# Predeclared TD3-KL candidate selection

Selected candidate: {selected}

The focused validation is a model-selection stage. The later 30-seed,
full-perturbation run is the confirmatory stage. Each training seed is the
bootstrap cluster, all perturbation levels for that seed remain together,
and the four focused axes receive equal weight.

A finite-radius candidate passes only when:

- median nominal retention is at least {args.min_nominal_retention:.2f};
- reliability AUC improves over vanilla with bootstrap probability at least
  {args.min_prob_auc_vs_vanilla:.2f};
- axis-balanced CVaR20 improves over vanilla with bootstrap probability at
  least {args.min_prob_cvar_vs_vanilla:.2f};
- catastrophe probability at threshold 0.5 is no worse than vanilla; and
- reliability AUC improves over the rho=0 physical-ensemble control with
  bootstrap probability at least {args.min_prob_auc_vs_control:.2f}.

Among passing radii, the selected candidate is the one with the largest
axis-balanced reliability AUC. This rule was fixed before the five-seed
validation evaluation completed.
"""
    (out_dir / "README.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"selected candidate: {selected}")


if __name__ == "__main__":
    main()
