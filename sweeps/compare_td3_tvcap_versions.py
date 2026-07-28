#!/usr/bin/env python3
"""Compare axis-balanced TD3 TV-cap effects across environment versions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    "reliability_auc",
    "cvar20",
    "catastrophe_prob_t0p50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-metrics", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--reference-name", default="Walker2d-v4")
    parser.add_argument("--candidate-name", default="Walker2d-v5")
    parser.add_argument("--reference-model", default="tvc250")
    parser.add_argument("--candidate-model", default="tvc400")
    parser.add_argument("--baseline-model-label", default="vanilla")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
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
    return 0.0 if axis in {"action_noise", "action_replace"} else 1.0


def lower_tail_mean(values: pd.Series, fraction: float = 0.2) -> float:
    ordered = np.sort(values.to_numpy(dtype=float))
    count = max(1, int(np.ceil(fraction * len(ordered))))
    return float(ordered[:count].mean())


def prepare(
    path: Path,
    version: str,
    baseline: str,
) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "axis" not in df or "factor" not in df:
        parsed = df["scenario_label"].map(parse_scenario)
        df["axis"] = parsed.map(lambda item: item[0])
        df["factor"] = parsed.map(lambda item: item[1])
    df["version"] = version
    df["nominal_factor"] = df["axis"].map(nominal_factor)
    df["is_nominal"] = np.isclose(df["factor"], df["nominal_factor"])
    reference = (
        df[(df["model_label"] == baseline) & df["is_nominal"]]
        .groupby("axis")["mean_return"]
        .median()
    )
    if set(reference.index) != set(df["axis"].unique()):
        raise ValueError(f"{version}: missing a vanilla nominal reference")
    df["vanilla_nominal_median"] = df["axis"].map(reference)
    df["normalized_return"] = df["mean_return"] / df["vanilla_nominal_median"]
    return df


def compute_seed_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (version, model, seed, axis), group in df.groupby(
        ["version", "model_label", "seed", "axis"],
        sort=False,
    ):
        nominal = group[group["is_nominal"]]["normalized_return"]
        shifted = group[~group["is_nominal"]]["normalized_return"]
        if nominal.empty or shifted.empty:
            raise ValueError(
                f"Incomplete axis: {version}, {model}, seed={seed}, {axis}"
            )
        rows.append(
            {
                "version": version,
                "model_label": model,
                "seed": int(seed),
                "axis": axis,
                "nominal_retention": float(nominal.median()),
                "reliability_auc": float(np.clip(shifted, 0.0, 1.0).mean()),
                "cvar20": lower_tail_mean(shifted),
                "catastrophe_prob_t0p50": float((shifted < 0.5).mean()),
            }
        )
    axis_metrics = pd.DataFrame(rows)
    seed_metrics = (
        axis_metrics.groupby(["version", "model_label", "seed"], as_index=False)
        .agg(
            nominal_retention=("nominal_retention", "median"),
            reliability_auc=("reliability_auc", "mean"),
            cvar20=("cvar20", "mean"),
            catastrophe_prob_t0p50=("catastrophe_prob_t0p50", "mean"),
        )
    )
    return seed_metrics, axis_metrics


def bootstrap_effect(
    model_values: np.ndarray,
    baseline_values: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    model_indices = rng.integers(
        0,
        len(model_values),
        size=(replicates, len(model_values)),
    )
    baseline_indices = rng.integers(
        0,
        len(baseline_values),
        size=(replicates, len(baseline_values)),
    )
    return (
        model_values[model_indices].mean(axis=1)
        - baseline_values[baseline_indices].mean(axis=1)
    )


def summarize_models(
    seed_metrics: pd.DataFrame,
    baseline: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (version, model), group in seed_metrics.groupby(
        ["version", "model_label"],
        sort=False,
    ):
        baseline_group = seed_metrics[
            (seed_metrics["version"] == version)
            & (seed_metrics["model_label"] == baseline)
        ]
        row: dict[str, object] = {
            "version": version,
            "model_label": model,
            "n_seeds": len(group),
            "nominal_retention_median": float(
                group["nominal_retention"].median()
            ),
        }
        for metric in METRICS:
            model_values = group[metric].to_numpy(dtype=float)
            baseline_values = baseline_group[metric].to_numpy(dtype=float)
            effects = bootstrap_effect(
                model_values,
                baseline_values,
                replicates,
                rng,
            )
            row[metric] = float(model_values.mean())
            row[f"delta_{metric}_vs_vanilla"] = float(
                model_values.mean() - baseline_values.mean()
            )
            row[f"delta_{metric}_vs_vanilla_ci95_low"] = float(
                np.quantile(effects, 0.025)
            )
            row[f"delta_{metric}_vs_vanilla_ci95_high"] = float(
                np.quantile(effects, 0.975)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_selected(
    seed_metrics: pd.DataFrame,
    reference_name: str,
    candidate_name: str,
    reference_model: str,
    candidate_model: str,
    baseline: str,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    specifications = (
        (reference_name, reference_model),
        (candidate_name, candidate_model),
    )
    effects_by_version: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    for version, model in specifications:
        model_group = seed_metrics[
            (seed_metrics["version"] == version)
            & (seed_metrics["model_label"] == model)
        ]
        baseline_group = seed_metrics[
            (seed_metrics["version"] == version)
            & (seed_metrics["model_label"] == baseline)
        ]
        if model_group.empty or baseline_group.empty:
            raise ValueError(f"Missing selected comparison: {version}, {model}")
        effects_by_version[version] = {}
        row: dict[str, object] = {
            "version": version,
            "model_label": model,
            "n_model_seeds": len(model_group),
            "n_vanilla_seeds": len(baseline_group),
            "nominal_retention_median": float(
                model_group["nominal_retention"].median()
            ),
        }
        for metric in METRICS:
            effects = bootstrap_effect(
                model_group[metric].to_numpy(dtype=float),
                baseline_group[metric].to_numpy(dtype=float),
                replicates,
                rng,
            )
            effects_by_version[version][metric] = effects
            row[f"delta_{metric}_vs_vanilla"] = float(
                model_group[metric].mean() - baseline_group[metric].mean()
            )
            row[f"delta_{metric}_vs_vanilla_ci95_low"] = float(
                np.quantile(effects, 0.025)
            )
            row[f"delta_{metric}_vs_vanilla_ci95_high"] = float(
                np.quantile(effects, 0.975)
            )
        rows.append(row)

    interaction_rows: list[dict[str, object]] = []
    for metric in METRICS:
        interaction = (
            effects_by_version[candidate_name][metric]
            - effects_by_version[reference_name][metric]
        )
        reference_point = rows[0][f"delta_{metric}_vs_vanilla"]
        candidate_point = rows[1][f"delta_{metric}_vs_vanilla"]
        interaction_rows.append(
            {
                "metric": metric,
                "reference_effect": reference_point,
                "candidate_effect": candidate_point,
                "candidate_minus_reference_effect": (
                    candidate_point - reference_point
                ),
                "candidate_minus_reference_ci95_low": float(
                    np.quantile(interaction, 0.025)
                ),
                "candidate_minus_reference_ci95_high": float(
                    np.quantile(interaction, 0.975)
                ),
                "probability_candidate_effect_is_larger": float(
                    (interaction > 0).mean()
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(interaction_rows)


def selected_axis_effects(
    axis_metrics: pd.DataFrame,
    reference_name: str,
    candidate_name: str,
    reference_model: str,
    candidate_model: str,
    baseline: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for version, model in (
        (reference_name, reference_model),
        (candidate_name, candidate_model),
    ):
        version_df = axis_metrics[axis_metrics["version"] == version]
        for axis in sorted(version_df["axis"].unique()):
            model_df = version_df[
                (version_df["model_label"] == model)
                & (version_df["axis"] == axis)
            ]
            baseline_df = version_df[
                (version_df["model_label"] == baseline)
                & (version_df["axis"] == axis)
            ]
            row: dict[str, object] = {
                "version": version,
                "model_label": model,
                "axis": axis,
            }
            for metric in METRICS:
                row[f"delta_{metric}_vs_vanilla"] = float(
                    model_df[metric].mean() - baseline_df[metric].mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    for extension in formats:
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_selected_effects(
    selected: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
) -> None:
    labels = {
        "reliability_auc": "Reliability AUC",
        "cvar20": "Axis-balanced CVaR20",
        "catastrophe_prob_t0p50": "Catastrophe probability",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for axis, metric in zip(axes, METRICS):
        point = selected[f"delta_{metric}_vs_vanilla"].to_numpy(dtype=float)
        low = selected[
            f"delta_{metric}_vs_vanilla_ci95_low"
        ].to_numpy(dtype=float)
        high = selected[
            f"delta_{metric}_vs_vanilla_ci95_high"
        ].to_numpy(dtype=float)
        y = np.arange(len(selected))
        axis.errorbar(
            point,
            y,
            xerr=[point - low, high - point],
            fmt="o",
            capsize=3,
        )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(
            y,
            [
                f"{row.version}: {row.model_label}"
                for row in selected.itertuples()
            ],
        )
        axis.set_title(labels[metric])
        axis.set_xlabel("Effect relative to version-matched vanilla")
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_dir / "selected_cap_version_effects", formats)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        prepare(
            Path(args.reference_metrics).expanduser(),
            args.reference_name,
            args.baseline_model_label,
        ),
        prepare(
            Path(args.candidate_metrics).expanduser(),
            args.candidate_name,
            args.baseline_model_label,
        ),
    ]
    seed_metrics, axis_metrics = compute_seed_metrics(
        pd.concat(frames, ignore_index=True)
    )
    model_summary = summarize_models(
        seed_metrics,
        args.baseline_model_label,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    selected, interaction = compare_selected(
        seed_metrics,
        args.reference_name,
        args.candidate_name,
        args.reference_model,
        args.candidate_model,
        args.baseline_model_label,
        args.bootstrap_replicates,
        args.bootstrap_seed + 1,
    )
    axis_effects = selected_axis_effects(
        axis_metrics,
        args.reference_name,
        args.candidate_name,
        args.reference_model,
        args.candidate_model,
        args.baseline_model_label,
    )

    seed_metrics.to_csv(out_dir / "seed_metrics.csv", index=False)
    axis_metrics.to_csv(out_dir / "seed_axis_metrics.csv", index=False)
    model_summary.to_csv(out_dir / "model_summary.csv", index=False)
    selected.to_csv(out_dir / "selected_cap_effects.csv", index=False)
    interaction.to_csv(out_dir / "cross_version_effect_interaction.csv", index=False)
    axis_effects.to_csv(out_dir / "selected_cap_axis_effects.csv", index=False)
    plot_selected_effects(selected, out_dir, args.formats)

    report = f"""# TD3 TV-Cap Cross-Version Comparison

Reference: {args.reference_name}, model `{args.reference_model}`

Candidate: {args.candidate_name}, model `{args.candidate_model}`

All returns are normalized by the corresponding environment version's
cross-seed vanilla nominal median. Perturbation axes receive equal weight.
Bootstrap samples treat each independently trained policy seed as one cluster.
The cross-version interaction is the candidate-version robust-minus-vanilla
effect minus the reference-version robust-minus-vanilla effect. It therefore
does not confuse a raw reward-scale change with replication of the clipping
effect.
"""
    (out_dir / "README.md").write_text(report, encoding="utf-8")
    print(selected.to_string(index=False))
    print(interaction.to_string(index=False))


if __name__ == "__main__":
    main()
