from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EvalRow:
    timestamp: int
    model_label: str
    scenario_label: str
    axis: str
    factor: float
    seed: int
    mean_return: float
    std_return: float
    median_return: float
    iqm_return: float
    min_return: float
    max_return: float
    eval_episodes: int
    model_path: str
    norm_stats_path: str
    eval_raw_rewards: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package alpha robustness eval raw_metrics into summary tables, plots, and a README."
    )
    parser.add_argument("--raw-metrics-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="PPO Alpha Robustness Evaluation")
    parser.add_argument("--baseline-model-label", default="vanilla")
    parser.add_argument("--comparison-model-labels", nargs="*", default=[])
    parser.add_argument("--model-order", nargs="*", default=[])
    parser.add_argument(
        "--display-label",
        action="append",
        default=[],
        help="Model label remap in the form raw=Display Name. Can be passed multiple times.",
    )
    parser.add_argument("--report-date", default=date.today().isoformat())
    parser.add_argument("--eval-metrics-name", default="eval_metrics_final.csv")
    parser.add_argument("--summary-name", default="summary_by_scenario.csv")
    parser.add_argument("--drop-name", default="drop_summary.csv")
    parser.add_argument("--gain-name", default="gain_summary.csv")
    parser.add_argument("--axis-overview-name", default="axis_overview.csv")
    parser.add_argument("--curve-points-name", default="curve_points.csv")
    parser.add_argument("--gain-curve-points-name", default="gain_curve_points.csv")
    parser.add_argument("--readme-name", default="README.md")
    parser.add_argument("--nominal-factor", type=float, default=1.0)
    parser.add_argument(
        "--exclude-axes",
        nargs="*",
        default=[],
        help="Optional axis names to exclude from summaries and plots.",
    )
    parser.add_argument(
        "--disable-variance-whiskers",
        action="store_true",
        help="Disable per-line variance whiskers on return/gain plots.",
    )
    return parser.parse_args()


def parse_display_map(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {
        "vanilla": "Vanilla",
        "a1e9": "No-op (1e9)",
        "noop": "No-op (1e9)",
        "a3p05": "TV cap=3.05",
    }
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --display-label {item!r}; expected raw=Display Name.")
        raw, display = item.split("=", 1)
        out[raw.strip()] = display.strip()
    return out


def parse_factor_token(token: str) -> float:
    sign = -1.0 if token.startswith("m") else 1.0
    if token.startswith("m"):
        token = token[1:]
    return sign * float(token.replace("p", "."))


def split_scenario_label(label: str) -> tuple[str, float]:
    if "_" not in label:
        if label == "nominal":
            return "nominal", 1.0
        raise ValueError(f"Unsupported scenario label format: {label!r}")
    axis, factor_token = label.rsplit("_", 1)
    return axis, parse_factor_token(factor_token)


def mean_ci95(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    std = float(np.std(arr, ddof=1))
    return mean, 1.96 * std / math.sqrt(arr.size)


def iqm(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    lower = np.quantile(arr, 0.25)
    upper = np.quantile(arr, 0.75)
    middle = arr[(arr >= lower) & (arr <= upper)]
    if middle.size == 0:
        middle = arr
    return float(np.mean(middle))


def load_latest_rows(raw_metrics_dir: Path) -> list[EvalRow]:
    latest: dict[tuple[str, str, int], EvalRow] = {}
    csv_paths = sorted(raw_metrics_dir.glob("*.csv"))
    if not csv_paths:
        raise RuntimeError(f"No CSV files found in {raw_metrics_dir}")
    skipped_rows = 0
    skipped_examples: list[str] = []

    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as fobj:
            reader = csv.DictReader(fobj)
            for line_no, row in enumerate(reader, start=2):
                try:
                    scenario_label = row["scenario_label"]
                    axis, factor = split_scenario_label(scenario_label)
                    eval_row = EvalRow(
                        timestamp=int(row["timestamp"]),
                        model_label=row["model_label"],
                        scenario_label=scenario_label,
                        axis=axis,
                        factor=factor,
                        seed=int(row["seed"]),
                        mean_return=float(row["mean_return"]),
                        std_return=float(row["std_return"]),
                        median_return=float(row["median_return"]),
                        iqm_return=float(row["iqm_return"]),
                        min_return=float(row["min_return"]),
                        max_return=float(row["max_return"]),
                        eval_episodes=int(row["eval_episodes"]),
                        model_path=row["model_path"],
                        norm_stats_path=row["norm_stats_path"],
                        eval_raw_rewards=int(row["eval_raw_rewards"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    skipped_rows += 1
                    if len(skipped_examples) < 5:
                        skipped_examples.append(f"{csv_path.name}:{line_no} ({exc})")
                    continue
                key = (eval_row.model_label, eval_row.scenario_label, eval_row.seed)
                prev = latest.get(key)
                if prev is None or eval_row.timestamp > prev.timestamp:
                    latest[key] = eval_row

    if skipped_rows:
        print(
            f"warning: skipped {skipped_rows} malformed raw_metrics rows from {raw_metrics_dir}"
            f" (examples: {', '.join(skipped_examples)})"
        )
    if not latest:
        raise RuntimeError(f"No valid rows found in {raw_metrics_dir}")
    return sorted(latest.values(), key=lambda r: (r.axis, r.factor, r.model_label, r.seed))


def write_eval_metrics(rows: list[EvalRow], path: Path) -> None:
    fieldnames = [
        "timestamp",
        "model_label",
        "scenario_label",
        "axis",
        "factor",
        "seed",
        "mean_return",
        "std_return",
        "median_return",
        "iqm_return",
        "min_return",
        "max_return",
        "eval_episodes",
        "model_path",
        "norm_stats_path",
        "eval_raw_rewards",
    ]
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def build_group_summary(rows: list[EvalRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], list[EvalRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.model_label, row.axis, row.factor)].append(row)

    summary: list[dict[str, object]] = []
    for (model_label, axis, factor), grows in sorted(grouped.items()):
        mean_vals = [r.mean_return for r in grows]
        median_vals = [r.median_return for r in grows]
        iqm_vals = [r.iqm_return for r in grows]
        mean_mean, mean_ci = mean_ci95(mean_vals)
        median_mean, median_ci = mean_ci95(median_vals)
        iqm_mean, iqm_ci = mean_ci95(iqm_vals)
        summary.append(
            {
                "model_label": model_label,
                "axis": axis,
                "factor": factor,
                "scenario_label": f"{axis}_{format_factor_token(factor)}",
                "n": len(grows),
                "mean_return_mean": mean_mean,
                "mean_return_ci95": mean_ci,
                "median_return_mean": median_mean,
                "median_return_ci95": median_ci,
                "iqm_return_mean": iqm_mean,
                "iqm_return_ci95": iqm_ci,
                "seed_iqm_of_mean_return": iqm(mean_vals),
            }
        )
    return summary


def format_factor_token(value: float) -> str:
    text = f"{value:.10g}"
    if text.startswith("-"):
        return "m" + text[1:].replace(".", "p")
    return text.replace(".", "p")


def build_drop_summary(rows: list[EvalRow], nominal_factor: float = 1.0) -> list[dict[str, object]]:
    by_key = {(r.model_label, r.axis, r.factor, r.seed): r for r in rows}
    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        nominal = by_key.get((row.model_label, row.axis, nominal_factor, row.seed))
        if nominal is None:
            nominal = by_key.get((row.model_label, "nominal", nominal_factor, row.seed))
        if nominal is None:
            continue
        drop_mean = nominal.mean_return - row.mean_return
        drop_median = nominal.median_return - row.median_return
        drop_iqm = nominal.iqm_return - row.iqm_return
        rel_drop = drop_mean / nominal.mean_return if nominal.mean_return else float("nan")
        grouped[(row.model_label, row.axis, row.factor)].append(
            {
                "drop_mean": drop_mean,
                "drop_median": drop_median,
                "drop_iqm": drop_iqm,
                "rel_drop_mean": rel_drop,
            }
        )

    summary: list[dict[str, object]] = []
    for (model_label, axis, factor), drows in sorted(grouped.items()):
        dm = [r["drop_mean"] for r in drows]
        dmed = [r["drop_median"] for r in drows]
        diqm = [r["drop_iqm"] for r in drows]
        rel = [r["rel_drop_mean"] for r in drows]
        dm_mean, dm_ci = mean_ci95(dm)
        dmed_mean, dmed_ci = mean_ci95(dmed)
        diqm_mean, diqm_ci = mean_ci95(diqm)
        rel_mean, rel_ci = mean_ci95(rel)
        summary.append(
            {
                "model_label": model_label,
                "axis": axis,
                "factor": factor,
                "scenario_label": f"{axis}_{format_factor_token(factor)}",
                "n": len(drows),
                "drop_mean_return_mean": dm_mean,
                "drop_mean_return_ci95": dm_ci,
                "drop_median_return_mean": dmed_mean,
                "drop_median_return_ci95": dmed_ci,
                "drop_iqm_return_mean": diqm_mean,
                "drop_iqm_return_ci95": diqm_ci,
                "rel_drop_mean_return_mean": rel_mean,
                "rel_drop_mean_return_ci95": rel_ci,
            }
        )
    return summary


def build_gain_summary(
    rows: list[EvalRow],
    baseline_model_label: str,
    comparison_model_labels: list[str],
    nominal_factor: float = 1.0,
) -> list[dict[str, object]]:
    by_key = {(r.model_label, r.axis, r.factor, r.seed): r for r in rows}
    summary: list[dict[str, object]] = []
    axes = sorted({r.axis for r in rows})
    factors = sorted({r.factor for r in rows})
    seeds = sorted({r.seed for r in rows})

    for compare_label in comparison_model_labels:
        for axis in axes:
            for factor in factors:
                gain_mean_vals: list[float] = []
                gain_median_vals: list[float] = []
                gain_iqm_vals: list[float] = []
                for seed in seeds:
                    base_nom = by_key.get((baseline_model_label, axis, nominal_factor, seed))
                    if base_nom is None:
                        base_nom = by_key.get((baseline_model_label, "nominal", nominal_factor, seed))
                    base_row = by_key.get((baseline_model_label, axis, factor, seed))
                    cmp_nom = by_key.get((compare_label, axis, nominal_factor, seed))
                    if cmp_nom is None:
                        cmp_nom = by_key.get((compare_label, "nominal", nominal_factor, seed))
                    cmp_row = by_key.get((compare_label, axis, factor, seed))
                    if None in (base_nom, base_row, cmp_nom, cmp_row):
                        continue
                    base_drop_mean = base_nom.mean_return - base_row.mean_return
                    cmp_drop_mean = cmp_nom.mean_return - cmp_row.mean_return
                    base_drop_median = base_nom.median_return - base_row.median_return
                    cmp_drop_median = cmp_nom.median_return - cmp_row.median_return
                    base_drop_iqm = base_nom.iqm_return - base_row.iqm_return
                    cmp_drop_iqm = cmp_nom.iqm_return - cmp_row.iqm_return
                    gain_mean_vals.append(base_drop_mean - cmp_drop_mean)
                    gain_median_vals.append(base_drop_median - cmp_drop_median)
                    gain_iqm_vals.append(base_drop_iqm - cmp_drop_iqm)

                if not gain_mean_vals:
                    continue

                gm, gm_ci = mean_ci95(gain_mean_vals)
                gmed, gmed_ci = mean_ci95(gain_median_vals)
                giqm_mean, giqm_ci = mean_ci95(gain_iqm_vals)
                summary.append(
                    {
                        "baseline_model_label": baseline_model_label,
                        "compare_model_label": compare_label,
                        "axis": axis,
                        "factor": factor,
                        "scenario_label": f"{axis}_{format_factor_token(factor)}",
                        "n": len(gain_mean_vals),
                        "gain_mean_return_mean": gm,
                        "gain_mean_return_ci95": gm_ci,
                        "gain_median_return_mean": gmed,
                        "gain_median_return_ci95": gmed_ci,
                        "gain_iqm_return_mean": giqm_mean,
                        "gain_iqm_return_ci95": giqm_ci,
                    }
                )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    elif fieldnames is None:
        raise RuntimeError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def build_axis_overview(
    summary_rows: list[dict[str, object]],
    gain_rows: list[dict[str, object]],
    baseline_model_label: str,
    comparison_model_labels: list[str],
    nominal_factor: float = 1.0,
) -> list[dict[str, object]]:
    by_group = {
        (row["model_label"], row["axis"], row["factor"]): row
        for row in summary_rows
    }
    by_gain = defaultdict(list)
    for row in gain_rows:
        if row["factor"] == nominal_factor:
            continue
        by_gain[(row["compare_model_label"], row["axis"])].append(row)

    axes = sorted({row["axis"] for row in summary_rows})
    overview: list[dict[str, object]] = []
    for axis in axes:
        base_nom = by_group.get((baseline_model_label, axis, nominal_factor))
        if base_nom is None:
            base_nom = by_group.get((baseline_model_label, "nominal", nominal_factor))
        if base_nom is None:
            continue
        overview.append(
            {
                "axis": axis,
                "model_label": baseline_model_label,
                "nominal_mean_return": base_nom["mean_return_mean"],
                "nominal_mean_return_ci95": base_nom["mean_return_ci95"],
                "avg_gain_mean_return": 0.0,
                "avg_gain_mean_return_ci95": 0.0,
                "positive_gain_scenarios": 0,
                "perturbed_scenarios": 0,
            }
        )
        for compare_label in comparison_model_labels:
            cmp_nom = by_group.get((compare_label, axis, 1.0))
            if cmp_nom is None:
                cmp_nom = by_group.get((compare_label, "nominal", 1.0))
            if cmp_nom is None:
                continue
            grows = by_gain.get((compare_label, axis), [])
            gain_vals = [float(row["gain_mean_return_mean"]) for row in grows]
            gain_cis = [float(row["gain_mean_return_ci95"]) for row in grows]
            overview.append(
                {
                    "axis": axis,
                    "model_label": compare_label,
                    "nominal_mean_return": cmp_nom["mean_return_mean"],
                    "nominal_mean_return_ci95": cmp_nom["mean_return_ci95"],
                    "avg_gain_mean_return": float(np.mean(gain_vals)) if gain_vals else float("nan"),
                    "avg_gain_mean_return_ci95": float(np.mean(gain_cis)) if gain_cis else float("nan"),
                    "positive_gain_scenarios": sum(1 for value in gain_vals if value > 0),
                    "perturbed_scenarios": len(gain_vals),
                }
            )
    return overview


def parse_alpha_like_label(label: str) -> float | None:
    if label in {"vanilla", "noop", "a1e9"}:
        return None
    if not label.startswith("a"):
        return None
    token = label[1:]
    sign = -1.0 if token.startswith("m") else 1.0
    if token.startswith("m"):
        token = token[1:]
    try:
        return sign * float(token.replace("p", "."))
    except ValueError:
        return None


def build_model_styles(model_labels: list[str]) -> dict[str, dict[str, str]]:
    # Fixed, high-contrast styles for known variants to avoid any color collisions.
    styles: dict[str, dict[str, str]] = {
        "vanilla": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "a1e9": {"color": "#6e6e6e", "marker": "s", "linestyle": "--"},
        "noop": {"color": "#6e6e6e", "marker": "s", "linestyle": "--"},
        "a2p85": {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
        "a2p95": {"color": "#ff7f0e", "marker": "D", "linestyle": "-"},
        "a3p00": {"color": "#9467bd", "marker": "P", "linestyle": "-"},
        "a3p05": {"color": "#17becf", "marker": "X", "linestyle": "-"},
        "a3p10": {"color": "#d62728", "marker": "v", "linestyle": "-"},
        "a3p20": {"color": "#8c564b", "marker": ">", "linestyle": "-"},
        "a3p50": {"color": "#e377c2", "marker": "<", "linestyle": "-"},
        "a3p70": {"color": "#bcbd22", "marker": "h", "linestyle": "-"},
        "a4p00": {"color": "#393b79", "marker": "*", "linestyle": "-"},
    }

    cap_palette = [
        "#637939",  # moss
        "#8c6d31",  # mustard-brown
        "#843c39",  # brick
        "#7f7f7f",  # medium gray
        "#3182bd",  # alt blue
        "#31a354",  # alt green
        "#756bb1",  # alt purple
        "#636363",  # dark gray
    ]
    cap_markers = ["^", "D", "P", "X", "v", ">", "<", "h", "*"]

    cap_labels = [label for label in model_labels if label not in styles]
    cap_labels = sorted(
        cap_labels,
        key=lambda label: (
            parse_alpha_like_label(label) is None,
            parse_alpha_like_label(label) if parse_alpha_like_label(label) is not None else label,
        ),
    )

    for idx, label in enumerate(cap_labels):
        styles[label] = {
            "color": cap_palette[idx % len(cap_palette)],
            "marker": cap_markers[idx % len(cap_markers)],
            "linestyle": "-",
        }

    return styles


def draw_vertical_range_whisker(
    ax,
    x: float,
    y_low: float,
    y_high: float,
    color: str,
) -> None:
    if not np.isfinite(x) or not np.isfinite(y_low) or not np.isfinite(y_high):
        return
    if y_high < y_low:
        y_low, y_high = y_high, y_low
    if y_high <= y_low:
        return
    x_min, x_max = ax.get_xlim()
    cap_half = max(0.003 * (x_max - x_min), 0.005)
    ax.vlines(x, y_low, y_high, color=color, linewidth=2.2, alpha=0.95, zorder=6)
    ax.hlines([y_low, y_high], x - cap_half, x + cap_half, color=color, linewidth=2.0, alpha=0.95, zorder=6)


def add_max_return_spread_whisker(
    ax,
    rows: list[EvalRow],
    axis: str,
    model_label: str,
    color: str,
) -> None:
    factor_to_vals: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r.axis == axis and r.model_label == model_label:
            factor_to_vals[r.factor].append(r.mean_return)
    if not factor_to_vals:
        return
    best_factor = None
    best_spread = -1.0
    best_low = 0.0
    best_high = 0.0
    for factor, vals in factor_to_vals.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            continue
        low = float(np.min(arr))
        high = float(np.max(arr))
        spread = high - low
        if spread > best_spread:
            best_spread = spread
            best_factor = float(factor)
            best_low = low
            best_high = high
    if best_factor is None:
        return
    draw_vertical_range_whisker(ax, best_factor, best_low, best_high, color)


def add_max_gain_spread_whisker(
    ax,
    by_key: dict[tuple[str, str, float, int], EvalRow],
    seeds: list[int],
    baseline_model_label: str,
    compare_label: str,
    axis: str,
    nominal_factor: float,
    color: str,
) -> None:
    candidate_factors = sorted(
        {
            factor
            for (model_label, row_axis, factor, _seed) in by_key.keys()
            if model_label == compare_label and row_axis == axis
        }
    )
    if not candidate_factors:
        return
    best_factor = None
    best_spread = -1.0
    best_low = 0.0
    best_high = 0.0
    for factor in candidate_factors:
        gains: list[float] = []
        for seed in seeds:
            base_nom = by_key.get((baseline_model_label, axis, nominal_factor, seed))
            base_row = by_key.get((baseline_model_label, axis, factor, seed))
            cmp_nom = by_key.get((compare_label, axis, nominal_factor, seed))
            cmp_row = by_key.get((compare_label, axis, factor, seed))
            if None in (base_nom, base_row, cmp_nom, cmp_row):
                continue
            base_drop = base_nom.mean_return - base_row.mean_return
            cmp_drop = cmp_nom.mean_return - cmp_row.mean_return
            gains.append(base_drop - cmp_drop)
        if not gains:
            continue
        arr = np.asarray(gains, dtype=np.float64)
        low = float(np.min(arr))
        high = float(np.max(arr))
        spread = high - low
        if spread > best_spread:
            best_spread = spread
            best_factor = float(factor)
            best_low = low
            best_high = high
    if best_factor is None:
        return
    draw_vertical_range_whisker(ax, best_factor, best_low, best_high, color)


def plot_return_curves(
    rows: list[EvalRow],
    summary_rows: list[dict[str, object]],
    model_order: list[str],
    display_map: dict[str, str],
    out_dir: Path,
    nominal_factor: float = 1.0,
    show_variance_whiskers: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = build_model_styles(model_order)

    axes = sorted({row["axis"] for row in summary_rows})
    fig, axs = plt.subplots(1, len(axes), figsize=(6 * len(axes), 4.8), sharey=True)
    if len(axes) == 1:
        axs = [axs]

    for ax, axis in zip(axs, axes):
        axis_rows = [row for row in summary_rows if row["axis"] == axis]
        for model_label in model_order:
            grows = sorted(
                [row for row in axis_rows if row["model_label"] == model_label],
                key=lambda row: float(row["factor"]),
            )
            if not grows:
                continue
            xs = np.asarray([float(row["factor"]) for row in grows], dtype=np.float64)
            ys = np.asarray([float(row["mean_return_mean"]) for row in grows], dtype=np.float64)
            cis = np.asarray([float(row["mean_return_ci95"]) for row in grows], dtype=np.float64)
            style = styles[model_label]
            color = style["color"]
            label = display_map.get(model_label, model_label)
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                color=color,
                label=label,
            )
            ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.18)
            if show_variance_whiskers:
                add_max_return_spread_whisker(ax, rows, axis, model_label, color)
        ax.axvline(nominal_factor, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.9)
        ax.set_title(axis.capitalize())
        ax.set_xlabel(f"{axis.capitalize()} scale")
        ax.grid(alpha=0.25)

    axs[0].set_ylabel("Average reward")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "return_curves_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for axis in axes:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        axis_rows = [row for row in summary_rows if row["axis"] == axis]
        for model_label in model_order:
            grows = sorted(
                [row for row in axis_rows if row["model_label"] == model_label],
                key=lambda row: float(row["factor"]),
            )
            if not grows:
                continue
            xs = np.asarray([float(row["factor"]) for row in grows], dtype=np.float64)
            ys = np.asarray([float(row["mean_return_mean"]) for row in grows], dtype=np.float64)
            cis = np.asarray([float(row["mean_return_ci95"]) for row in grows], dtype=np.float64)
            style = styles[model_label]
            color = style["color"]
            label = display_map.get(model_label, model_label)
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.4,
                color=color,
                label=label,
            )
            ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.18)
            if show_variance_whiskers:
                add_max_return_spread_whisker(ax, rows, axis, model_label, color)
        ax.axvline(nominal_factor, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.9, label="Nominal scale")
        ax.set_xlabel(f"{axis.capitalize()} scale")
        ax.set_ylabel("Average reward")
        ax.set_title(f"{axis.capitalize()} robustness")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{axis}_return_curve.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_gain_curves(
    rows: list[EvalRow],
    gain_rows: list[dict[str, object]],
    baseline_model_label: str,
    comparison_model_labels: list[str],
    display_map: dict[str, str],
    out_dir: Path,
    nominal_factor: float = 1.0,
    show_variance_whiskers: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not gain_rows:
        return

    styles = build_model_styles(comparison_model_labels)
    axes = sorted({row["axis"] for row in gain_rows})
    fig, axs = plt.subplots(1, len(axes), figsize=(6 * len(axes), 4.8), sharey=True)
    if len(axes) == 1:
        axs = [axs]

    by_key = {(r.model_label, r.axis, r.factor, r.seed): r for r in rows}
    seeds = sorted({r.seed for r in rows})

    for ax, axis in zip(axs, axes):
        axis_rows = [row for row in gain_rows if row["axis"] == axis]
        for compare_label in comparison_model_labels:
            grows = sorted(
                [row for row in axis_rows if row["compare_model_label"] == compare_label],
                key=lambda row: float(row["factor"]),
            )
            if not grows:
                continue
            xs = np.asarray([float(row["factor"]) for row in grows], dtype=np.float64)
            ys = np.asarray([float(row["gain_mean_return_mean"]) for row in grows], dtype=np.float64)
            cis = np.asarray([float(row["gain_mean_return_ci95"]) for row in grows], dtype=np.float64)
            style = styles[compare_label]
            color = style["color"]
            label = f"{display_map.get(compare_label, compare_label)} vs Vanilla"
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                color=color,
                label=label,
            )
            ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.18)
            if show_variance_whiskers:
                add_max_gain_spread_whisker(
                    ax,
                    by_key,
                    seeds,
                    baseline_model_label,
                    compare_label,
                    axis,
                    nominal_factor,
                    color,
                )
        ax.axhline(0.0, color="#111111", linestyle="--", linewidth=1.4, alpha=0.8)
        ax.axvline(nominal_factor, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.9)
        ax.set_title(axis.capitalize())
        ax.set_xlabel(f"{axis.capitalize()} scale")
        ax.grid(alpha=0.25)

    axs[0].set_ylabel("Robust gain (vanilla drop - model drop)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "gain_curves_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for axis in axes:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        axis_rows = [row for row in gain_rows if row["axis"] == axis]
        for compare_label in comparison_model_labels:
            grows = sorted(
                [row for row in axis_rows if row["compare_model_label"] == compare_label],
                key=lambda row: float(row["factor"]),
            )
            if not grows:
                continue
            xs = np.asarray([float(row["factor"]) for row in grows], dtype=np.float64)
            ys = np.asarray([float(row["gain_mean_return_mean"]) for row in grows], dtype=np.float64)
            cis = np.asarray([float(row["gain_mean_return_ci95"]) for row in grows], dtype=np.float64)
            style = styles[compare_label]
            color = style["color"]
            label = f"{display_map.get(compare_label, compare_label)} vs Vanilla"
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.4,
                color=color,
                label=label,
            )
            ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.18)
            if show_variance_whiskers:
                add_max_gain_spread_whisker(
                    ax,
                    by_key,
                    seeds,
                    baseline_model_label,
                    compare_label,
                    axis,
                    nominal_factor,
                    color,
                )
        ax.axhline(0.0, color="#111111", linestyle="--", linewidth=1.4, alpha=0.8, label="Zero gain")
        ax.axvline(nominal_factor, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.9, label="Nominal scale")
        ax.set_xlabel(f"{axis.capitalize()} scale")
        ax.set_ylabel("Robust gain (higher is better)")
        ax.set_title(f"{axis.capitalize()} robust gain")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{axis}_gain_curve.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def build_readme(
    title: str,
    report_date: str,
    raw_metrics_dir: Path,
    outputs_dir: Path,
    plots_dir: Path,
    summary_rows: list[dict[str, object]],
    overview_rows: list[dict[str, object]],
    model_order: list[str],
    comparison_model_labels: list[str],
    display_map: dict[str, str],
    nominal_factor: float = 1.0,
) -> str:
    by_group = {(row["model_label"], row["axis"], row["factor"]): row for row in summary_rows}
    axes = sorted({row["axis"] for row in summary_rows})
    lines: list[str] = [
        f"# {title}",
        "",
        f"Date: {report_date}",
        "",
        "This folder packages the final pinned robustness evaluation for normalized PPO.",
        "",
        "## Contents",
        "",
        f"- Raw metrics: `{raw_metrics_dir.as_posix()}`",
        f"- Aggregated outputs: `{outputs_dir.as_posix()}`",
        f"- Plots: `{plots_dir.as_posix()}`",
        "",
        "## Evaluation protocol",
        "",
        "- Models are compared on the same perturbation grid for the configured axes.",
        f"- Nominal reference within each axis is the `factor={nominal_factor}` point.",
        "- Curves show mean return across seeds with `95% CI` shading.",
        "- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.",
        "",
        "## Model labels",
        "",
    ]
    for model_label in model_order:
        lines.append(f"- `{model_label}` -> {display_map.get(model_label, model_label)}")

    lines += [
        "",
        "## Nominal returns by axis",
        "",
    ]

    nominal_rows: list[list[str]] = []
    for axis in axes:
        row = [axis]
        for model_label in model_order:
            grow = by_group.get((model_label, axis, nominal_factor))
            if grow is None:
                grow = by_group.get((model_label, "nominal", nominal_factor))
            if grow is None:
                row.append("n/a")
                continue
            row.append(f"{grow['mean_return_mean']:.2f} +/- {grow['mean_return_ci95']:.2f}")
        nominal_rows.append(row)
    lines += markdown_table(["Axis"] + [display_map.get(m, m) for m in model_order], nominal_rows)

    lines += [
        "",
        "## Axis overview",
        "",
    ]
    overview_md_rows: list[list[str]] = []
    for row in overview_rows:
        if row["model_label"] not in comparison_model_labels:
            continue
        overview_md_rows.append(
            [
                str(row["axis"]),
                display_map.get(str(row["model_label"]), str(row["model_label"])),
                f"{float(row['nominal_mean_return']):.2f} +/- {float(row['nominal_mean_return_ci95']):.2f}",
                f"{float(row['avg_gain_mean_return']):+.2f}",
                f"{int(row['positive_gain_scenarios'])}/{int(row['perturbed_scenarios'])}",
            ]
        )
    lines += markdown_table(
        ["Axis", "Model", "Nominal return", "Mean gain over perturbed scenarios", "Positive gain scenarios"],
        overview_md_rows,
    )

    lines += [
        "",
        "## Plot files",
        "",
        "- `plots/return_curves_panel.png`",
        "- `plots/gain_curves_panel.png`",
    ]
    for axis in axes:
        lines.append(f"- `plots/{axis}_return_curve.png`")
        lines.append(f"- `plots/{axis}_gain_curve.png`")

    lines += [
        "",
        "## Output CSV files",
        "",
        "- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.",
        "- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.",
        "- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.",
        "- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.",
        "- `outputs/axis_overview.csv`: compact axis-level overview.",
        "- `outputs/curve_points.csv`: same data used for return plots.",
        "- `outputs/gain_curve_points.csv`: same data used for gain plots.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    raw_metrics_dir = Path(args.raw_metrics_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    outputs_dir = out_dir / "outputs"
    plots_dir = out_dir / "plots"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    display_map = parse_display_map(args.display_label)
    rows = load_latest_rows(raw_metrics_dir)
    if args.exclude_axes:
        excluded = set(args.exclude_axes)
        rows = [row for row in rows if row.axis not in excluded]
        if not rows:
            raise RuntimeError(f"All rows were excluded by --exclude-axes={sorted(excluded)}")
    models = sorted({row.model_label for row in rows})

    comparison_model_labels = (
        args.comparison_model_labels
        if args.comparison_model_labels
        else [model for model in models if model != args.baseline_model_label]
    )
    model_order = (
        args.model_order
        if args.model_order
        else [args.baseline_model_label] + [m for m in models if m != args.baseline_model_label]
    )
    model_order = [m for m in model_order if m in models]

    eval_metrics_path = outputs_dir / args.eval_metrics_name
    summary_path = outputs_dir / args.summary_name
    drop_path = outputs_dir / args.drop_name
    gain_path = outputs_dir / args.gain_name
    axis_overview_path = outputs_dir / args.axis_overview_name
    curve_points_path = outputs_dir / args.curve_points_name
    gain_curve_points_path = outputs_dir / args.gain_curve_points_name

    write_eval_metrics(rows, eval_metrics_path)
    summary_rows = build_group_summary(rows)
    drop_rows = build_drop_summary(rows, nominal_factor=args.nominal_factor)
    gain_rows = build_gain_summary(
        rows,
        args.baseline_model_label,
        comparison_model_labels,
        nominal_factor=args.nominal_factor,
    )
    overview_rows = build_axis_overview(
        summary_rows,
        gain_rows,
        args.baseline_model_label,
        comparison_model_labels,
        nominal_factor=args.nominal_factor,
    )

    write_csv(
        summary_path,
        summary_rows,
        fieldnames=[
            "model_label",
            "axis",
            "factor",
            "scenario_label",
            "n",
            "mean_return_mean",
            "mean_return_ci95",
            "median_return_mean",
            "median_return_ci95",
            "iqm_return_mean",
            "iqm_return_ci95",
            "seed_iqm_of_mean_return",
        ],
    )
    write_csv(
        drop_path,
        drop_rows,
        fieldnames=[
            "model_label",
            "axis",
            "factor",
            "scenario_label",
            "n",
            "drop_mean_return_mean",
            "drop_mean_return_ci95",
            "drop_median_return_mean",
            "drop_median_return_ci95",
            "drop_iqm_return_mean",
            "drop_iqm_return_ci95",
            "rel_drop_mean_return_mean",
            "rel_drop_mean_return_ci95",
        ],
    )
    write_csv(
        gain_path,
        gain_rows,
        fieldnames=[
            "baseline_model_label",
            "compare_model_label",
            "axis",
            "factor",
            "scenario_label",
            "n",
            "gain_mean_return_mean",
            "gain_mean_return_ci95",
            "gain_median_return_mean",
            "gain_median_return_ci95",
            "gain_iqm_return_mean",
            "gain_iqm_return_ci95",
        ],
    )
    write_csv(
        axis_overview_path,
        overview_rows,
        fieldnames=[
            "axis",
            "model_label",
            "nominal_mean_return",
            "nominal_mean_return_ci95",
            "avg_gain_mean_return",
            "avg_gain_mean_return_ci95",
            "positive_gain_scenarios",
            "perturbed_scenarios",
        ],
    )
    write_csv(
        curve_points_path,
        summary_rows,
        fieldnames=[
            "model_label",
            "axis",
            "factor",
            "scenario_label",
            "n",
            "mean_return_mean",
            "mean_return_ci95",
            "median_return_mean",
            "median_return_ci95",
            "iqm_return_mean",
            "iqm_return_ci95",
            "seed_iqm_of_mean_return",
        ],
    )
    write_csv(
        gain_curve_points_path,
        gain_rows,
        fieldnames=[
            "baseline_model_label",
            "compare_model_label",
            "axis",
            "factor",
            "scenario_label",
            "n",
            "gain_mean_return_mean",
            "gain_mean_return_ci95",
            "gain_median_return_mean",
            "gain_median_return_ci95",
            "gain_iqm_return_mean",
            "gain_iqm_return_ci95",
        ],
    )

    plot_return_curves(
        rows,
        summary_rows,
        model_order,
        display_map,
        plots_dir,
        nominal_factor=args.nominal_factor,
        show_variance_whiskers=not args.disable_variance_whiskers,
    )
    plot_gain_curves(
        rows,
        gain_rows,
        args.baseline_model_label,
        comparison_model_labels,
        display_map,
        plots_dir,
        nominal_factor=args.nominal_factor,
        show_variance_whiskers=not args.disable_variance_whiskers,
    )

    readme = build_readme(
        title=args.title,
        report_date=args.report_date,
        raw_metrics_dir=raw_metrics_dir,
        outputs_dir=outputs_dir,
        plots_dir=plots_dir,
        summary_rows=summary_rows,
        overview_rows=overview_rows,
        model_order=model_order,
        comparison_model_labels=comparison_model_labels,
        display_map=display_map,
        nominal_factor=args.nominal_factor,
    )
    (out_dir / args.readme_name).write_text(readme, encoding="utf-8")

    print(f"packaged eval metrics: {eval_metrics_path}")
    print(f"scenario summary:      {summary_path}")
    print(f"drop summary:          {drop_path}")
    print(f"gain summary:          {gain_path}")
    print(f"axis overview:         {axis_overview_path}")
    print(f"plots:                 {plots_dir}")
    print(f"readme:                {out_dir / args.readme_name}")


if __name__ == "__main__":
    main()
