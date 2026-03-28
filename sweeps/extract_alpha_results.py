"""
Extract alpha sweep results from W&B and optionally write a report bundle.

Typical usage:
    python sweeps/extract_alpha_results.py --group ppo-alpha-norm-coarse --use-summary
    python sweeps/extract_alpha_results.py --group ppo-alpha-norm-fine-expanded --out-dir sweeps/results/PPO_AlphaGridNorm_fine_20260328 --write-readme
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import DefaultDict
import warnings

import numpy as np

warnings.filterwarnings("ignore")

ENTITY = "arjun-mahesh-duke-university"
PROJECT = "fixed-alpha-randomness"

METRICS = {
    "episodic_return": "charts/episodic_return",
    "eval_episodic_return": "eval/episodic_return",
    "clip_fraction": "robust/tv_return_clip_fraction",
    "excess_mean": "robust/tv_return_excess_mean",
    "explained_variance": "losses/explained_variance",
    "value_loss": "losses/value_loss",
    "p95_pre_clip": "targets/returns_p95_pre_transform",
    "p99_pre_clip": "targets/returns_p99_pre_transform",
}

VARIANT_ORDER = [
    "vanilla",
    "noop",
    "2.85",
    "2.90",
    "2.95",
    "3.00",
    "3.05",
    "3.10",
    "3.20",
    "3.50",
    "3.70",
    "4.00",
]


def get_variant(run) -> str:
    cfg = run.config
    if not cfg.get("tv_clip_value_targets", False):
        return "vanilla"

    cap = cfg.get("tv_fixed_cap")
    if cap is None:
        return "noop"

    cap_f = float(cap)
    if cap_f >= 1e8:
        return "noop"
    return f"{cap_f:.2f}"


def late_mean_from_run(run, key: str, last_frac: float) -> float | None:
    values: list[float] = []
    for row in run.scan_history(keys=[key], page_size=1000):
        val = row.get(key)
        if val is None:
            continue
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    n = max(1, int(len(values) * last_frac))
    return float(np.mean(values[-n:]))


def summary_value(run, key: str) -> float | None:
    val = run.summary.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def iqm(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    if len(vals) < 2:
        return float(np.mean(vals))
    arr = np.sort(vals)
    lo = int(np.floor(0.25 * len(arr)))
    hi = int(np.ceil(0.75 * len(arr)))
    return float(arr[lo:hi].mean())


def fmt_sorted(vals: list[float], decimals: int) -> str:
    if not vals:
        return "[]"
    rounded = [round(v, decimals) for v in sorted(vals)]
    return str(rounded)


def mean_or_dash(vals: list[float], decimals: int = 3) -> str:
    if not vals:
        return "-"
    return str(round(float(np.mean(vals)), decimals))


def find_best_variant(data: dict[str, dict[str, list[float]]]) -> str | None:
    candidates: list[tuple[float, str]] = []
    for variant, metrics in data.items():
        if variant in {"vanilla", "noop"}:
            continue
        clip_vals = metrics.get("clip_fraction", [])
        ret_vals = metrics.get("episodic_return", [])
        if not clip_vals or not ret_vals:
            continue
        clip_iqm = iqm(clip_vals)
        ret_iqm = iqm(ret_vals)
        if clip_iqm <= 0.0:
            continue
        if 0.005 <= clip_iqm <= 0.050:
            candidates.append((ret_iqm, variant))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def build_report_text(
    args: argparse.Namespace,
    data: dict[str, dict[str, list[float]]],
    skipped: int,
) -> str:
    lines: list[str] = []
    extraction_mode = (
        "run.summary (final logged value per run)"
        if args.use_summary
        else f"last {int(args.last_frac * 100)}% of each run via scan_history"
    )

    lines.append(f"Alpha Grid Sweep - {args.group}")
    lines.append(f"Project: {ENTITY}/{PROJECT}")
    lines.append(f"Date: {args.report_date}")
    lines.append(f"Extraction mode: {extraction_mode}")
    lines.append(f"Finished runs processed: {sum(len(m.get('episodic_return', [])) > 0 for m in data.values())}")
    if skipped:
        lines.append(f"Runs skipped: {skipped}")
    lines.append("")

    lines.append("=" * 95)
    lines.append("TABLE 1: NOMINAL PERFORMANCE")
    lines.append("=" * 95)
    lines.append(f"{'VARIANT':<10}  {'EPISODIC RETURN (sorted seeds)':<52}  {'IQM':>7}  N")
    lines.append("-" * 95)
    for variant in VARIANT_ORDER:
        if variant not in data:
            continue
        vals = data[variant].get("episodic_return", [])
        if not vals:
            continue
        lines.append(
            f"{variant:<10}  {fmt_sorted(vals, 0):<52}  {round(iqm(vals), 0):>7}  {len(vals)}"
        )
    lines.append("")

    lines.append("=" * 95)
    lines.append("TABLE 2: VALUE LEARNING")
    lines.append("=" * 95)
    lines.append(f"{'VARIANT':<10}  {'EXPLAINED VARIANCE (sorted / mean)':<38}  {'VALUE LOSS (sorted / mean)':<38}")
    lines.append("-" * 95)
    for variant in VARIANT_ORDER:
        if variant not in data:
            continue
        ev = data[variant].get("explained_variance", [])
        vl = data[variant].get("value_loss", [])
        ev_str = f"{fmt_sorted(ev, 3)} mean={mean_or_dash(ev, 3)}" if ev else "-"
        vl_str = f"{fmt_sorted(vl, 4)} mean={mean_or_dash(vl, 4)}" if vl else "-"
        lines.append(f"{variant:<10}  {ev_str:<38}  {vl_str:<38}")
    lines.append("")

    lines.append("=" * 110)
    lines.append("TABLE 3: CLIPPING ACTIVITY")
    lines.append("=" * 110)
    lines.append(f"{'VARIANT':<10}  {'CLIP FRACTION (sorted / IQM)':<46}  {'EXCESS MEAN (sorted / IQM)':<46}")
    lines.append("-" * 110)
    for variant in VARIANT_ORDER:
        if variant not in data or variant == "vanilla":
            continue
        cf = data[variant].get("clip_fraction", [])
        em = data[variant].get("excess_mean", [])
        cf_str = f"{fmt_sorted(cf, 4)} iqm={round(iqm(cf), 4)}" if cf else "-"
        em_str = f"{fmt_sorted(em, 4)} iqm={round(iqm(em), 4)}" if em else "-"
        lines.append(f"{variant:<10}  {cf_str:<46}  {em_str:<46}")
    lines.append("")

    lines.append("=" * 90)
    lines.append("TABLE 4: TARGET DISTRIBUTION")
    lines.append("=" * 90)
    lines.append(f"{'VARIANT':<10}  {'P95 pre-clip mean':<22}  {'P99 pre-clip mean':<22}  {'Eval return IQM':<16}")
    lines.append("-" * 90)
    for variant in VARIANT_ORDER:
        if variant not in data:
            continue
        p95 = data[variant].get("p95_pre_clip", [])
        p99 = data[variant].get("p99_pre_clip", [])
        eval_ret = data[variant].get("eval_episodic_return", [])
        lines.append(
            f"{variant:<10}  {mean_or_dash(p95, 3):<22}  {mean_or_dash(p99, 3):<22}  "
            f"{('-' if not eval_ret else round(iqm(eval_ret), 0)):<16}"
        )
    lines.append("")

    best_variant = find_best_variant(data)
    if best_variant is not None:
        lines.append(f"Best active variant by current rule: {best_variant}")
    else:
        lines.append("Best active variant by current rule: none")

    return "\n".join(lines) + "\n"


def build_summary_rows(data: dict[str, dict[str, list[float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in VARIANT_ORDER:
        if variant not in data:
            continue
        metrics = data[variant]
        for metric_name, values in metrics.items():
            if not values:
                continue
            rows.append(
                {
                    "variant": variant,
                    "metric": metric_name,
                    "n": len(values),
                    "sorted_values": "|".join(str(v) for v in sorted(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "iqm": iqm(values),
                }
            )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "metric", "n", "sorted_values", "mean", "median", "iqm"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    path: Path,
    args: argparse.Namespace,
    data: dict[str, dict[str, list[float]]],
    plot_name: str,
    summary_txt_name: str,
    summary_csv_name: str,
) -> None:
    extraction_mode = (
        "final run.summary values"
        if args.use_summary
        else f"late-window means over the last {int(args.last_frac * 100)}% of each run"
    )
    vanilla_iqm = iqm(data.get("vanilla", {}).get("episodic_return", []))
    noop_iqm = iqm(data.get("noop", {}).get("episodic_return", []))
    best_variant = find_best_variant(data)

    lines: list[str] = []
    lines.append(f"# {args.readme_title}")
    lines.append("")
    lines.append(f"**Date:** {args.report_date}")
    lines.append(f"**WandB group:** `{args.group}`")
    lines.append(f"**Project:** `{PROJECT}`")
    lines.append("")
    lines.append("## Extraction procedure")
    lines.append("")
    lines.append(f"- Runs are pulled from W&B group `{args.group}`.")
    lines.append(f"- Variant labels are derived from the training config (`vanilla`, `noop`, or the fixed cap value).")
    lines.append(f"- This report uses **{extraction_mode}**.")
    lines.append("- IQM means: sort the 5 seed-level values, drop the min and max, average the middle 3.")
    lines.append("")
    lines.append("## High-level readout")
    lines.append("")
    if not np.isnan(vanilla_iqm):
        lines.append(f"- `vanilla` nominal IQM return: **{round(vanilla_iqm, 0)}**.")
    if not np.isnan(noop_iqm):
        lines.append(f"- `noop` nominal IQM return: **{round(noop_iqm, 0)}**.")
    if best_variant is not None:
        lines.append(f"- Current best active variant by the scripted rule: **`{best_variant}`**.")
    else:
        lines.append("- Current best active variant by the scripted rule: none.")
    lines.append("")
    lines.append("## Per-variant summary")
    lines.append("")
    lines.append("| Variant | Nominal IQM return | IQM clip fraction | IQM excess mean | Eval IQM return |")
    lines.append("|---|---:|---:|---:|---:|")
    for variant in VARIANT_ORDER:
        if variant not in data:
            continue
        metrics = data[variant]
        ret_iqm = iqm(metrics.get("episodic_return", []))
        cf_iqm = iqm(metrics.get("clip_fraction", []))
        em_iqm = iqm(metrics.get("excess_mean", []))
        eval_iqm = iqm(metrics.get("eval_episodic_return", []))
        ret_cell = "-" if np.isnan(ret_iqm) else str(round(ret_iqm, 0))
        cf_cell = "-" if np.isnan(cf_iqm) else str(round(cf_iqm, 4))
        em_cell = "-" if np.isnan(em_iqm) else str(round(em_iqm, 4))
        eval_cell = "-" if np.isnan(eval_iqm) else str(round(eval_iqm, 0))
        lines.append(f"| {variant} | {ret_cell} | {cf_cell} | {em_cell} | {eval_cell} |")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("```text")
    lines.append("outputs/")
    lines.append(f"  {summary_txt_name}")
    lines.append(f"  {summary_csv_name}")
    lines.append(f"  {plot_name}")
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Use the W&B training curves for detailed timing of when clipping turns on or fades out.")
    lines.append("- Use the last checkpoint policy for final robustness evaluation.")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(path: Path, args: argparse.Namespace, data: dict[str, dict[str, list[float]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    variants = [variant for variant in VARIANT_ORDER if variant in data]
    if not variants:
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    color_map = dict(zip(variants, colors))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Alpha grid sweep - {args.group}", fontsize=13)
    plot_specs = [
        ("episodic_return", axes[0], "Nominal return"),
        ("eval_episodic_return", axes[1], "Eval return"),
        ("clip_fraction", axes[2], "Clip fraction"),
        ("explained_variance", axes[3], "Explained variance"),
    ]

    for metric, ax, title in plot_specs:
        xs: list[str] = []
        ys: list[float] = []
        errs: list[float] = []
        cols: list[np.ndarray] = []
        for variant in variants:
            vals = data[variant].get(metric, [])
            if not vals:
                continue
            xs.append(variant)
            ys.append(float(np.mean(vals)))
            errs.append(float(np.std(vals)))
            cols.append(color_map[variant])
        if xs:
            ax.bar(xs, ys, yerr=errs, color=cols, capsize=4, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="ppo-alpha-norm-coarse")
    parser.add_argument("--last-frac", type=float, default=0.25)
    parser.add_argument("--use-summary", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out-dir", default=None, help="Optional results directory for txt/csv/png/README outputs")
    parser.add_argument("--summary-name", default="alpha_grid_summary.txt")
    parser.add_argument("--csv-name", default="alpha_grid_summary.csv")
    parser.add_argument("--plot-name", default="alpha_grid_summary.png")
    parser.add_argument("--write-readme", action="store_true")
    parser.add_argument("--readme-name", default="README.md")
    parser.add_argument("--readme-title", default="PPO Alpha Grid Results")
    parser.add_argument("--report-date", default=str(date.today()))
    args = parser.parse_args()

    import wandb

    api = wandb.Api(timeout=args.timeout)

    print(f"Fetching runs: {ENTITY}/{PROJECT} group={args.group} ...")
    runs = list(
        api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={"group": args.group, "state": "finished"},
        )
    )
    print(f"  {len(runs)} finished runs found.")

    data: DefaultDict[str, DefaultDict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for run in runs:
        if run.summary.get("charts/SPS", 0) < 10:
            skipped += 1
            continue

        variant = get_variant(run)
        had_any_metric = False
        for metric_name, wandb_key in METRICS.items():
            if args.use_summary:
                val = summary_value(run, wandb_key)
            else:
                try:
                    val = late_mean_from_run(run, wandb_key, args.last_frac)
                except Exception:
                    val = summary_value(run, wandb_key)
            if val is None:
                continue
            data[variant][metric_name].append(val)
            had_any_metric = True
        if not had_any_metric:
            skipped += 1

    report = build_report_text(args, data, skipped)
    print(report, end="")

    summary_rows = build_summary_rows(data)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir = out_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        summary_txt_path = outputs_dir / args.summary_name
        summary_csv_path = outputs_dir / args.csv_name
        plot_path = outputs_dir / args.plot_name

        summary_txt_path.write_text(report, encoding="utf-8")
        write_summary_csv(summary_csv_path, summary_rows)
        save_plot(plot_path, args, data)

        if args.write_readme:
            readme_path = out_dir / args.readme_name
            write_readme(
                readme_path,
                args,
                data,
                args.plot_name,
                args.summary_name,
                args.csv_name,
            )

        print(f"Report bundle written to {out_dir}")
    else:
        plot_path = Path("sweeps") / args.plot_name
        save_plot(plot_path, args, data)
        if plot_path.exists():
            print(f"Plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
