"""
Extract nominal performance and clipping activity from the alpha grid sweep.
Usage:
    python sweeps/extract_alpha_results.py
    python sweeps/extract_alpha_results.py --group ppo-alpha-norm-coarse --last-frac 0.25
"""

import argparse
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

ENTITY  = "arjun-mahesh-duke-university"
PROJECT = "fixed-alpha-randomness"

METRICS = {
    "episodic_return":    "charts/episodic_return",
    "clip_fraction":      "robust/tv_return_clip_fraction",
    "excess_mean":        "robust/tv_return_excess_mean",
    "explained_variance": "losses/explained_variance",
    "value_loss":         "losses/value_loss",
    "p95_pre_clip":       "targets/returns_p95_pre_transform",
    "p99_pre_clip":       "targets/returns_p99_pre_transform",
}

VARIANT_ORDER = ["vanilla", "noop", "2.85", "2.95", "3.05", "3.20", "3.50"]


def get_variant(run) -> str:
    """Derive variant label from run config.

    Actual exp_name format used by the training script:
        ppo_alpha_vanilla   ppo_alpha_noop
        ppo_alpha_a2p85     ppo_alpha_a2p95
        ppo_alpha_a3p05     ppo_alpha_a3p20   ppo_alpha_a3p50
    Fallback: read tv_fixed_cap directly from config.
    """
    cfg = run.config
    # Primary: use tv flags from config (most reliable)
    if not cfg.get("tv_clip_value_targets", False):
        return "vanilla"
    cap = cfg.get("tv_fixed_cap", None)
    if cap is None or float(cap) >= 1e8:
        return "noop"
    # Format float cap to match VARIANT_ORDER keys e.g. 3.5 -> "3.50"
    cap_f = float(cap)
    formatted = f"{cap_f:.2f}"
    if formatted in VARIANT_ORDER:
        return formatted
    # Fallback: nearest match
    return formatted


def late_mean(history, key: str, last_frac: float) -> float | None:
    if key not in history.columns:
        return None
    col = history[key].dropna()
    if len(col) == 0:
        return None
    n = max(1, int(len(col) * last_frac))
    return float(col.iloc[-n:].mean())


def late_mean_from_run(run, key: str, last_frac: float) -> float | None:
    """Fetch one metric at a time from W&B history.

    W&B's `run.history(keys=[...])` can return an empty frame when different
    keys are logged on different steps and the intersection is sparse. Fetching
    each key independently is slower but reliable for this sweep size.
    """
    values = []
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group",     default="ppo-alpha-norm-coarse")
    parser.add_argument("--last-frac", type=float, default=0.25,
                        help="Fraction of training steps considered 'late'")
    parser.add_argument("--min-steps", type=int, default=50_000,
                        help="Skip runs shorter than this (incomplete)")
    parser.add_argument("--use-summary", action="store_true",
                        help="Fast mode: read final value from run.summary instead of scan_history")
    parser.add_argument("--timeout", type=int, default=180,
                        help="W&B API timeout in seconds")
    args = parser.parse_args()

    import wandb
    api = wandb.Api(timeout=args.timeout)

    print(f"Fetching runs: {ENTITY}/{PROJECT}  group={args.group} …")
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"group": args.group, "state": "finished"},
    )
    runs = list(runs)
    print(f"  {len(runs)} finished runs found.")

    # Accumulate per-variant per-metric lists
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for run in runs:
        # Completeness check: state=finished already filters crashed runs.
        # Also require SPS > 0 (proves training actually ran) and
        # a minimum number of logged iterations via charts/SPS being present.
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
            if val is not None:
                data[variant][metric_name].append(val)
                had_any_metric = True
        if not had_any_metric:
            skipped += 1

    if skipped:
        print(f"  {skipped} runs skipped (incomplete / no history).\n")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def iqm(vals: list[float]) -> float:
        """Interquartile mean: drop bottom and top 25%, average the rest."""
        if len(vals) < 2:
            return float(np.mean(vals)) if vals else float("nan")
        arr = np.sort(vals)
        lo, hi = int(np.floor(0.25 * len(arr))), int(np.ceil(0.75 * len(arr)))
        return float(arr[lo:hi].mean())

    def fmt(vals: list[float], decimals: int = 1) -> str:
        if not vals:
            return "[ — ]"
        rounded = [round(v, decimals) for v in sorted(vals)]
        return f"{rounded}  iqm={round(iqm(vals), decimals)}"

    # ── Table 1: Nominal performance ──────────────────────────────────────────
    print(f"\n{'─'*95}")
    print(f"{'VARIANT':<10}  {'EPISODIC RETURN (sorted seeds)':<52}  {'IQM':>7}  N")
    print(f"{'─'*95}")
    for v in VARIANT_ORDER:
        if v not in data:
            continue
        vals = data[v]["episodic_return"]
        print(f"{v:<10}  {fmt(vals, 0):<52}  {round(iqm(vals),0):>7}  {len(vals)}")

    # ── Table 2: Value learning (mechanism proof) ─────────────────────────────
    print(f"\n{'─'*95}")
    print(f"{'VARIANT':<10}  {'EXPLAINED VARIANCE (mean)':<30}  {'VALUE LOSS (mean)':<25}")
    print(f"{'─'*95}")
    for v in VARIANT_ORDER:
        if v not in data:
            continue
        ev  = data[v]["explained_variance"]
        vl  = data[v]["value_loss"]
        ev_str = f"{[round(x,3) for x in sorted(ev)]}  mean={round(np.mean(ev),3)}" if ev else "—"
        vl_str = f"{[round(x,4) for x in sorted(vl)]}  mean={round(np.mean(vl),4)}" if vl else "—"
        print(f"{v:<10}  {ev_str:<30}  {vl_str:<25}")

    # ── Table 3: Clipping activity ────────────────────────────────────────────
    print(f"\n{'─'*110}")
    print(f"{'VARIANT':<10}  {'CLIP FRACTION (sorted seeds)':<46}  {'EXCESS MEAN':<40}")
    print(f"{'─'*110}")
    for v in VARIANT_ORDER:
        if v not in data or v == "vanilla":
            continue
        cf = data[v]["clip_fraction"]
        em = data[v]["excess_mean"]
        print(f"{v:<10}  {fmt(cf, 4):<46}  {fmt(em, 4):<40}")

    # ── Table 4: Value target distribution ───────────────────────────────────
    print(f"\n{'─'*90}")
    print(f"{'VARIANT':<10}  {'P95 pre-clip (mean)':<30}  {'P99 pre-clip (mean)':<30}")
    print(f"{'─'*90}")
    for v in VARIANT_ORDER:
        if v not in data or v == "vanilla":
            continue
        p95 = data[v]["p95_pre_clip"]
        p99 = data[v]["p99_pre_clip"]
        p95_str = f"{round(np.mean(p95),3)}" if p95 else "—"
        p99_str = f"{round(np.mean(p99),3)}" if p99 else "—"
        print(f"{v:<10}  {p95_str:<30}  {p99_str:<30}")

    print()

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        variants_with_data = [v for v in VARIANT_ORDER if v in data]
        colors = plt.cm.tab10(np.linspace(0, 1, len(variants_with_data)))
        color_map = dict(zip(variants_with_data, colors))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Alpha grid sweep — {args.group}", fontsize=13)

        plot_specs = [
            ("episodic_return",  axes[0], "Nominal return (late-training mean)", 0),
            ("clip_fraction",    axes[1], "Clip fraction (late-training mean)",  4),
            ("explained_variance", axes[2], "Explained variance (late-training mean)", 3),
        ]

        for metric, ax, title, dec in plot_specs:
            xs, ys, errs, cols = [], [], [], []
            for v in variants_with_data:
                vals = data[v].get(metric, [])
                if not vals:
                    continue
                xs.append(v)
                ys.append(np.mean(vals))
                errs.append(np.std(vals))
                cols.append(color_map[v])
            ax.bar(xs, ys, yerr=errs, color=cols, capsize=4, alpha=0.85)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Alpha variant")
            ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        out = "sweeps/alpha_grid_summary.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out}")
        plt.show()

    except ImportError:
        print("matplotlib not available — skipping plots.")


if __name__ == "__main__":
    main()
