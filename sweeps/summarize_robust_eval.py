import argparse
import csv
import math
from collections import defaultdict

import numpy as np


def mean_std_ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, std, ci95


def main():
    parser = argparse.ArgumentParser(description="Summarize multi-seed robust eval metrics.")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--baseline-model-label", default="vanilla")
    parser.add_argument("--robust-model-label", default="tv90")
    parser.add_argument("--nominal-scenario-label", default="nominal")
    parser.add_argument("--perturbed-scenario-label", default="adversarial")
    parser.add_argument("--out-csv", default="sweeps/robust_eval_summary.csv")
    args = parser.parse_args()

    rows = []
    with open(args.metrics_csv, newline="", encoding="utf-8") as fobj:
        reader = csv.DictReader(fobj)
        for row in reader:
            row["seed"] = int(row["seed"])
            row["timestamp"] = int(row["timestamp"])
            for key in ("mean_return", "median_return", "iqm_return", "std_return", "min_return", "max_return"):
                row[key] = float(row[key])
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No rows found in {args.metrics_csv}")

    # Keep only the latest row for each (model, scenario, seed) to avoid stale reruns.
    latest: dict[tuple[str, str, int], dict] = {}
    for row in rows:
        key = (row["model_label"], row["scenario_label"], row["seed"])
        prev = latest.get(key)
        if prev is None or row["timestamp"] > prev["timestamp"]:
            latest[key] = row

    grouped = defaultdict(list)
    for key, row in latest.items():
        grouped[(key[0], key[1])].append(row)

    print("Per-group summary (across seeds):")
    summary_rows = []
    for (model, scenario), grows in sorted(grouped.items()):
        mean_vals = [r["mean_return"] for r in grows]
        med_vals = [r["median_return"] for r in grows]
        iqm_vals = [r["iqm_return"] for r in grows]
        mean_mean, _, mean_ci = mean_std_ci95(mean_vals)
        med_mean, _, med_ci = mean_std_ci95(med_vals)
        iqm_mean, _, iqm_ci = mean_std_ci95(iqm_vals)
        print(
            f"- model={model}, scenario={scenario}, n={len(grows)} | "
            f"mean={mean_mean:.3f}±{mean_ci:.3f}, median={med_mean:.3f}±{med_ci:.3f}, iqm={iqm_mean:.3f}±{iqm_ci:.3f}"
        )
        summary_rows.append(
            {
                "kind": "group",
                "model_label": model,
                "scenario_label": scenario,
                "n": len(grows),
                "mean_return_mean": mean_mean,
                "mean_return_ci95": mean_ci,
                "median_return_mean": med_mean,
                "median_return_ci95": med_ci,
                "iqm_return_mean": iqm_mean,
                "iqm_return_ci95": iqm_ci,
            }
        )

    # Per-seed drops by model
    drops_by_model = defaultdict(list)
    models = {key[0] for key in latest.keys()}
    seeds = {key[2] for key in latest.keys()}
    for model in models:
        for seed in seeds:
            nominal = latest.get((model, args.nominal_scenario_label, seed))
            perturbed = latest.get((model, args.perturbed_scenario_label, seed))
            if nominal is None or perturbed is None:
                continue
            drop_mean = nominal["mean_return"] - perturbed["mean_return"]
            drop_median = nominal["median_return"] - perturbed["median_return"]
            rel_drop = drop_mean / nominal["mean_return"] if nominal["mean_return"] != 0 else float("nan")
            drops_by_model[model].append(
                {
                    "seed": seed,
                    "drop_mean": drop_mean,
                    "drop_median": drop_median,
                    "rel_drop_mean": rel_drop,
                }
            )

    print("\nDrop summary (nominal - perturbed, lower is better):")
    for model, drows in sorted(drops_by_model.items()):
        dm = [r["drop_mean"] for r in drows]
        dmed = [r["drop_median"] for r in drows]
        rd = [r["rel_drop_mean"] for r in drows]
        dm_mean, _, dm_ci = mean_std_ci95(dm)
        dmed_mean, _, dmed_ci = mean_std_ci95(dmed)
        rd_mean, _, rd_ci = mean_std_ci95(rd)
        print(
            f"- model={model}, n={len(drows)} | "
            f"drop_mean={dm_mean:.3f}±{dm_ci:.3f}, drop_median={dmed_mean:.3f}±{dmed_ci:.3f}, rel_drop={rd_mean:.3f}±{rd_ci:.3f}"
        )
        summary_rows.append(
            {
                "kind": "drop",
                "model_label": model,
                "scenario_label": f"{args.nominal_scenario_label}->{args.perturbed_scenario_label}",
                "n": len(drows),
                "mean_return_mean": dm_mean,
                "mean_return_ci95": dm_ci,
                "median_return_mean": dmed_mean,
                "median_return_ci95": dmed_ci,
                "iqm_return_mean": rd_mean,
                "iqm_return_ci95": rd_ci,
            }
        )

    # Robust gain: how much less drop robust has than baseline.
    b = args.baseline_model_label
    r = args.robust_model_label
    if b in drops_by_model and r in drops_by_model:
        b_by_seed = {row["seed"]: row for row in drops_by_model[b]}
        r_by_seed = {row["seed"]: row for row in drops_by_model[r]}
        common = sorted(set(b_by_seed.keys()) & set(r_by_seed.keys()))
        if common:
            gain_mean = [b_by_seed[s]["drop_mean"] - r_by_seed[s]["drop_mean"] for s in common]
            gain_med = [b_by_seed[s]["drop_median"] - r_by_seed[s]["drop_median"] for s in common]
            gm, _, gm_ci = mean_std_ci95(gain_mean)
            gmed, _, gmed_ci = mean_std_ci95(gain_med)
            print("\nRobust gain (baseline_drop - robust_drop, higher is better):")
            print(f"- seeds={len(common)} | gain_mean={gm:.3f}±{gm_ci:.3f}, gain_median={gmed:.3f}±{gmed_ci:.3f}")
            summary_rows.append(
                {
                    "kind": "gain",
                    "model_label": f"{b}_vs_{r}",
                    "scenario_label": f"{args.nominal_scenario_label}->{args.perturbed_scenario_label}",
                    "n": len(common),
                    "mean_return_mean": gm,
                    "mean_return_ci95": gm_ci,
                    "median_return_mean": gmed,
                    "median_return_ci95": gmed_ci,
                    "iqm_return_mean": float("nan"),
                    "iqm_return_ci95": float("nan"),
                }
            )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fobj:
        fieldnames = [
            "kind",
            "model_label",
            "scenario_label",
            "n",
            "mean_return_mean",
            "mean_return_ci95",
            "median_return_mean",
            "median_return_ci95",
            "iqm_return_mean",
            "iqm_return_ci95",
        ]
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nsummary written to {args.out_csv}")


if __name__ == "__main__":
    main()
