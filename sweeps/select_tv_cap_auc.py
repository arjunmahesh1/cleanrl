import argparse
import csv
import math
import os
import re
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_seeds(spec: str) -> set[int]:
    seeds = set()
    for token in spec.split(","):
        token = token.strip()
        if token:
            seeds.add(int(token))
    return seeds


def cap_from_exp_name(exp_name: str, exp_prefix: str) -> float | None:
    if not exp_name.startswith(exp_prefix):
        return None
    suffix = exp_name[len(exp_prefix) :]
    match = re.match(r"([0-9]+(?:p[0-9]+)?)", suffix)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def load_curve(run_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        acc.Reload()
    except Exception:
        return None
    if tag not in acc.Tags().get("scalars", []):
        return None
    events = acc.Scalars(tag)
    if len(events) < 2:
        return None
    steps = np.array([float(e.step) for e in events], dtype=np.float64)
    vals = np.array([float(e.value) for e in events], dtype=np.float64)
    order = np.argsort(steps)
    return steps[order], vals[order]


def mean_std_ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, std, ci95


def main():
    parser = argparse.ArgumentParser(description="Select fixed TV cap by normalized AUC.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--env-id", default="Walker2d-v4")
    parser.add_argument("--exp-prefix", default="ppo_cont_tv_fixedcap")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--tag", default="charts/episodic_return")
    parser.add_argument("--out-csv", default="sweeps/tv_cap_auc_sweep.csv")
    args = parser.parse_args()

    seed_filter = parse_seeds(args.seeds)
    rows = []

    for entry in os.scandir(args.run_dir):
        if not entry.is_dir():
            continue
        parts = entry.name.split("__")
        if len(parts) < 4:
            continue
        env_id, exp_name, seed_str = parts[0], parts[1], parts[2]
        if env_id != args.env_id:
            continue
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        if seed_filter and seed not in seed_filter:
            continue

        cap = cap_from_exp_name(exp_name, args.exp_prefix)
        if cap is None:
            continue

        curve = load_curve(entry.path, args.tag)
        if curve is None:
            continue
        steps, vals = curve
        auc = float(np.trapz(vals, steps))
        max_step = float(np.max(steps))
        auc_norm = float(auc / max_step) if max_step > 0 else float("nan")
        rows.append(
            {
                "run_name": entry.name,
                "exp_name": exp_name,
                "seed": seed,
                "tv_fixed_cap": cap,
                "max_step": max_step,
                "auc": auc,
                "auc_norm": auc_norm,
                "final_return": float(vals[-1]),
            }
        )

    if not rows:
        raise RuntimeError("No matching runs found. Check --run-dir/--env-id/--exp-prefix/--seeds.")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote per-run AUC rows: {args.out_csv}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["tv_fixed_cap"]].append(row["auc_norm"])

    summary = []
    for cap, vals in grouped.items():
        mean, std, ci95 = mean_std_ci95(vals)
        summary.append(
            {
                "tv_fixed_cap": cap,
                "n_runs": len(vals),
                "mean_auc_norm": mean,
                "std_auc_norm": std,
                "ci95_auc_norm": ci95,
            }
        )

    summary.sort(key=lambda r: r["mean_auc_norm"], reverse=True)
    print("\nAUC summary by fixed cap (higher is better):")
    for i, row in enumerate(summary, start=1):
        print(
            f"{i:2d}. cap={row['tv_fixed_cap']:.4f}, n={row['n_runs']}, "
            f"mean_auc_norm={row['mean_auc_norm']:.4f}, ci95={row['ci95_auc_norm']:.4f}"
        )

    best = summary[0]
    print(
        f"\nselected cap={best['tv_fixed_cap']:.4f} "
        f"(mean_auc_norm={best['mean_auc_norm']:.4f}, n={best['n_runs']})"
    )


if __name__ == "__main__":
    main()
