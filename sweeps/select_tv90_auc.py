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
        if not token:
            continue
        seeds.add(int(token))
    return seeds


def keep_prob_from_exp_name(exp_name: str) -> float | None:
    # Recommended naming convention: ..._kp0p99 or ..._kp0p995
    m = re.search(r"kp([0-9]+p[0-9]+)", exp_name)
    if m:
        return float(m.group(1).replace("p", "."))

    # Back-compat: kp099 / kp0995
    m = re.search(r"kp([0-9]{2,5})", exp_name)
    if m:
        digits = m.group(1)
        if digits.startswith("0"):
            return float("0." + digits.lstrip("0"))
        return float(digits)
    return None


def load_scalar(run_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()
    except Exception:
        return None

    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return None
    events = ea.Scalars(tag)
    if len(events) < 2:
        return None

    steps = np.array([float(e.step) for e in events], dtype=np.float64)
    values = np.array([float(e.value) for e in events], dtype=np.float64)
    order = np.argsort(steps)
    return steps[order], values[order]


def mean_std_ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, std, ci95


def main():
    parser = argparse.ArgumentParser(description="Select TV-90 keep_prob by AUC over training curves.")
    parser.add_argument("--run-dir", required=True, help="Directory containing TensorBoard run folders.")
    parser.add_argument("--env-id", default="Walker2d-v4")
    parser.add_argument("--exp-prefix", default="ppo_cont_tv90_upper_kp")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--tag", default="charts/episodic_return")
    parser.add_argument("--out-csv", default="sweeps/tv90_auc_sweep.csv")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    seed_filter = parse_seeds(args.seeds)
    rows: list[dict] = []

    for entry in os.scandir(args.run_dir):
        if not entry.is_dir():
            continue
        parts = entry.name.split("__")
        if len(parts) < 4:
            continue
        env_id, exp_name, seed_str = parts[0], parts[1], parts[2]
        if env_id != args.env_id:
            continue
        if not exp_name.startswith(args.exp_prefix):
            continue
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        if seed_filter and seed not in seed_filter:
            continue

        keep_prob = keep_prob_from_exp_name(exp_name)
        if keep_prob is None:
            continue

        loaded = load_scalar(entry.path, args.tag)
        if loaded is None:
            continue
        steps, values = loaded
        auc = float(np.trapz(values, steps))
        max_step = float(np.max(steps))
        auc_norm = float(auc / max_step) if max_step > 0 else float("nan")
        rows.append(
            {
                "run_name": entry.name,
                "exp_name": exp_name,
                "seed": seed,
                "keep_prob": keep_prob,
                "max_step": max_step,
                "auc": auc,
                "auc_norm": auc_norm,
                "final_return": float(values[-1]),
            }
        )

    if not rows:
        raise RuntimeError("No matching runs found. Check --run-dir, --exp-prefix, and --seeds.")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote per-run AUC rows: {args.out_csv}")

    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["keep_prob"]].append(row["auc_norm"])

    summary = []
    for kp, vals in grouped.items():
        mean, std, ci95 = mean_std_ci95(vals)
        summary.append(
            {
                "keep_prob": kp,
                "n_runs": len(vals),
                "mean_auc_norm": mean,
                "std_auc_norm": std,
                "ci95_auc_norm": ci95,
            }
        )

    summary.sort(key=lambda x: x["mean_auc_norm"], reverse=True)
    best = summary[0]
    print("\nAUC summary by keep_prob (higher is better):")
    for i, row in enumerate(summary[: args.top_k], start=1):
        print(
            f"{i:2d}. keep_prob={row['keep_prob']:.6f}, n={row['n_runs']}, "
            f"mean_auc_norm={row['mean_auc_norm']:.4f}, ci95={row['ci95_auc_norm']:.4f}"
        )

    print(
        f"\nselected keep_prob={best['keep_prob']:.6f} "
        f"(mean_auc_norm={best['mean_auc_norm']:.4f}, n={best['n_runs']})"
    )


if __name__ == "__main__":
    main()
