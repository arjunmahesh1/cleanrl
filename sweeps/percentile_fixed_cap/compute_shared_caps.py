import argparse
import csv
import math
import os
from dataclasses import dataclass

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class RunRecord:
    path: str
    run_name: str
    env_id: str
    exp_name: str
    seed: int
    timestamp: int


def parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def parse_seeds(spec: str) -> set[int]:
    return set(parse_int_list(spec))


def mean_std_ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, std, ci95


def format_cap_tag(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def load_scalar_curve(run_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
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
    steps = np.asarray([float(e.step) for e in events], dtype=np.float64)
    vals = np.asarray([float(e.value) for e in events], dtype=np.float64)
    order = np.argsort(steps)
    return steps[order], vals[order]


def select_runs(run_dir: str, env_id: str, exp_name_prefix: str, seeds: set[int]) -> list[RunRecord]:
    latest_by_seed: dict[int, RunRecord] = {}
    for entry in os.scandir(run_dir):
        if not entry.is_dir():
            continue
        parts = entry.name.split("__")
        if len(parts) < 4:
            continue
        run_env_id, exp_name, seed_str, timestamp_str = parts[0], parts[1], parts[2], parts[3]
        if run_env_id != env_id or not exp_name.startswith(exp_name_prefix):
            continue
        try:
            seed = int(seed_str)
            timestamp = int(timestamp_str)
        except ValueError:
            continue
        if seeds and seed not in seeds:
            continue
        candidate = RunRecord(
            path=entry.path,
            run_name=entry.name,
            env_id=run_env_id,
            exp_name=exp_name,
            seed=seed,
            timestamp=timestamp,
        )
        prev = latest_by_seed.get(seed)
        if prev is None or candidate.timestamp > prev.timestamp:
            latest_by_seed[seed] = candidate
    return [latest_by_seed[seed] for seed in sorted(latest_by_seed)]


def main():
    parser = argparse.ArgumentParser(
        description="Compute shared percentile-based fixed caps from converged vanilla PPO training returns."
    )
    parser.add_argument("--run-dir", required=True, help="Directory containing PPO training run folders.")
    parser.add_argument("--env-id", default="Walker2d-v4")
    parser.add_argument("--exp-name-prefix", default="ppo_cont_vanilla_final")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--percentiles", default="75,85,90,95")
    parser.add_argument("--tag", default="charts/episodic_return")
    parser.add_argument(
        "--tail-frac",
        type=float,
        default=0.25,
        help="Fraction of the training curve to treat as convergence phase (e.g. 0.25 = last 25%%).",
    )
    parser.add_argument("--min-tail-points", type=int, default=10)
    parser.add_argument("--out-dir", required=True, help="New output directory for this experiment setup.")
    args = parser.parse_args()

    if args.tail_frac <= 0.0 or args.tail_frac > 1.0:
        raise ValueError("--tail-frac must be in (0, 1].")

    percentiles = parse_int_list(args.percentiles)
    if not percentiles:
        raise ValueError("No percentiles parsed from --percentiles.")

    seeds = parse_seeds(args.seeds)
    runs = select_runs(args.run_dir, args.env_id, args.exp_name_prefix, seeds)
    if not runs:
        raise RuntimeError("No matching vanilla runs found. Check --run-dir/--env-id/--exp-name-prefix/--seeds.")

    os.makedirs(args.out_dir, exist_ok=True)

    per_seed_rows: list[dict[str, object]] = []
    caps_by_percentile: dict[int, list[float]] = {pct: [] for pct in percentiles}

    for run in runs:
        curve = load_scalar_curve(run.path, args.tag)
        if curve is None:
            raise RuntimeError(f"Missing scalar tag {args.tag!r} for run {run.run_name}.")
        steps, vals = curve
        max_step = float(np.max(steps))
        cutoff_step = max_step * (1.0 - args.tail_frac)
        tail_mask = steps >= cutoff_step
        tail_steps = steps[tail_mask]
        tail_vals = vals[tail_mask]
        if tail_vals.size < args.min_tail_points:
            tail_vals = vals[-args.min_tail_points :]
            tail_steps = steps[-args.min_tail_points :]
            cutoff_step = float(tail_steps[0])

        for pct in percentiles:
            cap = float(np.percentile(tail_vals, pct))
            caps_by_percentile[pct].append(cap)
            per_seed_rows.append(
                {
                    "run_name": run.run_name,
                    "seed": run.seed,
                    "percentile": pct,
                    "tv_fixed_cap": cap,
                    "tv_fixed_cap_tag": format_cap_tag(cap),
                    "n_curve_points": int(vals.size),
                    "n_tail_points": int(tail_vals.size),
                    "max_step": max_step,
                    "tail_start_step": cutoff_step,
                    "tail_mean_return": float(np.mean(tail_vals)),
                    "tail_median_return": float(np.median(tail_vals)),
                    "tail_min_return": float(np.min(tail_vals)),
                    "tail_max_return": float(np.max(tail_vals)),
                }
            )

    per_seed_csv = os.path.join(args.out_dir, "per_seed_percentile_caps.csv")
    with open(per_seed_csv, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    shared_rows: list[dict[str, object]] = []
    for pct in percentiles:
        values = caps_by_percentile[pct]
        mean_cap, std_cap, ci95_cap = mean_std_ci95(values)
        shared_cap = float(np.median(np.asarray(values, dtype=np.float64)))
        shared_row = {
            "percentile": pct,
            "n_seeds": len(values),
            "shared_cap_median_across_seeds": shared_cap,
            "shared_cap_tag": format_cap_tag(shared_cap),
            "mean_seed_cap": mean_cap,
            "std_seed_cap": std_cap,
            "ci95_seed_cap": ci95_cap,
            "min_seed_cap": float(np.min(values)),
            "max_seed_cap": float(np.max(values)),
        }
        shared_rows.append(shared_row)

        txt_path = os.path.join(args.out_dir, f"cap_p{pct}.txt")
        with open(txt_path, "w", encoding="utf-8") as fobj:
            fobj.write(f"{shared_cap:.10f}\n")

    shared_csv = os.path.join(args.out_dir, "shared_percentile_caps.csv")
    with open(shared_csv, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(shared_rows[0].keys()))
        writer.writeheader()
        writer.writerows(shared_rows)

    env_path = os.path.join(args.out_dir, "shared_caps.env")
    with open(env_path, "w", encoding="utf-8") as fobj:
        for row in shared_rows:
            pct = int(row["percentile"])
            cap = float(row["shared_cap_median_across_seeds"])
            fobj.write(f"P{pct}_CAP={cap:.10f}\n")

    readme_path = os.path.join(args.out_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as fobj:
        fobj.write("Percentile-based fixed-cap setup\n")
        fobj.write(f"env_id={args.env_id}\n")
        fobj.write(f"exp_name_prefix={args.exp_name_prefix}\n")
        fobj.write(f"seeds={','.join(str(seed) for seed in sorted(seeds))}\n")
        fobj.write(f"tag={args.tag}\n")
        fobj.write(f"tail_frac={args.tail_frac}\n")
        fobj.write(f"min_tail_points={args.min_tail_points}\n")
        fobj.write(f"runs_used={len(runs)}\n")

    print(f"selected {len(runs)} vanilla runs")
    for run in runs:
        print(f"- seed={run.seed} run={run.run_name}")
    print(f"\nwrote {per_seed_csv}")
    print(f"wrote {shared_csv}")
    print(f"wrote {env_path}")
    print("\nShared caps (median across seeds):")
    for row in shared_rows:
        print(
            f"- p{int(row['percentile'])}: cap={float(row['shared_cap_median_across_seeds']):.4f} "
            f"(tag={row['shared_cap_tag']}, mean_seed_cap={float(row['mean_seed_cap']):.4f}, "
            f"ci95={float(row['ci95_seed_cap']):.4f})"
        )


if __name__ == "__main__":
    main()
