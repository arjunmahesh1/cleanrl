import argparse
import glob
import os
import subprocess
import time
from typing import Dict, List, Tuple


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_thresholds(value: str) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    if not value:
        return thresholds
    for item in parse_csv_list(value):
        env, thresh = item.split("=", 1)
        thresholds[env] = float(thresh)
    return thresholds


def find_new_run_dirs(before: set, runs_dir: str) -> List[str]:
    after = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()
    return sorted(after - before)


def load_last_returns(run_dir: str, tag: str, window: int) -> Tuple[float | None, int]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None, 0

    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        return None, 0
    event_files.sort(key=os.path.getmtime, reverse=True)
    acc = EventAccumulator(event_files[0])
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None, 0
    scalars = acc.Scalars(tag)
    if not scalars:
        return None, 0
    values = [s.value for s in scalars][-window:]
    return sum(values) / len(values), len(values)


def main():
    parser = argparse.ArgumentParser(description="Screen value-based algorithms across envs/seeds.")
    parser.add_argument("--algos", default="dqn,c51,pqn", help="comma-separated algos: dqn,c51,pqn")
    parser.add_argument("--envs", default="CartPole-v1", help="comma-separated env ids")
    parser.add_argument("--seeds", default="1,2,3", help="comma-separated seeds")
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--window", type=int, default=10, help="episode window for mean return")
    parser.add_argument("--thresholds", default="", help="env=threshold,comma-separated")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-project-name", default="cleanRL")
    parser.add_argument("--run-dir", default="runs", help="base directory for TensorBoard logs")
    parser.add_argument("--extra-args", default="", help="extra args appended to each run")
    args = parser.parse_args()

    algo_map = {
        "dqn": os.path.join("cleanrl", "dqn.py"),
        "c51": os.path.join("cleanrl", "c51.py"),
        "pqn": os.path.join("cleanrl", "pqn.py"),
    }
    algos = parse_csv_list(args.algos)
    envs = parse_csv_list(args.envs)
    seeds = [int(s) for s in parse_csv_list(args.seeds)]
    thresholds = parse_thresholds(args.thresholds)

    runs_dir = args.run_dir
    results = []
    for algo in algos:
        if algo not in algo_map:
            raise SystemExit(f"Unknown algo '{algo}'. Known: {', '.join(algo_map.keys())}")
        script = algo_map[algo]
        for env_id in envs:
            for seed in seeds:
                before = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()
                cmd = [
                    "python",
                    script,
                    "--env-id",
                    env_id,
                    "--seed",
                    str(seed),
                    "--total-timesteps",
                    str(args.total_timesteps),
                    "--run-dir",
                    args.run_dir,
                ]
                if args.track:
                    cmd += ["--track"]
                if args.wandb_entity:
                    cmd += ["--wandb-entity", args.wandb_entity]
                if args.wandb_project_name:
                    cmd += ["--wandb-project-name", args.wandb_project_name]
                if args.extra_args:
                    cmd += args.extra_args.split()

                print(f"\n[run] algo={algo} env={env_id} seed={seed}")
                subprocess.run(cmd, check=True)
                time.sleep(1.0)

                new_dirs = find_new_run_dirs(before, runs_dir)
                if not new_dirs:
                    results.append((algo, env_id, seed, None, 0, "no-run-dir"))
                    continue
                run_dir = os.path.join(runs_dir, new_dirs[-1])
                mean_return, count = load_last_returns(run_dir, "charts/episodic_return", args.window)
                status = "unknown"
                if mean_return is not None:
                    threshold = thresholds.get(env_id)
                    if threshold is None:
                        status = "measured"
                    else:
                        status = "pass" if mean_return >= threshold else "fail"
                results.append((algo, env_id, seed, mean_return, count, status))

    print("\nResults (last-window mean episodic return):")
    for algo, env_id, seed, mean_return, count, status in results:
        mean_str = "n/a" if mean_return is None else f"{mean_return:.2f}"
        print(f"{algo:6s} {env_id:18s} seed={seed} mean={mean_str} n={count} status={status}")


if __name__ == "__main__":
    main()
