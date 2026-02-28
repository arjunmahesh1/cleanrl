import argparse
import csv
import os


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    return seeds


def load_caps(caps_csv: str) -> list[dict[str, str]]:
    with open(caps_csv, newline="", encoding="utf-8") as fobj:
        return list(csv.DictReader(fobj))


def main():
    parser = argparse.ArgumentParser(
        description="Build a phase-1 percentile-cap training manifest from shared percentile cap definitions."
    )
    parser.add_argument("--caps-csv", required=True, help="shared_percentile_caps.csv from compute_shared_caps.py")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--env-id", default="Walker2d-v4")
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--run-dir", default="~/rl_runs_pctcap_phase1")
    parser.add_argument("--wandb-project-name", default="cleanrl")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="ppo-tv-percentile-cap-phase1")
    parser.add_argument("--python-bin", default="$ROOT/.venv/bin/python")
    parser.add_argument("--root", default="$HOME/cleanrl")
    parser.add_argument("--exp-prefix", default="ppo_cont_tv_pctcap")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    cap_rows = load_caps(args.caps_csv)
    if not cap_rows:
        raise RuntimeError(f"No cap rows found in {args.caps_csv}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for cap_row in cap_rows:
        percentile = int(cap_row["percentile"])
        shared_cap = float(cap_row["shared_cap_median_across_seeds"])
        cap_tag = cap_row["shared_cap_tag"]
        pct_label = f"p{percentile}"
        exp_name = f"{args.exp_prefix}_{pct_label}"
        for seed in seeds:
            command = (
                f'{args.python_bin} cleanrl/ppo_continuous_action.py '
                f'--exp-name {exp_name} '
                f'--env-id {args.env_id} '
                f'--total-timesteps {args.total_timesteps} '
                f'--seed {seed} '
                f'--track '
                f'--wandb-project-name {args.wandb_project_name} '
                + (f'--wandb-entity {args.wandb_entity} ' if args.wandb_entity else "")
                + (f'--wandb-group {args.wandb_group} ' if args.wandb_group else "")
                + f'--run-dir {args.run_dir} '
                f'--save-model '
                f'--tv-clip-value-targets '
                f'--tv-mode fixed_cap '
                f'--tv-fixed-cap {shared_cap:.10f}'
            )
            manifest_rows.append(
                {
                    "percentile": percentile,
                    "percentile_label": pct_label,
                    "tv_fixed_cap": f"{shared_cap:.10f}",
                    "tv_fixed_cap_tag": cap_tag,
                    "seed": seed,
                    "exp_name": exp_name,
                    "run_dir": args.run_dir,
                    "command": command,
                }
            )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"wrote {args.out_csv}")
    print(f"rows={len(manifest_rows)}")
    print("percentile labels:", ", ".join(sorted({row['percentile_label'] for row in manifest_rows})))


if __name__ == "__main__":
    main()
