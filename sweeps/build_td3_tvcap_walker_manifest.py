#!/usr/bin/env python3
"""Build the full Walker2d TD3 TV-cap robustness-evaluation manifest."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path


CAPS = ["vanilla", "100", "150", "200", "225", "250", "275", "300"]
NONMASS_FACTORS = "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.3 1.5 1.7 2.0".split()
MASS_FACTORS = "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.3 1.5 1.7 2.0".split()
GAUSSIAN_FACTORS = "0 0.05 0.1 0.2 0.3 0.5".split()
BERNOULLI_FACTORS = "0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5".split()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--train-run-dir", required=True)
    parser.add_argument("--caps", nargs="+", default=CAPS)
    parser.add_argument("--seeds", nargs="+", default=[str(seed) for seed in range(1, 31)])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser()
    train_dir = str(Path(args.train_run_dir).expanduser())
    manifest = result_dir / "manifests" / "walker_full_eval_manifest.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, ...]] = []

    def add(axes: list[str], factors: list[str], category: str) -> None:
        for cap, axis, factor, seed in product(args.caps, axes, factors, args.seeds):
            rows.append(("Walker2d-v4", train_dir, cap, seed, axis, factor, category))

    add("friction damping actuator_gain".split(), NONMASS_FACTORS, "single_axis_perturbations")
    add(["mass"], MASS_FACTORS, "single_axis_perturbations")
    add(
        "foot_left_actuator_gain foot_left_damping foot_left_friction "
        "leg_left_actuator_gain leg_left_damping thigh_left_actuator_gain thigh_left_damping".split(),
        NONMASS_FACTORS,
        "targeted_localized_perturbations",
    )
    add("foot_left_mass leg_left_mass thigh_left_mass".split(), MASS_FACTORS, "targeted_localized_perturbations")
    add(["friction_damping"], NONMASS_FACTORS, "combos")
    add("friction_mass friction_mass_damping mass_damping".split(), MASS_FACTORS, "combos")
    add(["action_noise"], GAUSSIAN_FACTORS, "gaussian_action_noise")
    add(["action_replace"], BERNOULLI_FACTORS, "bernoulli_action_noise")

    manifest.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    print(f"manifest: {manifest}")
    print(f"rows: {len(rows)}")
    print(f"chunks@100: {(len(rows) + 99) // 100}")


if __name__ == "__main__":
    main()
