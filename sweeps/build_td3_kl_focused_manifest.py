#!/usr/bin/env python3
"""Build a focused physical-support TD3-KL robustness manifest."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path


FACTOR_GRID = {
    "mass": ["0.5", "0.8", "1.0", "1.2", "1.5"],
    "actuator_gain": ["0.5", "0.8", "1.0", "1.2", "1.5"],
    "friction_mass_damping": ["0.5", "0.8", "1.0", "1.2", "1.5"],
    "action_replace": ["0", "0.05", "0.1", "0.2", "0.3"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--train-run-dir", required=True)
    parser.add_argument("--env-id", default="Walker2d-v5")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["vanilla", "0", "0.01", "0.05", "0.1", "0.2", "0.5"],
        help="Radius tokens consumed by the generic TD3 manifest evaluator.",
    )
    parser.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--axes", nargs="+", default=list(FACTOR_GRID))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.axes) - set(FACTOR_GRID))
    if unknown:
        raise SystemExit(f"Unsupported focused axes: {unknown}")

    result_dir = Path(args.result_dir).expanduser()
    train_dir = str(Path(args.train_run_dir).expanduser())
    manifest = result_dir / "manifests" / "focused_eval_manifest.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, ...]] = []
    for axis in args.axes:
        for variant, factor, seed in product(
            args.variants,
            FACTOR_GRID[axis],
            args.seeds,
        ):
            rows.append(
                (
                    args.env_id,
                    train_dir,
                    variant,
                    seed,
                    axis,
                    factor,
                    "focused_diagnostic",
                )
            )

    manifest.write_text(
        "".join("\t".join(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"manifest: {manifest}")
    print(f"rows: {len(rows)}")
    print(f"chunks@20: {(len(rows) + 19) // 20}")


if __name__ == "__main__":
    main()
