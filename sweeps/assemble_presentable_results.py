from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble final presentable Walker and HalfCheetah plot folders from packaged result folders."
    )
    parser.add_argument(
        "--results-dir",
        default="sweeps/results",
        help="Base results directory containing packaged experiment folders.",
    )
    parser.add_argument(
        "--walker-out",
        default="PPO_Walker_final_presentable_20260430",
        help="Output folder name under results-dir for Walker presentation plots.",
    )
    parser.add_argument(
        "--halfcheetah-out",
        default="PPO_HalfCheetah_final_presentable_20260430",
        help="Output folder name under results-dir for HalfCheetah presentation plots.",
    )
    return parser.parse_args()


def ensure_plot_variant_dir(base: Path, variant: str) -> Path:
    candidate = base / "plots" / variant
    if candidate.is_dir():
        return candidate
    return base / "plots"


def reset_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def make_final_dir(root: Path, categories: list[str]) -> None:
    reset_dir(root)
    for variant in ("with_variance", "without_variance"):
        for category in categories:
            (root / "plots" / variant / category).mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_globs(src_dir: Path, dst_dir: Path, patterns: list[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for src in src_dir.glob(pattern):
            if src.is_file():
                shutil.copy2(src, dst_dir / src.name)


def copy_raw_metrics_snapshot(source_root: Path, bundle_root: Path, snapshot_name: str) -> None:
    src = source_root / "raw_metrics"
    if not src.is_dir():
        return
    dst = bundle_root / "regeneration_sources" / snapshot_name / "raw_metrics"
    shutil.rmtree(dst.parent, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def write_regeneration_readme(bundle_root: Path, source_map: dict[str, Path]) -> None:
    regen_root = bundle_root / "regeneration_sources"
    regen_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Regeneration Sources",
        "",
        "This directory stores raw-metric snapshots copied from the packaged source experiment folders used to build the presentation bundle.",
        "",
        "Each subfolder contains a `raw_metrics/` directory that can be fed back into `sweeps/package_alpha_robust_eval.py` to regenerate category-level packaged plots.",
        "",
        "Snapshots included:",
    ]
    for name, src in source_map.items():
        lines.append(f"- `{name}` <- `{src.as_posix()}`")
    (regen_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, title: str, categories: list[str]) -> None:
    lines = [
        f"# {title}",
        "",
        "Presentation bundle of plots organized for slides and figures.",
        "",
        "Structure:",
        "- `plots/with_variance/`",
        "- `plots/without_variance/`",
        "- `regeneration_sources/`",
        "",
        "Categories:",
    ]
    lines.extend(f"- {category}" for category in categories)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assemble_walker(results_dir: Path, out_name: str) -> Path:
    categories = [
        "Single-axis perturbations",
        "Targeted localized perturbations",
        "combos",
        "Bernoulli action noise",
        "Gaussian action noise",
    ]
    out_root = results_dir / out_name
    make_final_dir(out_root, categories)

    sources = {
        "single_axis_nonmass": results_dir / "PPO_Walker_single_axis_physical_nonmass_0p0_2p0_20260430",
        "single_axis_mass": results_dir / "PPO_Walker_single_axis_mass_0p1_2p0_20260430",
        "targeted_nonmass": results_dir / "PPO_Walker_targeted_nonmass_0p0_2p0_20260430",
        "targeted_mass": results_dir / "PPO_Walker_targeted_mass_0p1_2p0_20260430",
        "combo_nonmass": results_dir / "PPO_Walker_fmd_nonmass_0p0_2p0_20260430",
        "combo_mass": results_dir / "PPO_Walker_fmd_mass_0p1_2p0_20260430",
        "gaussian": results_dir / "PPO_Adversarial_action_noise_20260406_repack_20260430",
        "bernoulli": results_dir / "PPO_Adversarial_action_bernoulli_20260408_clean_v3_repack_20260430",
    }

    for variant in ("with_variance", "without_variance"):
        sa_nonmass = ensure_plot_variant_dir(sources["single_axis_nonmass"], variant)
        sa_mass = ensure_plot_variant_dir(sources["single_axis_mass"], variant)
        tgt_nonmass = ensure_plot_variant_dir(sources["targeted_nonmass"], variant)
        tgt_mass = ensure_plot_variant_dir(sources["targeted_mass"], variant)
        combo_nonmass = ensure_plot_variant_dir(sources["combo_nonmass"], variant)
        combo_mass = ensure_plot_variant_dir(sources["combo_mass"], variant)
        gaussian = ensure_plot_variant_dir(sources["gaussian"], variant)
        bernoulli = ensure_plot_variant_dir(sources["bernoulli"], variant)

        dst = out_root / "plots" / variant

        copy_if_exists(sa_nonmass / "return_curves_panel.png", dst / "Single-axis perturbations" / "single_axis_nonmass_return_curves_panel.png")
        copy_if_exists(sa_nonmass / "gain_curves_panel.png", dst / "Single-axis perturbations" / "single_axis_nonmass_gain_curves_panel.png")
        copy_if_exists(sa_mass / "return_curves_panel.png", dst / "Single-axis perturbations" / "single_axis_mass_return_curves_panel.png")
        copy_if_exists(sa_mass / "gain_curves_panel.png", dst / "Single-axis perturbations" / "single_axis_mass_gain_curves_panel.png")
        copy_globs(sa_nonmass, dst / "Single-axis perturbations", ["friction_*", "damping_*", "actuator_gain_*"])
        copy_globs(sa_mass, dst / "Single-axis perturbations", ["mass_*"])

        copy_if_exists(tgt_nonmass / "return_curves_panel.png", dst / "Targeted localized perturbations" / "targeted_nonmass_return_curves_panel.png")
        copy_if_exists(tgt_nonmass / "gain_curves_panel.png", dst / "Targeted localized perturbations" / "targeted_nonmass_gain_curves_panel.png")
        copy_if_exists(tgt_mass / "return_curves_panel.png", dst / "Targeted localized perturbations" / "targeted_mass_return_curves_panel.png")
        copy_if_exists(tgt_mass / "gain_curves_panel.png", dst / "Targeted localized perturbations" / "targeted_mass_gain_curves_panel.png")
        copy_globs(tgt_nonmass, dst / "Targeted localized perturbations", ["thigh_left_*", "leg_left_*", "foot_left_*"])
        copy_globs(tgt_mass, dst / "Targeted localized perturbations", ["thigh_left_*", "leg_left_*", "foot_left_*"])

        copy_if_exists(combo_nonmass / "return_curves_panel.png", dst / "combos" / "combos_nonmass_return_curves_panel.png")
        copy_if_exists(combo_nonmass / "gain_curves_panel.png", dst / "combos" / "combos_nonmass_gain_curves_panel.png")
        copy_if_exists(combo_mass / "return_curves_panel.png", dst / "combos" / "combos_mass_return_curves_panel.png")
        copy_if_exists(combo_mass / "gain_curves_panel.png", dst / "combos" / "combos_mass_gain_curves_panel.png")
        copy_globs(combo_nonmass, dst / "combos", ["friction_damping_*"])
        copy_globs(combo_mass, dst / "combos", ["friction_mass_*", "mass_damping_*", "friction_mass_damping_*"])

        copy_if_exists(gaussian / "return_curves_panel.png", dst / "Gaussian action noise" / "return_curves_panel.png")
        copy_if_exists(gaussian / "gain_curves_panel.png", dst / "Gaussian action noise" / "gain_curves_panel.png")
        copy_if_exists(gaussian / "action_noise_return_curve.png", dst / "Gaussian action noise" / "action_noise_return_curve.png")
        copy_if_exists(gaussian / "action_noise_gain_curve.png", dst / "Gaussian action noise" / "action_noise_gain_curve.png")

        copy_if_exists(bernoulli / "return_curves_panel.png", dst / "Bernoulli action noise" / "return_curves_panel.png")
        copy_if_exists(bernoulli / "gain_curves_panel.png", dst / "Bernoulli action noise" / "gain_curves_panel.png")
        copy_if_exists(bernoulli / "action_replace_return_curve.png", dst / "Bernoulli action noise" / "bernoulli_action_noise_return_curve.png")
        copy_if_exists(bernoulli / "action_replace_gain_curve.png", dst / "Bernoulli action noise" / "bernoulli_action_noise_gain_curve.png")
        copy_if_exists(bernoulli / "action_noise_bernoulli_return_curve.png", dst / "Bernoulli action noise" / "bernoulli_action_noise_return_curve.png")
        copy_if_exists(bernoulli / "action_noise_bernoulli_gain_curve.png", dst / "Bernoulli action noise" / "bernoulli_action_noise_gain_curve.png")

    for snapshot_name, source_root in sources.items():
        copy_raw_metrics_snapshot(source_root, out_root, snapshot_name)
    write_regeneration_readme(out_root, sources)
    write_readme(out_root / "README.md", out_name, categories)
    return out_root


def assemble_halfcheetah(results_dir: Path, out_name: str) -> Path:
    categories = [
        "Single-axis perturbations",
        "Targeted localized perturbations",
        "combos",
        "Bernoulli action noise",
        "Gaussian action noise",
        "Observation noise",
    ]
    out_root = results_dir / out_name
    make_final_dir(out_root, categories)

    sources = {
        "nonmass": results_dir / "PPO_HalfCheetah_expanded_nonmass_0p0_2p0_20260430",
        "mass": results_dir / "PPO_HalfCheetah_expanded_mass_0p1_2p0_20260430",
        "signal": results_dir / "PPO_HalfCheetah_signal_all_caps_20260430",
    }

    for variant in ("with_variance", "without_variance"):
        nonmass = ensure_plot_variant_dir(sources["nonmass"], variant)
        mass = ensure_plot_variant_dir(sources["mass"], variant)
        signal = ensure_plot_variant_dir(sources["signal"], variant)

        dst = out_root / "plots" / variant

        copy_if_exists(nonmass / "return_curves_panel.png", dst / "Single-axis perturbations" / "single_axis_nonmass_return_curves_panel.png")
        copy_if_exists(nonmass / "gain_curves_panel.png", dst / "Single-axis perturbations" / "single_axis_nonmass_gain_curves_panel.png")
        copy_if_exists(mass / "return_curves_panel.png", dst / "Single-axis perturbations" / "single_axis_mass_return_curves_panel.png")
        copy_if_exists(mass / "gain_curves_panel.png", dst / "Single-axis perturbations" / "single_axis_mass_gain_curves_panel.png")
        copy_globs(nonmass, dst / "Single-axis perturbations", ["friction_*", "damping_*", "gravity_*"])
        copy_if_exists(nonmass / "gear_return_curve.png", dst / "Single-axis perturbations" / "actuator_gain_return_curve.png")
        copy_if_exists(nonmass / "gear_gain_curve.png", dst / "Single-axis perturbations" / "actuator_gain_gain_curve.png")
        copy_globs(mass, dst / "Single-axis perturbations", ["mass_*"])

        copy_if_exists(nonmass / "return_curves_panel.png", dst / "Targeted localized perturbations" / "localized_nonmass_return_curves_panel.png")
        copy_if_exists(nonmass / "gain_curves_panel.png", dst / "Targeted localized perturbations" / "localized_nonmass_gain_curves_panel.png")
        copy_if_exists(mass / "return_curves_panel.png", dst / "Targeted localized perturbations" / "localized_mass_return_curves_panel.png")
        copy_if_exists(mass / "gain_curves_panel.png", dst / "Targeted localized perturbations" / "localized_mass_gain_curves_panel.png")
        copy_globs(nonmass, dst / "Targeted localized perturbations", ["bthigh_*", "bshin_*", "bfoot_*", "fthigh_*", "fshin_*", "ffoot_*"])
        copy_globs(mass, dst / "Targeted localized perturbations", ["bthigh_*", "bshin_*", "bfoot_*", "fthigh_*", "fshin_*", "ffoot_*"])

        copy_if_exists(nonmass / "return_curves_panel.png", dst / "combos" / "combos_nonmass_return_curves_panel.png")
        copy_if_exists(nonmass / "gain_curves_panel.png", dst / "combos" / "combos_nonmass_gain_curves_panel.png")
        copy_if_exists(mass / "return_curves_panel.png", dst / "combos" / "combos_mass_return_curves_panel.png")
        copy_if_exists(mass / "gain_curves_panel.png", dst / "combos" / "combos_mass_gain_curves_panel.png")
        copy_globs(nonmass, dst / "combos", ["friction_damping_*"])
        copy_globs(mass, dst / "combos", ["friction_mass_*", "mass_damping_*", "friction_mass_damping_*"])

        copy_if_exists(signal / "return_curves_panel.png", dst / "Gaussian action noise" / "all_signal_return_curves_panel.png")
        copy_if_exists(signal / "gain_curves_panel.png", dst / "Gaussian action noise" / "all_signal_gain_curves_panel.png")
        copy_if_exists(signal / "action_noise_return_curve.png", dst / "Gaussian action noise" / "action_noise_return_curve.png")
        copy_if_exists(signal / "action_noise_gain_curve.png", dst / "Gaussian action noise" / "action_noise_gain_curve.png")

        copy_if_exists(signal / "return_curves_panel.png", dst / "Bernoulli action noise" / "all_signal_return_curves_panel.png")
        copy_if_exists(signal / "gain_curves_panel.png", dst / "Bernoulli action noise" / "all_signal_gain_curves_panel.png")
        copy_if_exists(signal / "action_noise_bernoulli_return_curve.png", dst / "Bernoulli action noise" / "bernoulli_action_noise_return_curve.png")
        copy_if_exists(signal / "action_noise_bernoulli_gain_curve.png", dst / "Bernoulli action noise" / "bernoulli_action_noise_gain_curve.png")

        copy_if_exists(signal / "return_curves_panel.png", dst / "Observation noise" / "all_signal_return_curves_panel.png")
        copy_if_exists(signal / "gain_curves_panel.png", dst / "Observation noise" / "all_signal_gain_curves_panel.png")
        copy_if_exists(signal / "state_noise_return_curve.png", dst / "Observation noise" / "obs_noise_return_curve.png")
        copy_if_exists(signal / "state_noise_gain_curve.png", dst / "Observation noise" / "obs_noise_gain_curve.png")

    for snapshot_name, source_root in sources.items():
        copy_raw_metrics_snapshot(source_root, out_root, snapshot_name)
    write_regeneration_readme(out_root, sources)
    write_readme(out_root / "README.md", out_name, categories)
    return out_root


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    walker_out = assemble_walker(results_dir, args.walker_out)
    halfcheetah_out = assemble_halfcheetah(results_dir, args.halfcheetah_out)
    print(f"Built Walker presentation bundle: {walker_out}")
    print(f"Built HalfCheetah presentation bundle: {halfcheetah_out}")


if __name__ == "__main__":
    main()
