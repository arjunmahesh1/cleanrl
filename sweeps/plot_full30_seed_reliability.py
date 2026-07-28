from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIGNAL_AXES = {"action_noise", "action_replace", "action_noise_bernoulli", "state_noise"}

SELECTED_AXES = {
    "Walker2d-v4": [
        "mass",
        "actuator_gain",
        "action_replace",
        "friction_mass_damping",
        "friction_mass",
        "friction",
    ],
    "Walker2d-v5": [
        "mass",
        "actuator_gain",
        "action_replace",
        "friction_mass_damping",
        "friction_mass",
        "friction",
    ],
    "HalfCheetah-v4": [
        "mass",
        "gear",
        "action_noise_bernoulli",
        "state_noise",
        "friction_mass_damping",
        "mass_damping",
        "damping",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate seed-level robustness and reliability plots."
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Result directory containing outputs/combined_metrics.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to RESULT_DIR/analysis_plots.",
    )
    parser.add_argument(
        "--catastrophe-threshold-frac",
        type=float,
        default=0.5,
        help="Catastrophe threshold as a fraction of vanilla nominal median return.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--all-scatter-axes",
        action="store_true",
        help="Generate raw/delta scatter and reliability plots for every axis. Default uses selected high-signal axes.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write summary CSVs and README; skip figure generation.",
    )
    parser.add_argument(
        "--fixed-model-label",
        default=None,
        help="Preselected model for the deployment-style seed-conditioned gain panel, e.g. tvc250.",
    )
    return parser.parse_args()


def parse_factor_from_scenario(label: str) -> tuple[str, float]:
    axis, token = str(label).rsplit("_", 1)
    sign = -1.0 if token.startswith("m") else 1.0
    if token.startswith("m"):
        token = token[1:]
    return axis, sign * float(token.replace("p", "."))


def model_sort_key(label: str) -> tuple[int, float, str]:
    if label == "vanilla":
        return (0, -1.0, label)
    match = re.fullmatch(r"a(\d+)p(\d+)", label)
    if match:
        value = float(f"{match.group(1)}.{match.group(2)}")
        return (1, value, label)
    match = re.fullmatch(r"(klb|kle)(\d+)(?:p(\d+))?", label)
    if match:
        frac = match.group(3) or "0"
        value = float(f"{match.group(2)}.{frac}")
        return (2 if match.group(1) == "klb" else 3, value, label)
    match = re.fullmatch(r"q(\d+)(?:p(\d+))?", label)
    if match:
        frac = match.group(2) or "0"
        value = float(f"{match.group(1)}.{frac}")
        return (4, value, label)
    match = re.fullmatch(r"tvc(\d+)(?:p(\d+))?", label)
    if match:
        frac = match.group(2) or "0"
        value = float(f"{match.group(1)}.{frac}")
        return (5, value, label)
    match = re.fullmatch(r"klprho(\d+)(?:p(\d+))?", label)
    if match:
        frac = match.group(2) or "0"
        value = float(f"{match.group(1)}.{frac}")
        return (6, value, label)
    return (7, math.inf, label)


def display_model(label: str) -> str:
    if label == "vanilla":
        return "Vanilla"
    match = re.fullmatch(r"a(\d+)p(\d+)", label)
    if match:
        return f"c={match.group(1)}.{match.group(2)}"
    match = re.fullmatch(r"(klb|kle)(\d+)(?:p(\d+))?", label)
    if match:
        frac = match.group(3)
        beta = match.group(2) if frac is None else f"{match.group(2)}.{frac}"
        prefix = r"$\beta" if match.group(1) == "klb" else r"$K$-KL $\beta"
        return rf"{prefix}={beta}$"
    match = re.fullmatch(r"q(\d+)(?:p(\d+))?", label)
    if match:
        frac = match.group(2)
        cap = match.group(1) if frac is None else f"{match.group(1)}.{frac}"
        return f"q={cap}"
    match = re.fullmatch(r"tvc(\d+)(?:p(\d+))?", label)
    if match:
        frac = match.group(2)
        cap = match.group(1) if frac is None else f"{match.group(1)}.{frac}"
        return f"TV c={cap}"
    match = re.fullmatch(r"klprho(\d+)(?:p(\d+))?", label)
    if match:
        frac = match.group(2)
        radius = match.group(1) if frac is None else f"{match.group(1)}.{frac}"
        return rf"KL $\rho={radius}$"
    return label


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def nominal_factor(axis: str) -> float:
    return 0.0 if axis in SIGNAL_AXES else 1.0


def choose_stress_factor(factors: list[float], axis: str) -> float:
    arr = np.asarray(sorted(set(float(x) for x in factors)), dtype=float)
    if arr.size == 0:
        raise ValueError(f"No factors for axis {axis}")
    nom = nominal_factor(axis)
    if nom == 0.0:
        return float(arr[-1])
    if np.any(np.isclose(arr, 0.5)):
        return 0.5
    below = arr[arr < nom]
    if below.size:
        return float(below[np.argmin(np.abs(below - 0.5))])
    return float(arr[np.argmin(np.abs(arr - nom))])


def figure_grid(n: int, max_cols: int = 4) -> tuple[int, int]:
    ncols = min(max_cols, max(1, math.ceil(math.sqrt(n))))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.parent / f"{stem.name}.{fmt}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_metrics(result_dir: Path) -> pd.DataFrame:
    path = result_dir / "outputs" / "combined_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "factor" not in df.columns:
        parsed = df["scenario_label"].map(parse_factor_from_scenario)
        df["axis"] = parsed.map(lambda x: x[0])
        df["factor"] = parsed.map(lambda x: x[1])
    model_pattern = (
        r"^(a\d+p\d+|klb\d+(p\d+)?|kle\d+(p\d+)?|q\d+(p\d+)?|"
        r"tvc\d+(p\d+)?|klprho\d+(p\d+)?)$"
    )
    df = df[df["model_label"].isin(["vanilla"]) | df["model_label"].str.match(model_pattern, na=False)].copy()
    df["factor"] = df["factor"].astype(float)
    df["seed"] = df["seed"].astype(int)
    df["mean_return"] = df["mean_return"].astype(float)
    return df


def vanilla_nominal_baselines(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for (env, axis), sub in df[df["model_label"] == "vanilla"].groupby(["env_id", "axis"]):
        nom = nominal_factor(axis)
        exact = sub[np.isclose(sub["factor"], nom)]
        if exact.empty:
            closest_factor = sub["factor"].iloc[np.argmin(np.abs(sub["factor"].to_numpy() - nom))]
            exact = sub[np.isclose(sub["factor"], closest_factor)]
        out[(env, axis)] = float(exact["mean_return"].median())
    return out


def stress_factor_map(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for (env, axis), sub in df.groupby(["env_id", "axis"]):
        out[(env, axis)] = choose_stress_factor(sorted(sub["factor"].unique()), axis)
    return out


def same_seed_vanilla(stress: pd.DataFrame) -> pd.DataFrame:
    vanilla = stress[stress["model_label"] == "vanilla"][
        ["env_id", "axis", "factor", "seed", "mean_return"]
    ].rename(columns={"mean_return": "vanilla_return"})
    return stress.merge(vanilla, on=["env_id", "axis", "factor", "seed"], how="left")


def reliability_auc(values: pd.Series, baseline: float, max_frac: float) -> float:
    if baseline <= 0:
        return float("nan")
    arr = values.to_numpy(dtype=float)
    thresholds = np.linspace(0.0, max_frac, 501)
    reliability = np.asarray([(arr >= frac * baseline).mean() for frac in thresholds], dtype=float)
    return float(np.trapezoid(reliability, thresholds) / max_frac)


def write_stress_summaries(
    df: pd.DataFrame,
    out_dir: Path,
    baselines: dict[tuple[str, str], float],
    stress_factors: dict[tuple[str, str], float],
    threshold_frac: float,
) -> None:
    rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    best_auc_rows: list[dict[str, object]] = []
    per_seed_rows: list[dict[str, object]] = []

    for (env, axis), sub in df.groupby(["env_id", "axis"]):
        factor = stress_factors[(env, axis)]
        stress = sub[np.isclose(sub["factor"], factor)].copy()
        stress = same_seed_vanilla(stress)
        baseline = baselines[(env, axis)]
        threshold = threshold_frac * baseline

        for model, msub in stress.groupby("model_label"):
            deltas = msub["mean_return"] - msub["vanilla_return"]
            rows.append(
                {
                    "env_id": env,
                    "axis": axis,
                    "stress_factor": factor,
                    "model_label": model,
                    "baseline_vanilla_nominal_median": baseline,
                    "catastrophe_threshold": threshold,
                    "mean_return": float(msub["mean_return"].mean()),
                    "median_return": float(msub["mean_return"].median()),
                    "reliability_at_threshold": float((msub["mean_return"] >= threshold).mean()),
                    "failure_rate_at_threshold": float((msub["mean_return"] < threshold).mean()),
                    "reliability_auc_0_to_1x_nominal": reliability_auc(msub["mean_return"], baseline, 1.0),
                    "reliability_auc_0_to_1p25x_nominal": reliability_auc(msub["mean_return"], baseline, 1.25),
                    "median_delta_vs_same_seed_vanilla": float(deltas.median()),
                    "mean_delta_vs_same_seed_vanilla": float(deltas.mean()),
                    "win_rate_vs_same_seed_vanilla": float((deltas > 0).mean()),
                }
            )

        grouped = (
            stress.groupby("model_label")["mean_return"]
            .median()
            .sort_values(ascending=False)
        )
        best_model = str(grouped.index[0])
        best_rows.append(
            {
                "env_id": env,
                "axis": axis,
                "stress_factor": factor,
                "best_model_by_median_return": best_model,
                "best_median_return": float(grouped.iloc[0]),
                "vanilla_median_return": float(grouped.get("vanilla", np.nan)),
                "best_minus_vanilla_median": float(grouped.iloc[0] - grouped.get("vanilla", np.nan)),
            }
        )

        auc_table = {
            model: reliability_auc(msub["mean_return"], baseline, 1.0)
            for model, msub in stress.groupby("model_label")
        }
        best_auc_model = max(auc_table, key=lambda model: (-math.inf if math.isnan(auc_table[model]) else auc_table[model]))
        best_auc_rows.append(
            {
                "env_id": env,
                "axis": axis,
                "stress_factor": factor,
                "best_model_by_reliability_auc_0_to_1x_nominal": best_auc_model,
                "best_reliability_auc_0_to_1x_nominal": float(auc_table[best_auc_model]),
                "vanilla_reliability_auc_0_to_1x_nominal": float(auc_table.get("vanilla", np.nan)),
                "best_minus_vanilla_reliability_auc": float(auc_table[best_auc_model] - auc_table.get("vanilla", np.nan)),
            }
        )

        for seed, ssub in stress.groupby("seed"):
            seed_sorted = ssub.sort_values("mean_return", ascending=False)
            seed_best = seed_sorted.iloc[0]
            vanilla_value = float(ssub.loc[ssub["model_label"] == "vanilla", "mean_return"].iloc[0])
            per_seed_rows.append(
                {
                    "env_id": env,
                    "axis": axis,
                    "stress_factor": factor,
                    "seed": int(seed),
                    "best_model": seed_best["model_label"],
                    "best_return": float(seed_best["mean_return"]),
                    "vanilla_return": vanilla_value,
                    "best_minus_vanilla": float(seed_best["mean_return"] - vanilla_value),
                    "num_models_above_vanilla": int((ssub["mean_return"] > vanilla_value).sum()),
                }
            )

    summary = pd.DataFrame(rows)
    best = pd.DataFrame(best_rows)
    best_auc = pd.DataFrame(best_auc_rows)
    per_seed = pd.DataFrame(per_seed_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "stress_scenario_summary.csv", index=False)
    best.to_csv(out_dir / "best_caps_by_axis.csv", index=False)
    best_auc.to_csv(out_dir / "best_caps_by_reliability_auc.csv", index=False)
    per_seed.to_csv(out_dir / "per_seed_best_caps.csv", index=False)


def plot_seed_spaghetti(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    plot_root = out_dir / "seed_spaghetti_by_axis"
    for (env, axis), sub in df.groupby(["env_id", "axis"]):
        models = sorted(sub["model_label"].unique(), key=model_sort_key)
        nrows, ncols = figure_grid(len(models), max_cols=4)
        fig, axs = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
        fig.suptitle(f"{env} {axis}: seed return curves by model", fontsize=14)

        y_min = float(sub["mean_return"].min())
        y_max = float(sub["mean_return"].max())
        pad = 0.04 * max(1.0, y_max - y_min)
        for ax, model in zip(axs.flat, models):
            msub = sub[sub["model_label"] == model]
            for _, ssub in msub.groupby("seed"):
                ssub = ssub.sort_values("factor")
                ax.plot(ssub["factor"], ssub["mean_return"], color="#2f6f9f", alpha=0.22, linewidth=0.8)
            median_curve = msub.groupby("factor", as_index=False)["mean_return"].median().sort_values("factor")
            ax.plot(median_curve["factor"], median_curve["mean_return"], color="black", linewidth=2.0, label="median")
            ax.set_title(display_model(model), fontsize=10)
            ax.set_xlabel("Perturbation factor")
            ax.set_ylabel("Eval return")
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.grid(alpha=0.2)
        for ax in axs.flat[len(models) :]:
            ax.axis("off")
        fig.tight_layout()
        save_figure(fig, plot_root / env / f"{slug(axis)}_seed_spaghetti", formats)


def plot_fixed_seed_caps(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    plot_root = out_dir / "fixed_seed_all_caps"
    for env, env_df in df.groupby("env_id"):
        available_axes = set(env_df["axis"].unique())
        axes = [a for a in SELECTED_AXES.get(env, sorted(available_axes)) if a in available_axes]
        if not axes:
            continue
        models = sorted(env_df["model_label"].unique(), key=model_sort_key)
        cmap = plt.get_cmap("tab20")
        colors = {m: ("black" if m == "vanilla" else cmap(i % 20)) for i, m in enumerate(models)}
        seeds = sorted(env_df["seed"].unique())
        for seed in seeds:
            seed_df = env_df[env_df["seed"] == seed]
            nrows, ncols = figure_grid(len(axes), max_cols=3)
            fig, axs = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), squeeze=False)
            fig.suptitle(f"{env} seed {seed}: all caps on selected perturbations", fontsize=14)
            for ax, axis in zip(axs.flat, axes):
                asub = seed_df[seed_df["axis"] == axis]
                for model in models:
                    msub = asub[asub["model_label"] == model].sort_values("factor")
                    if msub.empty:
                        continue
                    ax.plot(
                        msub["factor"],
                        msub["mean_return"],
                        color=colors[model],
                        linewidth=2.2 if model == "vanilla" else 1.4,
                        linestyle="--" if model == "vanilla" else "-",
                        alpha=0.95 if model == "vanilla" else 0.8,
                        label=display_model(model),
                    )
                ax.set_title(axis)
                ax.set_xlabel("Perturbation factor")
                ax.set_ylabel("Eval return")
                ax.grid(alpha=0.2)
            for ax in axs.flat[len(axes) :]:
                ax.axis("off")
            handles, labels = axs.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
            fig.tight_layout(rect=(0, 0, 0.88, 0.96))
            save_figure(fig, plot_root / env / f"seed_{seed:02d}_selected_axes_all_caps", formats)


def axes_for_compact_plots(df: pd.DataFrame, all_axes: bool) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for env, sub in df.groupby("env_id"):
        available = sorted(sub["axis"].unique())
        if all_axes:
            out[env] = available
        else:
            selected = [a for a in SELECTED_AXES.get(env, available) if a in available]
            out[env] = selected or available
    return out


def plot_seed_scatter_and_reliability(
    df: pd.DataFrame,
    out_dir: Path,
    baselines: dict[tuple[str, str], float],
    stress_factors: dict[tuple[str, str], float],
    threshold_frac: float,
    formats: list[str],
    all_axes: bool,
) -> None:
    selected = axes_for_compact_plots(df, all_axes)
    models_by_env = {env: sorted(sub["model_label"].unique(), key=model_sort_key) for env, sub in df.groupby("env_id")}
    cmap = plt.get_cmap("tab20")

    for env, axes in selected.items():
        env_df = df[df["env_id"] == env]
        models = models_by_env[env]
        model_positions = {m: i for i, m in enumerate(models)}
        model_colors = {m: ("black" if m == "vanilla" else cmap(i % 20)) for i, m in enumerate(models)}
        for axis in axes:
            factor = stress_factors[(env, axis)]
            stress = env_df[(env_df["axis"] == axis) & np.isclose(env_df["factor"], factor)].copy()
            stress = same_seed_vanilla(stress)
            stress["delta_vs_vanilla"] = stress["mean_return"] - stress["vanilla_return"]

            for ycol, ylabel, suffix in [
                ("mean_return", "Eval return", "raw"),
                ("delta_vs_vanilla", "Return minus same-seed vanilla", "vanilla_subtracted"),
            ]:
                fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(models)), 5.0))
                for model in models:
                    msub = stress[stress["model_label"] == model]
                    if msub.empty:
                        continue
                    rng = np.random.default_rng(abs(hash((env, axis, model, suffix))) % (2**32))
                    x = model_positions[model] + rng.uniform(-0.16, 0.16, size=len(msub))
                    ax.scatter(
                        x,
                        msub[ycol],
                        s=24,
                        alpha=0.72,
                        color=model_colors[model],
                        edgecolors="none",
                    )
                    ax.hlines(
                        float(msub[ycol].median()),
                        model_positions[model] - 0.24,
                        model_positions[model] + 0.24,
                        color="red",
                        linewidth=2.0,
                    )
                if suffix == "vanilla_subtracted":
                    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels([display_model(m) for m in models], rotation=45, ha="right")
                ax.set_title(f"{env} {axis} factor={factor:g}: seed scatter ({suffix})")
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", alpha=0.22)
                fig.tight_layout()
                save_figure(fig, out_dir / "seed_scatter" / env / f"{slug(axis)}_factor_{factor:g}_{suffix}", formats)

            baseline = baselines[(env, axis)]
            threshold_grid = np.linspace(0.0, max(1.25, threshold_frac * 1.5), 151)
            fig, ax = plt.subplots(figsize=(8.0, 5.4))
            reliability_rows: list[dict[str, object]] = []
            for model in models:
                values = stress.loc[stress["model_label"] == model, "mean_return"].to_numpy(dtype=float)
                if values.size == 0:
                    continue
                reliability = [(values >= frac * baseline).mean() for frac in threshold_grid]
                ax.plot(
                    threshold_grid,
                    reliability,
                    color=model_colors[model],
                    linewidth=2.4 if model == "vanilla" else 1.5,
                    linestyle="--" if model == "vanilla" else "-",
                    label=display_model(model),
                )
                for frac, rel in zip(threshold_grid, reliability):
                    reliability_rows.append(
                        {
                            "env_id": env,
                            "axis": axis,
                            "stress_factor": factor,
                            "model_label": model,
                            "threshold_fraction_of_vanilla_nominal_median": float(frac),
                            "threshold_return": float(frac * baseline),
                            "reliability": float(rel),
                        }
                    )
            ax.axvline(threshold_frac, color="black", linestyle=":", linewidth=1.2)
            ax.set_title(f"{env} {axis} factor={factor:g}: robustness reliability curve")
            ax.set_xlabel("Catastrophe threshold / vanilla nominal median return")
            ax.set_ylabel("Reliability = P(return >= threshold)")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.22)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            save_figure(fig, out_dir / "reliability_curves" / env / f"{slug(axis)}_factor_{factor:g}_reliability", formats)
            pd.DataFrame(reliability_rows).to_csv(
                out_dir / "reliability_curves" / env / f"{slug(axis)}_factor_{factor:g}_reliability.csv",
                index=False,
            )


def plot_seed_conditioned_effect(
    df: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    fixed_model_label: str | None,
) -> None:
    group_colors = {
        "weak vanilla": "#d95f02",
        "middle": "#7570b3",
        "elite vanilla": "#1b9e77",
    }

    for env, env_df in df.groupby("env_id"):
        available_axes = set(env_df["axis"].unique())
        axes = [axis for axis in SELECTED_AXES.get(env, sorted(available_axes)) if axis in available_axes]
        if not axes:
            continue
        selected = env_df[env_df["axis"].isin(axes)].copy()

        axis_curve = selected.groupby(["seed", "model_label", "axis"], as_index=False)["mean_return"].mean()
        curve_average = (
            axis_curve.groupby(["seed", "model_label"], as_index=False)["mean_return"]
            .mean()
            .rename(columns={"mean_return": "curve_average"})
        )
        vanilla_curve = curve_average[curve_average["model_label"] == "vanilla"][["seed", "curve_average"]].rename(
            columns={"curve_average": "vanilla_curve_average"}
        )
        gains = curve_average.merge(vanilla_curve, on="seed", how="inner")
        gains["gain"] = gains["curve_average"] - gains["vanilla_curve_average"]

        nominal_parts: list[pd.DataFrame] = []
        for axis in axes:
            nominal = nominal_factor(axis)
            nominal_parts.append(
                selected[
                    (selected["axis"] == axis)
                    & (selected["model_label"] == "vanilla")
                    & np.isclose(selected["factor"], nominal)
                ][["seed", "axis", "mean_return"]]
            )
        nominal_rows = pd.concat(nominal_parts, ignore_index=True)
        vanilla_nominal = (
            nominal_rows.groupby(["seed", "axis"], as_index=False)["mean_return"]
            .mean()
            .groupby("seed", as_index=False)["mean_return"]
            .mean()
            .rename(columns={"mean_return": "vanilla_nominal"})
        )

        robust = gains[gains["model_label"] != "vanilla"].copy()
        if robust.empty:
            continue
        best_idx = robust.groupby("seed")["gain"].idxmax()
        best = robust.loc[best_idx, ["seed", "model_label", "gain"]].rename(
            columns={"model_label": "best_menu_model", "gain": "best_menu_gain"}
        )

        fixed = fixed_model_label if fixed_model_label in set(robust["model_label"]) else None
        if fixed is None:
            fixed = str(robust.groupby("model_label")["gain"].median().idxmax())
        fixed_gain = robust[robust["model_label"] == fixed][["seed", "gain"]].rename(
            columns={"gain": "fixed_gain"}
        )
        summary = vanilla_nominal.merge(best, on="seed", how="inner").merge(fixed_gain, on="seed", how="inner")

        ranked = summary["vanilla_nominal"].rank(method="first", pct=True)
        summary["group"] = pd.cut(
            ranked,
            bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            labels=["weak vanilla", "middle", "elite vanilla"],
            include_lowest=True,
        ).astype(str)

        def correlation(ycol: str) -> float:
            if len(summary) < 2 or summary[ycol].nunique() < 2:
                return float("nan")
            return float(np.corrcoef(summary["vanilla_nominal"], summary[ycol])[0, 1])

        fig, axs = plt.subplots(1, 2, figsize=(15.5, 5.8), sharex=True)
        panels = [
            ("best_menu_gain", "Best robust cap gain\ndiagnostic: ex post over cap menu", correlation("best_menu_gain")),
            ("fixed_gain", f"Fixed {display_model(fixed)} gain\ndeployment-style: one preselected cap", correlation("fixed_gain")),
        ]
        for ax, (ycol, title, corr) in zip(axs, panels):
            for group, gsub in summary.groupby("group"):
                ax.scatter(
                    gsub["vanilla_nominal"],
                    gsub[ycol],
                    s=54,
                    alpha=0.85,
                    color=group_colors[group],
                    label=group,
                )
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
            ax.set_title(f"{title}; r={corr:.2f}")
            ax.set_xlabel("Vanilla nominal return, averaged over selected axes")
            ax.set_ylabel("Robust minus vanilla curve-average return")
            ax.grid(alpha=0.22)
        axs[0].legend(frameon=False)
        fig.suptitle(f"{env}: seed-conditioned effect of robust target clipping", fontsize=15)
        fig.tight_layout()

        result_dir = out_dir / "seed_conditioned_effect" / env
        save_figure(fig, result_dir / "vanilla_nominal_vs_robust_gain", formats)
        summary.to_csv(result_dir / "vanilla_nominal_vs_robust_gain_summary.csv", index=False)


def write_readme(
    out_dir: Path,
    result_dir: Path,
    all_axes: bool,
    threshold_frac: float,
    fixed_model_label: str | None,
) -> None:
    lines = [
        "# Seed-Level Analysis Plots",
        "",
        f"Source result directory: `{result_dir.as_posix()}`",
        "",
        "Outputs:",
        "- `seed_spaghetti_by_axis/`: one figure per environment/axis. Each model panel shows all seed return curves over perturbation level, plus a thick median curve.",
        "- `fixed_seed_all_caps/`: one figure per environment/seed. Each panel is a selected perturbation axis, with all models shown together.",
        "- `seed_scatter/`: one-point-per-seed scatter plots at a stress factor, both raw return and same-seed vanilla-subtracted return.",
        "- `reliability_curves/`: reliability survival curves, `P(return >= threshold)`, where threshold is normalized by vanilla nominal median return.",
        "- `seed_conditioned_effect/`: vanilla nominal quality versus robust curve-average gain, shown both ex post over the cap menu and for one preselected cap.",
        "- `stress_scenario_summary.csv`: model-level stress-factor summary with reliability/failure/win-rate statistics.",
        "- `best_caps_by_axis.csv`: best model by median stress-factor return for each axis.",
        "- `best_caps_by_reliability_auc.csv`: best model by area under the reliability curve between 0 and 1x vanilla nominal median return.",
        "- `per_seed_best_caps.csv`: per-seed best model and number of robust models beating same-seed vanilla at the stress factor.",
        "",
        f"Default catastrophe threshold fraction: `{threshold_frac}`.",
        f"Scatter/reliability axis mode: `{'all axes' if all_axes else 'selected high-signal axes'}`.",
        f"Requested fixed model for seed-conditioned analysis: `{fixed_model_label or 'auto'}`.",
        "",
        "ClipFraction note:",
        "`robust/tv_return_clip_fraction` is logged during training, not evaluation. These eval CSVs are enough for reliability and seed-return analyses, but ClipFraction trajectory/scatter plots require the training TensorBoard event files or exported W&B scalars.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir) if args.out_dir else result_dir / "analysis_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(result_dir)
    baselines = vanilla_nominal_baselines(df)
    stress_factors = stress_factor_map(df)

    write_stress_summaries(df, out_dir, baselines, stress_factors, args.catastrophe_threshold_frac)
    if not args.summary_only:
        plot_seed_spaghetti(df, out_dir, args.formats)
        plot_fixed_seed_caps(df, out_dir, args.formats)
        plot_seed_scatter_and_reliability(
            df,
            out_dir,
            baselines,
            stress_factors,
            args.catastrophe_threshold_frac,
            args.formats,
            args.all_scatter_axes,
        )
        plot_seed_conditioned_effect(df, out_dir, args.formats, args.fixed_model_label)
    write_readme(
        out_dir,
        result_dir,
        args.all_scatter_axes,
        args.catastrophe_threshold_frac,
        args.fixed_model_label,
    )
    print(f"wrote {out_dir}")
    print(f"rows: {len(df)}")
    print(f"envs: {sorted(df['env_id'].unique())}")


if __name__ == "__main__":
    main()
