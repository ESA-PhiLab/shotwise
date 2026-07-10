"""Aggregate analysis for toy adaptive-vs-uniform experiments.

This script scans experiment run directories and writes a compact analysis to
``outputs/toy_analysis`` by default.

Expected run directory contents
-------------------------------
Each run directory should contain at least:
    metrics.csv

The following files are used when available:
    config.json
    dataset_and_reference.npz
    adaptive_supportshot_history.npz
    adaptive_supportshot_history.json
    adaptive_measurements.npz
    uniform_measurements.npz

Example
-------
python examples/analyze_toy_experiments.py \
    --root outputs/toy \
    --out outputs/toy_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator, NullLocator

try:
    from sklearn.svm import SVC
except Exception:  # pragma: no cover
    SVC = None


ERROR_METRICS = [
    "rmse_K",
    "rmse_K_SV",
    "margin_err",
    "func_rmse",
    "margin_err_normalized",
    "func_rmse_normalized",
]

HIGHER_IS_BETTER_METRICS = [
    "jaccard",
    "jaccard_weighted",
    "acc_train_vs_true",
]

PRIMARY_METRICS = [
    "rmse_K",
    "rmse_K_SV",
    "jaccard",
    "jaccard_weighted",
    "margin_err",
    "func_rmse",
    "margin_err_normalized",
    "func_rmse_normalized",
    "acc_train_vs_true",
    "sv_ratio_est",
]

# Metrics used for the paper-style IBM hardware panels.
# These are intended to be combined in LaTeX as a 2 x 2 figure:
#   (a) RMSE(K)
#   (b) RMSE(K_SV)
#   (c) weighted Jaccard
#   (d) normalized decision-function RMSE
STAGE_PLOT_METRICS = [
    "rmse_K",
    "rmse_K_SV",
    "jaccard_weighted",
    "func_rmse_normalized",
]

# Metrics shown on log scale in stage plots.
LOG_STAGE_METRICS = {
    "rmse_K",
    "rmse_K_SV",
    "func_rmse_normalized",
}

PLOT_STYLE = "white"
PLOT_CONTEXT = "paper"

BOX_COLOR = "#2C7FB8"
POINT_COLOR = "black"

PRETTY_METHOD_NAMES = {
    "pilot": "pilot",
    "round_1": "round 1",
    "round_2": "round 2",
    "round_3": "round 3",
    "round_4": "round 4",
    "round_5": "round 5",
    "uniform": "uniform",
}

PRETTY_METRIC_NAMES = {
    "rmse_K": r"$\mathrm{RMSE}(K)$",
    "rmse_K_SV": r"$\mathrm{RMSE}(K_{\mathrm{SV}})$",
    "jaccard": r"$J_{\mathrm{SV}}$",
    "jaccard_weighted": r"$J_{\mathrm{SV}}^{w}$",
    "margin_err": r"$\delta_{\mathrm{margin}}$",
    "func_rmse": r"$\mathrm{RMSE}_{f}$",
    "margin_err_normalized": r"$\delta_{\mathrm{margin}}$",
    "func_rmse_normalized": r"$\delta_{\mathrm{RMSE},f}$",
    "acc_train_vs_true": "Training accuracy",
    "sv_ratio_est": "Estimated SV ratio",
}


def set_plot_style() -> None:
    """Set a paper-like plotting style close to the original manuscript figures."""
    sns.set_theme(
        style=PLOT_STYLE,
        context=PLOT_CONTEXT,
        font_scale=0.95,
    )

    plt.rcParams.update(
        {
            # LaTeX-like serif styling without requiring an external LaTeX install.
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "Times New Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            # Figure quality.
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            # Full framed axes, matching the original plot style.
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            # Font sizes.
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            # No gridlines.
            "axes.grid": False,
            "grid.alpha": 0.0,
            "grid.linewidth": 0.0,
            # Ticks.
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def pretty_method_name(method: str) -> str:
    return PRETTY_METHOD_NAMES.get(method, str(method).replace("_", " "))


def pretty_metric_name(metric: str) -> str:
    return PRETTY_METRIC_NAMES.get(metric, metric.replace("_", " "))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate toy adaptive-vs-uniform experiment outputs.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/toy"),
        help="Root directory containing one folder per experiment run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/toy_analysis"),
        help="Output directory for analysis products.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for metrics.csv files.",
    )
    parser.add_argument(
        "--primary-source",
        default="ibm",
        help="Source used for representative-run selection and highlighted plots.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples for paired confidence intervals.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_source(run_dir: Path, config: dict[str, Any]) -> str:
    name = run_dir.name.lower()
    backend = str(config.get("backend_name") or config.get("backend") or "")
    backend = backend.lower()

    if "ibm" in name or backend.startswith("ibm_"):
        return "ibm"
    if "fake" in name or "manila" in name or "fake" in backend:
        return "fake"
    if "synthetic" in name or "bernoulli" in name or "probability" in name:
        return "synthetic"
    return "unknown"


def find_run_dirs(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/metrics.csv" if recursive else "*/metrics.csv"
    return sorted(path.parent for path in root.glob(pattern))


def load_all_metrics(root: Path, recursive: bool) -> pd.DataFrame:
    rows = []

    for folder_index, run_dir in enumerate(find_run_dirs(root, recursive)):
        metrics_path = run_dir / "metrics.csv"
        config = load_json(run_dir / "config.json")
        source = infer_source(run_dir, config)

        df = pd.read_csv(metrics_path)
        df.insert(0, "source", source)
        df.insert(1, "folder_index", folder_index)
        df.insert(2, "run_name", run_dir.name)
        df.insert(3, "run_dir", str(run_dir))

        if "run" not in df.columns:
            df["run"] = 0

        for key, value in config.items():
            col = f"config_{key}"
            if col not in df.columns:
                df[col] = value

        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No metrics.csv files found under {root}")

    return pd.concat(rows, ignore_index=True)


def method_sort_key(method: str) -> tuple[int, int]:
    if method == "pilot":
        return (0, 0)

    if isinstance(method, str) and method.startswith("round_"):
        try:
            return (1, int(method.split("_")[1]))
        except ValueError:
            return (1, 999)

    if method == "uniform":
        return (2, 0)

    return (3, 0)


def ordered_methods(methods: pd.Series) -> list[str]:
    unique = [m for m in methods.dropna().unique() if isinstance(m, str)]
    return sorted(unique, key=method_sort_key)


def final_adaptive_method(methods: pd.Series) -> str | None:
    rounds = []

    for method in methods.dropna().unique():
        if isinstance(method, str) and method.startswith("round_"):
            try:
                rounds.append((int(method.split("_")[1]), method))
            except ValueError:
                continue

    if not rounds:
        return None

    return max(rounds)[1]


def extract_final_rows(all_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["source", "run_name", "run"]

    for _, group in all_metrics.groupby(group_cols, dropna=False):
        final_method = final_adaptive_method(group["method"])

        if final_method is not None:
            adaptive_rows = group[group["method"] == final_method].copy()
            adaptive_rows["strategy"] = "adaptive"
            adaptive_rows["final_method"] = final_method
            rows.append(adaptive_rows)

        uniform_rows = group[group["method"] == "uniform"].copy()

        if not uniform_rows.empty:
            uniform_rows["strategy"] = "uniform"
            uniform_rows["final_method"] = "uniform"
            rows.append(uniform_rows)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def winner_lower(a: float, u: float, *, tol: float = 1e-12) -> str:
    if pd.isna(a) or pd.isna(u):
        return "missing"

    if abs(a - u) <= tol:
        return "tie"

    return "adaptive" if a < u else "uniform"


def winner_higher(a: float, u: float, *, tol: float = 1e-12) -> str:
    if pd.isna(a) or pd.isna(u):
        return "missing"

    if abs(a - u) <= tol:
        return "tie"

    return "adaptive" if a > u else "uniform"


def paired_comparisons(final_rows: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    group_cols = ["source", "run_name", "run"]

    for _, group in final_rows.groupby(group_cols, dropna=False):
        adaptive = group[group["strategy"] == "adaptive"]
        uniform = group[group["strategy"] == "uniform"]

        if adaptive.empty or uniform.empty:
            continue

        adaptive_row = adaptive.iloc[0]
        uniform_row = uniform.iloc[0]

        item: dict[str, Any] = {
            "source": adaptive_row["source"],
            "run_name": adaptive_row["run_name"],
            "run_dir": adaptive_row["run_dir"],
            "run": adaptive_row["run"],
            "adaptive_method": adaptive_row["method"],
        }

        for metric in PRIMARY_METRICS:
            if metric not in group.columns:
                continue

            a = adaptive_row[metric]
            u = uniform_row[metric]

            item[f"adaptive_{metric}"] = a
            item[f"uniform_{metric}"] = u

            if metric in HIGHER_IS_BETTER_METRICS:
                item[f"delta_{metric}"] = a - u
                item[f"winner_{metric}"] = winner_higher(a, u)
            else:
                item[f"delta_{metric}"] = u - a
                item[f"winner_{metric}"] = winner_lower(a, u)

            if metric in ERROR_METRICS and pd.notna(u) and u != 0:
                item[f"relative_improvement_{metric}"] = (u - a) / u

        comparisons.append(item)

    return pd.DataFrame(comparisons)


def summarize_final_mean_std(final_rows: pd.DataFrame) -> pd.DataFrame:
    available = [m for m in PRIMARY_METRICS if m in final_rows.columns]

    summary = (
        final_rows.groupby(["source", "strategy"])[available]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    return summary


def summarize_round_mean_std(all_metrics: pd.DataFrame) -> pd.DataFrame:
    available = [m for m in PRIMARY_METRICS if m in all_metrics.columns]

    summary = (
        all_metrics.groupby(["source", "method"])[available]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary["method_order"] = summary["method"].map(lambda x: method_sort_key(str(x)))
    summary = summary.sort_values(["source", "method_order"]).drop(
        columns="method_order"
    )

    return summary


def win_counts(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric in PRIMARY_METRICS:
        winner_col = f"winner_{metric}"

        if winner_col not in comparisons.columns:
            continue

        for source, group in comparisons.groupby("source"):
            counts = group[winner_col].value_counts().to_dict()
            n = int(len(group))

            adaptive_wins = int(counts.get("adaptive", 0))
            uniform_wins = int(counts.get("uniform", 0))
            ties = int(counts.get("tie", 0))

            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "n": n,
                    "adaptive_wins": adaptive_wins,
                    "uniform_wins": uniform_wins,
                    "ties": ties,
                    "adaptive_win_rate": adaptive_wins / n if n else np.nan,
                    "uniform_win_rate": uniform_wins / n if n else np.nan,
                }
            )

    return pd.DataFrame(rows)


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int = 0,
) -> tuple[float, float]:
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)

    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_effect_table(comparisons: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []

    for metric in PRIMARY_METRICS:
        delta_col = f"delta_{metric}"

        if delta_col not in comparisons.columns:
            continue

        for source, group in comparisons.groupby("source"):
            values = group[delta_col].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if values.size == 0:
                continue

            ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot)

            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "n": int(values.size),
                    "mean_delta_positive_adaptive_better": float(values.mean()),
                    "std_delta": (
                        float(values.std(ddof=1)) if values.size > 1 else np.nan
                    ),
                    "median_delta": float(np.median(values)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )

    return pd.DataFrame(rows)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def style_stage_axis(ax: plt.Axes) -> None:
    """Apply manuscript-style axis formatting."""
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.tick_params(axis="both", which="major", width=0.8, length=3.0)
    ax.margins(x=0.02)


def apply_metric_scale(ax: plt.Axes, metric: str) -> None:
    """Apply log scale to RMSE-like panels and linear scale otherwise."""
    if metric in LOG_STAGE_METRICS:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.set_yscale("linear")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def draw_stage_boxplot(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    metric: str,
    methods: list[str],
) -> None:
    """Draw one stage-wise boxplot panel."""
    sns.boxplot(
        data=plot_df,
        x="method",
        y=metric,
        order=methods,
        color=BOX_COLOR,
        width=0.52,
        linewidth=0.8,
        showfliers=False,
        whis=1.5,
        ax=ax,
        boxprops={"edgecolor": "0.2", "linewidth": 0.8, "alpha": 0.95},
        medianprops={"color": "0.2", "linewidth": 1.0},
        whiskerprops={"color": "0.2", "linewidth": 0.8},
        capprops={"color": "0.2", "linewidth": 0.8},
    )

    sns.stripplot(
        data=plot_df,
        x="method",
        y=metric,
        order=methods,
        color=POINT_COLOR,
        size=2.0,
        alpha=0.28,
        jitter=0.18,
        linewidth=0,
        ax=ax,
    )

    uniform_values = plot_df.loc[plot_df["method"] == "uniform", metric].dropna()

    if not uniform_values.empty:
        uniform_median = float(uniform_values.median())
        ax.axhline(
            uniform_median,
            color="black",
            linestyle="--",
            linewidth=0.9,
            alpha=0.9,
            zorder=10,
        )

    ax.set_xlabel("")
    ax.set_ylabel(pretty_metric_name(metric))
    ax.set_title("")

    ax.set_xticklabels(
        [pretty_method_name(method) for method in methods],
        rotation=0,
        ha="center",
    )

    apply_metric_scale(ax, metric)
    style_stage_axis(ax)


def plot_stage_boxplots(all_metrics: pd.DataFrame, out: Path) -> None:
    """Save individual paper-style stage boxplots.

    The intended LaTeX workflow is to combine these separate panels manually
    using subfigures/subcaption in the manuscript.
    """
    plot_dir = ensure_dir(out / "plots" / "stage_boxplots")

    for source, source_df in all_metrics.groupby("source"):
        methods = ordered_methods(source_df["method"])

        source_metrics = [
            metric for metric in STAGE_PLOT_METRICS if metric in source_df.columns
        ]

        if not source_metrics:
            continue

        for metric in source_metrics:
            plot_df = source_df[["method", metric]].dropna().copy()
            plot_df = plot_df[plot_df["method"].isin(methods)]

            if plot_df.empty:
                continue

            plot_df["method"] = pd.Categorical(
                plot_df["method"],
                categories=methods,
                ordered=True,
            )

            fig_width = max(4.3, 0.75 * len(methods) + 1.0)
            fig, ax = plt.subplots(figsize=(fig_width, 2.75))

            draw_stage_boxplot(ax, plot_df, metric, methods)

            fig.tight_layout(pad=0.4)

            fig.savefig(plot_dir / f"{source}_{metric}.pdf", bbox_inches="tight")
            fig.savefig(plot_dir / f"{source}_{metric}.png", bbox_inches="tight")

            plt.close(fig)


def load_npz_if_exists(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None

    data = np.load(path)
    return {key: data[key] for key in data.files}


def save_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel("sample index")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_grid(
    matrices: list[np.ndarray],
    labels: list[str],
    path: Path,
    title: str,
    *,
    cmap: str | None = None,
    symmetric: bool = False,
) -> None:
    if not matrices:
        return

    n_panels = len(matrices)

    vmin = min(float(np.nanmin(m)) for m in matrices)
    vmax = max(float(np.nanmax(m)) for m in matrices)

    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4 * n_panels, 3.8),
        squeeze=False,
    )

    im = None

    for ax, matrix, label in zip(axes[0], matrices, labels, strict=True):
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(label)
        ax.set_xlabel("j")
        ax.set_ylabel("i")

    fig.suptitle(title)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def history_stage_labels(run_dir: Path, n_stages: int) -> list[str]:
    metadata = load_json(run_dir / "adaptive_supportshot_history.json")
    labels = []

    if isinstance(metadata, list):
        for item in metadata:
            stage = item.get("stage", "stage")
            round_index = item.get("round_index", len(labels))

            if stage == "pilot":
                labels.append("pilot")
            else:
                labels.append(f"round_{round_index}")

    if len(labels) == n_stages:
        return labels

    fallback = ["pilot"] + [f"round_{idx}" for idx in range(1, n_stages)]
    return fallback[:n_stages]


def aggregate_allocations(all_metrics: pd.DataFrame, out: Path) -> None:
    alloc_dir = ensure_dir(out / "plots" / "allocations")
    table_dir = ensure_dir(out / "tables")

    run_info = all_metrics[["source", "run_name", "run_dir"]].drop_duplicates()

    stage_data: dict[str, dict[str, list[np.ndarray]]] = {}
    cumulative_data: dict[str, list[np.ndarray]] = {}
    uniform_data: dict[str, list[np.ndarray]] = {}

    for _, row in run_info.iterrows():
        source = row["source"]
        run_dir = Path(row["run_dir"])

        hist = load_npz_if_exists(run_dir / "adaptive_supportshot_history.npz")

        if hist is not None and "allocation_matrices" in hist:
            allocations = hist["allocation_matrices"]
            labels = history_stage_labels(run_dir, allocations.shape[0])

            for label, matrix in zip(labels, allocations, strict=True):
                stage_data.setdefault(source, {}).setdefault(label, []).append(matrix)

            cumulative_data.setdefault(source, []).append(allocations.sum(axis=0))

        uniform = load_npz_if_exists(run_dir / "uniform_measurements.npz")

        if uniform is not None and "requested_shots" in uniform:
            requested = uniform["requested_shots"]
            uniform_alloc = requested.sum(axis=0) if requested.ndim == 3 else requested
            uniform_data.setdefault(source, []).append(uniform_alloc)

    rows = []

    for source, stages in stage_data.items():
        labels = sorted(stages, key=method_sort_key)
        mean_matrices = []
        std_matrices = []

        for label in labels:
            stack = np.stack(stages[label], axis=0)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)

            mean_matrices.append(mean)
            std_matrices.append(std)

            rows.append(
                {
                    "source": source,
                    "stage": label,
                    "n": stack.shape[0],
                    "total_shots_mean": float(mean.sum()),
                    "total_shots_std": float(stack.sum(axis=(1, 2)).std()),
                    "max_entry_mean": float(mean.max()),
                }
            )

        save_heatmap_grid(
            mean_matrices,
            labels,
            alloc_dir / f"{source}_allocation_stage_means.png",
            f"{source}: mean adaptive allocation per stage",
        )

        save_heatmap_grid(
            std_matrices,
            labels,
            alloc_dir / f"{source}_allocation_stage_stds.png",
            f"{source}: std of adaptive allocation per stage",
        )

    for source, matrices in cumulative_data.items():
        stack = np.stack(matrices, axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)

        np.savez_compressed(
            table_dir / f"adaptive_cumulative_allocations_{source}.npz",
            allocations=stack,
            mean=mean,
            std=std,
        )

        save_heatmap(
            mean,
            alloc_dir / f"{source}_adaptive_cumulative_mean.png",
            f"{source}: adaptive cumulative mean",
        )

        save_heatmap(
            std,
            alloc_dir / f"{source}_adaptive_cumulative_std.png",
            f"{source}: adaptive cumulative std",
        )

    for source, matrices in uniform_data.items():
        stack = np.stack(matrices, axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)

        np.savez_compressed(
            table_dir / f"uniform_allocations_{source}.npz",
            allocations=stack,
            mean=mean,
            std=std,
        )

        save_heatmap(
            mean,
            alloc_dir / f"{source}_uniform_mean.png",
            f"{source}: uniform mean",
        )

    for source in sorted(set(cumulative_data) & set(uniform_data)):
        adaptive_mean = np.stack(cumulative_data[source], axis=0).mean(axis=0)
        uniform_mean = np.stack(uniform_data[source], axis=0).mean(axis=0)
        diff = adaptive_mean - uniform_mean

        save_heatmap(
            diff,
            alloc_dir / f"{source}_adaptive_minus_uniform_mean.png",
            f"{source}: adaptive - uniform allocation",
            cmap="coolwarm",
        )

    pd.DataFrame(rows).to_csv(table_dir / "allocation_stage_summary.csv", index=False)


def reference_q_from_svm(
    kernel_true: np.ndarray, y: np.ndarray, c: float
) -> np.ndarray:
    if SVC is None:
        raise RuntimeError("scikit-learn is required for margin-variance analysis")

    clf = SVC(kernel="precomputed", C=c)
    clf.fit(kernel_true, y)

    q = np.zeros(len(y), dtype=float)

    # sklearn dual_coef_ contains y_i * alpha_i for support vectors.
    q[clf.support_] = clf.dual_coef_[0]

    return q


def margin_variance_from_allocation(
    kernel_true: np.ndarray,
    q: np.ndarray,
    allocation: np.ndarray,
) -> tuple[float, float]:
    n = kernel_true.shape[0]
    rows, cols = np.triu_indices(n, k=1)

    s_value = float(q @ kernel_true @ q)
    var_s = 0.0

    for i, j in zip(rows, cols, strict=True):
        shots = float(allocation[i, j])

        if shots <= 0:
            continue

        kij = float(kernel_true[i, j])
        grad = 2.0 * q[i] * q[j]
        var_k = kij * (1.0 - kij) / shots
        var_s += (grad * grad) * var_k

    var_margin = var_s / (4.0 * s_value) if s_value > 0 else np.nan

    return float(var_s), float(var_margin)


def oracle_allocation_for_margin(
    kernel_true: np.ndarray,
    q: np.ndarray,
    total_budget: float,
) -> np.ndarray:
    n = kernel_true.shape[0]
    rows, cols = np.triu_indices(n, k=1)

    weights = []

    for i, j in zip(rows, cols, strict=True):
        kij = float(kernel_true[i, j])
        grad = abs(2.0 * q[i] * q[j])
        weights.append(grad * np.sqrt(max(kij * (1.0 - kij), 0.0)))

    weights_arr = np.asarray(weights, dtype=float)
    alloc = np.zeros((n, n), dtype=float)

    if weights_arr.sum() <= 0:
        alloc[rows, cols] = total_budget / len(rows)
    else:
        alloc[rows, cols] = total_budget * weights_arr / weights_arr.sum()

    return alloc


def margin_variance_analysis(all_metrics: pd.DataFrame, out: Path) -> None:
    table_dir = ensure_dir(out / "tables")
    plot_dir = ensure_dir(out / "plots" / "variance")

    run_info = all_metrics[["source", "run_name", "run_dir"]].drop_duplicates()
    rows = []

    for _, row in run_info.iterrows():
        source = row["source"]
        run_name = row["run_name"]
        run_dir = Path(row["run_dir"])

        ref = load_npz_if_exists(run_dir / "dataset_and_reference.npz")
        hist = load_npz_if_exists(run_dir / "adaptive_supportshot_history.npz")
        uniform = load_npz_if_exists(run_dir / "uniform_measurements.npz")

        if ref is None or hist is None or uniform is None:
            continue

        if "kernel_true" not in ref or "y" not in ref:
            continue

        if "allocation_matrices" not in hist or "requested_shots" not in uniform:
            continue

        kernel_true = ref["kernel_true"]
        y = ref["y"]

        config = load_json(run_dir / "config.json")
        c_value = float(config.get("C", config.get("c", 10.0)))

        q = reference_q_from_svm(kernel_true, y, c_value)

        adaptive_alloc = hist["allocation_matrices"].sum(axis=0)

        requested = uniform["requested_shots"]
        uniform_alloc = requested.sum(axis=0) if requested.ndim == 3 else requested

        total_budget = float(uniform_alloc.sum())
        oracle_alloc = oracle_allocation_for_margin(kernel_true, q, total_budget)

        var_s_adapt, var_m_adapt = margin_variance_from_allocation(
            kernel_true,
            q,
            adaptive_alloc,
        )

        var_s_uniform, var_m_uniform = margin_variance_from_allocation(
            kernel_true,
            q,
            uniform_alloc,
        )

        var_s_oracle, var_m_oracle = margin_variance_from_allocation(
            kernel_true,
            q,
            oracle_alloc,
        )

        rows.append(
            {
                "source": source,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "total_budget": total_budget,
                "adaptive_var_s": var_s_adapt,
                "uniform_var_s": var_s_uniform,
                "oracle_var_s": var_s_oracle,
                "adaptive_var_margin": var_m_adapt,
                "uniform_var_margin": var_m_uniform,
                "oracle_var_margin": var_m_oracle,
                "adaptive_over_uniform_var_margin": (
                    var_m_adapt / var_m_uniform if var_m_uniform > 0 else np.nan
                ),
                "oracle_over_uniform_var_margin": (
                    var_m_oracle / var_m_uniform if var_m_uniform > 0 else np.nan
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(table_dir / "margin_variance.csv", index=False)

    if df.empty:
        return

    for source, source_df in df.groupby("source"):
        ratio_df = pd.DataFrame(
            {
                "strategy": (
                    ["adaptive / uniform"] * len(source_df)
                    + ["oracle / uniform"] * len(source_df)
                ),
                "margin_variance_ratio": pd.concat(
                    [
                        source_df["adaptive_over_uniform_var_margin"],
                        source_df["oracle_over_uniform_var_margin"],
                    ],
                    ignore_index=True,
                ),
            }
        ).dropna()

        if ratio_df.empty:
            continue

        order = ["adaptive / uniform", "oracle / uniform"]

        fig, ax = plt.subplots(figsize=(4.2, 3.2))

        sns.boxplot(
            data=ratio_df,
            x="strategy",
            y="margin_variance_ratio",
            order=order,
            color=BOX_COLOR,
            width=0.52,
            linewidth=0.8,
            showfliers=False,
            whis=1.5,
            ax=ax,
            boxprops={"edgecolor": "0.2", "linewidth": 0.8, "alpha": 0.95},
            medianprops={"color": "0.2", "linewidth": 1.0},
            whiskerprops={"color": "0.2", "linewidth": 0.8},
            capprops={"color": "0.2", "linewidth": 0.8},
        )

        sns.stripplot(
            data=ratio_df,
            x="strategy",
            y="margin_variance_ratio",
            order=order,
            color=POINT_COLOR,
            size=2.0,
            alpha=0.28,
            jitter=0.14,
            linewidth=0,
            ax=ax,
        )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)

        ax.set_xlabel("")
        ax.set_ylabel("Margin-variance ratio")
        ax.set_title("")
        ax.set_xticklabels(order, rotation=20, ha="right")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        style_stage_axis(ax)

        fig.tight_layout()

        fig.savefig(
            plot_dir / f"{source}_margin_variance_ratio.pdf", bbox_inches="tight"
        )
        fig.savefig(
            plot_dir / f"{source}_margin_variance_ratio.png", bbox_inches="tight"
        )

        plt.close(fig)


def select_representative_runs(comparisons: pd.DataFrame, out: Path) -> None:
    metric = "delta_func_rmse"

    if metric not in comparisons.columns:
        return

    rows = []

    for _, group in comparisons.groupby("source"):
        valid = group[np.isfinite(group[metric])].copy()

        if valid.empty:
            continue

        median_delta = valid[metric].median()
        valid["distance_to_median_delta_func_rmse"] = (
            valid[metric] - median_delta
        ).abs()

        rows.append(valid.sort_values("distance_to_median_delta_func_rmse").iloc[0])

    if rows:
        pd.DataFrame(rows).to_csv(
            out / "tables" / "representative_runs.csv",
            index=False,
        )


def main() -> None:
    args = parse_args()

    ensure_dir(args.out)
    table_dir = ensure_dir(args.out / "tables")

    set_plot_style()

    all_metrics = load_all_metrics(args.root, args.recursive)
    all_metrics.to_csv(table_dir / "combined_metrics.csv", index=False)

    final_rows = extract_final_rows(all_metrics)
    final_rows.to_csv(table_dir / "final_strategy_metrics.csv", index=False)

    comparisons = paired_comparisons(final_rows)
    comparisons.to_csv(table_dir / "paired_final_comparisons.csv", index=False)

    summary = summarize_final_mean_std(final_rows)
    summary.to_csv(table_dir / "summary_final_mean_std.csv", index=False)

    round_summary = summarize_round_mean_std(all_metrics)
    round_summary.to_csv(table_dir / "summary_round_mean_std.csv", index=False)

    wins = win_counts(comparisons)
    wins.to_csv(table_dir / "win_counts.csv", index=False)

    effects = paired_effect_table(comparisons, n_boot=args.bootstrap_samples)
    effects.to_csv(table_dir / "paired_effects_bootstrap_ci.csv", index=False)

    plot_stage_boxplots(all_metrics, args.out)

    # Paired plots are intentionally disabled because they are not used in the
    # manuscript and are visually less informative than the stage-summary panels.
    aggregate_allocations(all_metrics, args.out)
    margin_variance_analysis(all_metrics, args.out)
    select_representative_runs(comparisons, args.out)

    print("Analysis complete.")
    print(f"Input root: {args.root}")
    print(f"Output dir: {args.out}")
    print(f"Metric rows: {len(all_metrics)}")
    print(f"Run folders: {all_metrics['run_name'].nunique()}")
    print(f"Paired adaptive/uniform comparisons: {len(comparisons)}")
    print("Key outputs:")
    print(f"- {table_dir / 'summary_round_mean_std.csv'}")
    print(f"- {table_dir / 'summary_final_mean_std.csv'}")
    print(f"- {table_dir / 'win_counts.csv'}")
    print(f"- {table_dir / 'paired_effects_bootstrap_ci.csv'}")
    print(f"- {table_dir / 'margin_variance.csv'}")
    print(f"- {args.out / 'plots'}")


if __name__ == "__main__":
    main()
