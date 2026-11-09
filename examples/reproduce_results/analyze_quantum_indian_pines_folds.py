"""Analyze Indian Pines quantum-kernel results separately by fold.

The measurement experiment has a hierarchical structure:

    qubit count
        -> four distinct Indian Pines folds / kernel matrices
            -> repeated finite-shot measurement realizations.

The four folds are treated as different learning problems rather than pooled
measurement repetitions. This script first aggregates the repeated
measurement realizations within every fold and qubit count, then examines
variation between folds.

The script also computes exact-kernel structural diagnostics:

- off-diagonal kernel standard deviation,
- oracle support-vector fraction,
- oracle dual-coefficient Gini coefficient,
- oracle measurement-allocation-weight Gini coefficient.

Example
-------
From the repository root:

    python examples/reproduce_results/analyze_quantum_indian_pines_folds.py \
        --results \
        outputs/quantum_indian_pines_no_bandwidth/quantum_indian_pines_results.csv \
        --data-dir \
        data/quantum_kernels/indian_pines_no_bandwidth \
        --output-dir \
        outputs/quantum_indian_pines_no_bandwidth/fold_analysis
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from shotwise.utils.matrix import upper_triangle_indices


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Indian Pines quantum-kernel measurement results "
            "separately by fold."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Raw CSV produced by quantum_indian_pines.py.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Indian Pines reproduction-data directory containing "
            "bandwidth/ and Indian_train_labels.npz."
        ),
    )
    parser.add_argument(
        "--kernel-subdir",
        default="bandwidth",
        help=(
            "Kernel subdirectory inside data-dir. The no-bandwidth "
            "download currently also uses the local name 'bandwidth'."
        ),
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "median"],
        default="mean",
        help=(
            "Aggregation over measurement realizations within each "
            "fold/qubit condition. Use mean to reproduce the historical "
            "analysis."
        ),
    )
    return parser.parse_args()


def validate_results(df: pd.DataFrame) -> None:
    """Validate required raw-result columns."""
    required = {
        "run",
        "fold",
        "n_qubits",
        "relative_func_rmse_improvement",
        "adaptive_func_rmse_normalized",
        "uniform_func_rmse_normalized",
        "offdiag_std",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Results CSV is missing columns: " + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("Results CSV is empty.")

    for column in required.difference({"run", "fold", "n_qubits"}):
        values = df[column].to_numpy(dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(f"Column {column!r} contains non-finite values.")


def load_labels(data_dir: Path) -> dict[int, np.ndarray]:
    """Load historical labels and convert them from {0,1} to {-1,+1}."""
    path = data_dir / "Indian_train_labels.npz"

    if not path.exists():
        raise FileNotFoundError(f"Label archive not found: {path}")

    labels: dict[int, np.ndarray] = {}

    with np.load(path) as archive:
        for fold in range(4):
            historical = np.asarray(
                archive[f"fold_{fold}"],
                dtype=int,
            ).reshape(-1)

            values = set(np.unique(historical).tolist())

            if values != {0, 1}:
                raise ValueError(
                    f"Fold {fold} historical labels contain "
                    f"{sorted(values)}, expected [0, 1]."
                )

            labels[fold] = 2 * historical - 1

    return labels


def parse_filename(path: Path) -> tuple[int, int]:
    """Return fold and qubit count parsed from a kernel filename."""
    fold_match = re.search(
        r"fold_(\d+)",
        path.name,
    )
    qubit_match = re.search(
        r"_n_(\d+)_",
        path.name,
    )

    if fold_match is None or qubit_match is None:
        raise ValueError(f"Could not parse fold/qubits from {path.name!r}.")

    return (
        int(fold_match.group(1)),
        int(qubit_match.group(1)),
    )


def gini_coefficient(values: np.ndarray) -> float:
    """Return the Gini coefficient of finite non-negative values."""
    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError("Gini input must not be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError("Gini input contains non-finite values.")

    if np.any(array < 0.0):
        raise ValueError("Gini input must be non-negative.")

    total = float(array.sum())

    if total == 0.0:
        return 0.0

    ordered = np.sort(array)
    n = ordered.size
    indices = np.arange(
        1,
        n + 1,
        dtype=float,
    )

    value = 2.0 * np.sum(indices * ordered) / (n * total) - (n + 1.0) / n

    return float(np.clip(value, 0.0, 1.0))


def oracle_diagnostics(
    kernel: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
) -> dict[str, float | int]:
    """Compute exact-kernel SVM and allocation-weight diagnostics."""
    kernel = np.asarray(
        kernel,
        dtype=float,
    )
    kernel = 0.5 * (kernel + kernel.T)
    kernel = np.clip(
        kernel,
        0.0,
        1.0,
    )
    np.fill_diagonal(
        kernel,
        1.0,
    )

    model = SVC(
        kernel="precomputed",
        C=c_value,
    )
    model.fit(
        kernel,
        y,
    )

    alpha = np.zeros(
        y.size,
        dtype=float,
    )
    alpha[model.support_] = np.abs(model.dual_coef_[0])

    rows, cols = upper_triangle_indices(kernel.shape[0])

    probabilities = kernel[
        rows,
        cols,
    ]

    measurement_std = np.sqrt(probabilities * (1.0 - probabilities))

    oracle_weights = alpha[rows] * alpha[cols] * measurement_std

    oracle_weight_sum = float(oracle_weights.sum())

    oracle_weight_sq_sum = float(np.sum(oracle_weights**2))

    n_entries = int(oracle_weights.size)

    if oracle_weight_sum > 0.0:
        oracle_variance_ratio = float(
            n_entries * oracle_weight_sq_sum / (oracle_weight_sum**2)
        )
    else:
        oracle_variance_ratio = 1.0

    return {
        "sv_fraction": (len(model.support_) / y.size),
        "n_support_vectors": len(model.support_),
        "alpha_gini": gini_coefficient(alpha),
        "weight_gini": gini_coefficient(oracle_weights),
        "offdiag_mean": float(probabilities.mean()),
        "offdiag_std": float(probabilities.std()),
        "oracle_weight_sum": oracle_weight_sum,
        "oracle_weight_sq_sum": oracle_weight_sq_sum,
        "oracle_variance_ratio": oracle_variance_ratio,
    }


def create_structural_table(
    data_dir: Path,
    *,
    kernel_subdir: str,
    c_value: float,
) -> pd.DataFrame:
    """Compute structural diagnostics for every fold/qubit kernel."""
    kernel_dir = data_dir / kernel_subdir

    if not kernel_dir.exists():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

    labels = load_labels(data_dir)

    records: list[dict[str, float | int | str]] = []

    paths = sorted(kernel_dir.glob("quantum_train_*.npy"))

    if not paths:
        raise FileNotFoundError(f"No quantum training kernels found in {kernel_dir}.")

    for path in paths:
        fold, qubits = parse_filename(path)

        kernel = np.load(path)
        y = labels[fold]

        if kernel.shape != (
            y.size,
            y.size,
        ):
            raise ValueError(
                f"{path.name}: kernel shape {kernel.shape} "
                f"does not match label length {y.size}."
            )

        record: dict[
            str,
            float | int | str,
        ] = {
            "kernel_filename": path.name,
            "fold": fold,
            "n_qubits": qubits,
            "c_value": c_value,
        }

        record.update(
            oracle_diagnostics(
                kernel,
                y,
                c_value=c_value,
            )
        )
        records.append(record)

    return (
        pd.DataFrame.from_records(records)
        .sort_values(["fold", "n_qubits"])
        .reset_index(drop=True)
    )


def aggregate_measurements(
    df: pd.DataFrame,
    *,
    aggregation: str,
) -> pd.DataFrame:
    """Aggregate measurement repetitions inside every fold/qubit."""
    grouped = df.groupby(
        ["fold", "n_qubits"],
        sort=True,
    )

    metrics = [
        "relative_func_rmse_improvement",
        "adaptive_func_rmse_normalized",
        "uniform_func_rmse_normalized",
    ]

    if aggregation == "mean":
        center = grouped[metrics].mean()
    else:
        center = grouped[metrics].median()

    center = center.reset_index()

    q25 = (
        grouped[metrics]
        .quantile(0.25)
        .reset_index()
        .rename(columns={metric: f"{metric}_q25" for metric in metrics})
    )
    q75 = (
        grouped[metrics]
        .quantile(0.75)
        .reset_index()
        .rename(columns={metric: f"{metric}_q75" for metric in metrics})
    )
    std = (
        grouped[metrics]
        .std()
        .reset_index()
        .rename(columns={metric: f"{metric}_std" for metric in metrics})
    )
    count = grouped.size().reset_index(name="measurement_repetitions")

    result = center.merge(
        q25,
        on=["fold", "n_qubits"],
    )
    result = result.merge(
        q75,
        on=["fold", "n_qubits"],
    )
    result = result.merge(
        std,
        on=["fold", "n_qubits"],
    )
    result = result.merge(
        count,
        on=["fold", "n_qubits"],
    )

    return result


def aggregate_across_folds(
    fold_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate fold-level centers across the four folds."""
    metrics = [
        "relative_func_rmse_improvement",
        "adaptive_func_rmse_normalized",
        "uniform_func_rmse_normalized",
    ]

    grouped = fold_summary.groupby(
        "n_qubits",
        sort=True,
    )

    result = grouped[metrics].agg(["mean", "std", "min", "max"])

    result.columns = [f"{metric}_{statistic}" for metric, statistic in result.columns]

    return result.reset_index()


def fold_colors() -> dict[int, str]:
    """Return restrained fold-specific colors."""
    return {
        0: "#1f4e79",
        1: "#377eb8",
        2: "#6baed6",
        3: "#9ecae1",
    }


def save_relative_facets(
    fold_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save 2x2 fold-faceted relative-improvement figure."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 5.0),
        sharex=True,
        sharey=True,
    )

    colors = fold_colors()

    for fold, ax in enumerate(axes.flat):
        selected = fold_summary.loc[fold_summary["fold"] == fold].sort_values(
            "n_qubits"
        )

        x = selected["n_qubits"].to_numpy(dtype=float)
        y = selected["relative_func_rmse_improvement"].to_numpy(dtype=float)

        q25 = selected["relative_func_rmse_improvement_q25"].to_numpy(dtype=float)
        q75 = selected["relative_func_rmse_improvement_q75"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            color=colors[fold],
            linewidth=1.2,
            marker="o",
            markersize=3.8,
        )
        ax.fill_between(
            x,
            q25,
            q75,
            color=colors[fold],
            alpha=0.12,
            linewidth=0.0,
        )
        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_xscale("log")
        ax.set_title(
            f"Fold {fold}",
            fontsize=9,
        )

    axes[1, 0].set_xlabel(
        "Number of qubits",
        fontsize=8,
    )
    axes[1, 1].set_xlabel(
        "Number of qubits",
        fontsize=8,
    )
    axes[0, 0].set_ylabel(
        r"Relative improvement in $\mathrm{RMSE}_f$",
        fontsize=8,
    )
    axes[1, 0].set_ylabel(
        r"Relative improvement in $\mathrm{RMSE}_f$",
        fontsize=8,
    )

    style_axes(axes.flat)

    fig.tight_layout(
        pad=0.55,
        w_pad=0.8,
        h_pad=0.8,
    )

    save_pdf_png(
        fig,
        output_dir,
        "quantum_fold_relative_improvement",
    )


def save_absolute_facets(
    fold_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save fold-faceted adaptive/uniform absolute-error figure."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 5.0),
        sharex=True,
        sharey=True,
    )

    colors = fold_colors()

    for fold, ax in enumerate(axes.flat):
        selected = fold_summary.loc[fold_summary["fold"] == fold].sort_values(
            "n_qubits"
        )

        x = selected["n_qubits"].to_numpy(dtype=float)

        ax.plot(
            x,
            selected["adaptive_func_rmse_normalized"].to_numpy(dtype=float),
            color=colors[fold],
            linestyle="-",
            linewidth=1.2,
            marker="o",
            markersize=3.8,
            label="Adaptive",
        )

        ax.plot(
            x,
            selected["uniform_func_rmse_normalized"].to_numpy(dtype=float),
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            marker="s",
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor="#666666",
            markeredgewidth=0.7,
            label="Uniform",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"Fold {fold}",
            fontsize=9,
        )

    axes[0, 1].legend(
        frameon=False,
        fontsize=7,
        loc="best",
    )

    axes[1, 0].set_xlabel(
        "Number of qubits",
        fontsize=8,
    )
    axes[1, 1].set_xlabel(
        "Number of qubits",
        fontsize=8,
    )
    axes[0, 0].set_ylabel(
        r"$\mathrm{RMSE}_f$",
        fontsize=8,
    )
    axes[1, 0].set_ylabel(
        r"$\mathrm{RMSE}_f$",
        fontsize=8,
    )

    style_axes(axes.flat)

    fig.tight_layout(
        pad=0.55,
        w_pad=0.8,
        h_pad=0.8,
    )

    save_pdf_png(
        fig,
        output_dir,
        "quantum_fold_absolute_rmse",
    )


def save_fold_overlay(
    fold_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save thin fold curves with the across-fold mean."""
    fig, ax = plt.subplots(figsize=(4.1, 2.9))

    colors = fold_colors()

    for fold in range(4):
        selected = fold_summary.loc[fold_summary["fold"] == fold].sort_values(
            "n_qubits"
        )

        ax.plot(
            selected["n_qubits"],
            selected["relative_func_rmse_improvement"],
            color=colors[fold],
            linewidth=0.9,
            marker="o",
            markersize=3.1,
            alpha=0.75,
            label=f"Fold {fold}",
        )

    mean_curve = (
        fold_summary.groupby(
            "n_qubits",
            sort=True,
        )["relative_func_rmse_improvement"]
        .mean()
        .reset_index()
    )

    ax.plot(
        mean_curve["n_qubits"],
        mean_curve["relative_func_rmse_improvement"],
        color="black",
        linewidth=1.6,
        marker="D",
        markersize=3.8,
        label="Fold mean",
        zorder=5,
    )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    ax.set_xscale("log")
    ax.set_xlabel(
        "Number of qubits",
        fontsize=8,
    )
    ax.set_ylabel(
        r"Relative improvement in $\mathrm{RMSE}_f$",
        fontsize=8,
    )
    ax.legend(
        frameon=False,
        fontsize=6.7,
        ncol=2,
        loc="best",
    )

    style_axes([ax])

    fig.tight_layout(pad=0.5)

    save_pdf_png(
        fig,
        output_dir,
        "quantum_fold_overlay",
    )


def save_structure_figure(
    structural: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save fold-specific exact-kernel structural diagnostics."""
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.0, 7.0),
        sharex=True,
    )

    colors = fold_colors()

    specs = [
        (
            "offdiag_std",
            r"Off-diagonal std. $\sigma(K_{ij})$",
        ),
        (
            "sv_fraction",
            r"SV fraction $|S^\star|/n$",
        ),
        (
            "alpha_gini",
            r"Dual Gini $G(|\alpha^\star|)$",
        ),
        (
            "weight_gini",
            r"Allocation-weight Gini $G(w^\star)$",
        ),
        (
            "oracle_variance_ratio",
            r"Oracle variance ratio $R_{\mathrm{oracle}}$",
        ),
        (
            "oracle_weight_sum",
            r"Total oracle weight $\sum w_{ij}^\star$",
        ),
    ]

    for ax, (
        metric,
        ylabel,
    ) in zip(
        axes.flat,
        specs,
        strict=True,
    ):
        for fold in range(4):
            selected = structural.loc[structural["fold"] == fold].sort_values(
                "n_qubits"
            )

            ax.plot(
                selected["n_qubits"],
                selected[metric],
                color=colors[fold],
                linewidth=1.05,
                marker="o",
                markersize=3.2,
                label=f"Fold {fold}",
            )

        ax.set_xscale("log")
        ax.set_ylabel(
            ylabel,
            fontsize=8,
        )

    axes[0, 1].legend(
        frameon=False,
        fontsize=6.7,
        loc="best",
    )

    axes[2, 0].set_xlabel(
        "Number of qubits",
        fontsize=8,
    )
    axes[2, 1].set_xlabel(
        "Number of qubits",
        fontsize=8,
    )

    style_axes(axes.flat)

    fig.tight_layout(
        pad=0.55,
        w_pad=0.9,
        h_pad=0.8,
    )

    save_pdf_png(
        fig,
        output_dir,
        "quantum_fold_structure",
    )


def style_axes(axes) -> None:
    """Apply restrained manuscript-compatible axis styling."""
    for ax in axes:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=7,
            width=0.6,
            length=3.0,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            width=0.5,
            length=2.0,
        )

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.65)

        ax.grid(False)
        ax.margins(x=0.03)


def save_pdf_png(
    fig,
    output_dir: Path,
    stem: str,
) -> None:
    """Save one figure as PNG and PDF."""
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def print_fold_table(
    fold_summary: pd.DataFrame,
) -> None:
    """Print fold-level relative-improvement centers."""
    table = fold_summary.pivot(
        index="n_qubits",
        columns="fold",
        values="relative_func_rmse_improvement",
    )

    table.columns = [f"fold_{int(fold)}" for fold in table.columns]

    table["fold_mean"] = table.mean(axis=1)
    table["fold_std"] = table[
        [
            "fold_0",
            "fold_1",
            "fold_2",
            "fold_3",
        ]
    ].std(axis=1)

    print(table.to_string(float_format=lambda value: f"{value: .4f}"))


def main() -> None:
    """Run fold-aware analysis of the quantum-kernel experiment."""
    args = parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results file not found: {args.results}")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = pd.read_csv(args.results)
    validate_results(raw)

    fold_summary = aggregate_measurements(
        raw,
        aggregation=args.aggregation,
    )

    structural = create_structural_table(
        args.data_dir,
        kernel_subdir=args.kernel_subdir,
        c_value=args.c,
    )

    fold_summary = fold_summary.merge(
        structural,
        on=["fold", "n_qubits"],
        how="left",
        validate="one_to_one",
    )

    aggregate = aggregate_across_folds(fold_summary)

    fold_path = args.output_dir / "quantum_fold_summary.csv"
    structural_path = args.output_dir / "quantum_fold_structure.csv"
    aggregate_path = args.output_dir / "quantum_across_fold_summary.csv"

    fold_summary.to_csv(
        fold_path,
        index=False,
    )
    structural.to_csv(
        structural_path,
        index=False,
    )
    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    print("=== Fold-level relative improvement ===")
    print_fold_table(fold_summary)

    save_relative_facets(
        fold_summary,
        args.output_dir,
    )
    save_absolute_facets(
        fold_summary,
        args.output_dir,
    )
    save_fold_overlay(
        fold_summary,
        args.output_dir,
    )
    save_structure_figure(
        structural,
        args.output_dir,
    )

    print()
    print("Saved:")
    print(f"- Fold summary:      {fold_path}")
    print(f"- Structural table:  {structural_path}")
    print(f"- Aggregate summary: {aggregate_path}")
    print(f"- Figures:           {args.output_dir}")


if __name__ == "__main__":
    main()
