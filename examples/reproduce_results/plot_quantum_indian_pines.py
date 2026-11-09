"""Create manuscript plots for the Indian Pines quantum-kernel experiment.

The system-size panel respects the hierarchical experimental design:

    qubit count
        -> four Indian Pines folds
            -> repeated finite-shot measurement realizations.

Measurement realizations are first averaged within each fold and qubit count.
The four resulting fold trajectories are shown explicitly, together with
their arithmetic mean.

Positive relative improvement means adaptive allocation produces smaller
normalized decision-function RMSE than uniform allocation.

Example
-------
From the repository root:

    python examples/reproduce_results/plot_quantum_indian_pines.py \
        --results \
        outputs/quantum_indian_pines_no_bandwidth/quantum_indian_pines_results.csv \
        --output-dir \
        outputs/quantum_indian_pines_no_bandwidth/manuscript
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create manuscript figures for the Indian Pines quantum-kernel experiment."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Raw CSV produced by quantum_indian_pines.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def validate_results(df: pd.DataFrame) -> None:
    """Validate columns required by the system-size panel."""
    required = {
        "run",
        "fold",
        "n_qubits",
        "relative_func_rmse_improvement",
        "adaptive_func_rmse_normalized",
        "uniform_func_rmse_normalized",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Results CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("Results CSV is empty.")

    values = df["relative_func_rmse_improvement"].to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("Relative-RMSE improvement contains non-finite values.")


def fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average measurement realizations within each fold/qubit condition."""
    return (
        df.groupby(
            ["fold", "n_qubits"],
            sort=True,
        )
        .agg(
            relative_improvement=(
                "relative_func_rmse_improvement",
                "mean",
            ),
            adaptive_rmse=(
                "adaptive_func_rmse_normalized",
                "mean",
            ),
            uniform_rmse=(
                "uniform_func_rmse_normalized",
                "mean",
            ),
            measurement_repetitions=(
                "run",
                "count",
            ),
        )
        .reset_index()
    )


def fold_colors() -> dict[int, str]:
    """Return restrained blue tones for the four folds."""
    return {
        0: "#4e79a7",
        1: "#5f95c8",
        2: "#86b9da",
        3: "#b2d4e7",
    }


def style_axis(ax) -> None:
    """Apply manuscript-compatible axis styling."""
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=8,
        width=0.65,
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
        spine.set_linewidth(0.7)

    ax.grid(False)
    ax.margins(x=0.03)


def plot_system_size(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot fold-aware relative improvement versus quantum-system size."""
    folds = sorted(int(value) for value in summary["fold"].unique())

    if folds != [0, 1, 2, 3]:
        raise ValueError(f"Expected folds [0, 1, 2, 3], found {folds}.")

    colors = fold_colors()

    fig, ax = plt.subplots(figsize=(3.65, 2.75))

    for fold in folds:
        selected = summary.loc[summary["fold"] == fold].sort_values("n_qubits")

        ax.plot(
            selected["n_qubits"],
            selected["relative_improvement"],
            color=colors[fold],
            linewidth=0.95,
            marker="o",
            markersize=3.3,
            markerfacecolor="none",
            markeredgewidth=0.8,
            label=f"Fold {fold}",
            alpha=0.95,
        )

    mean_curve = (
        summary.groupby(
            "n_qubits",
            sort=True,
        )["relative_improvement"]
        .mean()
        .reset_index()
    )

    ax.plot(
        mean_curve["n_qubits"],
        mean_curve["relative_improvement"],
        color="black",
        linewidth=1.55,
        marker="D",
        markersize=3.8,
        label="Fold mean",
        zorder=10,
    )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        zorder=0,
    )

    ax.set_xscale("log")

    qubits = mean_curve["n_qubits"].to_numpy(dtype=int)

    ax.set_xticks(qubits)
    ax.set_xticklabels(
        [str(value) for value in qubits],
        rotation=35,
        ha="right",
    )

    ax.set_xlabel(
        "Number of qubits",
        fontsize=8.5,
    )
    ax.set_ylabel(
        r"Relative improvement in $\mathrm{RMSE}_f$",
        fontsize=8.5,
    )

    ax.legend(
        frameon=False,
        fontsize=6.7,
        ncol=2,
        loc="lower left",
    )

    style_axis(ax)

    fig.tight_layout(pad=0.5)

    fig.savefig(
        output_dir / "quantum_system_size.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "quantum_system_size.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Create manuscript plots."""
    args = parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results file not found: {args.results}")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.results)
    validate_results(df)

    summary = fold_summary(df)

    summary_path = args.output_dir / "quantum_system_size_fold_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
    )

    plot_system_size(
        summary,
        args.output_dir,
    )

    print("Saved:")
    print(f"- Summary: {summary_path}")
    print(f"- Figure:  {args.output_dir / 'quantum_system_size.pdf'}")


if __name__ == "__main__":
    main()
