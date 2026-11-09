"""Create manuscript plots for the Indian Pines quantum-kernel experiments.

Produces two independent manuscript panels:

1. quantum_system_size.pdf
   Fold-resolved relative improvement in normalized decision-function RMSE
   versus quantum-system size.

2. quantum_budget_sweep.pdf
   Fold-resolved adaptive and uniform normalized decision-function RMSE
   versus measurement budget at fixed quantum-system size.

Measurement realizations are first averaged within each fold/condition.
The fold curves therefore represent genuinely different learning problems,
rather than finite-shot repetitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOLD_COLORS = {
    0: "#4e79a7",
    1: "#669cc8",
    2: "#8abbd8",
    3: "#b2d4e7",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system-results",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--budget-results",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def style_axis(ax) -> None:
    """Apply manuscript-compatible styling."""
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
    ax.margins(x=0.025)


def summarize_system(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Average measurement repetitions inside fold/qubit conditions."""
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
            repetitions=(
                "run",
                "count",
            ),
        )
        .reset_index()
    )


def summarize_budget(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Average measurement repetitions inside fold/budget conditions."""
    return (
        df.groupby(
            ["fold", "budget_per_entry"],
            sort=True,
        )
        .agg(
            adaptive_rmse=(
                "adaptive_func_rmse_normalized",
                "mean",
            ),
            uniform_rmse=(
                "uniform_func_rmse_normalized",
                "mean",
            ),
            repetitions=(
                "run",
                "count",
            ),
        )
        .reset_index()
    )


def plot_system_size(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save the fold-resolved system-size panel."""
    fig, ax = plt.subplots(figsize=(3.6, 2.75))

    for fold in range(4):
        selected = summary.loc[summary["fold"] == fold].sort_values("n_qubits")

        ax.plot(
            selected["n_qubits"],
            selected["relative_improvement"],
            color=FOLD_COLORS[fold],
            linewidth=0.95,
            marker="o",
            markersize=3.3,
            markerfacecolor="none",
            markeredgewidth=0.8,
            label=f"Fold {fold}",
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
        linewidth=1.65,
        marker="D",
        markersize=3.9,
        label="Fold mean",
        zorder=10,
    )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.75,
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
        fontsize=6.3,
        ncol=2,
        loc="lower left",
        columnspacing=0.9,
        handlelength=2.0,
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


def plot_budget(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save fold-resolved budget panel using paired fold observations."""
    fig, ax = plt.subplots(figsize=(3.6, 2.75))

    budgets = np.asarray(
        sorted(summary["budget_per_entry"].unique()),
        dtype=float,
    )

    # Small logarithmic horizontal displacement keeps paired adaptive
    # and uniform observations distinguishable without changing the
    # interpretation of the budget axis.
    adaptive_offset = 0.965
    uniform_offset = 1.035

    for budget in budgets:
        selected = summary.loc[summary["budget_per_entry"] == budget].sort_values(
            "fold"
        )

        if len(selected) != 4:
            raise ValueError(
                f"Expected four folds at budget {budget:g}, found {len(selected)}."
            )

        adaptive_x = budget * adaptive_offset
        uniform_x = budget * uniform_offset

        for _, row in selected.iterrows():
            fold = int(row["fold"])
            color = FOLD_COLORS[fold]

            adaptive_y = float(row["adaptive_rmse"])
            uniform_y = float(row["uniform_rmse"])

            # Connect the two allocation strategies for the same fold
            # and budget. These are paired observations on the same
            # exact kernel.
            ax.plot(
                [
                    adaptive_x,
                    uniform_x,
                ],
                [
                    adaptive_y,
                    uniform_y,
                ],
                color="#bbbbbb",
                linewidth=0.55,
                alpha=0.75,
                zorder=1,
            )

            ax.plot(
                adaptive_x,
                adaptive_y,
                linestyle="none",
                marker="o",
                markersize=3.5,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.6,
                zorder=3,
            )

            ax.plot(
                uniform_x,
                uniform_y,
                linestyle="none",
                marker="s",
                markersize=3.3,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                zorder=3,
            )

    mean_curve = (
        summary.groupby(
            "budget_per_entry",
            sort=True,
        )
        .agg(
            adaptive_rmse=(
                "adaptive_rmse",
                "mean",
            ),
            uniform_rmse=(
                "uniform_rmse",
                "mean",
            ),
        )
        .reset_index()
    )

    x_mean = mean_curve["budget_per_entry"].to_numpy(dtype=float)

    # Only the fold means are connected across measurement budgets.
    ax.plot(
        x_mean * adaptive_offset,
        mean_curve["adaptive_rmse"],
        color="#1f4e79",
        linestyle="-",
        linewidth=1.45,
        marker="o",
        markersize=4.2,
        markerfacecolor="#1f4e79",
        markeredgecolor="#1f4e79",
        label="Adaptive mean",
        zorder=6,
    )

    ax.plot(
        x_mean * uniform_offset,
        mean_curve["uniform_rmse"],
        color="black",
        linestyle="--",
        linewidth=1.25,
        marker="s",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.85,
        label="Uniform mean",
        zorder=6,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks(budgets)
    ax.set_xticklabels([str(int(value)) for value in budgets])

    ax.set_xlabel(
        "Measurements per kernel entry",
        fontsize=8.5,
    )
    ax.set_ylabel(
        r"$\mathrm{RMSE}_f$",
        fontsize=8.5,
    )

    ax.legend(
        frameon=False,
        fontsize=6.7,
        loc="upper right",
        handlelength=2.0,
    )

    style_axis(ax)

    fig.tight_layout(pad=0.5)

    fig.savefig(
        output_dir / "quantum_budget_sweep.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "quantum_budget_sweep.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Generate independent manuscript panels."""
    args = parse_args()

    if not args.system_results.exists():
        raise FileNotFoundError(f"System results not found: {args.system_results}")

    if not args.budget_results.exists():
        raise FileNotFoundError(f"Budget results not found: {args.budget_results}")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    system_raw = pd.read_csv(args.system_results)
    budget_raw = pd.read_csv(args.budget_results)

    system_summary = summarize_system(system_raw)
    budget_summary = summarize_budget(budget_raw)

    system_summary.to_csv(
        args.output_dir / "quantum_system_size_fold_summary.csv",
        index=False,
    )

    budget_summary.to_csv(
        args.output_dir / "quantum_budget_fold_summary.csv",
        index=False,
    )

    plot_system_size(
        system_summary,
        args.output_dir,
    )

    plot_budget(
        budget_summary,
        args.output_dir,
    )

    print("Saved:")
    print(f"- System-size panel: {args.output_dir / 'quantum_system_size.pdf'}")
    print(f"- Budget panel:      {args.output_dir / 'quantum_budget_sweep.pdf'}")


if __name__ == "__main__":
    main()
