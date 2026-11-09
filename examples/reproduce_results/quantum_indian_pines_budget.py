"""Run the Indian Pines quantum-kernel measurement-budget sweep.

The experiment fixes the quantum-system size and varies the average number of
measurements per independent off-diagonal kernel entry. The four Indian Pines
folds are treated as distinct learning problems. Repeated finite-shot
realizations quantify measurement uncertainty conditional on each fold.

The script saves:

- raw paired adaptive/uniform results,
- fold-level summaries,
- an across-fold summary,
- a manuscript-style aggregate figure,
- a four-panel fold-diagnostic figure.

Example
-------
From the ShotWise repository root:

    python examples/reproduce_results/quantum_indian_pines_budget.py \
        --data-dir data/quantum_kernels/indian_pines_no_bandwidth \
        --qubits 20 \
        --budgets-per-entry 5 10 20 40 80 160 \
        --experiment-rounds 100 \
        --c 1 \
        --pilot-shots 5 \
        --rounds 6 \
        --lambda-weight 0.5 \
        --allocation-method multinomial \
        --measurement-seed 42 \
        --output-dir outputs/quantum_indian_pines_budget
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from quantum_indian_pines import (  # noqa: E402
    estimate_adaptive_kernel,
    estimate_uniform_kernel,
    evaluate_pair,
    load_labels,
    load_source_revision,
    parse_kernel_metadata,
    total_budget_from_shots_per_entry,
    validate_and_prepare_kernel,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Run the Indian Pines quantum-kernel measurement-budget sweep.")
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/quantum_kernels/indian_pines_no_bandwidth"),
    )
    parser.add_argument(
        "--kernel-subdir",
        default="bandwidth",
        help=(
            "Local kernel directory inside data-dir. "
            "The current no-bandwidth download also uses "
            "the local directory name 'bandwidth'."
        ),
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--budgets-per-entry",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40, 80, 160],
    )
    parser.add_argument(
        "--experiment-rounds",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--pilot-shots",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--lambda-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--allocation-method",
        choices=[
            "multinomial",
            "deterministic",
        ],
        default="multinomial",
    )
    parser.add_argument(
        "--measurement-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/quantum_indian_pines_budget"),
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Validate the campaign configuration."""
    if args.qubits <= 0:
        raise ValueError("qubits must be positive.")

    if args.experiment_rounds <= 0:
        raise ValueError("experiment_rounds must be positive.")

    if args.c <= 0.0:
        raise ValueError("C must be positive.")

    if args.pilot_shots <= 0:
        raise ValueError("pilot_shots must be positive.")

    if args.rounds <= 0:
        raise ValueError("rounds must be positive.")

    if not 0.0 <= args.lambda_weight <= 1.0:
        raise ValueError("lambda_weight must lie in [0, 1].")

    budgets = sorted(set(args.budgets_per_entry))

    if not budgets:
        raise ValueError("At least one budget is required.")

    if any(budget < args.pilot_shots for budget in budgets):
        raise ValueError("Every final budget must be at least as large as pilot_shots.")

    args.budgets_per_entry = budgets


def find_fold_kernels(
    kernel_dir: Path,
    *,
    qubits: int,
) -> dict[int, Path]:
    """Find exactly one kernel per fold at a fixed qubit count."""
    result: dict[int, Path] = {}

    for path in sorted(kernel_dir.glob("quantum_train_*.npy")):
        metadata = parse_kernel_metadata(path)

        if metadata["n_qubits"] != qubits:
            continue

        fold = metadata["fold"]

        if fold in result:
            raise ValueError(f"Multiple q={qubits} kernels found for fold {fold}.")

        result[fold] = path

    expected = {0, 1, 2, 3}

    if set(result) != expected:
        raise FileNotFoundError(
            f"Expected q={qubits} kernels for folds "
            f"{sorted(expected)}, found {sorted(result)}."
        )

    return result


def run_campaign(
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Run the fixed-qubit budget sweep."""
    kernel_dir = args.data_dir / args.kernel_subdir

    if not kernel_dir.exists():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

    labels = load_labels(args.data_dir)
    source_revision = load_source_revision(args.data_dir)

    kernel_paths = find_fold_kernels(
        kernel_dir,
        qubits=args.qubits,
    )

    rng = np.random.default_rng(args.measurement_seed)

    total_conditions = (
        len(kernel_paths) * len(args.budgets_per_entry) * args.experiment_rounds
    )
    condition_index = 0

    records: list[dict[str, float | int | str]] = []

    print("=== Indian Pines quantum-kernel budget sweep ===")
    print(f"source_revision   = {source_revision}")
    print(f"qubits            = {args.qubits}")
    print(f"budgets_per_entry = {args.budgets_per_entry}")
    print(f"experiment_rounds = {args.experiment_rounds}")
    print(f"C                 = {args.c}")
    print(f"pilot_shots       = {args.pilot_shots}")
    print(f"rounds            = {args.rounds}")
    print(f"lambda_weight     = {args.lambda_weight}")
    print(f"allocation_method = {args.allocation_method}")
    print(f"measurement_seed  = {args.measurement_seed}")
    print()

    for fold in sorted(kernel_paths):
        path = kernel_paths[fold]

        kernel_true, diagnostics = validate_and_prepare_kernel(path)
        y = labels[fold]

        if kernel_true.shape != (
            y.size,
            y.size,
        ):
            raise ValueError(
                f"{path.name}: kernel shape "
                f"{kernel_true.shape} does not match "
                f"label length {y.size}."
            )

        metadata = parse_kernel_metadata(path)

        for budget_per_entry in args.budgets_per_entry:
            total_budget = total_budget_from_shots_per_entry(
                y.size,
                budget_per_entry,
            )

            print(
                f"Fold={fold}, "
                f"q={args.qubits}, "
                f"budget/entry={budget_per_entry}, "
                f"total={total_budget}"
            )

            for run in range(args.experiment_rounds):
                condition_index += 1

                if (
                    condition_index == 1
                    or condition_index % 100 == 0
                    or condition_index == total_conditions
                ):
                    print(f"Run {condition_index}/{total_conditions}")

                adaptive_seed = int(
                    rng.integers(
                        0,
                        2**32 - 1,
                    )
                )
                uniform_seed = int(
                    rng.integers(
                        0,
                        2**32 - 1,
                    )
                )

                (
                    kernel_adaptive,
                    adaptive_budget,
                ) = estimate_adaptive_kernel(
                    kernel_true,
                    y,
                    total_budget=total_budget,
                    pilot_shots=args.pilot_shots,
                    rounds=args.rounds,
                    lambda_weight=(args.lambda_weight),
                    c_value=args.c,
                    allocation_method=(args.allocation_method),
                    random_state=adaptive_seed,
                )

                (
                    kernel_uniform,
                    uniform_budget,
                ) = estimate_uniform_kernel(
                    kernel_true,
                    total_budget=(adaptive_budget),
                    random_state=uniform_seed,
                )

                if adaptive_budget != uniform_budget:
                    raise RuntimeError("Adaptive and uniform budgets do not match.")

                performance = evaluate_pair(
                    kernel_true,
                    kernel_adaptive,
                    kernel_uniform,
                    y,
                    c_value=args.c,
                )

                record: dict[
                    str,
                    float | int | str,
                ] = {
                    "run": run,
                    "fold": fold,
                    "n_qubits": (metadata["n_qubits"]),
                    "kernel_filename": path.name,
                    "source_revision": (source_revision),
                    "n_samples": y.size,
                    "class_03_count": int(np.count_nonzero(y == -1)),
                    "class_10_count": int(np.count_nonzero(y == 1)),
                    "c_value": args.c,
                    "budget_per_entry": (budget_per_entry),
                    "total_budget": (total_budget),
                    "total_shots_spent": (adaptive_budget),
                    "pilot_shots": (args.pilot_shots),
                    "adaptive_rounds": (args.rounds),
                    "lambda_weight": (args.lambda_weight),
                    "allocation_method": (args.allocation_method),
                    "adaptive_seed": (adaptive_seed),
                    "uniform_seed": (uniform_seed),
                }

                record.update(diagnostics)
                record.update(performance)
                records.append(record)

    return pd.DataFrame.from_records(records)


def summarize_by_fold(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Average measurement realizations within each fold and budget."""
    return (
        df.groupby(
            [
                "fold",
                "budget_per_entry",
            ],
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
            relative_improvement=(
                "relative_func_rmse_improvement",
                "mean",
            ),
            adaptive_rmse_std=(
                "adaptive_func_rmse_normalized",
                "std",
            ),
            uniform_rmse_std=(
                "uniform_func_rmse_normalized",
                "std",
            ),
            relative_improvement_std=(
                "relative_func_rmse_improvement",
                "std",
            ),
            measurement_repetitions=(
                "run",
                "count",
            ),
        )
        .reset_index()
    )


def summarize_across_folds(
    fold_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the four fold-level centers."""
    grouped = fold_summary.groupby(
        "budget_per_entry",
        sort=True,
    )

    return grouped.agg(
        adaptive_rmse=(
            "adaptive_rmse",
            "mean",
        ),
        adaptive_rmse_fold_std=(
            "adaptive_rmse",
            "std",
        ),
        uniform_rmse=(
            "uniform_rmse",
            "mean",
        ),
        uniform_rmse_fold_std=(
            "uniform_rmse",
            "std",
        ),
        relative_improvement=(
            "relative_improvement",
            "mean",
        ),
        relative_improvement_fold_std=(
            "relative_improvement",
            "std",
        ),
        folds=(
            "fold",
            "count",
        ),
    ).reset_index()


def fold_colors() -> dict[int, str]:
    """Return consistent colors for four folds."""
    return {
        0: "#4e79a7",
        1: "#5f95c8",
        2: "#86b9da",
        3: "#b2d4e7",
    }


def style_axis(ax) -> None:
    """Apply restrained manuscript styling."""
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


def save_main_figure(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save the aggregate historical-style budget panel."""
    blue = "#1f4e79"
    gray = "#666666"

    x = summary["budget_per_entry"].to_numpy(dtype=float)

    adaptive = summary["adaptive_rmse"].to_numpy(dtype=float)
    adaptive_sd = summary["adaptive_rmse_fold_std"].to_numpy(dtype=float)

    uniform = summary["uniform_rmse"].to_numpy(dtype=float)
    uniform_sd = summary["uniform_rmse_fold_std"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(3.65, 2.75))

    ax.plot(
        x,
        adaptive,
        color=blue,
        linewidth=1.2,
        marker="o",
        markersize=3.6,
        label="Adaptive",
    )
    ax.fill_between(
        x,
        np.maximum(
            adaptive - adaptive_sd,
            np.finfo(float).tiny,
        ),
        adaptive + adaptive_sd,
        color=blue,
        alpha=0.10,
        linewidth=0.0,
    )

    ax.plot(
        x,
        uniform,
        color=gray,
        linestyle="--",
        linewidth=1.0,
        marker="s",
        markersize=3.4,
        markerfacecolor="white",
        markeredgecolor=gray,
        markeredgewidth=0.75,
        label="Uniform",
    )
    ax.fill_between(
        x,
        np.maximum(
            uniform - uniform_sd,
            np.finfo(float).tiny,
        ),
        uniform + uniform_sd,
        color=gray,
        alpha=0.055,
        linewidth=0.0,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(value)}" for value in x])

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
        fontsize=7,
        loc="best",
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


def save_fold_figure(
    fold_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save four fold-specific adaptive/uniform budget panels."""
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
            "budget_per_entry"
        )

        x = selected["budget_per_entry"].to_numpy(dtype=float)

        ax.plot(
            x,
            selected["adaptive_rmse"].to_numpy(dtype=float),
            color=colors[fold],
            linewidth=1.15,
            marker="o",
            markersize=3.6,
            label="Adaptive",
        )

        ax.plot(
            x,
            selected["uniform_rmse"].to_numpy(dtype=float),
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            marker="s",
            markersize=3.3,
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

        style_axis(ax)

    axes[0, 1].legend(
        frameon=False,
        fontsize=7,
        loc="best",
    )

    axes[1, 0].set_xlabel(
        "Measurements per kernel entry",
        fontsize=8,
    )
    axes[1, 1].set_xlabel(
        "Measurements per kernel entry",
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

    fig.tight_layout(
        pad=0.55,
        w_pad=0.8,
        h_pad=0.8,
    )

    fig.savefig(
        output_dir / "quantum_budget_sweep_folds.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "quantum_budget_sweep_folds.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Run, summarize, and plot the quantum budget sweep."""
    args = parse_args()
    validate_args(args)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = run_campaign(args)

    raw_path = args.output_dir / "quantum_budget_results.csv"
    raw.to_csv(
        raw_path,
        index=False,
    )

    fold_summary = summarize_by_fold(raw)
    fold_path = args.output_dir / "quantum_budget_fold_summary.csv"
    fold_summary.to_csv(
        fold_path,
        index=False,
    )

    summary = summarize_across_folds(fold_summary)
    summary_path = args.output_dir / "quantum_budget_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
    )

    save_main_figure(
        summary,
        args.output_dir,
    )
    save_fold_figure(
        fold_summary,
        args.output_dir,
    )

    print()
    print("=== Across-fold budget summary ===")
    print(summary.to_string(index=False))

    print()
    print("Saved:")
    print(f"- Raw results:  {raw_path}")
    print(f"- Fold summary: {fold_path}")
    print(f"- Summary:      {summary_path}")
    print(f"- Figure:       {args.output_dir / 'quantum_budget_sweep.pdf'}")
    print(f"- Fold figure:  {args.output_dir / 'quantum_budget_sweep_folds.pdf'}")


if __name__ == "__main__":
    main()
