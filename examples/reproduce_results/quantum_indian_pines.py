"""Reproduce the Indian Pines quantum-kernel system-size experiment.

The experiment uses precomputed bandwidth-controlled quantum kernels from the
Large_scale_QSVM repository. Training labels are reconstructed from the
original fold ground-truth data by download_indian_pines_kernels.py.

For every fold and qubit count, the script compares:

1. SupportShot adaptive kernel estimation, and
2. uniform kernel estimation at the same total measurement budget.

Historical ShotWise settings are reproduced using the current implementation:

    C = 1
    final budget = 40 measurements per independent off-diagonal entry
    pilot = 5 measurements per entry
    adaptive rounds = 6
    lambda = 0.5

The script saves raw paired results, condition summaries, and a diagnostic
system-size figure.

Example
-------
From the repository root:

    python examples/reproduce_results/quantum_indian_pines.py \
        --experiment-rounds 100 \
        --output-dir outputs/quantum_indian_pines
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluation_utils import evaluate_estimated_kernel

from shotwise.algorithms.supportshot import SupportShot
from shotwise.backends.synthetic import SyntheticBernoulliBackend
from shotwise.storage.kernel_store import KernelStore
from shotwise.utils.matrix import upper_triangle_indices


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the Indian Pines quantum-kernel system-size experiment."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/quantum_kernels/indian_pines"),
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
        "--shots-per-entry",
        type=int,
        default=40,
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
        default=Path("outputs/quantum_indian_pines"),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Validate experiment arguments."""
    if args.experiment_rounds <= 0:
        raise ValueError("experiment_rounds must be positive.")

    if args.c <= 0.0:
        raise ValueError("C must be positive.")

    if args.shots_per_entry <= 0:
        raise ValueError("shots_per_entry must be positive.")

    if args.pilot_shots <= 0:
        raise ValueError("pilot_shots must be positive.")

    if args.pilot_shots > args.shots_per_entry:
        raise ValueError("pilot_shots cannot exceed shots_per_entry.")

    if args.rounds <= 0:
        raise ValueError("rounds must be positive.")

    if not 0.0 <= args.lambda_weight <= 1.0:
        raise ValueError("lambda_weight must lie in [0, 1].")


def parse_kernel_metadata(
    path: Path,
) -> dict[str, int]:
    """Parse fold, qubit count, and class IDs from a filename."""
    fold_match = re.search(
        r"fold_(\d+)",
        path.name,
    )
    qubit_match = re.search(
        r"_n_(\d+)_",
        path.name,
    )
    class_match = re.search(
        r"_classes_(\d+)_(\d+)\.npy$",
        path.name,
    )

    if fold_match is None or qubit_match is None or class_match is None:
        raise ValueError(f"Could not parse metadata from {path.name!r}.")

    return {
        "fold": int(fold_match.group(1)),
        "n_qubits": int(qubit_match.group(1)),
        "class_1": int(class_match.group(1)),
        "class_2": int(class_match.group(2)),
    }


def load_source_revision(
    data_dir: Path,
) -> str:
    """Load the external source-repository revision."""
    path = data_dir / "SOURCE_REVISION.txt"

    if not path.exists():
        return "unknown"

    for line in path.read_text().splitlines():
        if line.startswith("revision="):
            return line.split(
                "=",
                maxsplit=1,
            )[1].strip()

    return "unknown"


def load_labels(
    data_dir: Path,
) -> dict[int, np.ndarray]:
    """Load labels and convert historical {0, 1} encoding to {-1, +1}."""
    path = data_dir / "Indian_train_labels.npz"

    if not path.exists():
        raise FileNotFoundError(
            "Reconstructed labels were not found. Run "
            "download_indian_pines_kernels.py first."
        )

    labels: dict[int, np.ndarray] = {}

    with np.load(path) as archive:
        for fold in range(4):
            key = f"fold_{fold}"

            if key not in archive:
                raise KeyError(f"Missing {key!r} in {path}.")

            y_historical = np.asarray(
                archive[key],
                dtype=int,
            ).reshape(-1)

            historical_values = set(np.unique(y_historical).tolist())

            if historical_values != {0, 1}:
                raise ValueError(
                    f"Fold {fold} historical labels must contain "
                    f"exactly 0 and 1, found "
                    f"{sorted(historical_values)}."
                )

            # Historical Indian Pines preprocessing used:
            # class 3 -> 0 and class 10 -> 1.
            #
            # ShotWise requires:
            # class 3 -> -1 and class 10 -> +1.
            y = 2 * y_historical - 1

            shotwise_values = set(np.unique(y).tolist())

            if shotwise_values != {-1, 1}:
                raise RuntimeError(
                    f"Fold {fold} label conversion failed: {sorted(shotwise_values)}."
                )

            labels[fold] = y

    return labels


def validate_and_prepare_kernel(
    path: Path,
) -> tuple[np.ndarray, dict[str, float]]:
    """Load, validate, symmetrize, and clip one kernel matrix."""
    kernel_raw = np.asarray(
        np.load(path),
        dtype=float,
    )

    if kernel_raw.ndim != 2 or kernel_raw.shape[0] != kernel_raw.shape[1]:
        raise ValueError(
            f"{path.name}: expected square matrix, got {kernel_raw.shape}."
        )

    if not np.all(np.isfinite(kernel_raw)):
        raise ValueError(f"{path.name}: non-finite entries found.")

    asymmetry = float(np.max(np.abs(kernel_raw - kernel_raw.T)))

    kernel = 0.5 * (kernel_raw + kernel_raw.T)

    minimum = float(kernel.min())
    maximum = float(kernel.max())

    tolerance = 1e-8

    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"{path.name}: entries outside [0, 1]: [{minimum:.6g}, {maximum:.6g}]."
        )

    # Remove insignificant floating-point probability excursions.
    kernel = np.clip(
        kernel,
        0.0,
        1.0,
    )

    diagonal_error = float(np.max(np.abs(np.diag(kernel) - 1.0)))

    # Fidelity-kernel diagonals are known exactly.
    np.fill_diagonal(
        kernel,
        1.0,
    )

    rows, cols = upper_triangle_indices(kernel.shape[0])
    off_diagonal = kernel[
        rows,
        cols,
    ]

    diagnostics = {
        "kernel_min": minimum,
        "kernel_max": maximum,
        "kernel_asymmetry": asymmetry,
        "kernel_diagonal_error": diagonal_error,
        "offdiag_mean": float(off_diagonal.mean()),
        "offdiag_std": float(off_diagonal.std()),
    }

    return kernel, diagnostics


def total_budget_from_shots_per_entry(
    n_samples: int,
    shots_per_entry: int,
) -> int:
    """Convert average shots per independent entry into total budget."""
    n_pairs = n_samples * (n_samples - 1) // 2

    return n_pairs * shots_per_entry


def uniform_allocation_matrix(
    n_samples: int,
    total_budget: int,
) -> np.ndarray:
    """Construct an exact-budget uniform upper-triangular allocation."""
    rows, cols = upper_triangle_indices(n_samples)
    n_pairs = rows.size

    base = total_budget // n_pairs
    remainder = total_budget % n_pairs

    flat = np.full(
        n_pairs,
        base,
        dtype=int,
    )

    if remainder > 0:
        flat[:remainder] += 1

    allocation = np.zeros(
        (n_samples, n_samples),
        dtype=int,
    )
    allocation[
        rows,
        cols,
    ] = flat

    return allocation


def estimate_uniform_kernel(
    kernel_true: np.ndarray,
    *,
    total_budget: int,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """Estimate one kernel using uniform finite measurements."""
    backend = SyntheticBernoulliBackend(
        kernel_true,
        random_state=random_state,
    )

    store = KernelStore(
        kernel_true.shape[0],
        diagonal_value=1.0,
    )

    allocation = uniform_allocation_matrix(
        kernel_true.shape[0],
        total_budget,
    )

    counts = backend.sample_upper_triangle_counts(allocation)
    store.update(
        counts,
        allocation,
    )

    return (
        store.kernel_matrix(),
        store.total_shots_spent(),
    )


def estimate_adaptive_kernel(
    kernel_true: np.ndarray,
    y: np.ndarray,
    *,
    total_budget: int,
    pilot_shots: int,
    rounds: int,
    lambda_weight: float,
    c_value: float,
    allocation_method: str,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """Estimate one kernel using SupportShot."""
    backend = SyntheticBernoulliBackend(
        kernel_true,
        random_state=random_state,
    )

    model = SupportShot(
        total_budget=total_budget,
        pilot_shots_per_entry=pilot_shots,
        rounds=rounds,
        lambda_weight=lambda_weight,
        svm_c=c_value,
        stopping_epsilon=None,
        allocation_method=allocation_method,
        random_state=random_state,
    )

    model.fit(
        y,
        backend,
    )

    if not model.history_:
        raise RuntimeError("SupportShot produced no history records.")

    final_record = model.history_[-1]

    return (
        final_record.kernel_matrix,
        final_record.total_shots_spent,
    )


def evaluate_pair(
    kernel_true: np.ndarray,
    kernel_adaptive: np.ndarray,
    kernel_uniform: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
) -> dict[str, float]:
    """Evaluate adaptive and uniform estimates against the exact kernel."""
    adaptive = evaluate_estimated_kernel(
        kernel_true,
        kernel_adaptive,
        y,
        c=c_value,
        method="adaptive",
        round_id=-1,
    )
    uniform = evaluate_estimated_kernel(
        kernel_true,
        kernel_uniform,
        y,
        c=c_value,
        method="uniform",
        round_id=-1,
    )

    result: dict[str, float] = {}

    metrics = [
        "rmse_K",
        "rmse_K_SV",
        "jaccard",
        "jaccard_weighted",
        "margin_err",
        "margin_err_normalized",
        "func_rmse",
        "func_rmse_normalized",
        "acc_train_vs_true",
        "sv_ratio_est",
    ]

    for metric in metrics:
        if metric in adaptive:
            result[f"adaptive_{metric}"] = float(adaptive[metric])

        if metric in uniform:
            result[f"uniform_{metric}"] = float(uniform[metric])

    uniform_error = result["uniform_func_rmse_normalized"]
    adaptive_error = result["adaptive_func_rmse_normalized"]

    denominator = max(
        uniform_error,
        np.finfo(float).tiny,
    )

    result["relative_func_rmse_improvement"] = (
        uniform_error - adaptive_error
    ) / denominator

    return result


def summarize_results(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize results by qubit count."""
    metrics = [
        "relative_func_rmse_improvement",
        "adaptive_func_rmse_normalized",
        "uniform_func_rmse_normalized",
        "adaptive_jaccard_weighted",
        "uniform_jaccard_weighted",
        "offdiag_mean",
        "offdiag_std",
    ]

    grouped = df.groupby(
        "n_qubits",
        sort=True,
    )

    frames: list[pd.DataFrame] = []

    for metric in metrics:
        summary = grouped[metric].agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            mean="mean",
            std="std",
            count="count",
        )

        summary["metric"] = metric

        frames.append(summary.reset_index())

    return pd.concat(
        frames,
        ignore_index=True,
    )


def metric_summary(
    summary: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Select one metric from the long-form summary."""
    return summary.loc[summary["metric"] == metric].sort_values("n_qubits")


def save_figure(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save a diagnostic system-size figure."""
    blue = "#1f4e79"
    gray = "#666666"

    improvement = metric_summary(
        summary,
        "relative_func_rmse_improvement",
    )
    adaptive = metric_summary(
        summary,
        "adaptive_func_rmse_normalized",
    )
    uniform = metric_summary(
        summary,
        "uniform_func_rmse_normalized",
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.75),
    )

    # ----------------------------------------------------------
    # Relative adaptive improvement
    # ----------------------------------------------------------
    x = improvement["n_qubits"].to_numpy(dtype=float)

    axes[0].plot(
        x,
        improvement["median"].to_numpy(dtype=float),
        color=blue,
        linewidth=1.1,
        marker="o",
        markersize=3.5,
    )
    axes[0].fill_between(
        x,
        improvement["q25"].to_numpy(dtype=float),
        improvement["q75"].to_numpy(dtype=float),
        color=blue,
        alpha=0.10,
        linewidth=0.0,
    )
    axes[0].axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(
        "Number of qubits",
        fontsize=8,
    )
    axes[0].set_ylabel(
        r"Relative improvement in $\mathrm{RMSE}_f$",
        fontsize=8,
    )

    # ----------------------------------------------------------
    # Absolute classifier reconstruction
    # ----------------------------------------------------------
    x_adaptive = adaptive["n_qubits"].to_numpy(dtype=float)

    axes[1].plot(
        x_adaptive,
        adaptive["median"].to_numpy(dtype=float),
        color=blue,
        linestyle="-",
        linewidth=1.1,
        marker="o",
        markersize=3.5,
        label="Adaptive",
    )
    axes[1].fill_between(
        x_adaptive,
        adaptive["q25"].to_numpy(dtype=float),
        adaptive["q75"].to_numpy(dtype=float),
        color=blue,
        alpha=0.10,
        linewidth=0.0,
    )

    x_uniform = uniform["n_qubits"].to_numpy(dtype=float)

    axes[1].plot(
        x_uniform,
        uniform["median"].to_numpy(dtype=float),
        color=gray,
        linestyle="--",
        linewidth=1.0,
        marker="s",
        markersize=3.3,
        markerfacecolor="white",
        markeredgecolor=gray,
        markeredgewidth=0.7,
        label="Uniform",
    )
    axes[1].fill_between(
        x_uniform,
        uniform["q25"].to_numpy(dtype=float),
        uniform["q75"].to_numpy(dtype=float),
        color=gray,
        alpha=0.055,
        linewidth=0.0,
    )

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(
        "Number of qubits",
        fontsize=8,
    )
    axes[1].set_ylabel(
        r"$\mathrm{RMSE}_f$",
        fontsize=8,
    )
    axes[1].legend(
        fontsize=6.8,
        frameon=False,
        loc="best",
    )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{int(value)}" for value in x],
            rotation=35,
            ha="right",
        )

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

    fig.tight_layout(
        pad=0.45,
        w_pad=1.0,
    )

    fig.savefig(
        output_dir / "quantum_indian_pines_system_size.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "quantum_indian_pines_system_size.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Run the Indian Pines quantum-kernel system-size experiment."""
    args = parse_args()
    validate_args(args)

    kernel_dir = args.data_dir / "bandwidth"

    if not kernel_dir.exists():
        raise FileNotFoundError(
            f"Kernel directory not found: {kernel_dir}. "
            "Run download_indian_pines_kernels.py first."
        )

    kernel_files = sorted(kernel_dir.glob("quantum_train_*.npy"))

    if not kernel_files:
        raise FileNotFoundError(f"No quantum training kernels found in {kernel_dir}.")

    labels = load_labels(args.data_dir)
    source_revision = load_source_revision(args.data_dir)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(args.measurement_seed)

    records: list[dict[str, float | int | str]] = []

    total_conditions = len(kernel_files) * args.experiment_rounds
    condition_index = 0

    print("=== Indian Pines quantum-kernel experiment ===")
    print(f"source_revision  = {source_revision}")
    print(f"kernel_files     = {len(kernel_files)}")
    print(f"experiment_rounds = {args.experiment_rounds}")
    print(f"C                = {args.c}")
    print(f"shots_per_entry  = {args.shots_per_entry}")
    print(f"pilot_shots      = {args.pilot_shots}")
    print(f"rounds           = {args.rounds}")
    print(f"lambda_weight    = {args.lambda_weight}")
    print(f"allocation_method= {args.allocation_method}")
    print(f"measurement_seed = {args.measurement_seed}")
    print(f"output_dir       = {args.output_dir}")
    print()

    for kernel_path in kernel_files:
        metadata = parse_kernel_metadata(kernel_path)

        kernel_true, kernel_diagnostics = validate_and_prepare_kernel(kernel_path)

        fold = metadata["fold"]
        y = labels[fold]

        if kernel_true.shape != (
            y.size,
            y.size,
        ):
            raise ValueError(
                f"{kernel_path.name}: kernel shape "
                f"{kernel_true.shape} does not match "
                f"fold-{fold} labels of length {y.size}."
            )

        total_budget = total_budget_from_shots_per_entry(
            y.size,
            args.shots_per_entry,
        )

        print(
            f"Kernel: fold={fold}, "
            f"qubits={metadata['n_qubits']}, "
            f"n={y.size}, "
            f"budget={total_budget}"
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
                lambda_weight=args.lambda_weight,
                c_value=args.c,
                allocation_method=args.allocation_method,
                random_state=adaptive_seed,
            )

            (
                kernel_uniform,
                uniform_budget,
            ) = estimate_uniform_kernel(
                kernel_true,
                total_budget=adaptive_budget,
                random_state=uniform_seed,
            )

            if adaptive_budget != uniform_budget:
                raise RuntimeError(
                    "Adaptive and uniform budgets differ: "
                    f"{adaptive_budget} != {uniform_budget}."
                )

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
                "kernel_filename": kernel_path.name,
                "source_revision": source_revision,
                "fold": fold,
                "n_qubits": metadata["n_qubits"],
                "class_1": metadata["class_1"],
                "class_2": metadata["class_2"],
                "n_samples": y.size,
                "class_0_count": int(np.count_nonzero(y == 0)),
                "class_1_count": int(np.count_nonzero(y == 1)),
                "c_value": args.c,
                "shots_per_entry": args.shots_per_entry,
                "pilot_shots": args.pilot_shots,
                "adaptive_rounds": args.rounds,
                "lambda_weight": args.lambda_weight,
                "allocation_method": args.allocation_method,
                "total_budget": total_budget,
                "total_shots_spent": adaptive_budget,
                "adaptive_seed": adaptive_seed,
                "uniform_seed": uniform_seed,
            }

            record.update(kernel_diagnostics)
            record.update(performance)

            records.append(record)

    df = pd.DataFrame.from_records(records)

    raw_path = args.output_dir / "quantum_indian_pines_results.csv"
    df.to_csv(
        raw_path,
        index=False,
    )

    summary = summarize_results(df)

    summary_path = args.output_dir / "quantum_indian_pines_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
    )

    save_figure(
        summary,
        args.output_dir,
    )

    print()
    print("=== Median relative improvement by qubit count ===")

    display = (
        df.groupby(
            "n_qubits",
            sort=True,
        )[
            [
                "relative_func_rmse_improvement",
                "adaptive_func_rmse_normalized",
                "uniform_func_rmse_normalized",
                "offdiag_std",
            ]
        ]
        .median()
        .reset_index()
    )

    print(display.to_string(index=False))

    print()
    print("Saved:")
    print(f"- Raw results: {raw_path}")
    print(f"- Summary:     {summary_path}")
    print(f"- PNG figure:  {args.output_dir / 'quantum_indian_pines_system_size.png'}")
    print(f"- PDF figure:  {args.output_dir / 'quantum_indian_pines_system_size.pdf'}")


if __name__ == "__main__":
    main()
