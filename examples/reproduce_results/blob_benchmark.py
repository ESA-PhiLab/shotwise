"""Run a paper-style synthetic benchmark for SupportShot.

This script mirrors the spirit of the old notebook-based benchmark while
using the refactored ShotWise package. It compares a simple uniform baseline
against SupportShot on synthetic blob data, prints the aggregated results,
and saves boxplots plus a margin scatterplot.

Example
-------
From the repository root, after installing the example dependencies:

    python examples/blob_benchmark.py \
        --experiment-rounds 100 \
        --n-samples 50 \
        --gamma 1.0 \
        --c 100 \
        --shots-per-entry 100 \
        --pilot-shots 10 \
        --rounds 5 \
        --cluster-std 0.2 \
        --lambda-weight 0.5 \
        --output-dir outputs/blob_benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from evaluation_utils import (
    boxplot_metric,
    evaluate_estimated_kernel,
    margin_scatterplot,
    print_summary,
    records_to_dataframe,
)
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel

from shotwise.algorithms.supportshot import SupportShot
from shotwise.backends.synthetic import SyntheticBernoulliBackend
from shotwise.storage.kernel_store import KernelStore
from shotwise.utils.matrix import upper_triangle_indices


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a blob-classification benchmark for SupportShot."
    )
    parser.add_argument("--experiment-rounds", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=100.0)
    parser.add_argument("--shots-per-entry", type=int, default=100)
    parser.add_argument("--pilot-shots", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cluster-std", type=float, default=0.2)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument(
        "--allocation-method",
        choices=["multinomial", "deterministic"],
        default="multinomial",
    )
    parser.add_argument("--stopping-epsilon", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/blob_benchmark"),
    )
    return parser.parse_args()


def generate_blobs_dataset(
    n_samples: int,
    *,
    cluster_std: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a binary blob dataset with labels in {-1, +1}."""
    x, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    y = np.where(y == 0, -1, 1)
    return x, y


def total_budget_from_shots_per_entry(n_samples: int, shots_per_entry: int) -> int:
    """Return the total budget implied by uniform shots per off-diagonal entry."""
    n_pairs = n_samples * (n_samples - 1) // 2
    return n_pairs * shots_per_entry


def uniform_allocation_matrix(n: int, total_budget: int) -> np.ndarray:
    """Create an upper-triangular uniform allocation matrix for a fixed budget."""
    if total_budget < 0:
        raise ValueError(f"total_budget must be non-negative, got {total_budget}.")

    rows, cols = upper_triangle_indices(n)
    n_pairs = rows.size
    if n_pairs == 0:
        return np.zeros((n, n), dtype=int)

    base = total_budget // n_pairs
    remainder = total_budget % n_pairs
    flat = np.full(n_pairs, base, dtype=int)
    if remainder > 0:
        flat[:remainder] += 1

    allocation = np.zeros((n, n), dtype=int)
    allocation[rows, cols] = flat
    return allocation


def run_uniform_baseline(
    kernel_true: np.ndarray,
    *,
    total_budget: int,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """Estimate the kernel using a single uniform allocation pass."""
    backend = SyntheticBernoulliBackend(kernel_true, random_state=random_state)
    store = KernelStore(kernel_true.shape[0], diagonal_value=1.0)
    allocation = uniform_allocation_matrix(kernel_true.shape[0], total_budget)
    counts = backend.sample_upper_triangle_counts(allocation)
    store.update(counts, allocation)
    return store.kernel_matrix(), store.total_shots_spent()


def method_label_from_round(stage: str, round_index: int) -> str:
    """Map SupportShot history stages to human-readable method labels."""
    if stage == "pilot":
        return "pilot"
    return f"round_{round_index}"


def add_supportshot_records(
    records: list[dict[str, float | int | str]],
    *,
    kernel_true: np.ndarray,
    y: np.ndarray,
    model: SupportShot,
    c_value: float,
    run_id: int,
) -> None:
    """Append one evaluation record for each SupportShot history snapshot."""
    for record in model.history_:
        entry = evaluate_estimated_kernel(
            kernel_true,
            record.kernel_matrix,
            y,
            c=c_value,
            method=method_label_from_round(record.stage, record.round_index),
            round_id=record.round_index,
        )
        entry.update(
            {
                "run": run_id,
                "stage": record.stage,
                "budget": record.budget,
                "total_shots_spent": record.total_shots_spent,
                "stopped_early": record.stopped_early,
            }
        )
        records.append(entry)


def add_uniform_record(
    records: list[dict[str, float | int | str]],
    *,
    kernel_true: np.ndarray,
    kernel_uniform: np.ndarray,
    y: np.ndarray,
    c_value: float,
    run_id: int,
    total_shots_spent: int,
) -> None:
    """Append one evaluation record for the uniform baseline."""
    entry = evaluate_estimated_kernel(
        kernel_true,
        kernel_uniform,
        y,
        c=c_value,
        method="uniform",
        round_id=-1,
    )
    entry.update(
        {
            "run": run_id,
            "stage": "uniform",
            "budget": total_shots_spent,
            "total_shots_spent": total_shots_spent,
            "stopped_early": False,
        }
    )
    records.append(entry)


def save_standard_plots(df, output_dir: Path) -> None:
    """Save the standard family of benchmark plots."""
    metric_specs = [
        ("rmse_K", r"$\mathrm{RMSE}(K)$", "log"),
        ("rmse_K_SV", r"$\mathrm{RMSE}(K_{SV})$", "log"),
        ("jaccard", r"$J_{SV}$", "linear"),
        ("jaccard_weighted", r"$J_{SV}^{w}$", "linear"),
        ("margin_err", r"$\delta_{\mathrm{margin}}$", "log"),
        ("func_rmse_normalized", r"$\mathrm{RMSE}_f$", "log"),
        ("acc_train_vs_true", r"$\mathrm{Acc}_{\mathrm{vs\ true}}$", "linear"),
    ]

    for metric, label, yscale in metric_specs:
        boxplot_metric(
            df,
            metric=metric,
            metric_label=label,
            yscale=yscale,
            output_path=output_dir / f"{metric}.png",
        )

    margin_scatterplot(
        df,
        x_metric="rmse_K",
        y_metric="margin_err",
        output_path=output_dir / "margin_scatter.png",
    )


def main() -> None:
    """Run the benchmark."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    total_budget = total_budget_from_shots_per_entry(
        args.n_samples,
        args.shots_per_entry,
    )

    print("=== Benchmark configuration ===")
    print(f"experiment_rounds = {args.experiment_rounds}")
    print(f"n_samples         = {args.n_samples}")
    print(f"gamma             = {args.gamma}")
    print(f"C                 = {args.c}")
    print(f"shots_per_entry   = {args.shots_per_entry}")
    print(f"N_tot             = {total_budget}")
    print(f"pilot_shots       = {args.pilot_shots}")
    print(f"rounds            = {args.rounds}")
    print(f"cluster_std       = {args.cluster_std}")
    print(f"lambda_weight     = {args.lambda_weight}")
    print(f"allocation_method = {args.allocation_method}")
    print(f"stopping_epsilon  = {args.stopping_epsilon}")
    print(f"output_dir        = {output_dir}")
    print()

    records: list[dict[str, float | int | str]] = []

    for run in range(args.experiment_rounds):
        dataset_seed = int(rng.integers(0, 2**32 - 1))
        adaptive_seed = int(rng.integers(0, 2**32 - 1))
        uniform_seed = int(rng.integers(0, 2**32 - 1))

        print(f"Run {run + 1}/{args.experiment_rounds}")

        x, y = generate_blobs_dataset(
            args.n_samples,
            cluster_std=args.cluster_std,
            random_state=dataset_seed,
        )
        kernel_true = rbf_kernel(x, x, gamma=args.gamma)

        adaptive_backend = SyntheticBernoulliBackend(
            kernel_true,
            random_state=adaptive_seed,
        )
        model = SupportShot(
            total_budget=total_budget,
            pilot_shots_per_entry=args.pilot_shots,
            rounds=args.rounds,
            lambda_weight=args.lambda_weight,
            svm_c=args.c,
            stopping_epsilon=args.stopping_epsilon,
            allocation_method=args.allocation_method,
            random_state=adaptive_seed,
        )
        model.fit(y, adaptive_backend)

        add_supportshot_records(
            records,
            kernel_true=kernel_true,
            y=y,
            model=model,
            c_value=args.c,
            run_id=run,
        )

        kernel_uniform, spent_uniform = run_uniform_baseline(
            kernel_true,
            total_budget=total_budget,
            random_state=uniform_seed,
        )
        add_uniform_record(
            records,
            kernel_true=kernel_true,
            kernel_uniform=kernel_uniform,
            y=y,
            c_value=args.c,
            run_id=run,
            total_shots_spent=spent_uniform,
        )

    df = records_to_dataframe(records)
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== Summary ===")
    print_summary(df)
    save_standard_plots(df, output_dir)

    print("\nSaved outputs:")
    print(f"- CSV results: {csv_path}")
    print(f"- Plot directory: {output_dir}")


if __name__ == "__main__":
    main()
