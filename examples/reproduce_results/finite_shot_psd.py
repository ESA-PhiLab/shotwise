"""Study PSD violations caused by finite-shot kernel estimation.

This experiment isolates the effect of finite-shot measurement noise on the
positive-semidefiniteness of an estimated kernel matrix.

A single synthetic blob dataset is generated and converted into an exact RBF
kernel matrix. The exact kernel is then repeatedly estimated with a uniform
number of Bernoulli measurements per off-diagonal kernel entry.

For each shot budget, the experiment records spectral diagnostics of the
estimated kernel and summarizes how the magnitude of indefiniteness changes
as measurement precision increases.

Example
-------
From the repository root:

    python examples/reproduce_results/finite_shot_psd.py \
        --experiment-rounds 200 \
        --n-samples 50 \
        --gamma 1.0 \
        --cluster-std 0.2 \
        --shots-per-entry 1 2 5 10 20 50 100 200 500 1000 \
        --output-dir outputs/finite_shot_psd

Axis scales can be selected independently:

    python examples/reproduce_results/finite_shot_psd.py \
        --xscale-mass log \
        --yscale-mass log \
        --xscale-lambda log \
        --yscale-lambda linear
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psd_utils import psd_diagnostics
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel

from shotwise.backends.synthetic import SyntheticBernoulliBackend
from shotwise.storage.kernel_store import KernelStore
from shotwise.utils.matrix import upper_triangle_indices


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Study PSD violations caused by finite-shot kernel estimation."
    )
    parser.add_argument("--experiment-rounds", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--cluster-std", type=float, default=0.2)
    parser.add_argument(
        "--shots-per-entry",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        help="Uniform numbers of Bernoulli measurements per off-diagonal entry.",
    )
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--measurement-seed", type=int, default=1)

    parser.add_argument(
        "--xscale-mass",
        choices=["linear", "log"],
        default="linear",
        help="X-axis scale for the negative spectral mass panel.",
    )
    parser.add_argument(
        "--yscale-mass",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale for the negative spectral mass panel.",
    )
    parser.add_argument(
        "--xscale-lambda",
        choices=["linear", "log"],
        default="linear",
        help="X-axis scale for the minimum-eigenvalue panel.",
    )
    parser.add_argument(
        "--yscale-lambda",
        choices=["linear", "log"],
        default="linear",
        help=(
            "Y-axis scale for the minimum-eigenvalue panel. "
            "Note that ordinary log scale cannot display negative eigenvalues."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/finite_shot_psd"),
    )
    return parser.parse_args()


def generate_dataset(
    n_samples: int,
    *,
    cluster_std: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the fixed binary blob dataset used throughout the experiment."""
    x, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    y = np.where(y == 0, -1, 1)
    return x, y


def uniform_allocation_matrix(
    n: int,
    shots_per_entry: int,
) -> np.ndarray:
    """Allocate the same number of shots to each off-diagonal kernel entry."""
    if shots_per_entry < 0:
        raise ValueError(
            f"shots_per_entry must be non-negative, got {shots_per_entry}."
        )

    rows, cols = upper_triangle_indices(n)

    allocation = np.zeros((n, n), dtype=int)
    allocation[rows, cols] = shots_per_entry

    return allocation


def estimate_kernel(
    kernel_true: np.ndarray,
    *,
    shots_per_entry: int,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """Return one finite-shot estimate of an exact kernel matrix."""
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
        shots_per_entry,
    )

    counts = backend.sample_upper_triangle_counts(allocation)
    store.update(counts, allocation)

    return store.kernel_matrix(), store.total_shots_spent()


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PSD diagnostics over repeated measurement realizations."""
    grouped = df.groupby("shots_per_entry", sort=True)

    summary = grouped.agg(
        repetitions=("run", "count"),
        total_shots=("total_shots_spent", "first"),
        lambda_min_mean=("lambda_min", "mean"),
        lambda_min_median=("lambda_min", "median"),
        lambda_min_std=("lambda_min", "std"),
        relative_negative_mass_mean=(
            "relative_negative_spectral_mass",
            "mean",
        ),
        relative_negative_mass_median=(
            "relative_negative_spectral_mass",
            "median",
        ),
        relative_negative_mass_std=(
            "relative_negative_spectral_mass",
            "std",
        ),
        indefinite_probability=("is_psd", lambda x: 1.0 - x.mean()),
        negative_eigenvalue_count_mean=(
            "negative_eigenvalue_count",
            "mean",
        ),
    )

    return summary.reset_index()


def save_main_figure(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    xscale_mass: str,
    yscale_mass: str,
    xscale_lambda: str,
    yscale_lambda: str,
) -> None:
    """Save the two-panel spectral PSD figure."""
    shot_values = np.array(sorted(df["shots_per_entry"].unique()))

    relative_mass_data = [
        df.loc[
            df["shots_per_entry"] == shots,
            "relative_negative_spectral_mass",
        ].to_numpy()
        for shots in shot_values
    ]

    lambda_min_data = [
        df.loc[
            df["shots_per_entry"] == shots,
            "lambda_min",
        ].to_numpy()
        for shots in shot_values
    ]

    if yscale_lambda == "log":
        minimum_lambda = min(float(np.min(values)) for values in lambda_min_data)
        if minimum_lambda <= 0.0:
            raise ValueError(
                "--yscale-lambda log cannot be used because lambda_min "
                "contains zero or negative values. Use linear scaling."
            )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.0),
    )

    # For a logarithmic x-axis, boxplot positions must use the actual shot
    # values. For a linear categorical display, equally spaced positions make
    # the individual budgets easier to read.
    if xscale_mass == "log":
        mass_positions = shot_values.astype(float)
        mass_widths = np.maximum(0.12 * mass_positions, 0.1)
    else:
        mass_positions = np.arange(len(shot_values), dtype=float)
        mass_widths = 0.5

    axes[0].boxplot(
        relative_mass_data,
        positions=mass_positions,
        widths=mass_widths,
        showfliers=False,
    )
    axes[0].set_xscale(xscale_mass)
    axes[0].set_yscale(yscale_mass)

    if xscale_mass == "linear":
        axes[0].set_xticks(mass_positions)
        axes[0].set_xticklabels(shot_values, rotation=45)
    else:
        axes[0].set_xticks(shot_values)
        axes[0].set_xticklabels(shot_values, rotation=45)

    axes[0].set_xlabel("Shots per kernel entry")
    axes[0].set_ylabel("Relative negative spectral mass")
    axes[0].set_title("(a) Magnitude of PSD violation")

    if xscale_lambda == "log":
        lambda_positions = shot_values.astype(float)
        lambda_widths = np.maximum(0.12 * lambda_positions, 0.1)
    else:
        lambda_positions = np.arange(len(shot_values), dtype=float)
        lambda_widths = 0.5

    axes[1].boxplot(
        lambda_min_data,
        positions=lambda_positions,
        widths=lambda_widths,
        showfliers=False,
    )
    axes[1].set_xscale(xscale_lambda)
    axes[1].set_yscale(yscale_lambda)

    if xscale_lambda == "linear":
        axes[1].set_xticks(lambda_positions)
        axes[1].set_xticklabels(shot_values, rotation=45)
    else:
        axes[1].set_xticks(shot_values)
        axes[1].set_xticklabels(shot_values, rotation=45)

    axes[1].axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle="--",
    )
    axes[1].set_xlabel("Shots per kernel entry")
    axes[1].set_ylabel(r"Minimum eigenvalue $\lambda_{\min}(\widehat{K})$")
    axes[1].set_title("(b) Worst spectral PSD violation")

    fig.tight_layout()

    fig.savefig(
        output_dir / "finite_shot_psd.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "finite_shot_psd.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Run the finite-shot PSD experiment."""
    args = parse_args()

    if args.experiment_rounds <= 0:
        raise ValueError("experiment_rounds must be positive.")

    if args.n_samples < 2:
        raise ValueError("n_samples must be at least 2.")

    if any(shots <= 0 for shots in args.shots_per_entry):
        raise ValueError("All shots-per-entry values must be positive.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Finite-shot PSD experiment ===")
    print(f"experiment_rounds = {args.experiment_rounds}")
    print(f"n_samples         = {args.n_samples}")
    print(f"gamma             = {args.gamma}")
    print(f"cluster_std       = {args.cluster_std}")
    print(f"shots_per_entry   = {args.shots_per_entry}")
    print(f"dataset_seed      = {args.dataset_seed}")
    print(f"measurement_seed  = {args.measurement_seed}")
    print(f"xscale_mass       = {args.xscale_mass}")
    print(f"yscale_mass       = {args.yscale_mass}")
    print(f"xscale_lambda     = {args.xscale_lambda}")
    print(f"yscale_lambda     = {args.yscale_lambda}")
    print(f"output_dir        = {output_dir}")
    print()

    # Generate one dataset and one exact kernel. Keeping these fixed isolates
    # finite-shot measurement fluctuations from dataset-to-dataset variation.
    x, y = generate_dataset(
        args.n_samples,
        cluster_std=args.cluster_std,
        random_state=args.dataset_seed,
    )

    kernel_true = rbf_kernel(
        x,
        x,
        gamma=args.gamma,
    )

    true_diagnostics = psd_diagnostics(kernel_true)

    print("=== Exact kernel ===")
    print(f"lambda_min = {true_diagnostics['lambda_min']:.8e}")
    print(
        "relative_negative_spectral_mass = "
        f"{true_diagnostics['relative_negative_spectral_mass']:.8e}"
    )
    print(f"is_psd = {true_diagnostics['is_psd']}")
    print()

    np.savez(
        output_dir / "reference_kernel.npz",
        x=x,
        y=y,
        kernel_true=kernel_true,
    )

    rng = np.random.default_rng(args.measurement_seed)

    records: list[dict[str, float | int | bool]] = []

    for shots_per_entry in args.shots_per_entry:
        print(f"Shots per entry: {shots_per_entry}")

        for run in range(args.experiment_rounds):
            measurement_seed = int(rng.integers(0, 2**32 - 1))

            kernel_estimated, total_shots_spent = estimate_kernel(
                kernel_true,
                shots_per_entry=shots_per_entry,
                random_state=measurement_seed,
            )

            diagnostics = psd_diagnostics(kernel_estimated)

            record: dict[str, float | int | bool] = {
                "run": run,
                "measurement_seed": measurement_seed,
                "shots_per_entry": shots_per_entry,
                "total_shots_spent": total_shots_spent,
            }
            record.update(diagnostics)
            records.append(record)

    df = pd.DataFrame.from_records(records)

    raw_results_path = output_dir / "finite_shot_psd_results.csv"
    df.to_csv(raw_results_path, index=False)

    summary = summarize_results(df)
    summary_path = output_dir / "finite_shot_psd_summary.csv"
    summary.to_csv(summary_path, index=False)

    save_main_figure(
        df,
        output_dir,
        xscale_mass=args.xscale_mass,
        yscale_mass=args.yscale_mass,
        xscale_lambda=args.xscale_lambda,
        yscale_lambda=args.yscale_lambda,
    )

    print()
    print("=== Summary ===")
    print(summary.to_string(index=False))

    print()
    print("Saved outputs:")
    print(f"- Reference kernel: {output_dir / 'reference_kernel.npz'}")
    print(f"- Raw results:     {raw_results_path}")
    print(f"- Summary:         {summary_path}")
    print(f"- PNG figure:      {output_dir / 'finite_shot_psd.png'}")
    print(f"- PDF figure:      {output_dir / 'finite_shot_psd.pdf'}")


if __name__ == "__main__":
    main()
