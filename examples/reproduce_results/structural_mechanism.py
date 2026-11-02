"""Controlled experiment linking weight heterogeneity to adaptive advantage.

This experiment replaces the broad synthetic regime sweep with a controlled
family of binary blob-classification problems. It varies:

1. problem structure through the cluster standard deviation, and
2. symmetric training-label noise through an exact fraction of label flips.

For every condition, the exact-kernel SVM defines oracle support vectors,
signed dual coefficients, and ideal allocation weights

    w_ij = |alpha_i alpha_j| sqrt(K_ij (1 - K_ij)).

SupportShot and uniform allocation are then compared using the same exact
kernel, noisy training labels, and cumulative measurement budget.

The main structural-mechanism panel shows

    x = Gini coefficient of the oracle allocation weights,
    y = log10(RMSE_uniform / RMSE_adaptive).

Positive y values indicate an adaptive advantage. Point color represents the
label-noise rate, while marker size represents the oracle support-vector
fraction.

Example
-------
From the repository root:

    python examples/reproduce_results/structural_mechanism.py \
        --experiment-rounds 50 \
        --n-samples 100 \
        --cluster-stds 0.15 0.25 0.40 0.60 0.80 1.00 \
        --label-noise-rates 0.00 0.05 0.10 0.20 0.30 0.40 \
        --gamma 1.0 \
        --c 1.0 \
        --shots-per-entry 100 \
        --pilot-shots 10 \
        --rounds 5 \
        --lambda-weight 0.5 \
        --output-dir outputs/structural_mechanism
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluation_utils import evaluate_estimated_kernel
from matplotlib.colors import Normalize
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from shotwise.algorithms.supportshot import SupportShot
from shotwise.backends.synthetic import SyntheticBernoulliBackend
from shotwise.storage.kernel_store import KernelStore
from shotwise.utils.matrix import upper_triangle_indices


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Study the relationship between oracle allocation-weight "
            "heterogeneity and adaptive measurement advantage."
        )
    )

    parser.add_argument("--experiment-rounds", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument(
        "--cluster-stds",
        type=float,
        nargs="+",
        default=[0.15, 0.25, 0.40, 0.60, 0.80, 1.00],
    )
    parser.add_argument(
        "--label-noise-rates",
        type=float,
        nargs="+",
        default=[0.00, 0.05, 0.10, 0.20, 0.30, 0.40],
    )

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--shots-per-entry", type=int, default=100)
    parser.add_argument("--pilot-shots", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument(
        "--allocation-method",
        choices=["multinomial", "deterministic"],
        default="multinomial",
    )
    parser.add_argument("--stopping-epsilon", type=float, default=None)

    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--measurement-seed", type=int, default=1)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/structural_mechanism"),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate experimental arguments."""
    if args.experiment_rounds <= 0:
        raise ValueError("experiment_rounds must be positive.")

    if args.n_samples < 4:
        raise ValueError("n_samples must be at least 4.")

    if args.n_samples % 2 != 0:
        raise ValueError("n_samples must be even to generate balanced binary classes.")

    if args.gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    if args.c <= 0.0:
        raise ValueError("C must be positive.")

    if args.shots_per_entry <= 0:
        raise ValueError("shots_per_entry must be positive.")

    if args.pilot_shots <= 0:
        raise ValueError("pilot_shots must be positive.")

    if args.rounds <= 0:
        raise ValueError("rounds must be positive.")

    if any(value <= 0.0 for value in args.cluster_stds):
        raise ValueError("All cluster standard deviations must be positive.")

    if any(rate < 0.0 or rate >= 0.5 for rate in args.label_noise_rates):
        raise ValueError("Label-noise rates must satisfy 0 <= eta < 0.5.")


def generate_dataset(
    n_samples: int,
    *,
    cluster_std: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced binary blob dataset with labels in {-1, +1}."""
    x, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    y = np.where(y == 0, -1, 1)

    return x, y


def nested_noisy_labels(
    y_clean: np.ndarray,
    *,
    noise_rates: list[float],
    random_state: int,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Generate nested exact-count label flips for all requested noise rates.

    A single random permutation determines the flip ordering. Higher noise
    levels therefore contain all points flipped at lower noise levels, making
    the noise-rate trajectory easier to interpret.
    """
    rng = np.random.default_rng(random_state)
    permutation = rng.permutation(y_clean.size)

    noisy_labels: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for rate in sorted(noise_rates):
        n_flips = int(round(rate * y_clean.size))
        flip_indices = permutation[:n_flips]

        y_noisy = y_clean.copy()
        y_noisy[flip_indices] *= -1

        noisy_labels[rate] = (
            y_noisy,
            np.asarray(flip_indices, dtype=int),
        )

    return noisy_labels


def total_budget_from_shots_per_entry(
    n_samples: int,
    shots_per_entry: int,
) -> int:
    """Return the total budget implied by uniform off-diagonal allocation."""
    n_pairs = n_samples * (n_samples - 1) // 2
    return n_pairs * shots_per_entry


def uniform_allocation_matrix(
    n: int,
    total_budget: int,
) -> np.ndarray:
    """Return an upper-triangular uniform allocation for an exact budget."""
    if total_budget < 0:
        raise ValueError("total_budget must be non-negative.")

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


def estimate_uniform_kernel(
    kernel_true: np.ndarray,
    *,
    total_budget: int,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """Estimate the kernel using uniform allocation."""
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
    store.update(counts, allocation)

    return store.kernel_matrix(), store.total_shots_spent()


def fit_oracle_svm(
    kernel_true: np.ndarray,
    y_noisy: np.ndarray,
    *,
    c_value: float,
) -> tuple[SVC, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the exact-kernel SVM and return oracle dual information."""
    model = SVC(
        kernel="precomputed",
        C=c_value,
    )
    model.fit(kernel_true, y_noisy)

    support_indices = np.asarray(model.support_, dtype=int)

    signed_dual_coefficients = np.zeros(
        kernel_true.shape[0],
        dtype=float,
    )
    signed_dual_coefficients[support_indices] = model.dual_coef_[0]

    alpha = np.abs(signed_dual_coefficients)

    return (
        model,
        support_indices,
        signed_dual_coefficients,
        alpha,
    )


def gini_coefficient(values: np.ndarray) -> float:
    """Return the Gini coefficient of finite non-negative values."""
    array = np.asarray(values, dtype=float).ravel()

    if array.size == 0:
        raise ValueError("Gini input must not be empty.")

    if np.any(~np.isfinite(array)):
        raise ValueError("Gini input contains non-finite values.")

    if np.any(array < 0.0):
        raise ValueError("Gini input must be non-negative.")

    total = float(array.sum())

    if total == 0.0:
        return 0.0

    sorted_values = np.sort(array)
    n = sorted_values.size
    indices = np.arange(1, n + 1, dtype=float)

    gini = 2.0 * np.sum(indices * sorted_values) / (n * total) - (n + 1.0) / n

    # Protect against tiny floating-point excursions.
    return float(np.clip(gini, 0.0, 1.0))


def oracle_allocation_weights(
    kernel_true: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Return oracle Bernoulli Neyman weights on independent kernel entries."""
    rows, cols = upper_triangle_indices(kernel_true.shape[0])

    probabilities = np.clip(
        kernel_true[rows, cols],
        0.0,
        1.0,
    )

    bernoulli_std = np.sqrt(probabilities * (1.0 - probabilities))

    weights = alpha[rows] * alpha[cols] * bernoulli_std

    return np.asarray(weights, dtype=float)


def support_vector_diagnostics(
    alpha: np.ndarray,
    *,
    c_value: float,
    tolerance: float = 1e-7,
) -> dict[str, float | int]:
    """Return support-vector, free-SV, and bounded-SV diagnostics."""
    sv_mask = alpha > tolerance

    bounded_tolerance = max(
        tolerance,
        1e-6 * max(1.0, c_value),
    )
    bounded_mask = sv_mask & np.isclose(
        alpha,
        c_value,
        rtol=1e-5,
        atol=bounded_tolerance,
    )
    free_mask = sv_mask & ~bounded_mask

    n = alpha.size
    n_support = int(np.count_nonzero(sv_mask))
    n_free = int(np.count_nonzero(free_mask))
    n_bounded = int(np.count_nonzero(bounded_mask))

    return {
        "n_support_vectors": n_support,
        "sv_fraction": n_support / n,
        "n_free_support_vectors": n_free,
        "free_sv_fraction": n_free / n,
        "n_bounded_support_vectors": n_bounded,
        "bounded_sv_fraction": n_bounded / n,
    }


def safe_log_advantage(
    uniform_error: float,
    adaptive_error: float,
) -> float:
    """Return log10(uniform error / adaptive error) safely."""
    if uniform_error < 0.0 or adaptive_error < 0.0:
        raise ValueError("RMSE values must be non-negative.")

    floor = np.finfo(float).tiny

    return float(np.log10(max(uniform_error, floor) / max(adaptive_error, floor)))


def run_adaptive_estimate(
    kernel_true: np.ndarray,
    y_noisy: np.ndarray,
    *,
    total_budget: int,
    pilot_shots: int,
    rounds: int,
    lambda_weight: float,
    c_value: float,
    stopping_epsilon: float | None,
    allocation_method: str,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """Run SupportShot and return its final estimated kernel."""
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
        stopping_epsilon=stopping_epsilon,
        allocation_method=allocation_method,
        random_state=random_state,
    )
    model.fit(y_noisy, backend)

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
    y_noisy: np.ndarray,
    *,
    c_value: float,
) -> dict[str, float]:
    """Evaluate adaptive and uniform estimates against the same oracle."""
    adaptive_metrics = evaluate_estimated_kernel(
        kernel_true,
        kernel_adaptive,
        y_noisy,
        c=c_value,
        method="adaptive",
        round_id=-1,
    )

    uniform_metrics = evaluate_estimated_kernel(
        kernel_true,
        kernel_uniform,
        y_noisy,
        c=c_value,
        method="uniform",
        round_id=-1,
    )

    adaptive_rmse = float(adaptive_metrics["func_rmse_normalized"])
    uniform_rmse = float(uniform_metrics["func_rmse_normalized"])

    result = {
        "adaptive_func_rmse": adaptive_rmse,
        "uniform_func_rmse": uniform_rmse,
        "adaptive_advantage_log10": safe_log_advantage(
            uniform_rmse,
            adaptive_rmse,
        ),
    }

    optional_metrics = [
        "margin_err",
        "jaccard",
        "jaccard_weighted",
        "rmse_K",
        "rmse_K_SV",
        "acc_train_vs_true",
    ]

    for metric in optional_metrics:
        if metric in adaptive_metrics:
            result[f"adaptive_{metric}"] = float(adaptive_metrics[metric])

        if metric in uniform_metrics:
            result[f"uniform_{metric}"] = float(uniform_metrics[metric])

    return result


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired results by structural and label-noise condition."""
    grouped = df.groupby(
        ["cluster_std", "label_noise_rate"],
        sort=True,
    )

    summary = grouped.agg(
        repetitions=("run", "count"),
        weight_gini_median=("weight_gini", "median"),
        weight_gini_q25=(
            "weight_gini",
            lambda values: values.quantile(0.25),
        ),
        weight_gini_q75=(
            "weight_gini",
            lambda values: values.quantile(0.75),
        ),
        sv_fraction_median=("sv_fraction", "median"),
        sv_fraction_q25=(
            "sv_fraction",
            lambda values: values.quantile(0.25),
        ),
        sv_fraction_q75=(
            "sv_fraction",
            lambda values: values.quantile(0.75),
        ),
        advantage_median=(
            "adaptive_advantage_log10",
            "median",
        ),
        advantage_q25=(
            "adaptive_advantage_log10",
            lambda values: values.quantile(0.25),
        ),
        advantage_q75=(
            "adaptive_advantage_log10",
            lambda values: values.quantile(0.75),
        ),
    )

    return summary.reset_index()


def marker_size_from_sv_fraction(
    sv_fraction: np.ndarray | float,
) -> np.ndarray:
    """Map support-vector fraction to scatter-marker area."""
    values = np.asarray(sv_fraction, dtype=float)
    return 18.0 + 110.0 * np.clip(values, 0.0, 1.0)


def save_structural_mechanism_figure(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save the structural-mechanism panel in manuscript style."""
    figure_blue = "#4C78A8"

    noise_min = float(df["label_noise_rate"].min())
    noise_max = float(df["label_noise_rate"].max())

    if noise_max == noise_min:
        noise_max = noise_min + 1.0

    normalization = Normalize(
        vmin=noise_min,
        vmax=noise_max,
    )
    colormap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(4.8, 3.5),
    )

    # Individual paired realizations.
    scatter = ax.scatter(
        df["weight_gini"],
        df["adaptive_advantage_log10"],
        c=df["label_noise_rate"],
        cmap=colormap,
        norm=normalization,
        s=marker_size_from_sv_fraction(df["sv_fraction"]),
        alpha=0.16,
        linewidths=0.0,
        rasterized=True,
    )

    # Condition medians emphasize the controlled experimental grid.
    ax.scatter(
        summary["weight_gini_median"],
        summary["advantage_median"],
        c=summary["label_noise_rate"],
        cmap=colormap,
        norm=normalization,
        s=marker_size_from_sv_fraction(summary["sv_fraction_median"]),
        edgecolors="black",
        linewidths=0.65,
        alpha=0.95,
        zorder=3,
    )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )

    ax.set_xlabel(
        r"Oracle allocation-weight Gini coefficient $G(w^\star)$",
        fontsize=9,
    )
    ax.set_ylabel(
        r"Adaptive advantage "
        r"$\log_{10}(\mathrm{RMSE}_{f,\mathrm{unif}}/"
        r"\mathrm{RMSE}_{f,\mathrm{adapt}})$",
        fontsize=9,
    )
    ax.set_title(
        "Structural mechanism",
        fontsize=10,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=8,
        width=0.8,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.grid(False)
    ax.margins(x=0.04, y=0.08)

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        pad=0.025,
        fraction=0.055,
    )
    colorbar.set_label(
        r"Label-noise rate $\eta$",
        fontsize=8.5,
    )
    colorbar.ax.tick_params(labelsize=7.5)

    # Marker-size legend for representative support-vector fractions.
    size_handles = []
    size_labels = []

    for fraction in [0.25, 0.50, 0.75, 1.00]:
        handle = ax.scatter(
            [],
            [],
            s=float(marker_size_from_sv_fraction(fraction)),
            facecolor=figure_blue,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.75,
        )
        size_handles.append(handle)
        size_labels.append(f"{fraction:.2f}")

    ax.legend(
        size_handles,
        size_labels,
        title="SV fraction",
        fontsize=6.8,
        title_fontsize=7.2,
        frameon=False,
        loc="best",
        labelspacing=0.7,
        handletextpad=0.5,
    )

    fig.tight_layout(pad=0.7)

    fig.savefig(
        output_dir / "structural_mechanism.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "structural_mechanism.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Run the controlled structural-mechanism experiment."""
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_stds = sorted(set(args.cluster_stds))
    noise_rates = sorted(set(args.label_noise_rates))

    total_budget = total_budget_from_shots_per_entry(
        args.n_samples,
        args.shots_per_entry,
    )

    print("=== Structural-mechanism experiment ===")
    print(f"experiment_rounds = {args.experiment_rounds}")
    print(f"n_samples         = {args.n_samples}")
    print(f"cluster_stds      = {cluster_stds}")
    print(f"noise_rates       = {noise_rates}")
    print(f"gamma             = {args.gamma}")
    print(f"C                 = {args.c}")
    print(f"shots_per_entry   = {args.shots_per_entry}")
    print(f"total_budget      = {total_budget}")
    print(f"pilot_shots       = {args.pilot_shots}")
    print(f"rounds            = {args.rounds}")
    print(f"lambda_weight     = {args.lambda_weight}")
    print(f"allocation_method = {args.allocation_method}")
    print(f"stopping_epsilon  = {args.stopping_epsilon}")
    print(f"dataset_seed      = {args.dataset_seed}")
    print(f"measurement_seed  = {args.measurement_seed}")
    print(f"output_dir        = {output_dir}")
    print()

    dataset_rng = np.random.default_rng(args.dataset_seed)
    measurement_rng = np.random.default_rng(args.measurement_seed)

    records: list[dict[str, float | int]] = []

    total_conditions = args.experiment_rounds * len(cluster_stds) * len(noise_rates)
    condition_index = 0

    for run in range(args.experiment_rounds):
        for cluster_std in cluster_stds:
            dataset_seed = int(dataset_rng.integers(0, 2**32 - 1))
            label_seed = int(dataset_rng.integers(0, 2**32 - 1))

            x, y_clean = generate_dataset(
                args.n_samples,
                cluster_std=cluster_std,
                random_state=dataset_seed,
            )

            kernel_true = rbf_kernel(
                x,
                x,
                gamma=args.gamma,
            )

            noisy_label_sets = nested_noisy_labels(
                y_clean,
                noise_rates=noise_rates,
                random_state=label_seed,
            )

            for noise_rate in noise_rates:
                condition_index += 1

                if (
                    condition_index == 1
                    or condition_index % 25 == 0
                    or condition_index == total_conditions
                ):
                    print(f"Condition {condition_index}/{total_conditions}")

                y_noisy, flip_indices = noisy_label_sets[noise_rate]

                (
                    _oracle_model,
                    _support_indices,
                    signed_dual_coefficients,
                    alpha,
                ) = fit_oracle_svm(
                    kernel_true,
                    y_noisy,
                    c_value=args.c,
                )

                sv_diagnostics = support_vector_diagnostics(
                    alpha,
                    c_value=args.c,
                )

                oracle_weights = oracle_allocation_weights(
                    kernel_true,
                    alpha,
                )

                adaptive_seed = int(measurement_rng.integers(0, 2**32 - 1))
                uniform_seed = int(measurement_rng.integers(0, 2**32 - 1))

                (
                    kernel_adaptive,
                    adaptive_shots_spent,
                ) = run_adaptive_estimate(
                    kernel_true,
                    y_noisy,
                    total_budget=total_budget,
                    pilot_shots=args.pilot_shots,
                    rounds=args.rounds,
                    lambda_weight=args.lambda_weight,
                    c_value=args.c,
                    stopping_epsilon=args.stopping_epsilon,
                    allocation_method=args.allocation_method,
                    random_state=adaptive_seed,
                )

                (
                    kernel_uniform,
                    uniform_shots_spent,
                ) = estimate_uniform_kernel(
                    kernel_true,
                    total_budget=adaptive_shots_spent,
                    random_state=uniform_seed,
                )

                if uniform_shots_spent != adaptive_shots_spent:
                    raise RuntimeError(
                        "Adaptive and uniform final budgets differ: "
                        f"{adaptive_shots_spent} != "
                        f"{uniform_shots_spent}."
                    )

                performance = evaluate_pair(
                    kernel_true,
                    kernel_adaptive,
                    kernel_uniform,
                    y_noisy,
                    c_value=args.c,
                )

                record: dict[str, float | int] = {
                    "run": run,
                    "dataset_seed": dataset_seed,
                    "label_seed": label_seed,
                    "adaptive_seed": adaptive_seed,
                    "uniform_seed": uniform_seed,
                    "cluster_std": cluster_std,
                    "label_noise_rate": noise_rate,
                    "n_label_flips": flip_indices.size,
                    "realized_label_noise_rate": (flip_indices.size / args.n_samples),
                    "n_samples": args.n_samples,
                    "gamma": args.gamma,
                    "c_value": args.c,
                    "shots_per_entry": args.shots_per_entry,
                    "pilot_shots": args.pilot_shots,
                    "adaptive_rounds": args.rounds,
                    "lambda_weight": args.lambda_weight,
                    "total_budget": total_budget,
                    "total_shots_spent": adaptive_shots_spent,
                    "alpha_gini": gini_coefficient(alpha),
                    "weight_gini": gini_coefficient(oracle_weights),
                    "signed_dual_l1": float(np.sum(np.abs(signed_dual_coefficients))),
                    "oracle_weight_sum": float(oracle_weights.sum()),
                }

                record.update(sv_diagnostics)
                record.update(performance)
                records.append(record)

    df = pd.DataFrame.from_records(records)

    raw_path = output_dir / "structural_mechanism_results.csv"
    df.to_csv(raw_path, index=False)

    summary = summarize_results(df)
    summary_path = output_dir / "structural_mechanism_summary.csv"
    summary.to_csv(summary_path, index=False)

    save_structural_mechanism_figure(
        df,
        summary,
        output_dir,
    )

    print()
    print("=== Condition medians ===")
    display_columns = [
        "cluster_std",
        "label_noise_rate",
        "weight_gini_median",
        "sv_fraction_median",
        "advantage_median",
    ]
    print(
        summary[display_columns].to_string(
            index=False,
        )
    )

    print()
    print("Saved outputs:")
    print(f"- Raw results: {raw_path}")
    print(f"- Summary:     {summary_path}")
    print(f"- PNG figure:  {output_dir / 'structural_mechanism.png'}")
    print(f"- PDF figure:  {output_dir / 'structural_mechanism.pdf'}")


if __name__ == "__main__":
    main()
