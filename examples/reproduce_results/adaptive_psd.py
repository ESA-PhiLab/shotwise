"""Compare spectral and SVM-relevant kernel fidelity under adaptive and
uniform allocation.

This experiment compares SupportShot with uniform measurement allocation at
matched cumulative shot budgets. A single synthetic blob dataset and its exact
RBF kernel are kept fixed across repeated measurement realizations.

The exact-kernel SVM supplies an oracle support-vector set and signed dual
coefficients. At every SupportShot checkpoint, the adaptive estimate and an
independently generated uniform estimate with the same cumulative budget are
evaluated using:

- global relative negative spectral mass,
- global minimum eigenvalue,
- oracle support-vector-block relative negative spectral mass,
- oracle support-vector-block minimum eigenvalue,
- relative error of the ideal SVM dual quadratic form,
- relative Frobenius error of the oracle support-vector block.

Example
-------
From the repository root:

    python examples/reproduce_results/adaptive_psd.py \
        --experiment-rounds 200 \
        --n-samples 50 \
        --gamma 1.0 \
        --c 100 \
        --shots-per-entry 100 \
        --pilot-shots 10 \
        --rounds 5 \
        --cluster-std 0.2 \
        --lambda-weight 0.5 \
        --allocation-method multinomial \
        --output-dir outputs/adaptive_psd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator
from psd_utils import (
    principal_submatrix,
    psd_diagnostics,
    relative_dual_quadratic_error,
    support_vector_block_error,
)
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
            "Compare PSD and SVM-relevant kernel fidelity under adaptive "
            "and uniform allocation."
        )
    )

    parser.add_argument("--experiment-rounds", type=int, default=200)
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
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--measurement-seed", type=int, default=1)

    parser.add_argument(
        "--xscale",
        choices=["linear", "log"],
        default="linear",
    )
    parser.add_argument(
        "--yscale-spectral",
        choices=["linear", "log"],
        default="linear",
    )
    parser.add_argument(
        "--yscale-dual",
        choices=["linear", "log"],
        default="log",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/adaptive_psd"),
    )
    parser.add_argument(
        "--yscale-sv",
        choices=["linear", "log", "symlog"],
        default="symlog",
        help="Y-axis scale for the oracle SV-block PSD metric.",
    )
    parser.add_argument(
        "--sv-linthresh",
        type=float,
        default=1e-4,
        help=("Linear threshold around zero when --yscale-sv symlog is used."),
    )

    return parser.parse_args()


def generate_dataset(
    n_samples: int,
    *,
    cluster_std: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the fixed binary blob dataset used in the experiment."""
    x, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    y = np.where(y == 0, -1, 1)
    return x, y


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
    """Create an upper-triangular uniform allocation for an exact budget."""
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


def estimate_uniform_kernel(
    kernel_true: np.ndarray,
    *,
    total_budget: int,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """Estimate a kernel using uniform allocation at a specified budget."""
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


def fit_ideal_svm(
    kernel_true: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
) -> tuple[SVC, np.ndarray, np.ndarray]:
    """Fit the exact-kernel SVM and return its oracle active-set quantities."""
    model = SVC(
        kernel="precomputed",
        C=c_value,
    )
    model.fit(kernel_true, y)

    support_indices = np.asarray(model.support_, dtype=int)

    # For binary SVC, dual_coef_[0] already contains the signed coefficients
    # y_i alpha_i for the support vectors.
    signed_dual_coefficients = np.zeros(
        kernel_true.shape[0],
        dtype=float,
    )
    signed_dual_coefficients[support_indices] = model.dual_coef_[0]

    return model, support_indices, signed_dual_coefficients


def evaluate_kernel_estimate(
    kernel_estimated: np.ndarray,
    *,
    kernel_true: np.ndarray,
    support_indices: np.ndarray,
    signed_dual_coefficients: np.ndarray,
) -> dict[str, float | int | bool]:
    """Evaluate global, oracle-SV, and SVM-relevant kernel diagnostics."""
    global_diagnostics = psd_diagnostics(kernel_estimated)

    estimated_sv_block = principal_submatrix(
        kernel_estimated,
        support_indices,
    )
    sv_diagnostics = psd_diagnostics(estimated_sv_block)

    return {
        "global_lambda_min": global_diagnostics["lambda_min"],
        "global_negative_spectral_mass": global_diagnostics["negative_spectral_mass"],
        "global_relative_negative_mass": global_diagnostics[
            "relative_negative_spectral_mass"
        ],
        "global_negative_eigenvalue_count": global_diagnostics[
            "negative_eigenvalue_count"
        ],
        "global_is_psd": global_diagnostics["is_psd"],
        "sv_lambda_min": sv_diagnostics["lambda_min"],
        "sv_negative_spectral_mass": sv_diagnostics["negative_spectral_mass"],
        "sv_relative_negative_mass": sv_diagnostics["relative_negative_spectral_mass"],
        "sv_negative_eigenvalue_count": sv_diagnostics["negative_eigenvalue_count"],
        "sv_is_psd": sv_diagnostics["is_psd"],
        "dual_quadratic_error": relative_dual_quadratic_error(
            kernel_estimated,
            kernel_true,
            signed_dual_coefficients,
        ),
        "sv_block_frobenius_error": support_vector_block_error(
            kernel_estimated,
            kernel_true,
            support_indices,
        ),
    }


def stage_label(stage: str, round_index: int) -> str:
    """Return a concise label for a SupportShot history checkpoint."""
    if stage == "pilot":
        return "pilot"
    return f"round_{round_index}"


def append_record(
    records: list[dict[str, float | int | bool | str]],
    *,
    method: str,
    run_id: int,
    stage: str,
    round_index: int,
    requested_budget: int,
    total_shots_spent: int,
    measurement_seed: int,
    n_support_vectors: int,
    diagnostics: dict[str, float | int | bool],
) -> None:
    """Append one method/checkpoint record."""
    record: dict[str, float | int | bool | str] = {
        "run": run_id,
        "method": method,
        "stage": stage,
        "checkpoint": stage_label(stage, round_index),
        "round": round_index,
        "requested_budget": requested_budget,
        "total_shots_spent": total_shots_spent,
        "measurement_seed": measurement_seed,
        "n_support_vectors": n_support_vectors,
    }
    record.update(diagnostics)
    records.append(record)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate diagnostics by method and cumulative budget."""
    metrics = [
        "global_lambda_min",
        "global_relative_negative_mass",
        "sv_lambda_min",
        "sv_relative_negative_mass",
        "dual_quadratic_error",
        "sv_block_frobenius_error",
    ]

    grouped = df.groupby(
        ["method", "total_shots_spent"],
        sort=True,
    )

    summary_frames: list[pd.DataFrame] = []

    for metric in metrics:
        metric_summary = grouped[metric].agg(
            ["count", "mean", "std", "median", "min", "max"]
        )
        metric_summary["q25"] = grouped[metric].quantile(0.25)
        metric_summary["q75"] = grouped[metric].quantile(0.75)
        metric_summary["metric"] = metric
        summary_frames.append(metric_summary.reset_index())

    return pd.concat(summary_frames, ignore_index=True)


def metric_quantiles(
    df: pd.DataFrame,
    *,
    method: str,
    metric: str,
) -> pd.DataFrame:
    """Return median and interquartile range by budget for one method."""
    selected = df.loc[df["method"] == method]

    grouped = selected.groupby(
        "total_shots_spent",
        sort=True,
    )[metric]

    result = grouped.agg(
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    )

    return result.reset_index()


def plot_quantile_curve(
    ax,
    df: pd.DataFrame,
    *,
    method: str,
    metric: str,
    label: str,
    color: str,
    marker: str,
    log_floor: float | None = None,
) -> None:
    """Plot the median and interquartile range for one metric."""
    quantiles = metric_quantiles(
        df,
        method=method,
        metric=metric,
    )

    x = quantiles["total_shots_spent"].to_numpy(dtype=float)
    median = quantiles["median"].to_numpy(dtype=float)
    q25 = quantiles["q25"].to_numpy(dtype=float)
    q75 = quantiles["q75"].to_numpy(dtype=float)

    # Plotting transformation only. Raw experimental values are unchanged.
    if log_floor is not None:
        median = np.maximum(median, log_floor)
        q25 = np.maximum(q25, log_floor)
        q75 = np.maximum(q75, log_floor)

    ax.plot(
        x,
        median,
        color=color,
        linewidth=1.3,
        marker=marker,
        markersize=4.0,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.8,
        label=label,
    )

    ax.fill_between(
        x,
        q25,
        q75,
        color=color,
        alpha=0.14,
        linewidth=0.0,
    )


def save_main_figure(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    xscale: str,
    yscale_spectral: str,
    yscale_sv: str,
    sv_linthresh: float,
    yscale_dual: str,
) -> None:
    """Save the global-spectral versus SVM-relevant comparison."""
    adaptive_color = "#4C78A8"
    uniform_color = "#777777"

    # Convert cumulative total budget into average measurement budget per
    # off-diagonal kernel entry. For adaptive allocation this is a normalized
    # total budget, not the actual number of shots assigned to every entry.
    reference = np.load(output_dir / "reference_problem.npz")
    n_samples = int(reference["kernel_true"].shape[0])
    n_pairs = n_samples * (n_samples - 1) // 2

    df_plot = df.copy()
    df_plot["budget_per_entry"] = df_plot["total_shots_spent"] / n_pairs

    # plot_quantile_curve groups using total_shots_spent. Replace that
    # coordinate only in the plotting copy; the stored results are unchanged.
    df_plot["total_shots_spent"] = df_plot["budget_per_entry"]

    budgets = np.array(
        sorted(df_plot["budget_per_entry"].unique()),
        dtype=float,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
    )

    # ----------------------------------------------------------
    # (a) Global spectral PSD violation
    # ----------------------------------------------------------
    plot_quantile_curve(
        axes[0],
        df_plot,
        method="uniform",
        metric="global_relative_negative_mass",
        label="Uniform",
        color=uniform_color,
        marker="s",
    )
    plot_quantile_curve(
        axes[0],
        df_plot,
        method="adaptive",
        metric="global_relative_negative_mass",
        label="Adaptive",
        color=adaptive_color,
        marker="o",
    )

    axes[0].set_yscale(yscale_spectral)
    axes[0].set_ylabel(
        r"Relative negative spectral mass $\rho_-$",
        fontsize=9,
    )
    axes[0].set_title(
        "(a) Global spectral fidelity",
        fontsize=10,
    )
    axes[0].legend(
        fontsize=7.5,
        frameon=False,
        loc="best",
    )

    # ----------------------------------------------------------
    # (b) SVM-relevant kernel error
    # ----------------------------------------------------------
    plot_quantile_curve(
        axes[1],
        df_plot,
        method="uniform",
        metric="dual_quadratic_error",
        label="Uniform",
        color=uniform_color,
        marker="s",
    )
    plot_quantile_curve(
        axes[1],
        df_plot,
        method="adaptive",
        metric="dual_quadratic_error",
        label="Adaptive",
        color=adaptive_color,
        marker="o",
    )

    axes[1].set_yscale(yscale_dual)
    axes[1].set_ylabel(
        r"Relative dual-quadratic error $E_Q$",
        fontsize=9,
    )
    axes[1].set_title(
        "(b) SVM-relevant fidelity",
        fontsize=10,
    )
    axes[1].legend(
        fontsize=7.5,
        frameon=False,
        loc="best",
    )

    # ----------------------------------------------------------
    # Consistent logarithmic y-axis formatting.
    # Label decade ticks only; leave intermediate logarithmic
    # positions as unlabelled minor ticks.
    # ----------------------------------------------------------
    scale_pairs = [
        (axes[0], yscale_spectral),
        (axes[1], yscale_dual),
    ]

    for ax, scale in scale_pairs:
        if scale == "log":
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
            ax.yaxis.set_minor_locator(
                LogLocator(
                    base=10.0,
                    subs=np.arange(2, 10) * 0.1,
                )
            )
            ax.yaxis.set_minor_formatter(NullFormatter())

    # ----------------------------------------------------------
    # Common x-axis formatting
    # ----------------------------------------------------------
    for ax in axes:
        ax.set_xscale(xscale)

        if xscale == "log":
            # Label first, middle, and final measurement checkpoints.
            # All checkpoints remain visible through their data markers.
            indices = [0, len(budgets) // 2, len(budgets) - 1]
            ticks = budgets[indices]

            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [f"{value:.0f}" for value in ticks],
            )

            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())

        else:
            ax.set_xticks(budgets)
            ax.set_xticklabels(
                [f"{value:.0f}" for value in budgets],
            )

        ax.set_xlabel(
            "Measurement budget per kernel entry",
            fontsize=9,
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=8,
            width=0.8,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            width=0.6,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        ax.grid(False)
        ax.margins(x=0.04)

    fig.tight_layout(
        pad=0.7,
        w_pad=1.5,
    )

    fig.savefig(
        output_dir / "adaptive_psd.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "adaptive_psd.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Run the matched-budget adaptive-versus-uniform PSD experiment."""
    args = parse_args()

    if args.experiment_rounds <= 0:
        raise ValueError("experiment_rounds must be positive.")

    if args.n_samples < 2:
        raise ValueError("n_samples must be at least 2.")

    if args.shots_per_entry <= 0:
        raise ValueError("shots_per_entry must be positive.")

    if args.pilot_shots <= 0:
        raise ValueError("pilot_shots must be positive.")

    if args.rounds <= 0:
        raise ValueError("rounds must be positive.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_budget = total_budget_from_shots_per_entry(
        args.n_samples,
        args.shots_per_entry,
    )

    print("=== Adaptive PSD experiment ===")
    print(f"experiment_rounds = {args.experiment_rounds}")
    print(f"n_samples         = {args.n_samples}")
    print(f"gamma             = {args.gamma}")
    print(f"C                 = {args.c}")
    print(f"shots_per_entry   = {args.shots_per_entry}")
    print(f"total_budget      = {total_budget}")
    print(f"pilot_shots       = {args.pilot_shots}")
    print(f"rounds            = {args.rounds}")
    print(f"cluster_std       = {args.cluster_std}")
    print(f"lambda_weight     = {args.lambda_weight}")
    print(f"allocation_method = {args.allocation_method}")
    print(f"stopping_epsilon  = {args.stopping_epsilon}")
    print(f"dataset_seed      = {args.dataset_seed}")
    print(f"measurement_seed  = {args.measurement_seed}")
    print(f"output_dir        = {output_dir}")
    print()

    # Keep the dataset and exact kernel fixed to isolate allocation effects.
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

    ideal_model, support_indices, signed_dual_coefficients = fit_ideal_svm(
        kernel_true,
        y,
        c_value=args.c,
    )

    exact_global = psd_diagnostics(kernel_true)
    exact_sv = psd_diagnostics(
        principal_submatrix(
            kernel_true,
            support_indices,
        )
    )

    print("=== Exact reference ===")
    print(f"n_support_vectors        = {support_indices.size}")
    print(f"global_lambda_min        = {exact_global['lambda_min']:.8e}")
    print(
        "global_relative_neg_mass = "
        f"{exact_global['relative_negative_spectral_mass']:.8e}"
    )
    print(f"sv_lambda_min            = {exact_sv['lambda_min']:.8e}")
    print(
        f"sv_relative_neg_mass     = {exact_sv['relative_negative_spectral_mass']:.8e}"
    )
    print()

    np.savez(
        output_dir / "reference_problem.npz",
        x=x,
        y=y,
        kernel_true=kernel_true,
        support_indices=support_indices,
        signed_dual_coefficients=signed_dual_coefficients,
        ideal_support_vectors=ideal_model.support_vectors_,
    )

    rng = np.random.default_rng(args.measurement_seed)

    records: list[dict[str, float | int | bool | str]] = []

    for run in range(args.experiment_rounds):
        adaptive_seed = int(rng.integers(0, 2**32 - 1))

        print(f"Run {run + 1}/{args.experiment_rounds}")

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

        for history_record in model.history_:
            adaptive_diagnostics = evaluate_kernel_estimate(
                history_record.kernel_matrix,
                kernel_true=kernel_true,
                support_indices=support_indices,
                signed_dual_coefficients=signed_dual_coefficients,
            )

            append_record(
                records,
                method="adaptive",
                run_id=run,
                stage=history_record.stage,
                round_index=history_record.round_index,
                requested_budget=history_record.budget,
                total_shots_spent=history_record.total_shots_spent,
                measurement_seed=adaptive_seed,
                n_support_vectors=support_indices.size,
                diagnostics=adaptive_diagnostics,
            )

            # Construct an independent uniform estimate using precisely the
            # same cumulative number of shots as the adaptive checkpoint.
            uniform_seed = int(rng.integers(0, 2**32 - 1))

            kernel_uniform, uniform_shots_spent = estimate_uniform_kernel(
                kernel_true,
                total_budget=history_record.total_shots_spent,
                random_state=uniform_seed,
            )

            if uniform_shots_spent != history_record.total_shots_spent:
                raise RuntimeError(
                    "Uniform and adaptive cumulative shot budgets do not match: "
                    f"{uniform_shots_spent} != "
                    f"{history_record.total_shots_spent}."
                )

            uniform_diagnostics = evaluate_kernel_estimate(
                kernel_uniform,
                kernel_true=kernel_true,
                support_indices=support_indices,
                signed_dual_coefficients=signed_dual_coefficients,
            )

            append_record(
                records,
                method="uniform",
                run_id=run,
                stage=history_record.stage,
                round_index=history_record.round_index,
                requested_budget=history_record.total_shots_spent,
                total_shots_spent=uniform_shots_spent,
                measurement_seed=uniform_seed,
                n_support_vectors=support_indices.size,
                diagnostics=uniform_diagnostics,
            )

    df = pd.DataFrame.from_records(records)

    raw_results_path = output_dir / "adaptive_psd_results.csv"
    df.to_csv(raw_results_path, index=False)

    summary = summarize_results(df)
    summary_path = output_dir / "adaptive_psd_summary.csv"
    summary.to_csv(summary_path, index=False)

    save_main_figure(
        df,
        output_dir,
        xscale=args.xscale,
        yscale_spectral=args.yscale_spectral,
        yscale_sv=args.yscale_sv,
        sv_linthresh=args.sv_linthresh,
        yscale_dual=args.yscale_dual,
    )

    print()
    print("=== Median diagnostics ===")

    display_metrics = [
        "global_relative_negative_mass",
        "sv_relative_negative_mass",
        "dual_quadratic_error",
        "sv_block_frobenius_error",
    ]

    display = (
        df.groupby(
            ["method", "total_shots_spent"],
            sort=True,
        )[display_metrics]
        .median()
        .reset_index()
    )

    print(display.to_string(index=False))

    print()
    print("Saved outputs:")
    print(f"- Reference problem: {output_dir / 'reference_problem.npz'}")
    print(f"- Raw results:      {raw_results_path}")
    print(f"- Summary:          {summary_path}")
    print(f"- PNG figure:       {output_dir / 'adaptive_psd.png'}")
    print(f"- PDF figure:       {output_dir / 'adaptive_psd.pdf'}")


if __name__ == "__main__":
    main()
