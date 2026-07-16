#!/usr/bin/env python3
"""Fixed-budget lambda ablation for adaptive shot allocation.

This experiment evaluates the effect of the lambda parameter in the adaptive
allocation score

    score = (1 - lambda) * margin_sensitivity + lambda * active_set_instability

under a fixed total measurement budget.

The script reports, for each lambda and seed:

    - normalized decision-function RMSE
    - weighted Jaccard similarity of dual coefficients
    - support-vector Jaccard similarity
    - margin relative error
    - full-kernel RMSE
    - support-vector-block kernel RMSE

It also runs a uniform-allocation baseline once per seed, outside the lambda loop.

Outputs:
    outputs/lambda_ablation/lambda_ablation_raw.csv
    outputs/lambda_ablation/lambda_ablation_summary.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.datasets import make_moons
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC

EPS = 1e-12


def make_dataset(
    *,
    n_samples: int,
    noise: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the synthetic binary dataset used in the fixed-budget experiments.

    If your existing fixed-budget or early-stopping experiment uses a different
    dataset constructor, replace only this function and keep the rest unchanged.
    """
    x, y01 = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )
    y = np.where(y01 == 0, -1, 1)
    return x.astype(float), y.astype(int)


def rbf_kernel(x_left: np.ndarray, x_right: np.ndarray, gamma: float) -> np.ndarray:
    """Compute an RBF kernel matrix in [0, 1]."""
    sq_dists = cdist(x_left, x_right, metric="sqeuclidean")
    return np.exp(-gamma * sq_dists)


def default_gamma(x: np.ndarray) -> float:
    """Median-distance heuristic for the RBF gamma parameter."""
    dists = pairwise_distances(x)
    upper = dists[np.triu_indices_from(dists, k=1)]
    median_dist = np.median(upper)
    if median_dist <= 0:
        return 1.0
    return 1.0 / (2.0 * median_dist**2)


def independent_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-triangular off-diagonal indices."""
    return np.triu_indices(n, k=1)


def complete_symmetric_from_upper(
    values: np.ndarray,
    n: int,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    diagonal_value: float = 1.0,
) -> np.ndarray:
    """Build a symmetric matrix from upper-triangular off-diagonal values."""
    mat = np.eye(n) * diagonal_value
    mat[row_idx, col_idx] = values
    mat[col_idx, row_idx] = values
    return mat


def estimate_kernel_from_counts(
    successes: np.ndarray,
    counts: np.ndarray,
    n: int,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    """Construct kernel estimate from Bernoulli successes and shot counts."""
    estimates = successes / np.maximum(counts, 1)
    estimates = np.clip(estimates, 0.0, 1.0)
    return complete_symmetric_from_upper(estimates, n, row_idx, col_idx)


def allocate_integer_budget(
    weights: np.ndarray,
    budget: int,
    *,
    min_count: int = 0,
) -> np.ndarray:
    """Allocate an integer budget according to non-negative weights.

    The returned allocation sums exactly to budget.

    If min_count > 0, each entry receives at least min_count shots. In that case,
    budget must be at least min_count * len(weights).
    """
    n_entries = len(weights)

    if budget < 0:
        raise ValueError("Budget must be non-negative.")

    if min_count > 0:
        required = min_count * n_entries
        if budget < required:
            raise ValueError(
                f"Budget {budget} is too small for min_count={min_count} "
                f"over {n_entries} entries."
            )
        base = np.full(n_entries, min_count, dtype=int)
        remaining = budget - required
    else:
        base = np.zeros(n_entries, dtype=int)
        remaining = budget

    if remaining == 0:
        return base

    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        probs = np.full(n_entries, 1.0 / n_entries)
    else:
        probs = weights / weights.sum()

    raw = probs * remaining
    extra = np.floor(raw).astype(int)
    leftover = remaining - int(extra.sum())

    if leftover > 0:
        fractional = raw - extra
        order = np.argsort(-fractional)
        extra[order[:leftover]] += 1

    allocation = base + extra
    assert allocation.sum() == budget
    return allocation


def add_bernoulli_measurements(
    *,
    rng: np.random.Generator,
    true_upper_values: np.ndarray,
    successes: np.ndarray,
    counts: np.ndarray,
    additional_counts: np.ndarray,
) -> None:
    """Update successes and counts using Bernoulli measurements."""
    additional_counts = np.asarray(additional_counts, dtype=int)
    mask = additional_counts > 0

    if np.any(mask):
        new_successes = rng.binomial(
            n=additional_counts[mask],
            p=true_upper_values[mask],
        )
        successes[mask] += new_successes
        counts[mask] += additional_counts[mask]


def fit_precomputed_svm(
    kernel_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
) -> SVC:
    """Fit a precomputed-kernel SVM."""
    clf = SVC(
        C=c_value,
        kernel="precomputed",
        shrinking=False,
        tol=1e-8,
        cache_size=500,
    )
    clf.fit(kernel_train, y_train)
    return clf


def full_alpha(clf: SVC, n_train: int) -> np.ndarray:
    """Return full-length signed dual coefficients alpha_i.

    sklearn stores dual_coef_ as y_i * alpha_i for binary SVMs.
    This function returns non-negative alpha_i.
    """
    signed_alpha = np.zeros(n_train)
    signed_alpha[clf.support_] = clf.dual_coef_[0]

    alpha = np.abs(signed_alpha)
    return alpha


def decision_function_train(clf: SVC, kernel_train: np.ndarray) -> np.ndarray:
    """Decision function evaluated on the training points."""
    return clf.decision_function(kernel_train)


def svm_weight_norm(
    alpha: np.ndarray,
    y: np.ndarray,
    kernel_train: np.ndarray,
) -> float:
    """Compute ||w|| from alpha, labels, and the kernel matrix."""
    signed = alpha * y
    norm_sq = float(signed @ kernel_train @ signed)
    return math.sqrt(max(norm_sq, EPS))


def normalized_decision_values(
    clf: SVC,
    kernel_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return decision values normalized by ||w||.

    This corresponds to the geometric signed decision value f(x) / ||w||.
    Since the margin is 1 / ||w||, this is equivalently f(x) multiplied by
    the geometric margin.
    """
    alpha = full_alpha(clf, len(y_train))
    f_values = decision_function_train(clf, kernel_train)
    norm_w = svm_weight_norm(alpha, y_train, kernel_train)
    return f_values / norm_w, alpha, norm_w


def weighted_jaccard(alpha_ref: np.ndarray, alpha_est: np.ndarray) -> float:
    """Weighted Jaccard similarity between two non-negative alpha vectors."""
    numerator = np.minimum(alpha_ref, alpha_est).sum()
    denominator = np.maximum(alpha_ref, alpha_est).sum()
    if denominator <= EPS:
        return 1.0
    return float(numerator / denominator)


def support_jaccard(
    alpha_ref: np.ndarray,
    alpha_est: np.ndarray,
    *,
    threshold: float,
) -> float:
    """Binary support-vector Jaccard similarity."""
    ref = alpha_ref > threshold
    est = alpha_est > threshold

    union = np.logical_or(ref, est).sum()
    if union == 0:
        return 1.0

    intersection = np.logical_and(ref, est).sum()
    return float(intersection / union)


def normalize_score(values: np.ndarray) -> np.ndarray:
    """Normalize a non-negative score vector to [0, 1]-like scale."""
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0.0)

    max_value = values.max()
    if not np.isfinite(max_value) or max_value <= EPS:
        return np.zeros_like(values)

    return values / max_value


def adaptive_scores(
    *,
    kernel_est: np.ndarray,
    y: np.ndarray,
    clf: SVC,
    alpha: np.ndarray,
    counts_upper: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    lambda_value: float,
) -> np.ndarray:
    """Compute adaptive allocation scores for independent kernel entries."""
    n = len(y)

    f_values = decision_function_train(clf, kernel_est)
    margin_residual = y * f_values - 1.0

    signed_alpha = alpha * y

    # Margin sensitivity for K_ij.
    sensitivity_matrix = np.abs(np.outer(signed_alpha, signed_alpha))

    # Approximate variance of the Bernoulli kernel estimator.
    # We use plug-in K_hat(1 - K_hat) / N.
    variance_matrix = kernel_est * (1.0 - kernel_est)
    counts_matrix = np.ones((n, n), dtype=float)
    counts_matrix[row_idx, col_idx] = counts_upper
    counts_matrix[col_idx, row_idx] = counts_upper
    variance_matrix = variance_matrix / np.maximum(counts_matrix, 1.0)
    np.fill_diagonal(variance_matrix, 0.0)

    # Local variance of f(x_i), using the linearized decision function.
    decision_variance = (signed_alpha[None, :] ** 2 * variance_matrix).sum(axis=1)
    decision_std = np.sqrt(np.maximum(decision_variance, EPS))

    # Probability of being on or inside the margin.
    # P_i = Phi(-Delta_i / sigma_f,i)
    p_margin = norm.cdf(-margin_residual / decision_std)

    # Entry-level active-set instability signal.
    instability_matrix = 0.5 * (p_margin[:, None] + p_margin[None, :])

    sensitivity_upper = sensitivity_matrix[row_idx, col_idx]
    instability_upper = instability_matrix[row_idx, col_idx]

    sensitivity_upper = normalize_score(sensitivity_upper)
    instability_upper = normalize_score(instability_upper)

    scores = (1.0 - lambda_value) * sensitivity_upper
    scores += lambda_value * instability_upper

    scores = np.maximum(scores, EPS)
    return scores


def evaluate_solution(
    *,
    method: str,
    lambda_value: float | None,
    seed: int,
    true_kernel: np.ndarray,
    est_kernel: np.ndarray,
    y: np.ndarray,
    true_clf: SVC,
    true_norm_decision: np.ndarray,
    true_alpha: np.ndarray,
    true_norm_w: float,
    c_value: float,
    sv_threshold: float,
) -> dict[str, float | int | str | None]:
    """Fit SVM on estimated kernel and compute metrics."""
    est_clf = fit_precomputed_svm(est_kernel, y, c_value)
    est_norm_decision, est_alpha, est_norm_w = normalized_decision_values(
        est_clf,
        est_kernel,
        y,
    )

    norm_decision_rmse = float(
        np.sqrt(np.mean((est_norm_decision - true_norm_decision) ** 2))
    )

    wj = weighted_jaccard(true_alpha, est_alpha)
    sj = support_jaccard(true_alpha, est_alpha, threshold=sv_threshold)

    true_margin = 1.0 / true_norm_w
    est_margin = 1.0 / est_norm_w
    margin_relative_error = float(abs(est_margin - true_margin) / max(true_margin, EPS))

    full_kernel_rmse = float(np.sqrt(np.mean((est_kernel - true_kernel) ** 2)))

    true_sv = true_alpha > sv_threshold
    if true_sv.sum() >= 2:
        block_true = true_kernel[np.ix_(true_sv, true_sv)]
        block_est = est_kernel[np.ix_(true_sv, true_sv)]
        sv_block_kernel_rmse = float(np.sqrt(np.mean((block_est - block_true) ** 2)))
    else:
        sv_block_kernel_rmse = np.nan

    return {
        "seed": seed,
        "method": method,
        "lambda": lambda_value,
        "normalized_decision_rmse": norm_decision_rmse,
        "weighted_jaccard": wj,
        "support_jaccard": sj,
        "margin_relative_error": margin_relative_error,
        "full_kernel_rmse": full_kernel_rmse,
        "sv_block_kernel_rmse": sv_block_kernel_rmse,
        "n_support_true": int(np.sum(true_alpha > sv_threshold)),
        "n_support_est": int(np.sum(est_alpha > sv_threshold)),
    }


def run_uniform_baseline(
    *,
    rng: np.random.Generator,
    true_upper_values: np.ndarray,
    n_train: int,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    total_budget: int,
) -> np.ndarray:
    """Estimate kernel using uniform allocation over independent entries."""
    n_entries = len(true_upper_values)

    uniform_counts = allocate_integer_budget(
        np.ones(n_entries),
        total_budget,
        min_count=0,
    )
    successes = np.zeros(n_entries, dtype=float)
    counts = np.zeros(n_entries, dtype=int)

    add_bernoulli_measurements(
        rng=rng,
        true_upper_values=true_upper_values,
        successes=successes,
        counts=counts,
        additional_counts=uniform_counts,
    )

    return estimate_kernel_from_counts(
        successes,
        counts,
        n_train,
        row_idx,
        col_idx,
    )


def run_adaptive_fixed_budget(
    *,
    rng: np.random.Generator,
    true_upper_values: np.ndarray,
    y: np.ndarray,
    n_train: int,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    total_budget: int,
    pilot_shots_per_entry: int,
    n_rounds: int,
    lambda_value: float,
    c_value: float,
) -> np.ndarray:
    """Run fixed-budget adaptive allocation for one lambda value."""
    n_entries = len(true_upper_values)
    pilot_budget = pilot_shots_per_entry * n_entries

    if pilot_budget >= total_budget:
        raise ValueError(
            "Pilot budget must be smaller than total budget. "
            f"Got pilot_budget={pilot_budget}, total_budget={total_budget}."
        )

    successes = np.zeros(n_entries, dtype=float)
    counts = np.zeros(n_entries, dtype=int)

    pilot_counts = np.full(n_entries, pilot_shots_per_entry, dtype=int)
    add_bernoulli_measurements(
        rng=rng,
        true_upper_values=true_upper_values,
        successes=successes,
        counts=counts,
        additional_counts=pilot_counts,
    )

    remaining_budget = total_budget - pilot_budget

    round_budgets = allocate_integer_budget(
        np.ones(n_rounds),
        remaining_budget,
        min_count=0,
    )

    for round_budget in round_budgets:
        if round_budget <= 0:
            continue

        kernel_est = estimate_kernel_from_counts(
            successes,
            counts,
            n_train,
            row_idx,
            col_idx,
        )

        clf = fit_precomputed_svm(kernel_est, y, c_value)
        alpha = full_alpha(clf, n_train)

        scores = adaptive_scores(
            kernel_est=kernel_est,
            y=y,
            clf=clf,
            alpha=alpha,
            counts_upper=counts,
            row_idx=row_idx,
            col_idx=col_idx,
            lambda_value=lambda_value,
        )

        additional_counts = allocate_integer_budget(
            scores,
            int(round_budget),
            min_count=0,
        )

        add_bernoulli_measurements(
            rng=rng,
            true_upper_values=true_upper_values,
            successes=successes,
            counts=counts,
            additional_counts=additional_counts,
        )

    return estimate_kernel_from_counts(
        successes,
        counts,
        n_train,
        row_idx,
        col_idx,
    )


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Create median/IQR summary for plotting later."""
    metric_cols = [
        "normalized_decision_rmse",
        "weighted_jaccard",
        "support_jaccard",
        "margin_relative_error",
        "full_kernel_rmse",
        "sv_block_kernel_rmse",
        "n_support_est",
    ]

    grouped = results.groupby(["method", "lambda"], dropna=False)

    rows = []
    for key, group in grouped:
        method, lambda_value = key
        row = {
            "method": method,
            "lambda": lambda_value,
            "n_runs": len(group),
        }

        for metric in metric_cols:
            values = group[metric].dropna().to_numpy()
            if len(values) == 0:
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_q25"] = np.nan
                row[f"{metric}_q75"] = np.nan
                row[f"{metric}_iqr"] = np.nan
                continue

            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            row[f"{metric}_median"] = median
            row[f"{metric}_q25"] = q25
            row[f"{metric}_q75"] = q75
            row[f"{metric}_iqr"] = q75 - q25

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["method", "lambda"], na_position="first")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-budget lambda ablation experiment."
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/lambda_ablation")
    )

    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--dataset-noise", type=float, default=0.20)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=None)

    parser.add_argument("--total-budget", type=int, default=120_000)
    parser.add_argument("--pilot-shots-per-entry", type=int, default=4)
    parser.add_argument("--n-rounds", type=int, default=10)

    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)

    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    parser.add_argument("--sv-threshold", type=float, default=1e-6)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for run_id in range(args.n_runs):
        seed = args.seed_start + run_id

        x, y = make_dataset(
            n_samples=args.n_samples,
            noise=args.dataset_noise,
            random_state=seed,
        )

        gamma = args.gamma
        if gamma is None:
            gamma = default_gamma(x)

        true_kernel = rbf_kernel(x, x, gamma)
        np.fill_diagonal(true_kernel, 1.0)

        n_train = len(y)
        row_idx, col_idx = independent_pairs(n_train)
        true_upper_values = true_kernel[row_idx, col_idx]

        true_clf = fit_precomputed_svm(true_kernel, y, args.c_value)
        true_norm_decision, true_alpha, true_norm_w = normalized_decision_values(
            true_clf,
            true_kernel,
            y,
        )

        # Uniform baseline, run once per seed and outside the lambda loop.
        uniform_rng = np.random.default_rng(10_000_000 + seed)

        uniform_kernel = run_uniform_baseline(
            rng=uniform_rng,
            true_upper_values=true_upper_values,
            n_train=n_train,
            row_idx=row_idx,
            col_idx=col_idx,
            total_budget=args.total_budget,
        )

        uniform_record = evaluate_solution(
            method="uniform",
            lambda_value=np.nan,
            seed=seed,
            true_kernel=true_kernel,
            est_kernel=uniform_kernel,
            y=y,
            true_clf=true_clf,
            true_norm_decision=true_norm_decision,
            true_alpha=true_alpha,
            true_norm_w=true_norm_w,
            c_value=args.c_value,
            sv_threshold=args.sv_threshold,
        )
        uniform_record["gamma"] = gamma
        uniform_record["total_budget"] = args.total_budget
        uniform_record["pilot_shots_per_entry"] = np.nan
        uniform_record["n_rounds"] = np.nan
        records.append(uniform_record)

        print(
            f"[seed={seed:04d}] uniform "
            f"norm_decision_rmse={uniform_record['normalized_decision_rmse']:.6g} "
            f"weighted_jaccard={uniform_record['weighted_jaccard']:.6g}"
        )

        # Adaptive lambda sweep.
        for lambda_value in args.lambdas:
            adaptive_rng = np.random.default_rng(
                20_000_000 + 1000 * seed + int(round(lambda_value * 1000))
            )

            adaptive_kernel = run_adaptive_fixed_budget(
                rng=adaptive_rng,
                true_upper_values=true_upper_values,
                y=y,
                n_train=n_train,
                row_idx=row_idx,
                col_idx=col_idx,
                total_budget=args.total_budget,
                pilot_shots_per_entry=args.pilot_shots_per_entry,
                n_rounds=args.n_rounds,
                lambda_value=float(lambda_value),
                c_value=args.c_value,
            )

            adaptive_record = evaluate_solution(
                method="adaptive",
                lambda_value=float(lambda_value),
                seed=seed,
                true_kernel=true_kernel,
                est_kernel=adaptive_kernel,
                y=y,
                true_clf=true_clf,
                true_norm_decision=true_norm_decision,
                true_alpha=true_alpha,
                true_norm_w=true_norm_w,
                c_value=args.c_value,
                sv_threshold=args.sv_threshold,
            )
            adaptive_record["gamma"] = gamma
            adaptive_record["total_budget"] = args.total_budget
            adaptive_record["pilot_shots_per_entry"] = args.pilot_shots_per_entry
            adaptive_record["n_rounds"] = args.n_rounds
            records.append(adaptive_record)

            print(
                f"[seed={seed:04d}] lambda={lambda_value:.2f} "
                f"norm_decision_rmse="
                f"{adaptive_record['normalized_decision_rmse']:.6g} "
                f"weighted_jaccard={adaptive_record['weighted_jaccard']:.6g}"
            )

    raw = pd.DataFrame(records)

    raw_path = args.output_dir / "lambda_ablation_raw.csv"
    summary_path = args.output_dir / "lambda_ablation_summary.csv"

    raw.to_csv(raw_path, index=False)

    summary = summarize_results(raw)
    summary.to_csv(summary_path, index=False)

    print()
    print(f"Wrote raw results to: {raw_path}")
    print(f"Wrote summary results to: {summary_path}")
    print()
    print("Summary:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(
            summary[
                [
                    "method",
                    "lambda",
                    "n_runs",
                    "normalized_decision_rmse_median",
                    "normalized_decision_rmse_q25",
                    "normalized_decision_rmse_q75",
                    "weighted_jaccard_median",
                    "weighted_jaccard_q25",
                    "weighted_jaccard_q75",
                ]
            ]
        )


if __name__ == "__main__":
    main()
