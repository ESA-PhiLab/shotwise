"""ShotWise quickstart: adaptive measurement allocation with SupportShot.

This example demonstrates the basic ShotWise workflow on a small synthetic
kernel-learning problem.

It compares:

1. Uniform measurement allocation
2. SupportShot adaptive measurement allocation

The example uses a synthetic Bernoulli measurement backend. No Qiskit
installation is required.

Run
---
python examples/quickstart_supportshot.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from shotwise import SupportShot


class BernoulliKernelBackend:
    """Synthetic backend for Bernoulli measurements of kernel entries.

    Given a true kernel matrix K, this backend simulates repeated measurements

        X_ij ~ Bernoulli(K_ij)

    and returns the number of successes for each requested kernel entry.
    """

    def __init__(self, kernel_true: np.ndarray, *, seed: int = 0) -> None:
        self.kernel_true = np.asarray(kernel_true, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.n = self.kernel_true.shape[0]

    def sample_upper_triangle_counts(
        self,
        shot_matrix: np.ndarray,
        *,
        round_id: int | None = None,
    ) -> np.ndarray:
        """Sample counts for upper-triangular kernel entries."""
        del round_id

        shot_matrix = np.asarray(shot_matrix, dtype=int)
        counts = np.zeros_like(shot_matrix, dtype=int)

        rows, cols = np.triu_indices(self.n, k=1)
        for i, j in zip(rows, cols, strict=True):
            shots = int(shot_matrix[i, j])
            if shots <= 0:
                continue

            probability = float(np.clip(self.kernel_true[i, j], 0.0, 1.0))
            counts[i, j] = self.rng.binomial(shots, probability)

        return counts


def make_toy_problem(
    *,
    n_samples: int = 24,
    cluster_std: float = 0.45,
    gamma: float = 0.7,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a simple binary dataset and its exact RBF kernel."""
    x, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        cluster_std=cluster_std,
        random_state=seed,
    )
    y = np.where(y == 0, -1, 1)

    kernel_true = rbf_kernel(x, x, gamma=gamma)
    np.fill_diagonal(kernel_true, 1.0)

    return x, y, kernel_true


def uniform_allocation_matrix(n: int, total_budget: int) -> np.ndarray:
    """Allocate shots uniformly over independent off-diagonal entries."""
    rows, cols = np.triu_indices(n, k=1)
    n_pairs = len(rows)

    base = total_budget // n_pairs
    remainder = total_budget % n_pairs

    shots = np.full(n_pairs, base, dtype=int)
    if remainder > 0:
        shots[:remainder] += 1

    allocation = np.zeros((n, n), dtype=int)
    allocation[rows, cols] = shots
    return allocation


def kernel_from_counts(
    counts: np.ndarray,
    allocation: np.ndarray,
) -> np.ndarray:
    """Convert Bernoulli success counts into an empirical kernel matrix."""
    n = counts.shape[0]
    kernel = np.eye(n, dtype=float)

    rows, cols = np.triu_indices(n, k=1)
    for i, j in zip(rows, cols, strict=True):
        shots = allocation[i, j]
        value = counts[i, j] / shots if shots > 0 else 0.0
        kernel[i, j] = value
        kernel[j, i] = value

    return kernel


def dense_dual_from_svc(clf: SVC, n_samples: int) -> np.ndarray:
    """Return the dense signed dual vector q_i = alpha_i y_i."""
    q = np.zeros(n_samples, dtype=float)
    q[clf.support_] = clf.dual_coef_[0]
    return q


def decision_values(
    kernel: np.ndarray,
    q: np.ndarray,
    bias: float,
) -> np.ndarray:
    """Evaluate SVM decision values on the training kernel."""
    return kernel @ q + bias


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean-square error."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(diff * diff)))


def weighted_jaccard_from_duals(q_true: np.ndarray, q_est: np.ndarray) -> float:
    """Weighted overlap between true and estimated signed dual coefficients."""
    a = np.abs(q_true)
    b = np.abs(q_est)

    denominator = np.maximum(a, b).sum()
    if denominator <= 0:
        return 1.0

    return float(np.minimum(a, b).sum() / denominator)


def evaluate_kernel_estimate(
    *,
    kernel_est: np.ndarray,
    kernel_true: np.ndarray,
    y: np.ndarray,
    c: float,
    q_true: np.ndarray,
    bias_true: float,
    f_true: np.ndarray,
) -> dict[str, float]:
    """Evaluate an estimated kernel against the exact-kernel SVM."""
    del bias_true

    clf = SVC(kernel="precomputed", C=c)
    clf.fit(kernel_est, y)

    q_est = dense_dual_from_svc(clf, len(y))
    bias_est = float(clf.intercept_[0])

    f_est_on_true_kernel = decision_values(kernel_true, q_est, bias_est)

    return {
        "rmse_K": rmse(kernel_est, kernel_true),
        "func_rmse": rmse(f_est_on_true_kernel, f_true),
        "jaccard_weighted": weighted_jaccard_from_duals(q_true, q_est),
        "n_support": float(len(clf.support_)),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    """Pretty-print a compact metrics dictionary."""
    print(title)
    print("-" * len(title))
    print(f"Global kernel RMSE:        {metrics['rmse_K']:.4f}")
    print(f"Decision-function RMSE:   {metrics['func_rmse']:.4f}")
    print(f"Weighted support overlap: {metrics['jaccard_weighted']:.4f}")
    print(f"Estimated support count:  {metrics['n_support']:.0f}")
    print()


def main() -> None:
    seed = 7
    n_samples = 24
    gamma = 0.7
    c = 10.0

    shots_per_entry = 25
    pilot_shots = 4
    rounds = 3
    lambda_weight = 0.5

    _, y, kernel_true = make_toy_problem(
        n_samples=n_samples,
        gamma=gamma,
        seed=seed,
    )

    n_pairs = n_samples * (n_samples - 1) // 2
    total_budget = n_pairs * shots_per_entry

    print()
    print("ShotWise quickstart")
    print("===================")
    print(f"Samples:              {n_samples}")
    print(f"Independent entries:  {n_pairs}")
    print(f"Total budget:         {total_budget} measurements")
    print(f"Pilot shots/entry:    {pilot_shots}")
    print(f"Adaptive rounds:      {rounds}")
    print()

    # Exact reference SVM using the noise-free kernel.
    clf_true = SVC(kernel="precomputed", C=c)
    clf_true.fit(kernel_true, y)

    q_true = dense_dual_from_svc(clf_true, n_samples)
    bias_true = float(clf_true.intercept_[0])
    f_true = decision_values(kernel_true, q_true, bias_true)

    print("Reference exact-kernel SVM")
    print("--------------------------")
    print(f"True support count: {len(clf_true.support_)}")
    print()

    # Uniform baseline.
    uniform_backend = BernoulliKernelBackend(kernel_true, seed=seed + 1)
    uniform_allocation = uniform_allocation_matrix(n_samples, total_budget)
    uniform_counts = uniform_backend.sample_upper_triangle_counts(
        uniform_allocation,
        round_id=-1,
    )
    kernel_uniform = kernel_from_counts(uniform_counts, uniform_allocation)

    uniform_metrics = evaluate_kernel_estimate(
        kernel_est=kernel_uniform,
        kernel_true=kernel_true,
        y=y,
        c=c,
        q_true=q_true,
        bias_true=bias_true,
        f_true=f_true,
    )

    # Adaptive SupportShot.
    adaptive_backend = BernoulliKernelBackend(kernel_true, seed=seed + 2)

    model = SupportShot(
        total_budget=total_budget,
        pilot_shots_per_entry=pilot_shots,
        rounds=rounds,
        lambda_weight=lambda_weight,
        svm_c=c,
        allocation_method="deterministic",
        random_state=seed,
    )
    model.fit(y, adaptive_backend)

    final_kernel_adaptive = np.asarray(model.history_[-1].kernel_matrix)
    shots_spent_adaptive = int(model.history_[-1].total_shots_spent)

    adaptive_metrics = evaluate_kernel_estimate(
        kernel_est=final_kernel_adaptive,
        kernel_true=kernel_true,
        y=y,
        c=c,
        q_true=q_true,
        bias_true=bias_true,
        f_true=f_true,
    )

    print_metrics("Uniform allocation", uniform_metrics)
    print_metrics("SupportShot adaptive allocation", adaptive_metrics)

    improvement = (
        uniform_metrics["func_rmse"] - adaptive_metrics["func_rmse"]
    ) / uniform_metrics["func_rmse"]

    print("Summary")
    print("-------")
    print(f"Adaptive shots used: {shots_spent_adaptive} / {total_budget}")
    print(f"Relative shot use:   {shots_spent_adaptive / total_budget:.2%}")
    print(f"Function RMSE gain:  {100.0 * improvement:.1f}%")
    print()

    if improvement > 0:
        print("In this run, SupportShot improved the decision-function RMSE.")
    else:
        print("In this run, uniform allocation had lower decision-function RMSE.")

    print()
    print("Note:")
    print("Uniform allocation often gives lower global kernel RMSE.")
    print("SupportShot targets classifier-relevant quantities instead.")


if __name__ == "__main__":
    main()
