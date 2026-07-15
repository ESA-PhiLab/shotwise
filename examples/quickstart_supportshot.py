"""ShotWise quickstart: SupportShot on a synthetic noisy kernel.

This example demonstrates the minimal ShotWise workflow:

1. Build a small binary classification dataset.
2. Compute the exact RBF kernel.
3. Create a synthetic Bernoulli measurement backend.
4. Run SupportShot adaptive allocation.
5. Compare against a uniform measurement baseline.

No Qiskit installation is required.

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
from shotwise.backends import SyntheticBernoulliBackend
from shotwise.storage.kernel_store import KernelStore
from shotwise.utils.matrix import upper_triangle_indices


def make_problem(
    *,
    n_samples: int,
    cluster_std: float,
    gamma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a small binary dataset and its exact RBF kernel."""
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


def uniform_allocation(n_samples: int, total_budget: int) -> np.ndarray:
    """Uniformly allocate shots over independent off-diagonal kernel entries."""
    rows, cols = upper_triangle_indices(n_samples)
    n_pairs = len(rows)

    base = total_budget // n_pairs
    remainder = total_budget % n_pairs

    shots = np.full(n_pairs, base, dtype=int)
    shots[:remainder] += 1

    allocation = np.zeros((n_samples, n_samples), dtype=int)
    allocation[rows, cols] = shots
    return allocation


def estimate_uniform_kernel(
    backend: SyntheticBernoulliBackend,
    *,
    total_budget: int,
) -> tuple[np.ndarray, int]:
    """Estimate a kernel with uniform shot allocation."""
    allocation = uniform_allocation(backend.n, total_budget)
    counts = backend.sample_upper_triangle_counts(allocation)

    store = KernelStore(backend.n, diagonal_value=1.0)
    store.update(counts, allocation)

    return store.kernel_matrix(), store.total_shots_spent()


def signed_dual_from_svc(clf: SVC, n_samples: int) -> np.ndarray:
    """Return dense signed dual coefficients q_i = alpha_i y_i."""
    q = np.zeros(n_samples, dtype=float)
    q[clf.support_] = clf.dual_coef_[0]
    return q


def decision_values(kernel: np.ndarray, q: np.ndarray, bias: float) -> np.ndarray:
    """Evaluate SVM decision values on the training points."""
    return kernel @ q + bias


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean-square error."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(diff * diff)))


def weighted_dual_overlap(q_true: np.ndarray, q_est: np.ndarray) -> float:
    """Weighted overlap between reference and estimated dual coefficients."""
    a = np.abs(q_true)
    b = np.abs(q_est)

    denominator = np.maximum(a, b).sum()
    if denominator <= 0:
        return 1.0

    return float(np.minimum(a, b).sum() / denominator)


def evaluate_classifier_from_kernel(
    *,
    kernel_est: np.ndarray,
    kernel_true: np.ndarray,
    y: np.ndarray,
    c: float,
    q_true: np.ndarray,
    f_true: np.ndarray,
) -> dict[str, float]:
    """Evaluate the SVM induced by an estimated kernel."""
    clf = SVC(kernel="precomputed", C=c)
    clf.fit(kernel_est, y)

    q_est = signed_dual_from_svc(clf, len(y))
    bias_est = float(clf.intercept_[0])
    f_est_on_true_kernel = decision_values(kernel_true, q_est, bias_est)

    return {
        "kernel_rmse": rmse(kernel_est, kernel_true),
        "function_rmse": rmse(f_est_on_true_kernel, f_true),
        "weighted_dual_overlap": weighted_dual_overlap(q_true, q_est),
        "n_support": float(len(clf.support_)),
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    """Print a compact metric summary."""
    print(name)
    print("-" * len(name))
    print(f"Global kernel RMSE:        {metrics['kernel_rmse']:.4f}")
    print(f"Decision-function RMSE:   {metrics['function_rmse']:.4f}")
    print(f"Weighted dual overlap:    {metrics['weighted_dual_overlap']:.4f}")
    print(f"Estimated support count:  {metrics['n_support']:.0f}")
    print()


def main() -> None:
    seed = 7

    n_samples = 24
    cluster_std = 0.45
    gamma = 0.7
    c = 10.0

    shots_per_entry = 25
    pilot_shots = 4
    rounds = 3
    lambda_weight = 0.5

    _, y, kernel_true = make_problem(
        n_samples=n_samples,
        cluster_std=cluster_std,
        gamma=gamma,
        seed=seed,
    )

    n_pairs = n_samples * (n_samples - 1) // 2
    total_budget = n_pairs * shots_per_entry

    print()
    print("ShotWise quickstart")
    print("===================")
    print(f"Samples:             {n_samples}")
    print(f"Kernel entries:      {n_pairs}")
    print(f"Total budget:        {total_budget} measurements")
    print(f"Pilot shots/entry:   {pilot_shots}")
    print(f"Adaptive rounds:     {rounds}")
    print()

    # Reference exact-kernel SVM.
    clf_true = SVC(kernel="precomputed", C=c)
    clf_true.fit(kernel_true, y)

    q_true = signed_dual_from_svc(clf_true, n_samples)
    bias_true = float(clf_true.intercept_[0])
    f_true = decision_values(kernel_true, q_true, bias_true)

    print("Reference exact-kernel SVM")
    print("--------------------------")
    print(f"True support count: {len(clf_true.support_)}")
    print()

    # Uniform baseline.
    uniform_backend = SyntheticBernoulliBackend(
        kernel_true,
        random_state=seed + 1,
    )
    kernel_uniform, shots_uniform = estimate_uniform_kernel(
        uniform_backend,
        total_budget=total_budget,
    )
    uniform_metrics = evaluate_classifier_from_kernel(
        kernel_est=kernel_uniform,
        kernel_true=kernel_true,
        y=y,
        c=c,
        q_true=q_true,
        f_true=f_true,
    )

    # Adaptive SupportShot.
    adaptive_backend = SyntheticBernoulliBackend(
        kernel_true,
        random_state=seed + 2,
    )

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

    kernel_adaptive = np.asarray(model.history_[-1].kernel_matrix)
    shots_adaptive = int(model.history_[-1].total_shots_spent)

    adaptive_metrics = evaluate_classifier_from_kernel(
        kernel_est=kernel_adaptive,
        kernel_true=kernel_true,
        y=y,
        c=c,
        q_true=q_true,
        f_true=f_true,
    )

    print_metrics("Uniform allocation", uniform_metrics)
    print_metrics("SupportShot adaptive allocation", adaptive_metrics)

    if uniform_metrics["function_rmse"] > 0:
        improvement = (
            uniform_metrics["function_rmse"] - adaptive_metrics["function_rmse"]
        ) / uniform_metrics["function_rmse"]
    else:
        improvement = np.nan

    print("Summary")
    print("-------")
    print(f"Uniform shots used:  {shots_uniform} / {total_budget}")
    print(f"Adaptive shots used: {shots_adaptive} / {total_budget}")
    print(f"Relative shot use:   {shots_adaptive / total_budget:.2%}")
    print(f"Function RMSE gain:  {100.0 * improvement:.1f}%")
    print()
    print("Remark")
    print("------")
    print("Uniform allocation often estimates the full kernel more evenly.")
    print("SupportShot targets classifier-relevant entries of the kernel.")


if __name__ == "__main__":
    main()
