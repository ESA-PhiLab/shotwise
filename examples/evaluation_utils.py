"""Evaluation and plotting utilities for ShotWise example scripts.

This module contains lightweight helpers for reproducing paper-style
synthetic experiments with the refactored ShotWise package. The functions
here are intended for example/benchmark scripts rather than the minimal
package core.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shotwise.learning.svm import fit_precomputed_svm_state


def margin_from_signed_dual(
    kernel_matrix: np.ndarray, signed_dual: np.ndarray
) -> float:
    """Return the SVM weight norm ``||w||`` from a dense signed dual vector.

    Parameters
    ----------
    kernel_matrix
        Full training Gram matrix.
    signed_dual
        Dense vector ``y_i * alpha_i`` of length ``n``.

    Returns
    -------
    float
        The norm ``||w|| = sqrt((y*alpha)^T K (y*alpha))``.
    """
    kernel = np.asarray(kernel_matrix, dtype=float)
    ay = np.asarray(signed_dual, dtype=float).reshape(-1)
    value = float(ay @ (kernel @ ay))
    return float(np.sqrt(max(value, 0.0)))


def decision_values_from_signed_dual(
    kernel_matrix: np.ndarray,
    signed_dual: np.ndarray,
    bias: float,
) -> np.ndarray:
    """Return training decision values ``f = K @ (y*alpha) + b``."""
    kernel = np.asarray(kernel_matrix, dtype=float)
    ay = np.asarray(signed_dual, dtype=float).reshape(-1)
    return (kernel @ ay) + float(bias)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the root-mean-square error between two arrays."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    diff = x - y
    return float(np.sqrt(np.mean(diff * diff)))


def rmse_block(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> float:
    """Compute RMSE on the square block indexed by ``idx``.

    Returns ``np.nan`` if ``idx`` is empty.
    """
    indices = np.asarray(idx, dtype=int).reshape(-1)
    if indices.size == 0:
        return float("nan")
    block_a = np.asarray(a, dtype=float)[np.ix_(indices, indices)]
    block_b = np.asarray(b, dtype=float)[np.ix_(indices, indices)]
    return rmse(block_a, block_b)


def jaccard_index(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Jaccard index between two index arrays."""
    set_a = set(np.asarray(a, dtype=int).tolist())
    set_b = set(np.asarray(b, dtype=int).tolist())
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return float(len(set_a & set_b) / union) if union > 0 else 0.0


def weighted_jaccard_from_signed_duals(
    signed_dual_true: np.ndarray,
    signed_dual_est: np.ndarray,
) -> float:
    """Compute the weighted Jaccard similarity of dense signed dual vectors."""
    a = np.abs(np.asarray(signed_dual_true, dtype=float))
    b = np.abs(np.asarray(signed_dual_est, dtype=float))
    denom = np.maximum(a, b).sum()
    if denom == 0.0:
        return 1.0
    return float(np.minimum(a, b).sum() / denom)


def accuracy_against_true_classifier(
    decision_est: np.ndarray,
    decision_true: np.ndarray,
) -> float:
    """Compare sign predictions of an estimated and a true classifier."""
    pred_est = np.sign(np.asarray(decision_est, dtype=float))
    pred_true = np.sign(np.asarray(decision_true, dtype=float))
    return float(np.mean(pred_est == pred_true))


def evaluate_estimated_kernel(
    kernel_true: np.ndarray,
    kernel_est: np.ndarray,
    y: np.ndarray,
    *,
    c: float,
    method: str,
    round_id: int,
) -> dict[str, float | int | str]:
    """Evaluate an estimated kernel against the true-kernel SVM reference.

    The returned record is designed to plug directly into a benchmark
    ``pandas.DataFrame``.
    """
    state_true = fit_precomputed_svm_state(kernel_true, y, c=c)
    state_est = fit_precomputed_svm_state(kernel_est, y, c=c)

    norm_true = margin_from_signed_dual(kernel_true, state_true.signed_dual)
    norm_est = margin_from_signed_dual(kernel_est, state_est.signed_dual)
    decision_true = decision_values_from_signed_dual(
        kernel_true,
        state_true.signed_dual,
        state_true.bias,
    )
    decision_est = decision_values_from_signed_dual(
        kernel_est,
        state_est.signed_dual,
        state_est.bias,
    )

    record: dict[str, float | int | str] = {
        "method": method,
        "round": int(round_id),
        "rmse_K": rmse(kernel_est, kernel_true),
        "rmse_K_SV": rmse_block(
            kernel_est,
            kernel_true,
            state_true.support_indices,
        ),
        "jaccard": jaccard_index(
            state_true.support_indices,
            state_est.support_indices,
        ),
        "jaccard_weighted": weighted_jaccard_from_signed_duals(
            state_true.signed_dual,
            state_est.signed_dual,
        ),
        "margin_err": abs(norm_est - norm_true),
        "func_rmse": rmse(decision_est, decision_true),
        "acc_train_vs_true": accuracy_against_true_classifier(
            decision_est,
            decision_true,
        ),
        "sv_ratio_true": state_true.num_support / state_true.n_samples,
        "sv_ratio_est": state_est.num_support / state_est.n_samples,
        "num_support_true": state_true.num_support,
        "num_support_est": state_est.num_support,
    }
    return record


def records_to_dataframe(records: list[dict[str, float | int | str]]) -> pd.DataFrame:
    """Convert a list of experiment records into a tidy DataFrame."""
    return pd.DataFrame.from_records(records)


def print_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print a grouped mean/std summary for the most relevant metrics.

    Returns the summary DataFrame so callers can reuse it.
    """
    metrics = [
        "rmse_K",
        "rmse_K_SV",
        "jaccard",
        "jaccard_weighted",
        "margin_err",
        "func_rmse",
        "acc_train_vs_true",
        "sv_ratio_est",
    ]
    summary = df.groupby("method")[metrics].agg(["mean", "std"])
    print(summary)
    return summary


def boxplot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str | None = None,
    yscale: str = "log",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> None:
    """Create and optionally save a seaborn boxplot for one metric."""
    if metric not in df.columns:
        raise ValueError(f"Metric {metric!r} not found in DataFrame columns.")

    label = metric if metric_label is None else metric_label
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x="method", y=metric, ax=ax)
    ax.set_xlabel("Method")
    ax.set_ylabel(label)
    if yscale:
        ax.set_yscale(yscale)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def margin_scatterplot(
    df: pd.DataFrame,
    *,
    x_metric: str = "rmse_K",
    y_metric: str = "margin_err",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
) -> None:
    """Create a scatterplot relating kernel and margin errors by method."""
    for metric in (x_metric, y_metric, "method"):
        if metric not in df.columns:
            raise ValueError(f"Column {metric!r} not found in DataFrame.")

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=x_metric, y=y_metric, hue="method", ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
