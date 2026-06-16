"""SVM fitting utilities for ShotWise.

This module provides a thin, well-typed wrapper around scikit-learn's
precomputed-kernel SVM so that the rest of the package can work with a stable,
explicit representation of the training state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVC

from shotwise.utils.matrix import check_square_matrix


@dataclass(frozen=True)
class SVMState:
    """State of a binary SVM trained on a precomputed kernel matrix.

    Parameters
    ----------
    signed_dual
        Full-length vector of signed dual coefficients. For support vectors,
        this stores ``y_i * alpha_i`` as returned by scikit-learn; all other
        entries are zero.
    alphas
        Full-length vector of unsigned dual coefficients ``alpha_i``.
    support_indices
        Indices of the support vectors.
    support_mask
        Boolean mask selecting support vectors.
    decision_values
        Training-time decision-function values ``f(x_i)`` evaluated on the
        kernel matrix used for fitting.
    margin_residuals
        Per-sample residuals ``y_i * f(x_i) - 1``. These are useful for
        identifying near-margin samples and active-set instability.
    bias
        Intercept term of the trained classifier.
    labels
        Training labels in ``{-1, +1}``.
    c
        Regularization parameter used during training.
    """

    signed_dual: np.ndarray
    alphas: np.ndarray
    support_indices: np.ndarray
    support_mask: np.ndarray
    decision_values: np.ndarray
    margin_residuals: np.ndarray
    bias: float
    labels: np.ndarray
    c: float

    @property
    def n_samples(self) -> int:
        """Return the number of training samples."""
        return int(self.labels.shape[0])

    @property
    def num_support(self) -> int:
        """Return the number of support vectors."""
        return int(self.support_indices.size)

    @property
    def active_mask(self) -> np.ndarray:
        """Alias for ``support_mask`` kept for readability in downstream code."""
        return self.support_mask


def validate_binary_labels(y: np.ndarray) -> np.ndarray:
    """Validate and normalize binary labels.

    Parameters
    ----------
    y
        One-dimensional array of binary labels encoded as ``-1`` and ``+1``.

    Returns
    -------
    np.ndarray
        Normalized integer label array.

    Raises
    ------
    ValueError
        If the labels are not one-dimensional or are not exactly ``{-1, +1}``.
    """
    labels = np.asarray(y)

    if labels.ndim != 1:
        raise ValueError(f"y must be a 1D array, got shape={labels.shape}.")

    if labels.size == 0:
        raise ValueError("y must not be empty.")

    unique = np.unique(labels)
    if not np.array_equal(unique, np.array([-1, 1])):
        raise ValueError(
            "y must contain exactly the binary labels -1 and +1. "
            f"Got unique values {unique.tolist()}."
        )

    return labels.astype(int, copy=False)


def fit_precomputed_svm_state(
    kernel_matrix: np.ndarray,
    y: np.ndarray,
    *,
    c: float = 1.0,
) -> SVMState:
    """Fit a binary SVM on a precomputed kernel matrix and return its state.

    Parameters
    ----------
    kernel_matrix
        Symmetric training kernel matrix of shape ``(n, n)``.
    y
        Binary training labels encoded as ``-1`` and ``+1``.
    c
        Regularization parameter passed to ``sklearn.svm.SVC``.

    Returns
    -------
    SVMState
        Structured representation of the fitted SVM state.

    Raises
    ------
    ValueError
        If the kernel matrix or labels are invalid, or if ``c`` is not
        strictly positive.
    """
    check_square_matrix(np.asarray(kernel_matrix), name="kernel_matrix")
    labels = validate_binary_labels(y)
    kernel = np.asarray(kernel_matrix, dtype=float)

    if kernel.shape[0] != labels.shape[0]:
        raise ValueError(
            "kernel_matrix and y must have matching leading dimension. "
            f"Got kernel shape {kernel.shape} and y shape {labels.shape}."
        )

    if c <= 0:
        raise ValueError(f"c must be strictly positive, got {c}.")

    clf = SVC(kernel="precomputed", C=float(c))
    clf.fit(kernel, labels)

    support_indices = clf.support_.astype(int, copy=False)
    support_mask = np.zeros(labels.shape[0], dtype=bool)
    support_mask[support_indices] = True

    signed_dual = np.zeros(labels.shape[0], dtype=float)
    signed_dual[support_indices] = clf.dual_coef_.ravel()

    # scikit-learn stores signed coefficients y_i * alpha_i for support vectors.
    # Since y_i in {-1, +1}, multiplying by y_i recovers alpha_i >= 0.
    alphas = signed_dual * labels

    decision_values = np.asarray(clf.decision_function(kernel), dtype=float).reshape(-1)
    margin_residuals = labels * decision_values - 1.0
    bias = float(np.asarray(clf.intercept_).ravel()[0])

    return SVMState(
        signed_dual=signed_dual,
        alphas=alphas,
        support_indices=support_indices,
        support_mask=support_mask,
        decision_values=decision_values,
        margin_residuals=margin_residuals,
        bias=bias,
        labels=labels.copy(),
        c=float(c),
    )
