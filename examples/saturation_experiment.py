from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluation_utils import (
    evaluate_estimated_kernel,
)
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel

from shotwise.algorithms.supportshot import SupportShot
from shotwise.backends.synthetic import SyntheticBernoulliBackend

"""
Saturation experiment using ShotWise (based on blob_benchmark structure).

Tracks sklearn.datasets import make_blobsTracks:
from sklearn.metrics.pairwise import rbf_kernel

from shotwise.algorithms.supportshot import SupportShot
from shotwise.backends.synthetic import SyntheticBernoulliBackend

from evaluation_utils import evaluate_estimated_kernel
"""

# ============================================================
# Config (same style as your benchmark)
# ============================================================

EXPERIMENT_ROUNDS = 500
N_SAMPLES = 50
GAMMA = 1.0
C = 100
SHOTS_PER_ENTRY = 200
M0 = 50
ROUNDS = 50
CLUSTER_STD = 0.2
LAMBDA_WEIGHT = 0.5

OUTPUT_PATH = Path("plots/saturation.pdf")


# ============================================================
# Helpers
# ============================================================


def generate_blobs(n_samples, cluster_std, seed):
    x, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        cluster_std=cluster_std,
        random_state=seed,
    )
    y = np.where(y == 0, -1, 1)
    return x, y


def total_budget(n_samples, shots_per_entry):
    n_pairs = n_samples * (n_samples - 1) // 2
    return n_pairs * shots_per_entry


# ============================================================
# Main experiment
# ============================================================


def run_experiment():
    rng = np.random.default_rng(0)
    N_tot = total_budget(N_SAMPLES, SHOTS_PER_ENTRY)

    records = []

    for run in range(EXPERIMENT_ROUNDS):
        dataset_seed = int(rng.integers(0, 2**32 - 1))
        adaptive_seed = int(rng.integers(0, 2**32 - 1))

        print(f"Run {run + 1}/{EXPERIMENT_ROUNDS}")

        X, y = generate_blobs(N_SAMPLES, CLUSTER_STD, dataset_seed)
        K_true = rbf_kernel(X, X, gamma=GAMMA)

        backend = SyntheticBernoulliBackend(K_true, random_state=adaptive_seed)

        model = SupportShot(
            total_budget=N_tot,
            pilot_shots_per_entry=M0,
            rounds=ROUNDS,
            lambda_weight=LAMBDA_WEIGHT,
            svm_c=C,
            random_state=adaptive_seed,
        )

        model.fit(y, backend)

        # -----------------------------------
        # Collect per-round results
        # -----------------------------------
        for rec in model.history_:
            metrics = evaluate_estimated_kernel(
                K_true,
                rec.kernel_matrix,
                y,
                c=C,
                method=f"round_{rec.round_index}",
                round_id=rec.round_index,
            )

            metrics.update(
                {
                    "run": run,
                    "round": rec.round_index,
                    "func_rmse": metrics["func_rmse"],
                    "dual_change": getattr(rec, "dual_change", np.nan),
                }
            )

            records.append(metrics)

    return pd.DataFrame(records)


# ============================================================
# Analysis + plot
# ============================================================


def plot_saturation(df: pd.DataFrame):

    df = df[df["round"] >= 0]

    grouped = df.groupby("round")

    median = grouped["func_rmse"].median()
    q25 = grouped["func_rmse"].quantile(0.25)
    q75 = grouped["func_rmse"].quantile(0.75)

    dual = grouped["dual_change"].median()

    rounds = median.index

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # RMSE
    ax1.plot(rounds, median, marker="o", color="C0", label="median RMSE")
    ax1.fill_between(rounds, q25, q75, alpha=0.2, color="C0", label="IQR")

    ax1.set_yscale("log")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Decision-function RMSE")

    # shading
    ax1.axvspan(-0.5, 10.5, color="C7", alpha=0.15)
    ax1.axvspan(10.5, 50.5, color="C8", alpha=0.15)

    # dual stability
    ax2 = ax1.twinx()
    ax2.plot(rounds[1:], dual[1:], "--s", color="C1", label="dual change")
    ax2.set_yscale("log")
    ax2.set_ylabel("Dual stability")

    # legend
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2)

    plt.title("Saturation of Adaptive Measurement")
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    plt.show()

    print(f"Saved plot: {OUTPUT_PATH}")


# ============================================================
# Main
# ============================================================


def main():
    df = run_experiment()
    print(df["dual_change"].describe())
    print(df["dual_change"].isna().sum(), "NaNs")
    plot_saturation(df)


if __name__ == "__main__":
    main()
