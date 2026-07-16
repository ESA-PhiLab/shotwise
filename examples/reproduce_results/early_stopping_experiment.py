"""Early-stopping experiment for SupportShot.

This script recreates the early-stopping experiment from the original prototype
using the ShotWise package.

It sweeps the stopping tolerance epsilon and compares the final adaptive model
against a uniform allocation baseline under the same total measurement budget.

Outputs
-------
outputs/early_stopping/
    early_stopping_raw.csv
    early_stopping_summary.csv
    early_stopping.pdf
    early_stopping.png
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from shotwise.algorithms.supportshot import SupportShot


class BernoulliKernelBackend:
    """Synthetic backend that samples upper-triangular Bernoulli kernel entries."""

    def __init__(self, kernel_true: np.ndarray, rng: np.random.Generator) -> None:
        self.kernel_true = np.asarray(kernel_true, dtype=float)
        self.rng = rng
        self.n = self.kernel_true.shape[0]

    def sample_upper_triangle_counts(
        self,
        shot_matrix: np.ndarray,
        *,
        round_id: int | None = None,
    ) -> np.ndarray:
        del round_id

        shot_matrix = np.asarray(shot_matrix, dtype=int)
        if shot_matrix.shape != (self.n, self.n):
            raise ValueError(
                f"shot_matrix must have shape {(self.n, self.n)}, "
                f"got {shot_matrix.shape}."
            )

        counts = np.zeros_like(shot_matrix, dtype=int)
        rows, cols = np.triu_indices(self.n, k=1)

        for i, j in zip(rows, cols, strict=True):
            shots = int(shot_matrix[i, j])
            if shots <= 0:
                continue

            p = float(np.clip(self.kernel_true[i, j], 0.0, 1.0))
            counts[i, j] = self.rng.binomial(shots, p)

        return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SupportShot early-stopping experiment.",
    )
    parser.add_argument("--experiment-rounds", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=100.0)
    parser.add_argument("--shots-per-entry", type=int, default=100)
    parser.add_argument("--pilot-shots", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--cluster-std", type=float, default=0.2)
    parser.add_argument("--lambda-weight", type=float, default=0.2)
    parser.add_argument("--epsilon-min-exp", type=float, default=-3.5)
    parser.add_argument("--epsilon-max-exp", type=float, default=0.5)
    parser.add_argument("--epsilon-count", type=int, default=30)
    parser.add_argument("--allocation-method", default="deterministic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/early_stopping"),
    )
    return parser.parse_args()


def generate_blobs(
    n_samples: int,
    *,
    cluster_std: float = 0.2,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    y = np.where(y == 0, -1, 1)
    return x, y


def upper_triangle_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def uniform_allocation_matrix(n: int, total_budget: int) -> np.ndarray:
    rows, cols = upper_triangle_indices(n)
    n_pairs = rows.size

    base = total_budget // n_pairs
    remainder = total_budget % n_pairs

    flat = np.full(n_pairs, base, dtype=int)
    if remainder > 0:
        flat[:remainder] += 1

    allocation = np.zeros((n, n), dtype=int)
    allocation[rows, cols] = flat
    return allocation


def kernel_from_counts(
    counts: np.ndarray,
    allocation: np.ndarray,
    *,
    diagonal_value: float = 1.0,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    allocation = np.asarray(allocation, dtype=float)

    n = counts.shape[0]
    kernel = np.eye(n) * diagonal_value

    rows, cols = upper_triangle_indices(n)
    for i, j in zip(rows, cols, strict=True):
        shots = allocation[i, j]
        if shots > 0:
            value = counts[i, j] / shots
        else:
            value = 0.0
        kernel[i, j] = value
        kernel[j, i] = value

    return kernel


def dense_dual_from_svc(clf: SVC, n_samples: int) -> np.ndarray:
    """Return dense vector q_i = alpha_i y_i."""
    q = np.zeros(n_samples, dtype=float)
    q[clf.support_] = clf.dual_coef_[0]
    return q


def margin_norm_from_dual(kernel: np.ndarray, q: np.ndarray) -> float:
    value = float(q @ kernel @ q)
    return float(np.sqrt(max(value, 0.0)))


def decision_values_from_dual(
    kernel: np.ndarray,
    q: np.ndarray,
    bias: float,
) -> np.ndarray:
    return kernel @ q + bias


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(diff * diff)))


def rmse_block(
    kernel_est: np.ndarray,
    kernel_true: np.ndarray,
    indices: np.ndarray,
) -> float:
    if len(indices) == 0:
        return np.nan
    block_est = kernel_est[np.ix_(indices, indices)]
    block_true = kernel_true[np.ix_(indices, indices)]
    return rmse(block_est, block_true)


def jaccard_index(a: np.ndarray, b: np.ndarray) -> float:
    set_a = set(int(x) for x in a)
    set_b = set(int(x) for x in b)

    union = set_a | set_b
    if not union:
        return 1.0

    return len(set_a & set_b) / len(union)


def weighted_jaccard_from_duals(q_true: np.ndarray, q_est: np.ndarray) -> float:
    a = np.abs(q_true)
    b = np.abs(q_est)

    denominator = np.maximum(a, b).sum()
    if denominator <= 0:
        return 1.0

    return float(np.minimum(a, b).sum() / denominator)


def accuracy_against_true_classifier(
    f_est: np.ndarray,
    f_true: np.ndarray,
) -> float:
    return float(np.mean(np.sign(f_est) == np.sign(f_true)))


def make_supportshot(
    *,
    total_budget: int,
    pilot_shots: int,
    rounds: int,
    lambda_weight: float,
    c: float,
    allocation_method: str,
    random_state: int,
    stopping_epsilon: float,
) -> SupportShot:
    """Construct SupportShot while tolerating small API differences."""
    kwargs = {
        "total_budget": total_budget,
        "pilot_shots_per_entry": pilot_shots,
        "rounds": rounds,
        "lambda_weight": lambda_weight,
        "svm_c": c,
        "allocation_method": allocation_method,
        "random_state": random_state,
    }

    signature = inspect.signature(SupportShot)
    if "stopping_epsilon" in signature.parameters:
        kwargs["stopping_epsilon"] = stopping_epsilon
    elif "early_stopping_epsilon" in signature.parameters:
        kwargs["early_stopping_epsilon"] = stopping_epsilon
    elif "dual_stability_epsilon" in signature.parameters:
        kwargs["dual_stability_epsilon"] = stopping_epsilon
    else:
        raise TypeError(
            "SupportShot does not expose a recognized early-stopping "
            "epsilon parameter. Expected one of: stopping_epsilon, "
            "early_stopping_epsilon, dual_stability_epsilon."
        )

    return SupportShot(**kwargs)


def evaluate_kernel_estimate(
    *,
    kernel_est: np.ndarray,
    kernel_true: np.ndarray,
    y: np.ndarray,
    c: float,
    true_support: np.ndarray,
    q_true: np.ndarray,
    bias_true: float,
    margin_true: float,
    f_true: np.ndarray,
) -> dict[str, float]:
    del bias_true

    clf = SVC(kernel="precomputed", C=c)
    clf.fit(kernel_est, y)

    q_est = dense_dual_from_svc(clf, len(y))
    bias_est = float(clf.intercept_[0])
    f_est_on_true_kernel = decision_values_from_dual(
        kernel_true,
        q_est,
        bias_est,
    )

    margin_est_on_true_kernel = margin_norm_from_dual(kernel_true, q_est)

    return {
        "rmse_K": rmse(kernel_est, kernel_true),
        "rmse_K_SV": rmse_block(kernel_est, kernel_true, true_support),
        "jaccard": jaccard_index(true_support, clf.support_),
        "jaccard_weighted": weighted_jaccard_from_duals(q_true, q_est),
        "margin_err": abs(margin_est_on_true_kernel - margin_true)
        / (margin_true + 1e-12),
        "func_rmse": rmse(f_est_on_true_kernel, f_true) / (margin_true + 1e-12),
        "acc_train_vs_true": accuracy_against_true_classifier(
            f_est_on_true_kernel,
            f_true,
        ),
    }


def run_one_experiment(
    *,
    run: int,
    epsilon: float,
    n_samples: int,
    total_budget: int,
    rounds: int,
    lambda_weight: float,
    pilot_shots: int,
    gamma: float,
    c: float,
    cluster_std: float,
    allocation_method: str,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str | bool]]:
    records: list[dict[str, float | int | str | bool]] = []

    dataset_seed = int(rng.integers(0, 2**32 - 1))
    uniform_seed = int(rng.integers(0, 2**32 - 1))
    adaptive_seed = int(rng.integers(0, 2**32 - 1))

    x, y = generate_blobs(
        n_samples=n_samples,
        cluster_std=cluster_std,
        random_state=dataset_seed,
    )

    kernel_true = rbf_kernel(x, x, gamma=gamma)
    np.fill_diagonal(kernel_true, 1.0)

    clf_true = SVC(kernel="precomputed", C=c)
    clf_true.fit(kernel_true, y)

    q_true = dense_dual_from_svc(clf_true, n_samples)
    bias_true = float(clf_true.intercept_[0])
    support_true = clf_true.support_
    margin_true = margin_norm_from_dual(kernel_true, q_true)
    f_true = decision_values_from_dual(kernel_true, q_true, bias_true)
    sv_ratio = len(support_true) / n_samples

    # Uniform baseline.
    uniform_rng = np.random.default_rng(uniform_seed)
    uniform_backend = BernoulliKernelBackend(kernel_true, uniform_rng)
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
        true_support=support_true,
        q_true=q_true,
        bias_true=bias_true,
        margin_true=margin_true,
        f_true=f_true,
    )
    records.append(
        {
            "run": run,
            "round": -1,
            "method": "uniform",
            "stage": "uniform",
            "shots_spent": int(uniform_allocation.sum()),
            "sv_ratio": float(sv_ratio),
            "dual_change": 0.0,
            "stopped_early": False,
            **uniform_metrics,
        }
    )

    # Adaptive SupportShot.
    adaptive_rng = np.random.default_rng(adaptive_seed)
    adaptive_backend = BernoulliKernelBackend(kernel_true, adaptive_rng)

    model = make_supportshot(
        total_budget=total_budget,
        pilot_shots=pilot_shots,
        rounds=rounds,
        lambda_weight=lambda_weight,
        c=c,
        allocation_method=allocation_method,
        random_state=adaptive_seed,
        stopping_epsilon=epsilon,
    )
    model.fit(y, adaptive_backend)

    previous_q: np.ndarray | None = None

    for history_record in model.history_:
        kernel_est = np.asarray(history_record.kernel_matrix, dtype=float)

        metrics = evaluate_kernel_estimate(
            kernel_est=kernel_est,
            kernel_true=kernel_true,
            y=y,
            c=c,
            true_support=support_true,
            q_true=q_true,
            bias_true=bias_true,
            margin_true=margin_true,
            f_true=f_true,
        )

        clf_stage = SVC(kernel="precomputed", C=c)
        clf_stage.fit(kernel_est, y)
        q_stage = dense_dual_from_svc(clf_stage, n_samples)

        if previous_q is None:
            dual_change = 0.0
        else:
            dual_change = float(
                np.linalg.norm(q_stage - previous_q)
                / (np.linalg.norm(previous_q) + 1e-12)
            )
        previous_q = q_stage.copy()

        if history_record.stage == "pilot":
            round_id = 0
        else:
            round_id = int(history_record.round_index)

        records.append(
            {
                "run": run,
                "round": round_id,
                "method": "adaptive",
                "stage": history_record.stage,
                "shots_spent": int(history_record.total_shots_spent),
                "sv_ratio": float(sv_ratio),
                "dual_change": dual_change,
                "stopped_early": bool(history_record.stopped_early),
                **metrics,
            }
        )

    return records


def run_for_epsilon(
    *,
    epsilon: float,
    experiment_rounds: int,
    n_samples: int,
    total_budget: int,
    rounds: int,
    lambda_weight: float,
    pilot_shots: int,
    gamma: float,
    c: float,
    cluster_std: float,
    allocation_method: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for run in range(experiment_rounds):
        if (run + 1) % max(1, experiment_rounds // 10) == 0:
            print(f"  run {run + 1}/{experiment_rounds}")

        records.extend(
            run_one_experiment(
                run=run,
                epsilon=epsilon,
                n_samples=n_samples,
                total_budget=total_budget,
                rounds=rounds,
                lambda_weight=lambda_weight,
                pilot_shots=pilot_shots,
                gamma=gamma,
                c=c,
                cluster_std=cluster_std,
                allocation_method=allocation_method,
                rng=rng,
            )
        )

    df = pd.DataFrame.from_records(records)
    df["epsilon"] = epsilon
    return df


def summarize_epsilon(df: pd.DataFrame) -> dict[str, float]:
    metrics = ["run", "round", "shots_spent", "func_rmse"]

    df_adaptive = df[df["round"] != -1]
    df_max_round = df_adaptive.loc[df_adaptive.groupby("run")["round"].idxmax()][
        metrics
    ]
    df_uniform = df[df["round"] == -1][metrics]

    merged = pd.merge(
        df_uniform,
        df_max_round,
        on="run",
        suffixes=("_uniform", "_adaptive"),
    )

    merged["shots_ratio"] = (
        merged["shots_spent_adaptive"] / merged["shots_spent_uniform"]
    )
    merged["delta_rmse"] = merged["func_rmse_uniform"] - merged["func_rmse_adaptive"]
    merged["sign"] = np.sign(merged["delta_rmse"])

    success_rate = float(((merged["sign"] + 1.0) / 2.0).mean())

    return {
        "epsilon": float(df["epsilon"].iloc[0]),
        "success_rate": success_rate,
        "delta_rmse_median": float(merged["delta_rmse"].median()),
        "delta_rmse_q25": float(merged["delta_rmse"].quantile(0.25)),
        "delta_rmse_q75": float(merged["delta_rmse"].quantile(0.75)),
        "shots_ratio_median": float(merged["shots_ratio"].median()),
        "shots_ratio_q25": float(merged["shots_ratio"].quantile(0.25)),
        "shots_ratio_q75": float(merged["shots_ratio"].quantile(0.75)),
        "median_max_round": float(merged["round_adaptive"].median()),
    }


def crossing_point(
    epsilons: np.ndarray,
    values: np.ndarray,
) -> float | None:
    signs = np.sign(values)

    for idx in range(len(signs) - 1):
        if signs[idx] == 0:
            return float(epsilons[idx])
        if signs[idx] * signs[idx + 1] < 0:
            x1, x2 = epsilons[idx], epsilons[idx + 1]
            y1, y2 = values[idx], values[idx + 1]
            log_x1 = np.log10(x1)
            log_x2 = np.log10(x2)

            slope = (y2 - y1) / (log_x2 - log_x1)
            intercept = y1 - slope * log_x1
            log_x0 = -intercept / slope
            return float(10**log_x0)

    return None


def interpolate_logx(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
) -> float:
    interpolator = interp1d(
        np.log10(x),
        y,
        kind="linear",
        fill_value="extrapolate",
    )
    return float(interpolator(np.log10(x0)))


def plot_early_stopping(
    summary: pd.DataFrame,
    *,
    pilot_ratio: float,
    output_dir: Path,
) -> None:
    eps = summary["epsilon"].to_numpy()
    delta = summary["delta_rmse_median"].to_numpy()
    delta_q25 = summary["delta_rmse_q25"].to_numpy()
    delta_q75 = summary["delta_rmse_q75"].to_numpy()
    shot_median = summary["shots_ratio_median"].to_numpy()
    shot_q25 = summary["shots_ratio_q25"].to_numpy()
    shot_q75 = summary["shots_ratio_q75"].to_numpy()
    success = summary["success_rate"].to_numpy()
    median_round = summary["median_max_round"].to_numpy()

    x0 = crossing_point(eps, delta)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )
    ax1, ax2 = axes

    positive = delta >= 0

    ax1.plot(eps, delta, color="tab:blue", label=r"median $\Delta RMSE_f$")
    ax1.fill_between(eps, delta_q25, delta_q75, alpha=0.3, label="IQR")
    ax1.axhline(0.0, color="gray", linestyle="--")
    ax1.set_ylabel(r"$\Delta RMSE_f$")
    ax1.legend()

    if x0 is not None:
        ax1.axvline(x0, color="tab:brown")
        ax1.text(
            x0,
            0.02,
            rf"$\epsilon^\star={x0:.2g}$",
            rotation=90,
            va="bottom",
            ha="right",
            color="tab:brown",
        )

    if positive.any():
        ax1.fill_between(
            eps,
            delta_q25,
            delta_q75,
            where=positive,
            alpha=0.15,
            color="tab:blue",
        )
    if (~positive).any():
        ax1.fill_between(
            eps,
            delta_q25,
            delta_q75,
            where=~positive,
            alpha=0.15,
            color="tab:red",
        )

    ax2.plot(eps, shot_median, label="Median relative shot count")
    ax2.fill_between(eps, shot_q25, shot_q75, alpha=0.3)
    ax2.axhline(pilot_ratio, color="gray", linestyle="--")
    ax2.text(eps[0], pilot_ratio + 0.02, "Pilot measurement", color="gray")
    ax2.set_ylabel("Relative shot count\nSuccess rate")
    ax2.set_ylim(-0.05, 1.05)

    ax2.plot(
        eps,
        success,
        color="tab:blue",
        linestyle="dotted",
        label="Success rate",
    )
    ax2.axhline(0.5, color="gray", linestyle="--")
    ax2.text(eps[0], 0.52, "50% success rate", color="gray")

    if x0 is not None:
        rel_value = interpolate_logx(eps, shot_median, x0)
        ax2.axvline(x0, color="tab:brown")
        ax2.scatter(x0, rel_value, color="black", zorder=5)
        ax2.text(
            x0,
            -0.03,
            rf"$\epsilon^\star={x0:.2g}$",
            transform=ax2.get_xaxis_transform(),
            ha="center",
            va="top",
        )

    ax3 = ax2.twinx()
    ax3.plot(eps, median_round, color="maroon", label="Median max round")
    ax3.set_ylabel("Median max round", color="maroon")

    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_xscale("log")
    ax2.legend(loc="upper left")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "early_stopping.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "early_stopping.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_off_diagonal = args.n_samples * (args.n_samples - 1) // 2
    total_budget = n_off_diagonal * args.shots_per_entry
    # pilot_ratio = n_off_diagonal * args.pilot_shots / total_budget

    epsilons = np.logspace(
        args.epsilon_min_exp,
        args.epsilon_max_exp,
        num=args.epsilon_count,
    )

    all_raw = []
    summaries = []

    for idx, epsilon in enumerate(epsilons):
        print(f"Epsilon {idx + 1}/{len(epsilons)}: {epsilon:.4g}")

        df = run_for_epsilon(
            epsilon=float(epsilon),
            experiment_rounds=args.experiment_rounds,
            n_samples=args.n_samples,
            total_budget=total_budget,
            rounds=args.rounds,
            lambda_weight=args.lambda_weight,
            pilot_shots=args.pilot_shots,
            gamma=args.gamma,
            c=args.c,
            cluster_std=args.cluster_std,
            allocation_method=args.allocation_method,
            seed=args.seed + idx,
        )

        all_raw.append(df)
        summaries.append(summarize_epsilon(df))

    raw_df = pd.concat(all_raw, ignore_index=True)
    summary_df = pd.DataFrame.from_records(summaries)

    raw_path = args.output_dir / "early_stopping_raw.csv"
    summary_path = args.output_dir / "early_stopping_summary.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # plot_early_stopping(
    #    summary_df,
    #    pilot_ratio=pilot_ratio,
    #    output_dir=args.output_dir,
    # )

    print("Saved outputs:")
    print(f"- {raw_path}")
    print(f"- {summary_path}")
    # print(f"- {args.output_dir / 'early_stopping.pdf'}")
    # print(f"- {args.output_dir / 'early_stopping.png'}")


if __name__ == "__main__":
    main()
