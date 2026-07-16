#!/usr/bin/env python3
"""Plot lambda ablation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_FILE = "outputs/lambda_ablation/lambda_ablation_raw.csv"
OUTPUT_FILE = "outputs/lambda_ablation/lambda_ablation.pdf"


def summarize_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []

    for lam in sorted(df["lambda"].dropna().unique()):
        subset = df[df["lambda"] == lam]

        values = subset[metric].to_numpy()

        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])

        rows.append(
            {
                "lambda": lam,
                "median": median,
                "q25": q25,
                "q75": q75,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    adaptive = df[df["method"] == "adaptive"].copy()
    uniform = df[df["method"] == "uniform"].copy()

    rmse = summarize_metric(
        adaptive,
        "normalized_decision_rmse",
    )

    wj = summarize_metric(
        adaptive,
        "weighted_jaccard",
    )

    uniform_rmse = uniform["normalized_decision_rmse"].median()
    uniform_wj = uniform["weighted_jaccard"].median()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        constrained_layout=True,
    )

    # --------------------------------------------------
    # Panel (a): normalized decision RMSE
    # --------------------------------------------------

    ax = axes[0]

    ax.plot(
        rmse["lambda"],
        rmse["median"],
        color="tab:blue",
        linewidth=2,
        label="Adaptive allocation",
    )

    ax.fill_between(
        rmse["lambda"],
        rmse["q25"],
        rmse["q75"],
        color="tab:blue",
        alpha=0.20,
        linewidth=0,
        label="IQR",
    )

    uniform_rmse = uniform["normalized_decision_rmse"].median()

    ax.text(
        0.98,
        0.95,
        f"Uniform median = {uniform_rmse:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

    ax.axvline(
        0.5,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label=r"$\lambda=0.5$",
    )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Normalized decision RMSE")
    ax.set_title("(a) Decision-function accuracy")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.28, 0.65)
    ax.grid(alpha=0.3)

    # --------------------------------------------------
    # Panel (b): weighted Jaccard
    # --------------------------------------------------

    ax = axes[1]

    ax.plot(
        wj["lambda"],
        wj["median"],
        color="tab:blue",
        linewidth=2,
        label="Adaptive allocation",
    )

    ax.fill_between(
        wj["lambda"],
        wj["q25"],
        wj["q75"],
        color="tab:blue",
        alpha=0.20,
        linewidth=0,
        label="IQR",
    )

    ax.axhline(
        uniform_wj,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Uniform median",
    )

    ax.axvline(
        0.5,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label=r"$\lambda=0.5$",
    )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Weighted Jaccard")
    ax.set_title("(b) Support-vector recovery")

    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        frameon=False,
    )

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
