"""Plot early-stopping experiment from saved CSV files.

This script reads the summary CSV produced by early_stopping_experiment.py and
regenerates the early-stopping figure without rerunning the experiment.

Example
-------
python examples/plot_early_stopping.py \
    --summary outputs/early_stopping/early_stopping_summary.csv \
    --output-dir outputs/early_stopping \
    --pilot-ratio 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot early-stopping results from summary CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to early_stopping_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--pilot-ratio",
        type=float,
        default=0.05,
        help="Pilot measurement fraction, e.g. pilot_shots / shots_per_entry.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional x-axis minimum.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional x-axis maximum.",
    )
    parser.add_argument(
        "--y1-min",
        type=float,
        default=None,
        help="Optional upper-panel y-axis minimum.",
    )
    parser.add_argument(
        "--y1-max",
        type=float,
        default=None,
        help="Optional upper-panel y-axis maximum.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively.",
    )
    return parser.parse_args()


def crossing_point(
    epsilons: np.ndarray,
    values: np.ndarray,
) -> float | None:
    """Find zero crossing of median delta by interpolation in log10 epsilon."""
    signs = np.sign(values)

    for idx in range(len(signs) - 1):
        if signs[idx] == 0:
            return float(epsilons[idx])

        if signs[idx] * signs[idx + 1] < 0:
            x1 = float(epsilons[idx])
            x2 = float(epsilons[idx + 1])
            y1 = float(values[idx])
            y2 = float(values[idx + 1])

            log_x1 = np.log10(x1)
            log_x2 = np.log10(x2)

            slope = (y2 - y1) / (log_x2 - log_x1)
            intercept = y1 - slope * log_x1
            log_x0 = -intercept / slope

            return float(10.0**log_x0)

    return None


def interpolate_logx(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
) -> float:
    """Interpolate y as a function of log10(x)."""
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
    output_dir: Path,
    pilot_ratio: float,
    x_min: float | None = None,
    x_max: float | None = None,
    y1_min: float | None = None,
    y1_max: float | None = None,
    show: bool = False,
) -> None:
    """Create the early-stopping figure from a summary dataframe."""

    red = "#d6604d"
    blue = "#4393c3"
    brown = "tab:brown"

    summary = summary.sort_values("epsilon").reset_index(drop=True)

    eps = summary["epsilon"].to_numpy(dtype=float)

    delta = summary["delta_rmse_median"].to_numpy(dtype=float)
    delta_q25 = summary["delta_rmse_q25"].to_numpy(dtype=float)
    delta_q75 = summary["delta_rmse_q75"].to_numpy(dtype=float)

    shot_median = summary["shots_ratio_median"].to_numpy(dtype=float)
    shot_q25 = summary["shots_ratio_q25"].to_numpy(dtype=float)
    shot_q75 = summary["shots_ratio_q75"].to_numpy(dtype=float)

    success = summary["success_rate"].to_numpy(dtype=float)
    median_round = summary["median_max_round"].to_numpy(dtype=float)

    eps_star = crossing_point(eps, delta)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    ax1, ax2 = axes

    if eps_star is not None:
        eps_red = np.concatenate([eps[delta >= 0], [eps_star]])
        delta_red = np.concatenate([delta[delta >= 0], [0.0]])

        eps_blue = np.concatenate([[eps_star], eps[delta <= 0]])
        delta_blue = np.concatenate([[0.0], delta[delta <= 0]])

        ax1.plot(
            eps_red,
            delta_red,
            color=red,
            linewidth=2,
        )

        ax1.plot(
            eps_blue,
            delta_blue,
            color=blue,
            linewidth=2,
        )

    # Keep one neutral IQR band. This avoids visually implying that the
    # quartile band itself has a sharp sign split.
    # Positive part of IQR
    ax1.fill_between(
        eps,
        np.maximum(delta_q25, 0.0),
        np.maximum(delta_q75, 0.0),
        where=delta_q75 > 0,
        color=red,
        alpha=0.20,
    )

    # Negative part of IQR
    ax1.fill_between(
        eps,
        np.minimum(delta_q25, 0.0),
        np.minimum(delta_q75, 0.0),
        where=delta_q25 < 0,
        color=blue,
        alpha=0.20,
    )

    # Legend handle
    ax1.fill_between(
        [],
        [],
        [],
        color="gray",
        alpha=0.20,
        label="IQR",
    )

    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1.5)
    ax1.set_ylabel(r"$\Delta \mathrm{RMSE}$")
    ax1.legend(loc="lower left", frameon=True)

    if y1_min is not None or y1_max is not None:
        ax1.set_ylim(y1_min, y1_max)

    if eps_star is not None:
        ax1.axvline(eps_star, color=brown, linewidth=1.2, alpha=0.7)

        ymin, ymax = ax1.get_ylim()
        ax1.text(
            eps_star,
            ymin + 0.02 * (ymax - ymin),
            rf"$\epsilon^\star = {eps_star:.2f}$",
            color=brown,
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=10,
        )

    # Lower panel: relative shot count, success rate, and max round.
    ax2.plot(
        eps,
        shot_median,
        color="tab:blue",
        linewidth=2,
        label="Median relative shot count",
    )
    ax2.fill_between(
        eps,
        shot_q25,
        shot_q75,
        color="tab:blue",
        alpha=0.25,
    )

    ax2.plot(
        eps,
        success,
        color="tab:blue",
        linestyle=":",
        linewidth=2,
        label="Success rate",
    )

    ax2.axhline(
        pilot_ratio,
        color="gray",
        linestyle="--",
        linewidth=1.3,
    )
    text_x = x_min if x_min is not None else eps[0]
    text_x = text_x * 1.08

    ax2.text(
        text_x,
        pilot_ratio + 0.02,
        "Pilot measurement",
        color="gray",
        ha="left",
        va="bottom",
        fontsize=10,
    )

    ax2.axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1.3,
    )
    ax2.text(
        text_x,
        0.52,
        "50% success rate",
        color="gray",
        ha="left",
        va="bottom",
        fontsize=10,
    )

    if eps_star is not None:
        rel_value = interpolate_logx(eps, shot_median, eps_star)

        ax2.axvline(eps_star, color="tab:brown", linewidth=1.5)
        ax2.scatter(eps_star, rel_value, color="black", zorder=5)

        # ax2.text(
        #    eps_star * 0.97,
        #    0.09,
        #    rf"$\epsilon^\star = {eps_star:.2f}$",
        #    color=brown,
        #    ha="right",
        #    va="bottom",
        #    fontsize=10,
        # )
        # ax2.text(
        #    eps_star * 1.05,
        #    rel_value + 0.025,
        #    f"{rel_value:.2f}",
        #    color="black",
        #    ha="left",
        #    va="bottom",
        #    fontsize=10,
        # )

    ax2.set_ylabel("Relative shot count\nSuccess rate")
    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_ylim(-0.05, 1.05)

    ax2.legend(
        loc="upper right",
        frameon=True,
    )

    ax3 = ax2.twinx()
    ax3.plot(
        eps,
        median_round,
        color="maroon",
        linewidth=1.8,
        label="Median max round",
    )
    ax3.set_ylabel("Median max round", color="maroon")
    ax3.tick_params(axis="y", labelcolor="maroon")

    ax2.set_xscale("log")

    if x_min is not None or x_max is not None:
        ax2.set_xlim(x_min, x_max)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / "early_stopping_from_csv.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "early_stopping_from_csv.png", bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)

    plot_early_stopping(
        summary,
        output_dir=args.output_dir,
        pilot_ratio=args.pilot_ratio,
        x_min=args.x_min,
        x_max=args.x_max,
        y1_min=args.y1_min,
        y1_max=args.y1_max,
        show=args.show,
    )


if __name__ == "__main__":
    main()
