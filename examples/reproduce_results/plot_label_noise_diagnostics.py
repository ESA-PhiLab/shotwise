"""Diagnostic plots for interpreting adaptive allocation under label noise.

This script reads the raw output of structural_mechanism.py and produces
three diagnostic panels:

(a) absolute normalized decision-function RMSE for adaptive and uniform
    allocation,
(b) weighted support-vector Jaccard recovery for adaptive and uniform
    allocation,
(c) oracle bounded-support-vector fraction.

The diagnostics are intended to explain the non-monotonic adaptive-advantage
curve observed under label noise.

Example
-------
From the repository root:

    python examples/reproduce_results/plot_label_noise_diagnostics.py \
        --input \
        outputs/structural_mechanism_final/structural_mechanism_results.csv \
        --output-dir outputs/structural_mechanism_final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot label-noise diagnostics for SupportShot."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw CSV produced by structural_mechanism.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/structural_mechanism"),
    )
    parser.add_argument(
        "--rmse-scale",
        choices=["linear", "log"],
        default="log",
        help="Y-axis scale for absolute decision-function RMSE.",
    )
    return parser.parse_args()


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate columns required by the diagnostic plots."""
    required = {
        "cluster_std",
        "label_noise_rate",
        "adaptive_func_rmse",
        "uniform_func_rmse",
        "adaptive_jaccard_weighted",
        "uniform_jaccard_weighted",
        "bounded_sv_fraction",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("Input CSV contains no rows.")

    for column in required:
        values = df[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Column {column!r} contains non-finite values.")


def summarize_metric(
    df: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """Return median and interquartile range by experimental condition."""
    grouped = df.groupby(
        ["cluster_std", "label_noise_rate"],
        sort=True,
    )[metric]

    return grouped.agg(
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    ).reset_index()


def cluster_colors(cluster_stds: list[float]) -> dict[float, object]:
    """Assign consistent colors to structural regimes."""
    cmap = plt.get_cmap("viridis")

    if len(cluster_stds) == 1:
        positions = [0.5]
    else:
        positions = np.linspace(0.08, 0.88, len(cluster_stds))

    return {
        cluster_std: cmap(position)
        for cluster_std, position in zip(
            cluster_stds,
            positions,
            strict=True,
        )
    }


def plot_curve(
    ax,
    summary: pd.DataFrame,
    *,
    cluster_std: float,
    color,
    linestyle: str,
    marker: str,
    filled_marker: bool,
    label: str | None = None,
    alpha_band: float = 0.10,
) -> None:
    """Plot median and interquartile range for one condition trajectory."""
    selected = summary.loc[summary["cluster_std"] == cluster_std].sort_values(
        "label_noise_rate"
    )

    x = selected["label_noise_rate"].to_numpy(dtype=float)
    median = selected["median"].to_numpy(dtype=float)
    q25 = selected["q25"].to_numpy(dtype=float)
    q75 = selected["q75"].to_numpy(dtype=float)

    marker_face = color if filled_marker else "white"

    ax.plot(
        x,
        median,
        color=color,
        linestyle=linestyle,
        linewidth=1.3,
        marker=marker,
        markersize=3.6,
        markerfacecolor=marker_face,
        markeredgecolor=color,
        markeredgewidth=0.8,
        label=label,
        zorder=3,
    )

    ax.fill_between(
        x,
        q25,
        q75,
        color=color,
        alpha=alpha_band,
        linewidth=0.0,
        zorder=1,
    )


def save_figure(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    rmse_scale: str,
) -> None:
    """Create the three diagnostic panels."""
    cluster_stds = sorted(float(value) for value in df["cluster_std"].unique())
    noise_rates = sorted(float(value) for value in df["label_noise_rate"].unique())

    colors = cluster_colors(cluster_stds)

    adaptive_rmse = summarize_metric(
        df,
        metric="adaptive_func_rmse",
    )
    uniform_rmse = summarize_metric(
        df,
        metric="uniform_func_rmse",
    )

    adaptive_jaccard = summarize_metric(
        df,
        metric="adaptive_jaccard_weighted",
    )
    uniform_jaccard = summarize_metric(
        df,
        metric="uniform_jaccard_weighted",
    )

    bounded_sv = summarize_metric(
        df,
        metric="bounded_sv_fraction",
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.0, 3.1),
    )

    # ----------------------------------------------------------
    # (a) Absolute decision-function errors
    # ----------------------------------------------------------
    for cluster_std in cluster_stds:
        color = colors[cluster_std]

        plot_curve(
            axes[0],
            adaptive_rmse,
            cluster_std=cluster_std,
            color=color,
            linestyle="-",
            marker="o",
            filled_marker=True,
            alpha_band=0.09,
        )

        plot_curve(
            axes[0],
            uniform_rmse,
            cluster_std=cluster_std,
            color=color,
            linestyle="--",
            marker="s",
            filled_marker=False,
            alpha_band=0.05,
        )

    axes[0].set_yscale(rmse_scale)
    axes[0].set_xlabel(
        r"Label-noise rate $\eta$",
        fontsize=9,
    )
    axes[0].set_ylabel(
        r"Normalized decision RMSE $\mathrm{RMSE}_f$",
        fontsize=9,
    )
    axes[0].set_title(
        "(a) Absolute reconstruction error",
        fontsize=10,
    )

    # Allocation-strategy legend.
    (adaptive_handle,) = axes[0].plot(
        [],
        [],
        color="black",
        linestyle="-",
        marker="o",
        markersize=3.6,
        label="Adaptive",
    )
    (uniform_handle,) = axes[0].plot(
        [],
        [],
        color="black",
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markersize=3.6,
        label="Uniform",
    )

    axes[0].legend(
        handles=[adaptive_handle, uniform_handle],
        fontsize=7.0,
        frameon=False,
        loc="best",
    )

    # ----------------------------------------------------------
    # (b) Weighted support-vector recovery
    # ----------------------------------------------------------
    for cluster_std in cluster_stds:
        color = colors[cluster_std]

        plot_curve(
            axes[1],
            adaptive_jaccard,
            cluster_std=cluster_std,
            color=color,
            linestyle="-",
            marker="o",
            filled_marker=True,
            alpha_band=0.09,
        )

        plot_curve(
            axes[1],
            uniform_jaccard,
            cluster_std=cluster_std,
            color=color,
            linestyle="--",
            marker="s",
            filled_marker=False,
            alpha_band=0.05,
        )

    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel(
        r"Label-noise rate $\eta$",
        fontsize=9,
    )
    axes[1].set_ylabel(
        r"Weighted SV Jaccard $J_{\mathrm{SV}}^w$",
        fontsize=9,
    )
    axes[1].set_title(
        "(b) Support-vector recovery",
        fontsize=10,
    )

    axes[1].legend(
        handles=[adaptive_handle, uniform_handle],
        fontsize=7.0,
        frameon=False,
        loc="best",
    )

    # ----------------------------------------------------------
    # (c) Oracle bounded-SV fraction
    # ----------------------------------------------------------
    for cluster_std in cluster_stds:
        plot_curve(
            axes[2],
            bounded_sv,
            cluster_std=cluster_std,
            color=colors[cluster_std],
            linestyle="-",
            marker="o",
            filled_marker=True,
            label=rf"$\sigma={cluster_std:g}$",
            alpha_band=0.10,
        )

    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel(
        r"Label-noise rate $\eta$",
        fontsize=9,
    )
    axes[2].set_ylabel(
        r"Bounded-SV fraction",
        fontsize=9,
    )
    axes[2].set_title(
        r"(c) Box-constrained SVs",
        fontsize=10,
    )

    axes[2].legend(
        title=r"Cluster std. $\sigma$",
        fontsize=6.7,
        title_fontsize=7.1,
        frameon=False,
        loc="best",
    )

    # ----------------------------------------------------------
    # Shared style
    # ----------------------------------------------------------
    for ax in axes:
        ax.set_xticks(noise_rates)
        ax.set_xticklabels([f"{value:g}" for value in noise_rates])

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=8,
            width=0.8,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        ax.grid(False)
        ax.margins(x=0.03)

    fig.tight_layout(
        pad=0.7,
        w_pad=1.25,
    )

    fig.savefig(
        output_dir / "label_noise_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "label_noise_diagnostics.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def print_diagnostic_summary(df: pd.DataFrame) -> None:
    """Print compact medians useful for interpreting non-monotonic behavior."""
    metrics = [
        "adaptive_func_rmse",
        "uniform_func_rmse",
        "adaptive_jaccard_weighted",
        "uniform_jaccard_weighted",
        "bounded_sv_fraction",
    ]

    display = (
        df.groupby(
            ["cluster_std", "label_noise_rate"],
            sort=True,
        )[metrics]
        .median()
        .reset_index()
    )

    print("=== Diagnostic medians ===")
    print(display.to_string(index=False))


def main() -> None:
    """Load raw results and create diagnostic plots."""
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.input)
    validate_dataframe(df)

    print_diagnostic_summary(df)

    save_figure(
        df,
        args.output_dir,
        rmse_scale=args.rmse_scale,
    )

    print()
    print("Saved:")
    print(f"- {args.output_dir / 'label_noise_diagnostics.png'}")
    print(f"- {args.output_dir / 'label_noise_diagnostics.pdf'}")


if __name__ == "__main__":
    main()
