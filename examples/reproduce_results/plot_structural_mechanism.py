"""Create the manuscript figure for structure and label-noise experiments.

The figure uses three representative structural regimes throughout:

    sigma = 0.2, 0.6, 1.0.

Panel (a) shows:
    - oracle allocation-weight Gini coefficient G(w*),
    - oracle support-vector fraction |S*| / n.

Panel (b) shows:
    - normalized decision-function RMSE for adaptive allocation,
    - normalized decision-function RMSE for uniform allocation.

Curves show condition medians and shaded areas show interquartile ranges.
The visual style is intentionally restrained to match the other manuscript
figures.

Example
-------
From the repository root:

    python examples/reproduce_results/plot_structural_mechanism.py \
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
        description=(
            "Plot SVM structure and classifier reconstruction under label noise."
        )
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
        "--representative-stds",
        type=float,
        nargs="+",
        default=[0.2, 0.6, 1.0],
    )
    return parser.parse_args()


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate required columns."""
    required = {
        "cluster_std",
        "label_noise_rate",
        "weight_gini",
        "sv_fraction",
        "adaptive_func_rmse",
        "uniform_func_rmse",
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
    """Return median and interquartile range by condition."""
    grouped = df.groupby(
        ["cluster_std", "label_noise_rate"],
        sort=True,
    )[metric]

    return grouped.agg(
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    ).reset_index()


def manuscript_colors(
    values: list[float],
) -> dict[float, str]:
    """Return restrained blue tones matching the manuscript style."""
    palette = [
        "#1f4e79",
        "#377eb8",
        "#6baed6",
    ]

    if len(values) != 3:
        raise ValueError(
            "The manuscript figure is designed for exactly "
            "three representative structural regimes."
        )

    return {
        value: color
        for value, color in zip(
            values,
            palette,
            strict=True,
        )
    }


def select_summary(
    summary: pd.DataFrame,
    *,
    cluster_std: float,
) -> pd.DataFrame:
    """Select one structural trajectory."""
    selected = summary.loc[
        np.isclose(
            summary["cluster_std"],
            cluster_std,
        )
    ].sort_values("label_noise_rate")

    if selected.empty:
        raise ValueError(f"No data found for cluster_std={cluster_std:g}.")

    return selected


def plot_curve(
    ax,
    summary: pd.DataFrame,
    *,
    cluster_std: float,
    color: str,
    linestyle: str,
    marker: str,
    filled_marker: bool,
    linewidth: float = 1.1,
    band_alpha: float = 0.10,
) -> None:
    """Plot median and interquartile range."""
    selected = select_summary(
        summary,
        cluster_std=cluster_std,
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
        linewidth=linewidth,
        marker=marker,
        markersize=3.5,
        markerfacecolor=marker_face,
        markeredgecolor=color,
        markeredgewidth=0.7,
        zorder=3,
    )

    ax.fill_between(
        x,
        q25,
        q75,
        color=color,
        alpha=band_alpha,
        linewidth=0.0,
        zorder=1,
    )


def save_figure(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    representative_stds: list[float],
) -> None:
    """Create the final two-panel manuscript figure."""
    representative_stds = sorted(set(float(value) for value in representative_stds))

    if len(representative_stds) != 3:
        raise ValueError(
            "Please provide exactly three representative "
            "cluster-standard-deviation values."
        )

    observed_stds = sorted(float(value) for value in df["cluster_std"].unique())

    for value in representative_stds:
        if not any(np.isclose(value, observed) for observed in observed_stds):
            raise ValueError(
                f"Requested cluster std. {value:g} is not present in the input data."
            )

    noise_rates = sorted(float(value) for value in df["label_noise_rate"].unique())

    colors = manuscript_colors(representative_stds)

    gini_summary = summarize_metric(
        df,
        metric="weight_gini",
    )
    sv_summary = summarize_metric(
        df,
        metric="sv_fraction",
    )
    adaptive_rmse = summarize_metric(
        df,
        metric="adaptive_func_rmse",
    )
    uniform_rmse = summarize_metric(
        df,
        metric="uniform_func_rmse",
    )

    # Compact full-width two-column figure.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.75),
    )

    # ----------------------------------------------------------
    # (a) Structural response
    # ----------------------------------------------------------
    for cluster_std in representative_stds:
        color = colors[cluster_std]

        # Gini: solid, filled circles.
        plot_curve(
            axes[0],
            gini_summary,
            cluster_std=cluster_std,
            color=color,
            linestyle="-",
            marker="o",
            filled_marker=True,
            linewidth=1.1,
            band_alpha=0.08,
        )

        # Support-vector fraction: dashed, open squares.
        plot_curve(
            axes[0],
            sv_summary,
            cluster_std=cluster_std,
            color=color,
            linestyle="--",
            marker="s",
            filled_marker=False,
            linewidth=1.0,
            band_alpha=0.045,
        )

    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xlabel(
        r"Label-noise rate $\eta$",
        fontsize=8,
    )
    axes[0].set_ylabel(
        "Structural measure",
        fontsize=8,
    )

    # Color legend for sigma.
    sigma_handles = []

    for cluster_std in representative_stds:
        (handle,) = axes[0].plot(
            [],
            [],
            color=colors[cluster_std],
            linestyle="-",
            linewidth=1.1,
            label=rf"$\sigma={cluster_std:g}$",
        )
        sigma_handles.append(handle)

    sigma_legend = axes[0].legend(
        handles=sigma_handles,
        title=r"Cluster std. $\sigma$",
        fontsize=6.5,
        title_fontsize=6.8,
        frameon=False,
        loc="center left",
    )
    axes[0].add_artist(sigma_legend)

    # Line-style legend for metric.
    (gini_handle,) = axes[0].plot(
        [],
        [],
        color="black",
        linestyle="-",
        marker="o",
        markersize=3.2,
        linewidth=1.0,
        label=r"$G(w^\star)$",
    )
    (sv_handle,) = axes[0].plot(
        [],
        [],
        color="black",
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markersize=3.2,
        linewidth=1.0,
        label=r"$|S^\star|/n$",
    )

    axes[0].legend(
        handles=[gini_handle, sv_handle],
        fontsize=6.5,
        frameon=False,
        loc="lower left",
    )

    # ----------------------------------------------------------
    # (b) Absolute classifier reconstruction
    # ----------------------------------------------------------
    for cluster_std in representative_stds:
        color = colors[cluster_std]

        # Adaptive: solid, filled circles.
        plot_curve(
            axes[1],
            adaptive_rmse,
            cluster_std=cluster_std,
            color=color,
            linestyle="-",
            marker="o",
            filled_marker=True,
            linewidth=1.1,
            band_alpha=0.08,
        )

        # Uniform: dashed, open squares.
        plot_curve(
            axes[1],
            uniform_rmse,
            cluster_std=cluster_std,
            color=color,
            linestyle="--",
            marker="s",
            filled_marker=False,
            linewidth=1.0,
            band_alpha=0.035,
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel(
        r"Label-noise rate $\eta$",
        fontsize=8,
    )
    axes[1].set_ylabel(
        r"$\mathrm{RMSE}_f$",
        fontsize=8,
    )

    # Sigma/color legend.
    sigma_handles_right = []

    for cluster_std in representative_stds:
        (handle,) = axes[1].plot(
            [],
            [],
            color=colors[cluster_std],
            linestyle="-",
            linewidth=1.1,
            label=rf"$\sigma={cluster_std:g}$",
        )
        sigma_handles_right.append(handle)

    sigma_right_legend = axes[1].legend(
        handles=sigma_handles_right,
        title=r"Cluster std. $\sigma$",
        fontsize=6.5,
        title_fontsize=6.8,
        frameon=False,
        loc="upper right",
    )
    axes[1].add_artist(sigma_right_legend)

    (adaptive_handle,) = axes[1].plot(
        [],
        [],
        color="black",
        linestyle="-",
        marker="o",
        markersize=3.2,
        linewidth=1.0,
        label="Adaptive",
    )
    (uniform_handle,) = axes[1].plot(
        [],
        [],
        color="black",
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markersize=3.2,
        linewidth=1.0,
        label="Uniform",
    )

    axes[1].legend(
        handles=[
            adaptive_handle,
            uniform_handle,
        ],
        fontsize=6.5,
        frameon=False,
        loc="lower right",
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
            labelsize=7,
            width=0.6,
            length=3.0,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            width=0.5,
            length=2.0,
        )

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.65)

        ax.grid(False)
        ax.margins(x=0.025)

    fig.tight_layout(
        pad=0.45,
        w_pad=1.0,
    )

    fig.savefig(
        output_dir / "structural_mechanism_two_panel.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "structural_mechanism_two_panel.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Load campaign results and create the manuscript figure."""
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.input)
    validate_dataframe(df)

    save_figure(
        df,
        args.output_dir,
        representative_stds=args.representative_stds,
    )

    print("Saved:")
    print(f"- {args.output_dir / 'structural_mechanism_two_panel.png'}")
    print(f"- {args.output_dir / 'structural_mechanism_two_panel.pdf'}")


if __name__ == "__main__":
    main()
