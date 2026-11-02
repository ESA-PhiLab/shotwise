"""Analyze the sensitivity-vs-instability lambda sweep.

SupportShot combines normalized sensitivity and instability scores as

    score = (1 - lambda) * sensitivity + lambda * instability.

Therefore:
    lambda = 0 -> sensitivity-only allocation,
    lambda = 1 -> instability-only allocation.

This script combines structural_mechanism.py campaigns run at different
lambda values and examines whether instability-aware allocation improves
support-vector recovery and downstream decision-function reconstruction
under label noise.

The main figure contains:
    top row    : adaptive decision-function RMSE, with uniform reference,
    bottom row : weighted support-vector Jaccard recovery.

Columns correspond to controlled structural regimes.

Example
-------
From the repository root:

    python examples/reproduce_results/analyze_lambda_sweep.py \
        --base-dir outputs/structural_mechanism_lambda_sweep \
        --lambda-values 0 0.25 0.5 0.75 1 \
        --output-dir outputs/structural_mechanism_lambda_sweep/analysis
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
        description="Combine and analyze SupportShot lambda sweeps."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing lambda_<value>/"
            "structural_mechanism_results.csv directories."
        ),
    )
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/structural_mechanism_lambda_sweep/analysis"),
    )
    return parser.parse_args()


def lambda_directory_name(lambda_value: float) -> str:
    """Return campaign directory name for one lambda value."""
    if float(lambda_value).is_integer():
        return f"lambda_{int(lambda_value)}"

    return f"lambda_{lambda_value:g}"


def validate_dataframe(
    df: pd.DataFrame,
    *,
    expected_lambda: float,
) -> None:
    """Validate one lambda-sweep campaign."""
    required = {
        "run",
        "cluster_std",
        "label_noise_rate",
        "lambda_weight",
        "adaptive_func_rmse",
        "uniform_func_rmse",
        "adaptive_advantage_log10",
        "adaptive_jaccard_weighted",
        "uniform_jaccard_weighted",
        "adaptive_jaccard",
        "uniform_jaccard",
        "adaptive_margin_err",
        "uniform_margin_err",
        "sv_fraction",
        "bounded_sv_fraction",
        "weight_gini",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("Input CSV contains no rows.")

    observed_lambda = df["lambda_weight"].to_numpy(dtype=float)

    if not np.allclose(
        observed_lambda,
        expected_lambda,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"Expected lambda={expected_lambda:g}, but found "
            f"{sorted(df['lambda_weight'].unique())}."
        )


def load_campaigns(
    base_dir: Path,
    lambda_values: list[float],
) -> pd.DataFrame:
    """Load and concatenate all requested lambda campaigns."""
    frames: list[pd.DataFrame] = []

    for lambda_value in lambda_values:
        directory = base_dir / lambda_directory_name(lambda_value)
        csv_path = directory / "structural_mechanism_results.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing lambda={lambda_value:g} result file: {csv_path}"
            )

        print(f"Loading lambda={lambda_value:g}: {csv_path}")

        df = pd.read_csv(csv_path)

        validate_dataframe(
            df,
            expected_lambda=lambda_value,
        )

        df = df.copy()
        df["lambda_sweep_value"] = float(lambda_value)

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance by lambda, structure, and noise rate."""
    grouped = df.groupby(
        [
            "lambda_sweep_value",
            "cluster_std",
            "label_noise_rate",
        ],
        sort=True,
    )

    summary = grouped.agg(
        repetitions=("run", "count"),
        adaptive_rmse_median=(
            "adaptive_func_rmse",
            "median",
        ),
        adaptive_rmse_q25=(
            "adaptive_func_rmse",
            lambda values: values.quantile(0.25),
        ),
        adaptive_rmse_q75=(
            "adaptive_func_rmse",
            lambda values: values.quantile(0.75),
        ),
        uniform_rmse_median=(
            "uniform_func_rmse",
            "median",
        ),
        uniform_rmse_q25=(
            "uniform_func_rmse",
            lambda values: values.quantile(0.25),
        ),
        uniform_rmse_q75=(
            "uniform_func_rmse",
            lambda values: values.quantile(0.75),
        ),
        advantage_median=(
            "adaptive_advantage_log10",
            "median",
        ),
        advantage_q25=(
            "adaptive_advantage_log10",
            lambda values: values.quantile(0.25),
        ),
        advantage_q75=(
            "adaptive_advantage_log10",
            lambda values: values.quantile(0.75),
        ),
        adaptive_win_rate=(
            "adaptive_advantage_log10",
            lambda values: float((values > 0.0).mean()),
        ),
        weighted_jaccard_median=(
            "adaptive_jaccard_weighted",
            "median",
        ),
        weighted_jaccard_q25=(
            "adaptive_jaccard_weighted",
            lambda values: values.quantile(0.25),
        ),
        weighted_jaccard_q75=(
            "adaptive_jaccard_weighted",
            lambda values: values.quantile(0.75),
        ),
        uniform_weighted_jaccard_median=(
            "uniform_jaccard_weighted",
            "median",
        ),
        jaccard_median=(
            "adaptive_jaccard",
            "median",
        ),
        margin_error_median=(
            "adaptive_margin_err",
            "median",
        ),
        sv_fraction_median=(
            "sv_fraction",
            "median",
        ),
        bounded_sv_fraction_median=(
            "bounded_sv_fraction",
            "median",
        ),
        weight_gini_median=(
            "weight_gini",
            "median",
        ),
    )

    return summary.reset_index()


def lambda_colors(
    lambda_values: list[float],
) -> dict[float, object]:
    """Assign one color to each mixture weight."""
    cmap = plt.get_cmap("viridis")

    if len(lambda_values) == 1:
        positions = [0.5]
    else:
        positions = np.linspace(
            0.05,
            0.90,
            len(lambda_values),
        )

    return {
        value: cmap(position)
        for value, position in zip(
            lambda_values,
            positions,
            strict=True,
        )
    }


def plot_quantile_curve(
    ax,
    summary: pd.DataFrame,
    *,
    lambda_value: float,
    cluster_std: float,
    median_column: str,
    q25_column: str,
    q75_column: str,
    color,
    label: str | None = None,
) -> None:
    """Plot one lambda trajectory with an IQR band."""
    selected = summary.loc[
        (summary["lambda_sweep_value"] == lambda_value)
        & (summary["cluster_std"] == cluster_std)
    ].sort_values("label_noise_rate")

    x = selected["label_noise_rate"].to_numpy(dtype=float)
    median = selected[median_column].to_numpy(dtype=float)
    q25 = selected[q25_column].to_numpy(dtype=float)
    q75 = selected[q75_column].to_numpy(dtype=float)

    ax.plot(
        x,
        median,
        color=color,
        linewidth=1.35,
        marker="o",
        markersize=3.6,
        label=label,
        zorder=3,
    )

    ax.fill_between(
        x,
        q25,
        q75,
        color=color,
        alpha=0.10,
        linewidth=0.0,
        zorder=1,
    )


def uniform_reference_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one uniform RMSE trajectory per structure/noise condition.

    Uniform allocation is independent of lambda. Because the campaigns use
    matched seeds, we pool the uniform realizations across lambda values only
    for visualization of the reference trajectory.
    """
    grouped = df.groupby(
        [
            "cluster_std",
            "label_noise_rate",
        ],
        sort=True,
    )["uniform_func_rmse"]

    return grouped.agg(
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    ).reset_index()


def save_main_figure(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save the lambda-ablation diagnostic figure."""
    lambda_values = sorted(
        float(value) for value in summary["lambda_sweep_value"].unique()
    )
    cluster_stds = sorted(float(value) for value in summary["cluster_std"].unique())
    noise_rates = sorted(float(value) for value in summary["label_noise_rate"].unique())

    colors = lambda_colors(lambda_values)
    uniform_summary = uniform_reference_summary(df)

    fig, axes = plt.subplots(
        2,
        len(cluster_stds),
        figsize=(
            3.15 * len(cluster_stds),
            5.6,
        ),
        sharex=True,
        squeeze=False,
    )

    for column_index, cluster_std in enumerate(cluster_stds):
        rmse_ax = axes[0, column_index]
        jaccard_ax = axes[1, column_index]

        # ------------------------------------------------------
        # Top: adaptive absolute reconstruction error
        # ------------------------------------------------------
        for lambda_value in lambda_values:
            plot_quantile_curve(
                rmse_ax,
                summary,
                lambda_value=lambda_value,
                cluster_std=cluster_std,
                median_column=("adaptive_rmse_median"),
                q25_column=("adaptive_rmse_q25"),
                q75_column=("adaptive_rmse_q75"),
                color=colors[lambda_value],
                label=(rf"$\lambda={lambda_value:g}$"),
            )

        uniform_selected = uniform_summary.loc[
            uniform_summary["cluster_std"] == cluster_std
        ].sort_values("label_noise_rate")

        x_uniform = uniform_selected["label_noise_rate"].to_numpy(dtype=float)

        rmse_ax.plot(
            x_uniform,
            uniform_selected["median"].to_numpy(dtype=float),
            color="black",
            linestyle="--",
            linewidth=1.15,
            label="Uniform",
            zorder=2,
        )

        rmse_ax.fill_between(
            x_uniform,
            uniform_selected["q25"].to_numpy(dtype=float),
            uniform_selected["q75"].to_numpy(dtype=float),
            color="black",
            alpha=0.06,
            linewidth=0.0,
            zorder=1,
        )

        rmse_ax.set_yscale("log")
        rmse_ax.set_title(
            rf"$\sigma={cluster_std:g}$",
            fontsize=10,
        )

        if column_index == 0:
            rmse_ax.set_ylabel(
                r"Normalized decision RMSE "
                r"$\mathrm{RMSE}_f$",
                fontsize=8.8,
            )

        # ------------------------------------------------------
        # Bottom: weighted SV recovery
        # ------------------------------------------------------
        for lambda_value in lambda_values:
            plot_quantile_curve(
                jaccard_ax,
                summary,
                lambda_value=lambda_value,
                cluster_std=cluster_std,
                median_column=("weighted_jaccard_median"),
                q25_column=("weighted_jaccard_q25"),
                q75_column=("weighted_jaccard_q75"),
                color=colors[lambda_value],
            )

        # Uniform support-vector recovery as a reference.
        uniform_jaccard = (
            df.loc[df["cluster_std"] == cluster_std]
            .groupby(
                "label_noise_rate",
                sort=True,
            )["uniform_jaccard_weighted"]
            .median()
            .reset_index()
        )

        jaccard_ax.plot(
            uniform_jaccard["label_noise_rate"].to_numpy(dtype=float),
            uniform_jaccard["uniform_jaccard_weighted"].to_numpy(dtype=float),
            color="black",
            linestyle="--",
            linewidth=1.15,
            zorder=2,
        )

        jaccard_ax.set_ylim(
            -0.02,
            1.02,
        )

        if column_index == 0:
            jaccard_ax.set_ylabel(
                r"Weighted SV Jaccard "
                r"$J_{\mathrm{SV}}^w$",
                fontsize=8.8,
            )

        jaccard_ax.set_xlabel(
            r"Label-noise rate $\eta$",
            fontsize=8.8,
        )

        # ------------------------------------------------------
        # Common formatting
        # ------------------------------------------------------
        for ax in (
            rmse_ax,
            jaccard_ax,
        ):
            ax.set_xticks(noise_rates)
            ax.set_xticklabels([f"{value:g}" for value in noise_rates])

            ax.tick_params(
                axis="both",
                which="major",
                labelsize=7.5,
                width=0.8,
            )

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

            ax.grid(False)
            ax.margins(x=0.03)

    axes[0, -1].legend(
        title=(
            r"$\lambda$: "
            r"sensitivity $\rightarrow$ "
            r"instability"
        ),
        fontsize=6.7,
        title_fontsize=7.0,
        frameon=False,
        loc="best",
    )

    fig.tight_layout(
        pad=0.7,
        w_pad=0.8,
        h_pad=0.8,
    )

    fig.savefig(
        output_dir / "lambda_sweep_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "lambda_sweep_diagnostics.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def print_compact_summary(
    summary: pd.DataFrame,
) -> None:
    """Print primary lambda-ablation quantities."""
    columns = [
        "lambda_sweep_value",
        "cluster_std",
        "label_noise_rate",
        "adaptive_rmse_median",
        "advantage_median",
        "adaptive_win_rate",
        "weighted_jaccard_median",
        "margin_error_median",
    ]

    print("=== Lambda-sweep summary ===")
    print(summary[columns].to_string(index=False))


def main() -> None:
    """Combine lambda campaigns and create diagnostics."""
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lambda_values = sorted(set(float(value) for value in args.lambda_values))

    combined = load_campaigns(
        args.base_dir,
        lambda_values,
    )

    combined_path = args.output_dir / "lambda_sweep_combined_results.csv"
    combined.to_csv(
        combined_path,
        index=False,
    )

    summary = summarize_results(combined)

    summary_path = args.output_dir / "lambda_sweep_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
    )

    print_compact_summary(summary)

    save_main_figure(
        combined,
        summary,
        args.output_dir,
    )

    print()
    print("Saved:")
    print(f"- Combined raw data: {combined_path}")
    print(f"- Summary:           {summary_path}")
    print(f"- Figure:            {args.output_dir / 'lambda_sweep_diagnostics.pdf'}")


if __name__ == "__main__":
    main()
