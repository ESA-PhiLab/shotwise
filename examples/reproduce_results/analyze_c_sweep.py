"""Combine and analyze structural-mechanism campaigns across SVM C values.

The script expects one structural_mechanism_results.csv file for each tested
regularization value C. It combines the raw results, summarizes the main
quantities by C, cluster overlap, and label-noise rate, and creates a compact
four-panel robustness figure.

Panels:
(a) adaptive advantage,
(b) oracle support-vector fraction,
(c) bounded-support-vector fraction,
(d) oracle allocation-weight Gini coefficient.

Each column of the figure corresponds to one structural regime (cluster std.),
while color identifies C.

Example
-------
From the repository root:

    python examples/reproduce_results/analyze_c_sweep.py \
        --base-dir outputs/structural_mechanism_C_sweep \
        --c-values 0.1 1 10 100 \
        --output-dir outputs/structural_mechanism_C_sweep/analysis
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
        description="Combine and analyze structural-mechanism C sweeps."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing C_<value>/structural_mechanism_results.csv "
            "subdirectories."
        ),
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/structural_mechanism_C_sweep/analysis"),
    )
    return parser.parse_args()


def c_directory_name(c_value: float) -> str:
    """Return the directory name used by the campaign shell loop."""
    if float(c_value).is_integer():
        return f"C_{int(c_value)}"
    return f"C_{c_value:g}"


def validate_dataframe(
    df: pd.DataFrame,
    *,
    expected_c: float,
) -> None:
    """Validate one C-sweep raw result file."""
    required = {
        "run",
        "cluster_std",
        "label_noise_rate",
        "c_value",
        "sv_fraction",
        "bounded_sv_fraction",
        "weight_gini",
        "adaptive_advantage_log10",
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

    for column in required.difference({"run"}):
        values = df[column].to_numpy(dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(f"Column {column!r} contains non-finite values.")

    observed_c = df["c_value"].to_numpy(dtype=float)

    if not np.allclose(
        observed_c,
        expected_c,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"Expected C={expected_c:g}, but the CSV contains "
            f"C values {sorted(df['c_value'].unique())}."
        )


def load_campaigns(
    base_dir: Path,
    c_values: list[float],
) -> pd.DataFrame:
    """Load and concatenate all requested C-sweep campaigns."""
    frames: list[pd.DataFrame] = []

    for c_value in c_values:
        directory = base_dir / c_directory_name(c_value)
        csv_path = directory / "structural_mechanism_results.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing C={c_value:g} result file: {csv_path}")

        print(f"Loading C={c_value:g}: {csv_path}")

        df = pd.read_csv(csv_path)

        validate_dataframe(
            df,
            expected_c=c_value,
        )

        df = df.copy()
        df["c_sweep_value"] = float(c_value)

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the main C-robustness diagnostics."""
    group_columns = [
        "c_sweep_value",
        "cluster_std",
        "label_noise_rate",
    ]

    grouped = df.groupby(
        group_columns,
        sort=True,
    )

    summary = grouped.agg(
        repetitions=("run", "count"),
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
        sv_fraction_median=(
            "sv_fraction",
            "median",
        ),
        sv_fraction_q25=(
            "sv_fraction",
            lambda values: values.quantile(0.25),
        ),
        sv_fraction_q75=(
            "sv_fraction",
            lambda values: values.quantile(0.75),
        ),
        bounded_sv_fraction_median=(
            "bounded_sv_fraction",
            "median",
        ),
        bounded_sv_fraction_q25=(
            "bounded_sv_fraction",
            lambda values: values.quantile(0.25),
        ),
        bounded_sv_fraction_q75=(
            "bounded_sv_fraction",
            lambda values: values.quantile(0.75),
        ),
        weight_gini_median=(
            "weight_gini",
            "median",
        ),
        weight_gini_q25=(
            "weight_gini",
            lambda values: values.quantile(0.25),
        ),
        weight_gini_q75=(
            "weight_gini",
            lambda values: values.quantile(0.75),
        ),
        adaptive_rmse_median=(
            "adaptive_func_rmse",
            "median",
        ),
        uniform_rmse_median=(
            "uniform_func_rmse",
            "median",
        ),
    )

    return summary.reset_index()


def c_colors(
    c_values: list[float],
) -> dict[float, object]:
    """Assign consistent colors to C values."""
    cmap = plt.get_cmap("viridis")

    if len(c_values) == 1:
        positions = [0.5]
    else:
        positions = np.linspace(
            0.08,
            0.88,
            len(c_values),
        )

    return {
        c_value: cmap(position)
        for c_value, position in zip(
            c_values,
            positions,
            strict=True,
        )
    }


def plot_summary_curve(
    ax,
    summary: pd.DataFrame,
    *,
    c_value: float,
    cluster_std: float,
    median_column: str,
    q25_column: str,
    q75_column: str,
    color,
    label: str | None = None,
) -> None:
    """Plot one median C-sweep trajectory with its interquartile range."""
    selected = summary.loc[
        (summary["c_sweep_value"] == c_value) & (summary["cluster_std"] == cluster_std)
    ].sort_values("label_noise_rate")

    x = selected["label_noise_rate"].to_numpy(dtype=float)
    median = selected[median_column].to_numpy(dtype=float)
    q25 = selected[q25_column].to_numpy(dtype=float)
    q75 = selected[q75_column].to_numpy(dtype=float)

    ax.plot(
        x,
        median,
        color=color,
        linewidth=1.3,
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


def save_robustness_figure(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save a C-robustness figure with one column per structural regime."""
    c_values = sorted(float(value) for value in summary["c_sweep_value"].unique())
    cluster_stds = sorted(float(value) for value in summary["cluster_std"].unique())
    noise_rates = sorted(float(value) for value in summary["label_noise_rate"].unique())

    colors = c_colors(c_values)

    n_columns = len(cluster_stds)

    fig, axes = plt.subplots(
        4,
        n_columns,
        figsize=(3.0 * n_columns, 8.8),
        sharex=True,
        squeeze=False,
    )

    panel_specs = [
        (
            "advantage_median",
            "advantage_q25",
            "advantage_q75",
            r"Adaptive advantage $A_f$",
        ),
        (
            "sv_fraction_median",
            "sv_fraction_q25",
            "sv_fraction_q75",
            r"SV fraction $|S^\star|/n$",
        ),
        (
            "bounded_sv_fraction_median",
            "bounded_sv_fraction_q25",
            "bounded_sv_fraction_q75",
            "Bounded-SV fraction",
        ),
        (
            "weight_gini_median",
            "weight_gini_q25",
            "weight_gini_q75",
            r"Weight Gini $G(w^\star)$",
        ),
    ]

    for column_index, cluster_std in enumerate(cluster_stds):
        for row_index, (
            median_column,
            q25_column,
            q75_column,
            ylabel,
        ) in enumerate(panel_specs):
            ax = axes[row_index, column_index]

            for c_value in c_values:
                plot_summary_curve(
                    ax,
                    summary,
                    c_value=c_value,
                    cluster_std=cluster_std,
                    median_column=median_column,
                    q25_column=q25_column,
                    q75_column=q75_column,
                    color=colors[c_value],
                    label=rf"$C={c_value:g}$",
                )

            if row_index == 0:
                ax.set_title(
                    rf"$\sigma={cluster_std:g}$",
                    fontsize=10,
                )

            if column_index == 0:
                ax.set_ylabel(
                    ylabel,
                    fontsize=8.5,
                )

            if row_index == 0:
                ax.axhline(
                    0.0,
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    zorder=0,
                )
            else:
                ax.set_ylim(-0.02, 1.02)

            if row_index == len(panel_specs) - 1:
                ax.set_xlabel(
                    r"Label-noise rate $\eta$",
                    fontsize=8.5,
                )

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
        title=r"Regularization $C$",
        fontsize=7.0,
        title_fontsize=7.5,
        frameon=False,
        loc="best",
    )

    fig.tight_layout(
        pad=0.7,
        w_pad=0.8,
        h_pad=0.8,
    )

    fig.savefig(
        output_dir / "c_sweep_robustness.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "c_sweep_robustness.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def print_compact_summary(
    summary: pd.DataFrame,
) -> None:
    """Print the primary quantities used to assess qualitative robustness."""
    columns = [
        "c_sweep_value",
        "cluster_std",
        "label_noise_rate",
        "advantage_median",
        "adaptive_win_rate",
        "sv_fraction_median",
        "bounded_sv_fraction_median",
        "weight_gini_median",
    ]

    print("=== C-sweep summary ===")
    print(summary[columns].to_string(index=False))


def main() -> None:
    """Combine C-sweep campaigns and produce robustness analysis."""
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    c_values = sorted(set(float(value) for value in args.c_values))

    combined = load_campaigns(
        args.base_dir,
        c_values,
    )

    combined_path = args.output_dir / "c_sweep_combined_results.csv"
    combined.to_csv(
        combined_path,
        index=False,
    )

    summary = summarize_results(combined)

    summary_path = args.output_dir / "c_sweep_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
    )

    print_compact_summary(summary)

    save_robustness_figure(
        summary,
        args.output_dir,
    )

    print()
    print("Saved:")
    print(f"- Combined raw data: {combined_path}")
    print(f"- Summary:           {summary_path}")
    print(f"- Figure:            {args.output_dir / 'c_sweep_robustness.pdf'}")


if __name__ == "__main__":
    main()
