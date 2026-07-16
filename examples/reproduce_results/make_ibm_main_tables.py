import math

import numpy as np
import pandas as pd

FINAL_FILE = "outputs/toy_analysis/tables/summary_final_mean_std.csv"
WINS_FILE = "outputs/toy_analysis/tables/win_counts.csv"
VAR_FILE = "outputs/toy_analysis/tables/margin_variance.csv"


MAIN_METRICS = [
    ("rmse_K", "Full-kernel RMSE", "error"),
    ("rmse_K_SV", "SV-block RMSE", "error"),
    ("jaccard_weighted", "Weighted Jaccard", "score"),
    ("margin_err", "Margin error", "error"),
    ("func_rmse", "Decision RMSE", "error"),
]


def fmt_pm(mean, std, ndigits=4):
    return rf"${mean:.{ndigits}f} \pm {std:.{ndigits}f}$"


def fmt_num(x, ndigits=4):
    return rf"${x:.{ndigits}f}$"


def fmt_pvalue(p):
    """
    Compact p-value formatting for LaTeX tables.
    """
    if p < 1e-3:
        return rf"${p:.1e}$"
    else:
        return rf"${p:.4f}$"


def fmt_percent(x, ndigits=1):
    return rf"${100.0 * x:.{ndigits}f}\%$"


def compute_relative_effect(adaptive, uniform, kind):
    """
    Returns positive values when adaptive is better.

    For error metrics:
        relative improvement = (uniform - adaptive) / uniform

    For score metrics:
        relative improvement = (adaptive - uniform) / uniform
    """
    if kind == "error":
        return (uniform - adaptive) / uniform
    elif kind == "score":
        return (adaptive - uniform) / uniform
    else:
        raise ValueError(f"Unknown metric kind: {kind}")


def binom_cdf(k, n, p=0.5):
    """
    CDF of Binomial(n, p) at k, implemented directly to avoid scipy dependency.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0

    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))


def paired_sign_test_pvalue(adaptive_wins, uniform_wins):
    """
    Two-sided paired sign-test p-value.

    Ties are excluded. Under the null hypothesis, adaptive and uniform are
    equally likely to win each non-tied paired comparison.

    The two-sided p-value is:
        2 * min(P[X <= k], P[X >= k]),
    capped at 1.
    """
    adaptive_wins = int(adaptive_wins)
    uniform_wins = int(uniform_wins)

    n_eff = adaptive_wins + uniform_wins

    if n_eff == 0:
        return np.nan

    k = adaptive_wins

    left_tail = binom_cdf(k, n_eff, p=0.5)
    right_tail = 1.0 - binom_cdf(k - 1, n_eff, p=0.5)

    return min(1.0, 2.0 * min(left_tail, right_tail))


def load_inputs():
    final = pd.read_csv(FINAL_FILE)
    wins = pd.read_csv(WINS_FILE)
    var = pd.read_csv(VAR_FILE)
    return final, wins, var


def build_hardware_summary_table(final, wins):
    rows = []

    for metric, label, kind in MAIN_METRICS:
        a_mean = final.loc[final["strategy"] == "adaptive", f"{metric}_mean"].iloc[0]
        a_std = final.loc[final["strategy"] == "adaptive", f"{metric}_std"].iloc[0]

        u_mean = final.loc[final["strategy"] == "uniform", f"{metric}_mean"].iloc[0]
        u_std = final.loc[final["strategy"] == "uniform", f"{metric}_std"].iloc[0]

        w = wins.loc[wins["metric"] == metric].iloc[0]

        adaptive_wins = int(w["adaptive_wins"])
        uniform_wins = int(w["uniform_wins"])
        n_total = int(w["n"])

        rel = compute_relative_effect(a_mean, u_mean, kind)
        p_sign = paired_sign_test_pvalue(adaptive_wins, uniform_wins)

        rows.append(
            {
                "Metric": label,
                "Adaptive": fmt_pm(a_mean, a_std),
                "Uniform": fmt_pm(u_mean, u_std),
                "Rel. adaptive effect": fmt_percent(rel),
                "Adaptive wins": rf"${adaptive_wins}/{n_total}$",
                "Sign-test $p$": fmt_pvalue(p_sign),
            }
        )

    return pd.DataFrame(rows)


def write_hardware_summary_latex(
    table_df,
    out_file="outputs/toy_analysis/tables/ibm_hardware_summary_table.tex",
):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(
        r"IBM hardware results over 15 independent paired runs. "
        r"Values are reported as mean \(\pm\) standard deviation. "
        r"The relative adaptive effect is positive when adaptive allocation improves "
        r"over uniform allocation; for error metrics this corresponds to relative "
        r"error reduction, whereas for weighted Jaccard it corresponds to relative "
        r"score increase. The final column reports two-sided paired sign-test "
        r"\(p\)-values computed from the run-wise win counts, with ties excluded "
        r"from the test."
    )
    lines.append(r"}")
    lines.append(r"\label{tab:ibm_hardware_summary}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Metric & Adaptive & Uniform & Rel. adaptive effect & "
        r"Adaptive wins & Sign-test \(p\) \\"
    )
    lines.append(r"\midrule")

    for _, row in table_df.iterrows():
        lines.append(
            f"{row['Metric']} & {row['Adaptive']} & {row['Uniform']} & "
            f"{row['Rel. adaptive effect']} & {row['Adaptive wins']} & "
            f"{row['Sign-test $p$']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_margin_variance_table(var):
    ratio = var["adaptive_over_uniform_var_margin"]
    oracle_ratio = var["oracle_over_uniform_var_margin"]

    summary = {
        "Adaptive / uniform": {
            "Mean": ratio.mean(),
            "Std.": ratio.std(ddof=1),
            "Min.": ratio.min(),
            "Max.": ratio.max(),
        },
        "Oracle / uniform": {
            "Mean": oracle_ratio.mean(),
            "Std.": oracle_ratio.std(ddof=1),
            "Min.": oracle_ratio.min(),
            "Max.": oracle_ratio.max(),
        },
    }

    rows = []
    for name, stats in summary.items():
        rows.append(
            {
                "Ratio": name,
                "Mean": fmt_percent(stats["Mean"]),
                "Std.": fmt_percent(stats["Std."]),
                "Min.": fmt_percent(stats["Min."]),
                "Max.": fmt_percent(stats["Max."]),
            }
        )

    return pd.DataFrame(rows)


def write_margin_variance_latex(
    table_df,
    out_file="outputs/toy_analysis/tables/ibm_margin_variance_table.tex",
):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(
        r"Predicted margin-variance ratios on IBM hardware, computed using "
        r"the reference dual coefficients, the reference kernel, and the final "
        r"allocations from each run. Ratios are reported relative to uniform "
        r"allocation."
    )
    lines.append(r"}")
    lines.append(r"\label{tab:ibm_margin_variance}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Ratio & Mean & Std. & Min. & Max. \\")
    lines.append(r"\midrule")

    for _, row in table_df.iterrows():
        lines.append(
            f"{row['Ratio']} & {row['Mean']} & {row['Std.']} & "
            f"{row['Min.']} & {row['Max.']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    final, wins, var = load_inputs()

    hardware_summary = build_hardware_summary_table(final, wins)
    margin_variance_summary = build_margin_variance_table(var)

    write_hardware_summary_latex(hardware_summary)
    write_margin_variance_latex(margin_variance_summary)

    print("Wrote ibm_hardware_summary_table.tex")
    print("Wrote ibm_margin_variance_table.tex")
    print()
    print("Hardware summary:")
    print(hardware_summary.to_string(index=False))
    print()
    print("Margin-variance summary:")
    print(margin_variance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
