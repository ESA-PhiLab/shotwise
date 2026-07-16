from pathlib import Path

import pandas as pd

IN_FILE = Path("outputs/toy_analysis/tables/paired_final_comparisons.csv")
OUT_FILE = Path("outputs/toy_analysis/tables/ibm_run_by_run_table.tex")


METRICS = [
    ("rmse_K", r"$\mathrm{RMSE}(K)$", "lower"),
    ("rmse_K_SV", r"$\mathrm{RMSE}(K_{\mathrm{SV}})$", "lower"),
    ("jaccard_weighted", r"$J_{\mathrm{SV}}^{w}$", "higher"),
    ("margin_err", "Margin err.", "lower"),
    ("func_rmse", "Decision RMSE", "lower"),
]


def fmt_pair(a, u, ndigits=4):
    return rf"${a:.{ndigits}f}/{u:.{ndigits}f}$"


def simplify_run_name(name):
    # Example: ibm_toy8_b40_8_r3_run13 -> run13
    if "run" in name:
        return name[name.rfind("run") :]
    return name


def main():
    df = pd.read_csv(IN_FILE)

    # Keep IBM only.
    if "source" in df.columns:
        df = df[df["source"] == "ibm"].copy()

    # Sort naturally by run number if possible.
    def run_sort_key(name):
        text = str(name)
        if "run" in text:
            suffix = text[text.rfind("run") + 3 :]
            try:
                return int(suffix)
            except ValueError:
                return 9999
        return 9999

    df["run_sort"] = df["run_name"].map(run_sort_key)
    df = df.sort_values("run_sort")

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(
        r"Run-by-run IBM hardware results for the final adaptive allocation and "
        r"the uniform baseline. Each entry reports adaptive/uniform performance "
        r"for the same paired hardware run. Lower values are better for RMSE and "
        r"margin-error metrics, whereas higher values are better for weighted "
        r"Jaccard."
    )
    lines.append(r"}")
    lines.append(r"\label{tab:appendix_ibm_run_by_run}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")

    header = "Run"
    for _, label, _ in METRICS:
        header += f" & {label}"
    header += r" \\"
    lines.append(header)

    lines.append(r"\midrule")

    for _, row in df.iterrows():
        run_label = simplify_run_name(row["run_name"])
        cells = [run_label]

        for metric, _, _ in METRICS:
            a_col = f"adaptive_{metric}"
            u_col = f"uniform_{metric}"

            if a_col not in row or u_col not in row:
                cells.append("--")
            else:
                cells.append(fmt_pair(float(row[a_col]), float(row[u_col])))

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
