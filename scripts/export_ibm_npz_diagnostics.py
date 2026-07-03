from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def matrix_to_long_dataframe(
    matrix: np.ndarray, *, matrix_name: str, call: int | None = None
):
    rows = []
    n, m = matrix.shape
    for i in range(n):
        for j in range(m):
            rows.append(
                {
                    "matrix": matrix_name,
                    "call": call,
                    "i": i,
                    "j": j,
                    "value": float(matrix[i, j]),
                }
            )
    return pd.DataFrame(rows)


def export_measurements(base: Path) -> None:
    path = base / "adaptive_measurements.npz"
    if not path.exists():
        print(f"Missing {path}")
        return

    data = np.load(path)
    requested = data["requested_shots"]
    counts = data["measured_counts"]

    all_rows = []
    for call in range(requested.shape[0]):
        shots_call = requested[call]
        counts_call = counts[call]

        with np.errstate(divide="ignore", invalid="ignore"):
            probs_call = np.where(shots_call > 0, counts_call / shots_call, np.nan)

        all_rows.append(
            matrix_to_long_dataframe(
                shots_call, matrix_name="requested_shots", call=call
            )
        )
        all_rows.append(
            matrix_to_long_dataframe(
                counts_call, matrix_name="measured_counts", call=call
            )
        )
        all_rows.append(
            matrix_to_long_dataframe(
                probs_call, matrix_name="empirical_probabilities", call=call
            )
        )

        pd.DataFrame(shots_call).to_csv(
            base / f"adaptive_requested_shots_call_{call}.csv", index=False
        )
        pd.DataFrame(counts_call).to_csv(
            base / f"adaptive_counts_call_{call}.csv", index=False
        )
        pd.DataFrame(probs_call).to_csv(
            base / f"adaptive_probabilities_call_{call}.csv", index=False
        )

    long_df = pd.concat(all_rows, ignore_index=True)
    long_df.to_csv(base / "adaptive_measurements_long.csv", index=False)

    summary = []
    for call in range(requested.shape[0]):
        summary.append(
            {
                "call": int(call),
                "total_requested_shots": int(requested[call].sum()),
                "total_measured_successes": int(counts[call].sum()),
                "mean_empirical_probability": float(
                    np.nanmean(
                        np.where(
                            requested[call] > 0, counts[call] / requested[call], np.nan
                        )
                    )
                ),
                "n_active_pairs": int(np.sum(requested[call] > 0)),
            }
        )

    with (base / "adaptive_measurements_export_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"Exported adaptive measurement diagnostics to {base}")


def export_supportshot_history(base: Path) -> None:
    path = base / "adaptive_supportshot_history.npz"
    if not path.exists():
        print(f"Missing {path}")
        return

    data = np.load(path)
    allocations = data["allocation_matrices"]
    kernels = data["kernel_matrices"]

    all_rows = []
    for idx in range(allocations.shape[0]):
        all_rows.append(
            matrix_to_long_dataframe(
                allocations[idx], matrix_name="allocation", call=idx
            )
        )
        all_rows.append(
            matrix_to_long_dataframe(kernels[idx], matrix_name="kernel", call=idx)
        )

        pd.DataFrame(allocations[idx]).to_csv(
            base / f"history_allocation_{idx}.csv", index=False
        )
        pd.DataFrame(kernels[idx]).to_csv(
            base / f"history_kernel_{idx}.csv", index=False
        )

    long_df = pd.concat(all_rows, ignore_index=True)
    long_df.to_csv(base / "adaptive_supportshot_history_long.csv", index=False)

    print(f"Exported SupportShot history diagnostics to {base}")


def export_reference(base: Path) -> None:
    path = base / "dataset_and_reference.npz"
    if not path.exists():
        print(f"Missing {path}")
        return

    data = np.load(path)
    x = data["x"]
    y = data["y"]
    kernel_true = data["kernel_true"]

    pd.DataFrame(x).to_csv(base / "dataset_x.csv", index=False)
    pd.DataFrame({"y": y}).to_csv(base / "dataset_y.csv", index=False)
    pd.DataFrame(kernel_true).to_csv(
        base / "kernel_statevector_reference.csv", index=False
    )

    print(f"Exported dataset/reference diagnostics to {base}")


def main() -> None:
    base = Path("outputs/qiskit_toy6_ibm_shot_smoke")
    export_measurements(base)
    export_supportshot_history(base)
    export_reference(base)


if __name__ == "__main__":
    main()
