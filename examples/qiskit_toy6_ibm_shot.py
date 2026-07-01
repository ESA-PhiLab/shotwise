"""Toy6 IBM hardware experiment using QiskitShotBackend.

This script runs the true-shot version of the toy6 SupportShot experiment on an
actual IBM backend. It records enough provenance to reconstruct and analyse the
measurement campaign offline:

- exact statevector reference kernel,
- dataset and labels,
- adaptive and uniform requested shot matrices,
- measured all-zero count matrices,
- reconstructed kernel matrices from the stores,
- SupportShot history snapshots,
- Runtime job metadata recorded by QiskitShotBackend,
- metrics comparing hardware-estimated kernels to the statevector reference,
- diagnostic heatmaps for allocations, counts, and kernels.

The script uses job mode, not session or batch mode. This is the safest option
for IBM Open Plan users because adaptive rounds are sequential.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluation_utils import (
    evaluate_estimated_kernel,
    print_summary,
    records_to_dataframe,
)
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from shotwise.algorithms.supportshot import SupportShot
from shotwise.backends.qiskit_shot_backend import QiskitShotBackend
from shotwise.learning.svm import fit_precomputed_svm_state
from shotwise.storage.kernel_store import KernelStore
from shotwise.utils.matrix import upper_triangle_indices


class RecordingShotBackend:
    """Thin wrapper that records all shot requests and returned counts.

    SupportShot only needs a backend with ``sample_upper_triangle_counts``.
    This wrapper delegates the real work to QiskitShotBackend, but stores every
    requested shot matrix and every measured count matrix for offline analysis.
    """

    def __init__(self, backend: QiskitShotBackend, *, name: str) -> None:
        self.backend = backend
        self.name = name
        self.measurement_records: list[dict[str, Any]] = []

    @property
    def n(self) -> int:
        """Number of samples exposed for SupportShot dimension inference."""
        return self.backend.n

    @property
    def job_history_(self):
        return self.backend.job_history_

    def sample_upper_triangle_counts(
        self,
        shot_matrix: np.ndarray,
        *,
        round_id: int | None = None,
    ) -> np.ndarray:
        shot_matrix = np.asarray(shot_matrix, dtype=int)
        before_jobs = len(self.backend.job_history_)

        counts = self.backend.sample_upper_triangle_counts(
            shot_matrix,
            round_id=round_id,
        )
        counts = np.asarray(counts, dtype=int)

        after_jobs = len(self.backend.job_history_)
        new_jobs = self.backend.job_history_[before_jobs:after_jobs]

        record = {
            "backend_name": self.name,
            "call_index": len(self.measurement_records),
            "round_id": round_id,
            "requested_shots": shot_matrix.copy(),
            "measured_counts": counts.copy(),
            "total_requested_shots": int(shot_matrix.sum()),
            "total_measured_successes": int(counts.sum()),
            "job_history_start": before_jobs,
            "job_history_end": after_jobs,
            "job_records": [serialize_job_record(job) for job in new_jobs],
        }
        self.measurement_records.append(record)
        return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the toy6 SupportShot experiment on an IBM backend using "
            "direct sampler shots."
        ),
    )
    parser.add_argument("--backend", type=str, default="ibm_fez")
    parser.add_argument("--shots-per-entry", type=int, default=30)
    parser.add_argument("--pilot-shots", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument(
        "--allocation-method",
        choices=["multinomial", "deterministic"],
        default="deterministic",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-circuits-per-job",
        type=int,
        default=128,
        help="Maximum circuits submitted in one sampler call.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
    )
    parser.add_argument(
        "--group-by-shots",
        action="store_true",
        help=(
            "Use grouped run-level shots instead of PUB-specific shots. "
            "This can be safer but may create more jobs."
        ),
    )
    parser.add_argument(
        "--skip-uniform",
        action="store_true",
        help="Run only adaptive. Useful if IBM quota is limited.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qiskit_toy6_ibm_shot"),
    )
    return parser.parse_args()


def build_toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed 6-point toy dataset in angle space."""
    x = np.array(
        [
            [0.10, 0.15],
            [0.28, 0.10],
            [0.55, 0.55],
            [1.35, 1.25],
            [1.10, 1.38],
            [0.82, 0.88],
        ],
        dtype=float,
    )
    y = np.array([-1, -1, -1, 1, 1, 1], dtype=int)
    return x, y


def build_feature_map() -> tuple[QuantumCircuit, list[Any]]:
    """Build a shallow 2-qubit angle-encoding feature map."""
    theta = ParameterVector("x", 2)
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cz(0, 1)
    return qc, list(theta)


def total_budget_from_shots_per_entry(n_samples: int, shots_per_entry: int) -> int:
    n_pairs = n_samples * (n_samples - 1) // 2
    return n_pairs * shots_per_entry


def uniform_allocation_matrix(n: int, total_budget: int) -> np.ndarray:
    if total_budget < 0:
        raise ValueError(f"total_budget must be non-negative, got {total_budget}.")

    rows, cols = upper_triangle_indices(n)
    n_pairs = rows.size
    if n_pairs == 0:
        return np.zeros((n, n), dtype=int)

    base = total_budget // n_pairs
    remainder = total_budget % n_pairs
    flat = np.full(n_pairs, base, dtype=int)
    if remainder > 0:
        flat[:remainder] += 1

    allocation = np.zeros((n, n), dtype=int)
    allocation[rows, cols] = flat
    return allocation


def method_label_from_round(stage: str, round_index: int) -> str:
    if stage == "pilot":
        return "pilot"
    return f"round_{round_index}"


def make_reference_kernel(feature_map: QuantumCircuit) -> FidelityQuantumKernel:
    """Create exact local statevector reference kernel."""
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    return FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
        max_circuits_per_job=None,
    )


def make_ibm_shot_backend(
    x: np.ndarray,
    feature_map: QuantumCircuit,
    input_parameters: list[Any],
    *,
    ibm_backend,
    optimization_level: int,
    max_circuits_per_job: int,
    use_pub_shots: bool,
) -> QiskitShotBackend:
    """Create the direct-shot backend for an IBM backend in job mode."""
    pass_manager = generate_preset_pass_manager(
        backend=ibm_backend,
        optimization_level=optimization_level,
    )
    sampler = Sampler(mode=ibm_backend)

    return QiskitShotBackend(
        x,
        feature_map=feature_map,
        sampler=sampler,
        pass_manager=pass_manager,
        input_parameters=input_parameters,
        max_circuits_per_job=max_circuits_per_job,
        use_pub_shots=use_pub_shots,
    )


def run_uniform_baseline(
    x: np.ndarray,
    feature_map: QuantumCircuit,
    input_parameters: list[Any],
    *,
    ibm_backend,
    total_budget: int,
    optimization_level: int,
    max_circuits_per_job: int,
    use_pub_shots: bool,
) -> tuple[np.ndarray, int, RecordingShotBackend]:
    """Estimate the kernel uniformly using QiskitShotBackend."""
    raw_backend = make_ibm_shot_backend(
        x,
        feature_map,
        input_parameters,
        ibm_backend=ibm_backend,
        optimization_level=optimization_level,
        max_circuits_per_job=max_circuits_per_job,
        use_pub_shots=use_pub_shots,
    )
    backend = RecordingShotBackend(raw_backend, name="uniform")

    store = KernelStore(x.shape[0], diagonal_value=1.0)
    allocation = uniform_allocation_matrix(x.shape[0], total_budget)
    counts = backend.sample_upper_triangle_counts(allocation, round_id=-1)
    store.update(counts, allocation)

    return store.kernel_matrix(), store.total_shots_spent(), backend


def save_dataset_plot(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, marker, name in [(-1, "o", "class -1"), (1, "s", "class +1")]:
        mask = y == label
        ax.scatter(x[mask, 0], x[mask, 1], marker=marker, label=name)
    for idx, (a0, a1) in enumerate(x):
        ax.text(a0 + 0.015, a1 + 0.015, str(idx), fontsize=8)
    ax.set_xlabel("angle feature 1")
    ax.set_ylabel("angle feature 2")
    ax.set_title("Toy6 dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(matrix: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix)
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel("sample index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_kernel_heatmap(matrix: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel("sample index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_line_plot(
    df: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    if metric not in df.columns:
        return

    order = ["pilot"] + [f"round_{r}" for r in sorted_rounds(df)] + ["uniform"]
    plot_df = df[df["method"].isin(order)].copy()
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=order)
    plot_df = plot_df.sort_values("method")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(plot_df["method"].astype(str), plot_df[metric], marker="o")
    ax.set_title(title)
    ax.set_xlabel("method / round")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def sorted_rounds(df: pd.DataFrame) -> list[int]:
    rounds: list[int] = []
    for value in df["method"].unique():
        if isinstance(value, str) and value.startswith("round_"):
            rounds.append(int(value.split("_")[1]))
    return sorted(rounds)


def serialize_job_record(job_record: Any) -> dict[str, Any]:
    if is_dataclass(job_record):
        return asdict(job_record)
    if hasattr(job_record, "__dict__"):
        return dict(job_record.__dict__)
    return {"repr": repr(job_record)}


def save_measurement_records(
    records: list[dict[str, Any]],
    *,
    output_prefix: Path,
) -> None:
    """Save measurement records as NPZ plus JSON metadata."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    requested = np.stack([r["requested_shots"] for r in records], axis=0)
    counts = np.stack([r["measured_counts"] for r in records], axis=0)

    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        requested_shots=requested,
        measured_counts=counts,
    )

    metadata = []
    for r in records:
        item = {
            key: value
            for key, value in r.items()
            if key not in {"requested_shots", "measured_counts"}
        }
        metadata.append(item)

    with output_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_supportshot_history(model: SupportShot, output_dir: Path) -> None:
    """Save allocations and kernels from SupportShot history."""
    allocations = []
    kernels = []
    metadata = []

    for record in model.history_:
        allocations.append(np.asarray(record.allocation_matrix, dtype=int))
        kernels.append(np.asarray(record.kernel_matrix, dtype=float))
        metadata.append(
            {
                "stage": record.stage,
                "round_index": int(record.round_index),
                "budget": int(record.budget),
                "total_shots_spent": int(record.total_shots_spent),
                "stopped_early": bool(record.stopped_early),
            }
        )

    np.savez_compressed(
        output_dir / "adaptive_supportshot_history.npz",
        allocation_matrices=np.stack(allocations, axis=0),
        kernel_matrices=np.stack(kernels, axis=0),
    )
    with (output_dir / "adaptive_supportshot_history.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)


def save_diagnostics(
    *,
    output_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    kernel_true: np.ndarray,
    adaptive_model: SupportShot,
    adaptive_recorder: RecordingShotBackend,
    uniform_kernel: np.ndarray | None,
    uniform_recorder: RecordingShotBackend | None,
) -> None:
    """Save plots and raw arrays for offline analysis."""
    np.savez_compressed(
        output_dir / "dataset_and_reference.npz",
        x=x,
        y=y,
        kernel_true=kernel_true,
    )

    save_supportshot_history(adaptive_model, output_dir)
    save_measurement_records(
        adaptive_recorder.measurement_records,
        output_prefix=output_dir / "adaptive_measurements",
    )

    for idx, record in enumerate(adaptive_model.history_):
        label = f"{record.stage}_{record.round_index}"
        save_heatmap(
            record.allocation_matrix,
            output_dir / f"allocation_{idx}_{label}.png",
            f"Adaptive allocation: {label}",
        )
        save_kernel_heatmap(
            record.kernel_matrix,
            output_dir / f"kernel_{idx}_{label}.png",
            f"Adaptive kernel: {label}",
        )

    for idx, measurement in enumerate(adaptive_recorder.measurement_records):
        save_heatmap(
            measurement["measured_counts"],
            output_dir / f"counts_adaptive_call_{idx}.png",
            f"Adaptive measured counts call {idx}",
        )

    save_kernel_heatmap(
        kernel_true,
        output_dir / "kernel_statevector_reference.png",
        "Statevector reference kernel",
    )

    if uniform_kernel is not None:
        save_kernel_heatmap(
            uniform_kernel,
            output_dir / "kernel_uniform.png",
            "Uniform IBM-estimated kernel",
        )

    if uniform_recorder is not None:
        save_measurement_records(
            uniform_recorder.measurement_records,
            output_prefix=output_dir / "uniform_measurements",
        )
        for idx, measurement in enumerate(uniform_recorder.measurement_records):
            save_heatmap(
                measurement["requested_shots"],
                output_dir / f"allocation_uniform_call_{idx}.png",
                f"Uniform requested shots call {idx}",
            )
            save_heatmap(
                measurement["measured_counts"],
                output_dir / f"counts_uniform_call_{idx}.png",
                f"Uniform measured counts call {idx}",
            )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = build_toy_dataset()
    feature_map, input_parameters = build_feature_map()
    total_budget = total_budget_from_shots_per_entry(
        x.shape[0],
        args.shots_per_entry,
    )
    use_pub_shots = not args.group_by_shots

    print("=== Dataset and feature map ===")
    print("Feature map:")
    print(feature_map.draw())
    print("\nPoints:")
    print(x)
    print("\nLabels:")
    print(y)

    save_dataset_plot(x, y, output_dir / "dataset.png")

    service = QiskitRuntimeService()
    ibm_backend = service.backend(args.backend)

    ref_kernel = make_reference_kernel(feature_map)
    kernel_true = np.asarray(ref_kernel.evaluate(x), dtype=float)
    np.fill_diagonal(kernel_true, 1.0)

    state_true = fit_precomputed_svm_state(kernel_true, y, c=args.c)

    print("\n=== Statevector reference diagnostics ===")
    print(f"num_support_vectors = {state_true.num_support} / {state_true.n_samples}")
    print(f"support_vector_ratio = {state_true.num_support / state_true.n_samples:.3f}")
    print("unsigned dual coefficients:")
    print(np.round(state_true.alphas, 6))
    print("support indices:")
    print(state_true.support_indices)

    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "backend_name": getattr(ibm_backend, "name", None),
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "independent_entries": int(x.shape[0] * (x.shape[0] - 1) // 2),
        "shots_per_entry": int(args.shots_per_entry),
        "total_budget": int(total_budget),
        "pilot_shots": int(args.pilot_shots),
        "rounds": int(args.rounds),
        "lambda_weight": float(args.lambda_weight),
        "C": float(args.c),
        "allocation_method": args.allocation_method,
        "max_circuits_per_job": int(args.max_circuits_per_job),
        "optimization_level": int(args.optimization_level),
        "use_pub_shots": bool(use_pub_shots),
        "skip_uniform": bool(args.skip_uniform),
        "seed": int(args.seed),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n=== IBM experiment configuration ===")
    for key, value in config.items():
        print(f"{key:24s} = {value}")

    raw_adaptive_backend = make_ibm_shot_backend(
        x,
        feature_map,
        input_parameters,
        ibm_backend=ibm_backend,
        optimization_level=args.optimization_level,
        max_circuits_per_job=args.max_circuits_per_job,
        use_pub_shots=use_pub_shots,
    )
    adaptive_backend = RecordingShotBackend(raw_adaptive_backend, name="adaptive")

    model = SupportShot(
        total_budget=total_budget,
        pilot_shots_per_entry=args.pilot_shots,
        rounds=args.rounds,
        lambda_weight=args.lambda_weight,
        svm_c=args.c,
        allocation_method=args.allocation_method,
        random_state=args.seed,
    )

    print("\n=== Running adaptive SupportShot on IBM backend ===")
    model.fit(y, adaptive_backend)

    records: list[dict[str, float | int | str]] = []
    for record in model.history_:
        entry = evaluate_estimated_kernel(
            kernel_true,
            record.kernel_matrix,
            y,
            c=args.c,
            method=method_label_from_round(record.stage, record.round_index),
            round_id=record.round_index,
        )
        entry.update(
            {
                "run": 0,
                "stage": record.stage,
                "budget": record.budget,
                "total_shots_spent": record.total_shots_spent,
                "stopped_early": record.stopped_early,
            }
        )
        records.append(entry)

    uniform_kernel = None
    uniform_recorder = None

    if not args.skip_uniform:
        print("\n=== Running uniform baseline on IBM backend ===")
        uniform_kernel, spent_uniform, uniform_recorder = run_uniform_baseline(
            x,
            feature_map,
            input_parameters,
            ibm_backend=ibm_backend,
            total_budget=total_budget,
            optimization_level=args.optimization_level,
            max_circuits_per_job=args.max_circuits_per_job,
            use_pub_shots=use_pub_shots,
        )

        uniform_entry = evaluate_estimated_kernel(
            kernel_true,
            uniform_kernel,
            y,
            c=args.c,
            method="uniform",
            round_id=-1,
        )
        uniform_entry.update(
            {
                "run": 0,
                "stage": "uniform",
                "budget": spent_uniform,
                "total_shots_spent": spent_uniform,
                "stopped_early": False,
            }
        )
        records.append(uniform_entry)

    df = records_to_dataframe(records)
    csv_path = output_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)

    save_diagnostics(
        output_dir=output_dir,
        x=x,
        y=y,
        kernel_true=kernel_true,
        adaptive_model=model,
        adaptive_recorder=adaptive_backend,
        uniform_kernel=uniform_kernel,
        uniform_recorder=uniform_recorder,
    )

    summary_df = df.copy()
    save_line_plot(
        summary_df,
        metric="func_rmse",
        output_path=output_dir / "func_rmse_progress.png",
        title="Decision-function RMSE vs round",
    )
    save_line_plot(
        summary_df,
        metric="margin_err",
        output_path=output_dir / "margin_err_progress.png",
        title="Margin error vs round",
    )
    save_line_plot(
        summary_df,
        metric="jaccard",
        output_path=output_dir / "jaccard_progress.png",
        title="Support-vector Jaccard vs round",
    )

    print("\n=== Summary ===")
    print_summary(df)

    print("\n=== Job records ===")
    all_job_records = []

    for measurement in adaptive_backend.measurement_records:
        for job_record in measurement["job_records"]:
            job_record = dict(job_record)
            job_record["arm"] = "adaptive"
            job_record["call_index"] = measurement["call_index"]
            all_job_records.append(job_record)

    if uniform_recorder is not None:
        for measurement in uniform_recorder.measurement_records:
            for job_record in measurement["job_records"]:
                job_record = dict(job_record)
                job_record["arm"] = "uniform"
                job_record["call_index"] = measurement["call_index"]
                all_job_records.append(job_record)

    job_df = pd.DataFrame(all_job_records)
    job_csv = output_dir / "job_records.csv"
    job_df.to_csv(job_csv, index=False)
    print(job_df)

    final_methods = [f"round_{args.rounds}"]
    if not args.skip_uniform:
        final_methods.append("uniform")
    final_df = df[df["method"].isin(final_methods)]

    print("\n=== Final adaptive round vs uniform ===")
    print_summary(final_df)

    print("\nSaved outputs:")
    print(f"- Config: {output_dir / 'config.json'}")
    print(f"- Metrics: {csv_path}")
    print(f"- Jobs: {job_csv}")
    print(f"- Raw arrays and plots: {output_dir}")


if __name__ == "__main__":
    main()
