"""Small 8-point quantum-kernel experiment on IBM hardware.

This is the IBM Runtime version of the local simulator experiment:
- fixed 8-point binary toy dataset,
- shallow 2-qubit angle encoding,
- SupportShot vs uniform under the same total shot budget.

Notes
-----
- Open Plan users cannot use sessions; use job mode or batch mode instead.
- Runtime primitives require ISA-compatible circuits, so we pass a backend-
  specific pass manager into ComputeUncompute.
- Because adaptive rounds are sequential, this script still waits for each
  round before continuing. Batch mode can still help group the underlying
  primitive jobs more cleanly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from evaluation_utils import (
    evaluate_estimated_kernel,
    print_summary,
    records_to_dataframe,
)
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Batch, QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from shotwise.algorithms.supportshot import SupportShot
from shotwise.backends.qiskit_backend import QiskitBackend
from shotwise.learning.svm import fit_precomputed_svm_state
from shotwise.storage.kernel_store import KernelStore
from shotwise.utils.matrix import upper_triangle_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an 8-point quantum-kernel SupportShot experiment on IBM hardware."
        ),
    )
    parser.add_argument("--shots-per-entry", type=int, default=30)
    parser.add_argument("--pilot-shots", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument(
        "--allocation-method",
        choices=["multinomial", "deterministic"],
        default="deterministic",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help=(
            "IBM backend name. If omitted, the script selects the least busy "
            "operational non-simulator backend."
        ),
    )
    parser.add_argument(
        "--max-circuits-per-job",
        type=int,
        default=100,
        help=(
            "Passed both to FidelityQuantumKernel and the backend wrapper to "
            "limit per-job circuit count."
        ),
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Transpiler optimization level for hardware ISA compilation.",
    )
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help=(
            "Run the whole experiment inside an IBM Runtime Batch context. "
            "This is allowed on Open Plan, unlike sessions."
        ),
    )
    parser.add_argument(
        "--batch-max-time",
        type=str,
        default="10m",
        help=(
            "Maximum runtime TTL for Batch mode, e.g. '10m'. "
            "Open Plan defaults are short, so keep this modest."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qiskit_toy8_ibm"),
    )
    return parser.parse_args()


def build_toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed 8-point 2D toy dataset in angle space."""
    x = np.array(
        [
            [0.10, 0.15],
            # [0.18, 0.28],
            [0.28, 0.10],
            [0.55, 0.55],
            # [1.20, 1.10],
            [1.35, 1.25],
            [1.10, 1.38],
            [0.82, 0.88],
        ],
        dtype=float,
    )
    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=int)
    y = np.array([-1, -1, -1, 1, 1, 1], dtype=int)
    return x, y


def build_feature_map() -> QuantumCircuit:
    """Build a shallow 2-qubit angle-encoding feature map."""
    theta = ParameterVector("x", 2)
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cz(0, 1)
    return qc


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


def save_dataset_plot(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, marker, name in [(-1, "o", "class -1"), (1, "s", "class +1")]:
        mask = y == label
        ax.scatter(x[mask, 0], x[mask, 1], marker=marker, label=name)
    for idx, (a0, a1) in enumerate(x):
        ax.text(a0 + 0.015, a1 + 0.015, str(idx), fontsize=8)
    ax.set_xlabel("angle feature 1")
    ax.set_ylabel("angle feature 2")
    ax.set_title("8-point toy dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_kernel_heatmap(
    kernel_matrix: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(kernel_matrix, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel("sample index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_allocation_heatmaps(model: SupportShot, output_dir: Path) -> None:
    for record in model.history_:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(record.allocation_matrix)
        ax.set_title(f"Allocation: {record.stage} (round {record.round_index})")
        ax.set_xlabel("sample index")
        ax.set_ylabel("sample index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        name = f"allocation_{record.stage}_{record.round_index}.png"
        fig.savefig(output_dir / name, bbox_inches="tight")
        plt.close(fig)


def make_hardware_kernel(
    *,
    feature_map: QuantumCircuit,
    runtime_mode,
    backend,
    optimization_level: int,
    max_circuits_per_job: int,
) -> FidelityQuantumKernel:
    """Create a hardware-backed fidelity kernel for the chosen runtime mode."""
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
    )
    sampler = Sampler(mode=runtime_mode)
    fidelity = ComputeUncompute(
        sampler=sampler,
        pass_manager=pm,
    )
    return FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
        max_circuits_per_job=max_circuits_per_job,
    )


def run_uniform_baseline(
    x: np.ndarray,
    kernel: FidelityQuantumKernel,
    *,
    total_budget: int,
    max_circuits_per_job: int,
    rng_seed: int,
) -> tuple[np.ndarray, int]:
    backend = QiskitBackend(
        x,
        kernel=kernel,
        max_circuits_per_job=max_circuits_per_job,
        rng=np.random.default_rng(rng_seed),
    )
    store = KernelStore(x.shape[0], diagonal_value=1.0)
    allocation = uniform_allocation_matrix(x.shape[0], total_budget)
    counts = backend.sample_upper_triangle_counts(allocation, round_id=-1)
    store.update(counts, allocation)
    return store.kernel_matrix(), store.total_shots_spent()


def run_experiment_once(
    *,
    x: np.ndarray,
    y: np.ndarray,
    kernel_true: np.ndarray,
    kernel_hw: FidelityQuantumKernel,
    total_budget: int,
    pilot_shots: int,
    rounds: int,
    lambda_weight: float,
    c_value: float,
    allocation_method: str,
    max_circuits_per_job: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    """Run one adaptive and one uniform hardware experiment."""
    adaptive_seed = seed + 1
    uniform_seed = seed + 2

    adaptive_backend = QiskitBackend(
        x,
        kernel=kernel_hw,
        max_circuits_per_job=max_circuits_per_job,
        rng=np.random.default_rng(adaptive_seed),
    )
    model = SupportShot(
        total_budget=total_budget,
        pilot_shots_per_entry=pilot_shots,
        rounds=rounds,
        lambda_weight=lambda_weight,
        svm_c=c_value,
        allocation_method=allocation_method,
        random_state=adaptive_seed,
    )
    model.fit(y, adaptive_backend)

    kernel_uniform, spent_uniform = run_uniform_baseline(
        x,
        kernel_hw,
        total_budget=total_budget,
        max_circuits_per_job=max_circuits_per_job,
        rng_seed=uniform_seed,
    )

    records: list[dict[str, float | int | str]] = []

    for record in model.history_:
        entry = evaluate_estimated_kernel(
            kernel_true,
            record.kernel_matrix,
            y,
            c=c_value,
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

    uniform_entry = evaluate_estimated_kernel(
        kernel_true,
        kernel_uniform,
        y,
        c=c_value,
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

    save_allocation_heatmaps(model, output_dir)
    save_kernel_heatmap(
        model.history_[-1].kernel_matrix,
        output_dir / "adaptive_final_kernel.png",
        "Adaptive final kernel",
    )
    save_kernel_heatmap(
        kernel_uniform,
        output_dir / "uniform_final_kernel.png",
        "Uniform final kernel",
    )

    return {
        "records": records,
        "model": model,
        "kernel_uniform": kernel_uniform,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Dataset and logical feature map
    # ------------------------------------------------------------
    x, y = build_toy_dataset()
    feature_map = build_feature_map()

    print("=== Dataset and feature map ===")
    print("Feature map:")
    print(feature_map.draw())
    print("\nPoints:")
    print(x)
    print("\nLabels:")
    print(y)

    save_dataset_plot(x, y, output_dir / "dataset.png")

    # ------------------------------------------------------------
    # Select IBM backend
    # ------------------------------------------------------------
    service = QiskitRuntimeService()

    if args.backend is not None:
        ibm_backend = service.backend(args.backend)
    else:
        ibm_backend = service.least_busy(
            operational=True,
            simulator=False,
        )

    print("\n=== IBM backend ===")
    print(f"backend               = {ibm_backend.name}")
    print(f"use_batch             = {args.use_batch}")
    print(f"optimization_level    = {args.optimization_level}")
    print(f"max_circuits_per_job  = {args.max_circuits_per_job}")

    # ------------------------------------------------------------
    # Exact reference kernel from statevector (local)
    # ------------------------------------------------------------
    ref_sampler = StatevectorSampler()
    ref_fidelity = ComputeUncompute(sampler=ref_sampler)
    ref_kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=ref_fidelity,
        max_circuits_per_job=None,
    )
    kernel_true = np.asarray(ref_kernel.evaluate(x), dtype=float)
    np.fill_diagonal(kernel_true, 1.0)

    save_kernel_heatmap(
        kernel_true,
        output_dir / "true_kernel.png",
        "Exact reference quantum kernel",
    )

    state_true = fit_precomputed_svm_state(kernel_true, y, c=args.c)
    sv_ratio = state_true.num_support / state_true.n_samples
    print("\n=== True-kernel classifier diagnostics ===")
    print(f"num_support_vectors = {state_true.num_support} / {state_true.n_samples}")
    print(f"support_vector_ratio = {sv_ratio:.3f}")
    print("unsigned dual coefficients:")
    print(np.round(state_true.alphas, 6))
    print("support indices:")
    print(state_true.support_indices)

    total_budget = total_budget_from_shots_per_entry(
        x.shape[0],
        args.shots_per_entry,
    )
    print("\n=== Experiment configuration ===")
    print(f"n_samples            = {x.shape[0]}")
    print(f"independent_entries  = {x.shape[0] * (x.shape[0] - 1) // 2}")
    print(f"shots_per_entry      = {args.shots_per_entry}")
    print(f"total_budget         = {total_budget}")
    print(f"pilot_shots          = {args.pilot_shots}")
    print(f"rounds               = {args.rounds}")
    print(f"lambda_weight        = {args.lambda_weight}")
    print(f"C                    = {args.c}")
    print(f"allocation_method    = {args.allocation_method}")
    print(f"seed                 = {args.seed}")
    print(f"output_dir           = {output_dir}")

    # ------------------------------------------------------------
    # Hardware execution
    # ------------------------------------------------------------
    if args.use_batch:
        with Batch(backend=ibm_backend, max_time=args.batch_max_time) as batch:
            kernel_hw = make_hardware_kernel(
                feature_map=feature_map,
                runtime_mode=batch,
                backend=ibm_backend,
                optimization_level=args.optimization_level,
                max_circuits_per_job=args.max_circuits_per_job,
            )
            result = run_experiment_once(
                x=x,
                y=y,
                kernel_true=kernel_true,
                kernel_hw=kernel_hw,
                total_budget=total_budget,
                pilot_shots=args.pilot_shots,
                rounds=args.rounds,
                lambda_weight=args.lambda_weight,
                c_value=args.c,
                allocation_method=args.allocation_method,
                max_circuits_per_job=args.max_circuits_per_job,
                seed=args.seed,
                output_dir=output_dir,
            )
    else:
        kernel_hw = make_hardware_kernel(
            feature_map=feature_map,
            runtime_mode=ibm_backend,
            backend=ibm_backend,
            optimization_level=args.optimization_level,
            max_circuits_per_job=args.max_circuits_per_job,
        )
        result = run_experiment_once(
            x=x,
            y=y,
            kernel_true=kernel_true,
            kernel_hw=kernel_hw,
            total_budget=total_budget,
            pilot_shots=args.pilot_shots,
            rounds=args.rounds,
            lambda_weight=args.lambda_weight,
            c_value=args.c,
            allocation_method=args.allocation_method,
            max_circuits_per_job=args.max_circuits_per_job,
            seed=args.seed,
            output_dir=output_dir,
        )

    records = result["records"]
    df = records_to_dataframe(records)
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== Summary (all recorded metrics) ===")
    print_summary(df)

    print("\n=== Final-round vs uniform summary ===")
    final_df = df[df["method"].isin([f"round_{args.rounds}", "uniform"])]
    print_summary(final_df)

    print("\nSaved outputs:")
    print(f"- CSV results: {csv_path}")
    print(f"- Dataset plot: {output_dir / 'dataset.png'}")
    print(f"- True kernel heatmap: {output_dir / 'true_kernel.png'}")
    print(f"- Allocation heatmaps: {output_dir}")
    print(f"- Adaptive final kernel: {output_dir / 'adaptive_final_kernel.png'}")
    print(f"- Uniform final kernel: {output_dir / 'uniform_final_kernel.png'}")


if __name__ == "__main__":
    main()
