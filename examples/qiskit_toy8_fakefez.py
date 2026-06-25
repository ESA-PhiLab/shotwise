"""Small 8-point quantum-kernel experiment on FakeFez.

This is the noisy local-testing version of the toy-8 experiment:
- fixed 8-point binary toy dataset,
- shallow 2-qubit angle encoding,
- SupportShot vs uniform under the same total shot budget,
- fake IBM backend (FakeFez) to mimic hardware constraints and noise locally.

This script is intended as the intermediate step between:
1. exact local statevector simulation, and
2. real IBM hardware execution.
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
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeManilaV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from shotwise.algorithms.supportshot import SupportShot
from shotwise.backends.qiskit_backend import QiskitBackend
from shotwise.learning.svm import fit_precomputed_svm_state
from shotwise.storage.kernel_store import KernelStore
from shotwise.utils.matrix import upper_triangle_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an 8-point quantum-kernel SupportShot experiment on FakeFez.",
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--shots-per-entry", type=int, default=40)
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
        "--max-circuits-per-job",
        type=int,
        default=16,
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
        help="Transpiler optimization level for FakeFez ISA compilation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qiskit_toy8_fakefez"),
    )
    return parser.parse_args()


def build_toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed 8-point 2D toy dataset in angle space."""
    x = np.array(
        [
            [0.10, 0.15],
            [0.18, 0.28],
            [0.28, 0.10],
            [0.55, 0.55],
            [1.20, 1.10],
            [1.35, 1.25],
            [1.10, 1.38],
            [0.82, 0.88],
        ],
        dtype=float,
    )
    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=int)
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


def make_reference_kernel(feature_map: QuantumCircuit) -> FidelityQuantumKernel:
    """Create the exact local statevector reference kernel."""
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    return FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
        max_circuits_per_job=None,
    )


def make_fakefez_kernel(
    *,
    feature_map: QuantumCircuit,
    fake_backend: FakeFez,
    optimization_level: int,
    max_circuits_per_job: int,
) -> FidelityQuantumKernel:
    """Create a FakeFez-backed noisy fidelity kernel.

    FakeFez provides the hardware-like target and noise snapshot.
    AerSimulator.from_backend(fake_backend) is used as the actual local
    simulator to avoid the BasicSimulator qubit-limit issue.
    """
    aer_backend = AerSimulator.from_backend(fake_backend)

    pm = generate_preset_pass_manager(
        backend=fake_backend,
        optimization_level=optimization_level,
    )

    sampler = Sampler(mode=aer_backend)

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


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Dataset and feature map
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
    # Exact reference kernel
    # ------------------------------------------------------------
    ref_kernel = make_reference_kernel(feature_map)
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

    # ------------------------------------------------------------
    # FakeFez noisy kernel
    # ------------------------------------------------------------
    fake_backend = FakeManilaV2()
    kernel_noisy = make_fakefez_kernel(
        feature_map=feature_map,
        fake_backend=fake_backend,
        optimization_level=args.optimization_level,
        max_circuits_per_job=args.max_circuits_per_job,
    )

    total_budget = total_budget_from_shots_per_entry(
        x.shape[0],
        args.shots_per_entry,
    )
    print("\n=== Experiment configuration ===")
    print(f"backend              = {fake_backend.name}")
    print(f"repeats              = {args.repeats}")
    print(f"n_samples            = {x.shape[0]}")
    print(f"independent_entries  = {x.shape[0] * (x.shape[0] - 1) // 2}")
    print(f"shots_per_entry      = {args.shots_per_entry}")
    print(f"total_budget         = {total_budget}")
    print(f"pilot_shots          = {args.pilot_shots}")
    print(f"rounds               = {args.rounds}")
    print(f"lambda_weight        = {args.lambda_weight}")
    print(f"C                    = {args.c}")
    print(f"allocation_method    = {args.allocation_method}")
    print(f"max_circuits_per_job = {args.max_circuits_per_job}")
    print(f"optimization_level   = {args.optimization_level}")
    print(f"output_dir           = {output_dir}")

    master_rng = np.random.default_rng(args.seed)
    records: list[dict[str, float | int | str]] = []

    for run in range(args.repeats):
        adaptive_seed = int(master_rng.integers(0, 2**32 - 1))
        uniform_seed = int(master_rng.integers(0, 2**32 - 1))

        print(f"\nRepeat {run + 1}/{args.repeats}")

        adaptive_backend = QiskitBackend(
            x,
            kernel=kernel_noisy,
            max_circuits_per_job=args.max_circuits_per_job,
            rng=np.random.default_rng(adaptive_seed),
        )
        model = SupportShot(
            total_budget=total_budget,
            pilot_shots_per_entry=args.pilot_shots,
            rounds=args.rounds,
            lambda_weight=args.lambda_weight,
            svm_c=args.c,
            allocation_method=args.allocation_method,
            random_state=adaptive_seed,
        )
        model.fit(y, adaptive_backend)

        if run == 0:
            save_allocation_heatmaps(model, output_dir)
            save_kernel_heatmap(
                model.history_[-1].kernel_matrix,
                output_dir / "adaptive_final_kernel_run0.png",
                "Adaptive final kernel (run 0)",
            )

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
                    "run": run,
                    "stage": record.stage,
                    "budget": record.budget,
                    "total_shots_spent": record.total_shots_spent,
                    "stopped_early": record.stopped_early,
                }
            )
            records.append(entry)

        kernel_uniform, spent_uniform = run_uniform_baseline(
            x,
            kernel_noisy,
            total_budget=total_budget,
            max_circuits_per_job=args.max_circuits_per_job,
            rng_seed=uniform_seed,
        )
        uniform_entry = evaluate_estimated_kernel(
            kernel_true,
            kernel_uniform,
            y,
            c=args.c,
            method="uniform",
            round_id=-1,
        )
        uniform_entry.update(
            {
                "run": run,
                "stage": "uniform",
                "budget": spent_uniform,
                "total_shots_spent": spent_uniform,
                "stopped_early": False,
            }
        )
        records.append(uniform_entry)

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
    print(f"- Allocation heatmaps (run 0): {output_dir}")


if __name__ == "__main__":
    main()
