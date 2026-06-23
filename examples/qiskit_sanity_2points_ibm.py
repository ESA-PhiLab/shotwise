import argparse

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from shotwise.backends.qiskit_backend import QiskitBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 2-point Bloch-sphere sanity check on an IBM backend."
    )
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
        "--shots",
        type=int,
        default=200,
        help="Shots for the single off-diagonal kernel entry.",
    )
    parser.add_argument(
        "--max-circuits-per-job",
        type=int,
        default=10,
        help="Maximum number of circuits per job for the fidelity kernel.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Transpiler optimization level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ============================================================
    # 1. Define logical feature map (1-qubit RY encoding)
    # ============================================================
    theta = Parameter("θ")

    feature_map = QuantumCircuit(1)
    feature_map.ry(theta, 0)

    print("Logical feature map:")
    print(feature_map.draw())

    # ============================================================
    # 2. Connect to IBM Quantum and choose backend
    # ============================================================
    service = QiskitRuntimeService()

    if args.backend is not None:
        ibm_backend = service.backend(args.backend)
    else:
        ibm_backend = service.least_busy(
            operational=True,
            simulator=False,
        )

    print(f"\nUsing backend: {ibm_backend.name}")

    # ============================================================
    # 3. Build pass manager for ISA transpilation
    # ============================================================
    pm = generate_preset_pass_manager(
        backend=ibm_backend,
        optimization_level=args.optimization_level,
    )

    # ============================================================
    # 4. Runtime SamplerV2 + ComputeUncompute with pass manager
    # ============================================================
    sampler = Sampler(mode=ibm_backend)

    fidelity = ComputeUncompute(
        sampler=sampler,
        pass_manager=pm,
    )

    kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
        max_circuits_per_job=args.max_circuits_per_job,
    )

    # ============================================================
    # 5. Data (two points)
    # ============================================================
    X = np.array(
        [
            [0.0],  # |0>
            [np.pi / 2],  # |+>
        ]
    )

    print("\nInput points:")
    print(X)

    # ============================================================
    # 6. ShotWise backend wrapper
    # ============================================================
    backend = QiskitBackend(
        X,
        kernel=kernel,
        shots=args.shots,
        max_circuits_per_job=args.max_circuits_per_job,
        rng=np.random.default_rng(0),
    )

    # ============================================================
    # 7. Shot matrix (only one off-diagonal entry)
    # ============================================================
    shot_matrix = np.zeros((2, 2), dtype=int)
    shot_matrix[0, 1] = args.shots

    # ============================================================
    # 8. Run
    # ============================================================
    counts = backend.sample_upper_triangle_counts(
        shot_matrix,
        round_id=0,
    )

    print("\nCounts matrix:")
    print(counts)

    estimated_prob = counts[0, 1] / shot_matrix[0, 1]

    print("\nEstimated probability:")
    print(estimated_prob)

    print("\nExpected probability: 0.5")


if __name__ == "__main__":
    main()
