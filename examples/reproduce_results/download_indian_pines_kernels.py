"""Download Indian Pines quantum kernels and reconstruct training labels.

The precomputed quantum kernel matrices and the original Indian Pines fold
files are maintained in the external Large_scale_QSVM repository. They are
intentionally not committed to ShotWise.

This script:

1. sparsely checks out the bandwidth Indian Pines kernel matrices,
2. sparsely checks out the four original Indian Pines fold directories,
3. copies the quantum training kernels locally,
4. reconstructs the binary training labels used for classes 3 and 10,
5. validates the reconstructed labels against the kernel dimensions,
6. saves the labels and source provenance locally.

The historical data-preparation procedure maps class 3 to label 0 and class
10 to label 1, shuffles the available training/validation population using
random_state=0, and performs a stratified split with n_train=50 and n_val=50.

Example
-------
From the ShotWise repository root:

    python examples/reproduce_results/download_indian_pines_kernels.py

To pin the external repository to an immutable revision:

    python examples/reproduce_results/download_indian_pines_kernels.py \
        --source-ref <commit-or-tag>
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_REPOSITORY = "https://github.com/ArturMiroszewski/Large_scale_QSVM"
DEFAULT_KERNEL_PATH = "bandwidth/results/indian/kernel_matrices"
DEFAULT_FOLD_ROOT = "data/Indian Pines"
DEFAULT_OUTPUT_DIR = Path("data/quantum_kernels/indian_pines")

CLASS_1 = 3
CLASS_2 = 10
N_TRAIN = 50
N_VAL = 50
RANDOM_STATE = 0
N_FOLDS = 4


@dataclass(frozen=True)
class KernelMetadata:
    """Metadata parsed from an Indian Pines kernel filename."""

    path: Path
    split: str
    fold: int
    qubits: int
    class_1: int
    class_2: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download Indian Pines quantum kernels and reconstruct "
            "the original binary training labels."
        )
    )
    parser.add_argument(
        "--repository",
        default=DEFAULT_REPOSITORY,
    )
    parser.add_argument(
        "--source-ref",
        default=None,
        help=(
            "Optional Git commit, tag, or branch. If omitted, the "
            "repository default branch is used."
        ),
    )
    parser.add_argument(
        "--kernel-path",
        default=DEFAULT_KERNEL_PATH,
    )
    parser.add_argument(
        "--fold-root",
        default=DEFAULT_FOLD_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def run_git(
    *args: str,
    cwd: Path | None = None,
) -> str:
    """Run git and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def parse_kernel_filename(
    path: Path,
) -> KernelMetadata | None:
    """Parse metadata encoded in one quantum kernel filename."""
    name = path.name

    if (
        not name.endswith(".npy")
        or name.startswith(".")
        or not name.startswith("quantum_")
    ):
        return None

    split_match = re.search(
        r"quantum_(train|test)_",
        name,
    )
    fold_match = re.search(
        r"fold_(\d+)",
        name,
    )
    qubit_match = re.search(
        r"_n_(\d+)_",
        name,
    )
    class_match = re.search(
        r"_classes_(\d+)_(\d+)\.npy$",
        name,
    )

    if not all(
        [
            split_match,
            fold_match,
            qubit_match,
            class_match,
        ]
    ):
        return None

    return KernelMetadata(
        path=path,
        split=split_match.group(1),
        fold=int(fold_match.group(1)),
        qubits=int(qubit_match.group(1)),
        class_1=int(class_match.group(1)),
        class_2=int(class_match.group(2)),
    )


def sparse_checkout(
    *,
    repository: str,
    source_ref: str | None,
    paths: list[str],
    work_dir: Path,
) -> tuple[Path, str]:
    """Sparse-checkout the required external data paths."""
    repository_dir = work_dir / "Large_scale_QSVM"

    run_git(
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        repository,
        str(repository_dir),
    )

    run_git(
        "sparse-checkout",
        "init",
        "--cone",
        cwd=repository_dir,
    )

    run_git(
        "sparse-checkout",
        "set",
        *paths,
        cwd=repository_dir,
    )

    if source_ref is None:
        run_git(
            "checkout",
            cwd=repository_dir,
        )
    else:
        run_git(
            "checkout",
            source_ref,
            cwd=repository_dir,
        )

    revision = run_git(
        "rev-parse",
        "HEAD",
        cwd=repository_dir,
    )

    return repository_dir, revision


def patch_number(path: Path) -> int:
    """Return the numerical patch identifier for sorting."""
    match = re.search(
        r"patch_(\d+)(?:_gt)?\.npy$",
        path.name,
    )

    if match is None:
        raise ValueError(f"Could not parse patch number from {path.name!r}.")

    return int(match.group(1))


def load_fold_ground_truth(
    fold_dir: Path,
) -> np.ndarray:
    """Recreate the concatenated Y_all vector used by prepare_indian.py."""
    gt_files = sorted(
        fold_dir.glob("patch_*_gt.npy"),
        key=patch_number,
    )

    if not gt_files:
        raise FileNotFoundError(f"No patch ground-truth files found in {fold_dir}.")

    arrays: list[np.ndarray] = []

    for path in gt_files:
        ground_truth = np.load(path)
        arrays.append(np.asarray(ground_truth).reshape(-1))

    return np.concatenate(arrays)


def reconstruct_training_labels(
    y_all: np.ndarray,
    *,
    class_1: int,
    class_2: int,
    n_train: int,
    n_val: int,
    random_state: int,
) -> np.ndarray:
    """Reproduce the historical binary train-label split."""
    n_class_1 = int(np.count_nonzero(y_all == class_1))
    n_class_2 = int(np.count_nonzero(y_all == class_2))

    if n_class_1 == 0 or n_class_2 == 0:
        raise ValueError("One or both requested classes are absent from the fold.")

    # prepare_classes_fold() selected class-1 samples first and class-2
    # samples second, then mapped them to 0 and 1 respectively.
    labels = np.concatenate(
        [
            np.zeros(
                n_class_1,
                dtype=int,
            ),
            np.ones(
                n_class_2,
                dtype=int,
            ),
        ]
    )

    # prepare_folds() first shuffled the combined train/validation pool.
    rng = np.random.default_rng(seed=random_state)
    permutation = rng.permutation(labels.size)
    labels = labels[permutation]

    if n_train + n_val > labels.size:
        raise ValueError(
            f"Requested n_train={n_train} and n_val={n_val}, "
            f"but only {labels.size} class-selected samples are available."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_train,
        test_size=n_val,
        random_state=random_state,
    )

    # The split indices depend only on the stratification labels and random
    # state; feature values are irrelevant for reconstructing Y_train.
    dummy_features = np.zeros(
        (labels.size, 1),
        dtype=float,
    )

    train_index, _ = next(
        splitter.split(
            dummy_features,
            labels,
        )
    )

    return labels[train_index]


def validate_kernel(
    path: Path,
) -> tuple[int, float, float]:
    """Validate one downloaded kernel matrix."""
    kernel = np.load(path)

    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"{path.name}: expected square matrix, got {kernel.shape}.")

    if not np.all(np.isfinite(kernel)):
        raise ValueError(f"{path.name}: non-finite values found.")

    minimum = float(kernel.min())
    maximum = float(kernel.max())

    tolerance = 1e-8

    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"{path.name}: values outside expected [0, 1] range: "
            f"[{minimum:.6g}, {maximum:.6g}]."
        )

    return (
        kernel.shape[0],
        minimum,
        maximum,
    )


def main() -> None:
    """Download kernels and reconstruct the four training-label vectors."""
    args = parse_args()

    bandwidth_dir = args.output_dir / "bandwidth"
    bandwidth_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_paths = [f"{args.fold_root}/indiana_fold_{fold}" for fold in range(N_FOLDS)]

    sparse_paths = [
        args.kernel_path,
        *fold_paths,
    ]

    with tempfile.TemporaryDirectory(
        prefix="indian_pines_repro_"
    ) as temporary_directory:
        repository_dir, revision = sparse_checkout(
            repository=args.repository,
            source_ref=args.source_ref,
            paths=sparse_paths,
            work_dir=Path(temporary_directory),
        )

        # ------------------------------------------------------
        # Copy quantum TRAINING kernels only.
        # ------------------------------------------------------
        external_kernel_dir = repository_dir / args.kernel_path

        kernels: list[tuple[KernelMetadata, int, float, float]] = []

        for path in sorted(external_kernel_dir.glob("quantum_train_*.npy")):
            metadata = parse_kernel_filename(path)

            if metadata is None:
                continue

            if metadata.class_1 != CLASS_1 or metadata.class_2 != CLASS_2:
                continue

            destination = bandwidth_dir / path.name

            if not destination.exists() or args.overwrite:
                shutil.copy2(
                    path,
                    destination,
                )

            n_samples, minimum, maximum = validate_kernel(destination)

            kernels.append(
                (
                    metadata,
                    n_samples,
                    minimum,
                    maximum,
                )
            )

        if not kernels:
            raise RuntimeError("No matching quantum training kernels were found.")

        # ------------------------------------------------------
        # Reconstruct training labels from original fold GT data.
        # ------------------------------------------------------
        reconstructed: dict[
            str,
            np.ndarray,
        ] = {}

        for fold in range(N_FOLDS):
            fold_dir = repository_dir / args.fold_root / f"indiana_fold_{fold}"

            y_all = load_fold_ground_truth(fold_dir)

            y_train = reconstruct_training_labels(
                y_all,
                class_1=CLASS_1,
                class_2=CLASS_2,
                n_train=N_TRAIN,
                n_val=N_VAL,
                random_state=RANDOM_STATE,
            )

            reconstructed[f"fold_{fold}"] = y_train.astype(int)

            print(
                f"fold {fold}: "
                f"n={y_train.size}, "
                f"class counts="
                f"{dict(zip(*np.unique(y_train, return_counts=True), strict=True))}"
            )

        # ------------------------------------------------------
        # Validate label length against every kernel.
        # ------------------------------------------------------
        for (
            metadata,
            n_samples,
            _,
            _,
        ) in kernels:
            labels = reconstructed[f"fold_{metadata.fold}"]

            if labels.size != n_samples:
                raise ValueError(
                    f"{metadata.path.name}: kernel has n={n_samples}, "
                    f"but reconstructed fold-{metadata.fold} labels "
                    f"have n={labels.size}."
                )

        labels_path = args.output_dir / "Indian_train_labels.npz"

        np.savez(
            labels_path,
            **reconstructed,
            class_1=np.asarray(
                [CLASS_1],
                dtype=int,
            ),
            class_2=np.asarray(
                [CLASS_2],
                dtype=int,
            ),
            n_train=np.asarray(
                [N_TRAIN],
                dtype=int,
            ),
            n_val=np.asarray(
                [N_VAL],
                dtype=int,
            ),
            random_state=np.asarray(
                [RANDOM_STATE],
                dtype=int,
            ),
        )

    provenance_path = args.output_dir / "SOURCE_REVISION.txt"

    provenance_path.write_text(
        "\n".join(
            [
                f"repository={args.repository}",
                f"revision={revision}",
                f"kernel_path={args.kernel_path}",
                f"fold_root={args.fold_root}",
                f"class_1={CLASS_1}",
                f"class_2={CLASS_2}",
                f"n_train={N_TRAIN}",
                f"n_val={N_VAL}",
                f"random_state={RANDOM_STATE}",
                "",
            ]
        )
    )

    folds = sorted({metadata.fold for metadata, *_ in kernels})
    qubits = sorted({metadata.qubits for metadata, *_ in kernels})
    sample_sizes = sorted({n_samples for _, n_samples, _, _ in kernels})

    print()
    print("=== Reproduction data summary ===")
    print(f"Revision: {revision}")
    print(f"Kernels:  {len(kernels)}")
    print(f"Folds:    {folds}")
    print(f"Qubits:   {qubits}")
    print(f"Sizes:    {sample_sizes}")
    print(f"Labels:   {labels_path}")
    print(f"Source:   {provenance_path}")


if __name__ == "__main__":
    main()
