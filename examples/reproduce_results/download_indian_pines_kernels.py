"""Download Indian Pines quantum training kernels used by the manuscript.

The large precomputed kernel matrices are maintained in the external
Large_scale_QSVM repository and are intentionally not committed to ShotWise.

This script performs a sparse Git checkout of only the Indian Pines
bandwidth-kernel directory and copies the quantum training kernels into the
local ShotWise data directory.

By default, only files satisfying all of the following are retained:

- quantum kernels,
- training kernels,
- NumPy ``.npy`` files,
- not iCloud placeholder files.

The source repository can later be pinned to an immutable commit or tag using
``--source-ref``.

Example
-------
From the ShotWise repository root:

    python examples/reproduce_results/download_indian_pines_kernels.py

To pin an explicit revision:

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

DEFAULT_REPOSITORY = "https://github.com/ArturMiroszewski/Large_scale_QSVM"
DEFAULT_SOURCE_PATH = "bandwidth/results/indian/kernel_matrices"
DEFAULT_OUTPUT_DIR = Path("data/quantum_kernels/indian_pines/bandwidth")


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
            "Download Indian Pines quantum training kernels from Large_scale_QSVM."
        )
    )
    parser.add_argument(
        "--repository",
        default=DEFAULT_REPOSITORY,
        help="Git repository containing the precomputed kernels.",
    )
    parser.add_argument(
        "--source-ref",
        default=None,
        help=(
            "Optional immutable Git commit, tag, or branch to checkout. "
            "If omitted, the repository default branch is used."
        ),
    )
    parser.add_argument(
        "--source-path",
        default=DEFAULT_SOURCE_PATH,
        help="Repository-relative directory containing kernel matrices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local directory in which downloaded kernels are stored.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help=(
            "Also copy quantum test kernels. By default only training "
            "kernels required by the ShotWise experiment are retained."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local kernel files.",
    )
    return parser.parse_args()


def run_git(
    *args: str,
    cwd: Path | None = None,
) -> str:
    """Run Git and return stripped standard output."""
    command = ["git", *args]

    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return result.stdout.strip()


def parse_kernel_filename(path: Path) -> KernelMetadata | None:
    """Parse metadata encoded in a quantum kernel filename."""
    name = path.name

    if not name.endswith(".npy"):
        return None

    if name.startswith("."):
        return None

    if not name.startswith("quantum_"):
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


def validate_kernel(path: Path) -> tuple[int, float, float]:
    """Load and minimally validate one kernel matrix."""
    kernel = np.load(path)

    if kernel.ndim != 2:
        raise ValueError(
            f"{path.name}: expected a 2D matrix, got shape {kernel.shape}."
        )

    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"{path.name}: expected a square matrix, got {kernel.shape}.")

    if kernel.shape[0] == 0:
        raise ValueError(f"{path.name}: kernel matrix is empty.")

    if not np.all(np.isfinite(kernel)):
        raise ValueError(f"{path.name}: kernel contains non-finite values.")

    minimum = float(np.min(kernel))
    maximum = float(np.max(kernel))

    # SyntheticBernoulliBackend interprets kernel entries as Bernoulli
    # probabilities. Small floating-point excursions are tolerated, but
    # materially invalid probability values are rejected.
    tolerance = 1e-8

    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"{path.name}: kernel values fall outside the expected "
            f"[0, 1] probability range: min={minimum:.6g}, "
            f"max={maximum:.6g}."
        )

    return kernel.shape[0], minimum, maximum


def sparse_checkout(
    *,
    repository: str,
    source_ref: str | None,
    source_path: str,
    work_dir: Path,
) -> tuple[Path, str]:
    """Sparse-checkout only the requested source directory."""
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
        source_path,
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

    resolved_revision = run_git(
        "rev-parse",
        "HEAD",
        cwd=repository_dir,
    )

    return repository_dir, resolved_revision


def main() -> None:
    """Download and validate the required quantum training kernels."""
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="indian_pines_kernels_"
    ) as temporary_directory:
        work_dir = Path(temporary_directory)

        print("Sparse-checking external kernel repository...")

        repository_dir, revision = sparse_checkout(
            repository=args.repository,
            source_ref=args.source_ref,
            source_path=args.source_path,
            work_dir=work_dir,
        )

        source_dir = repository_dir / args.source_path

        if not source_dir.exists():
            raise FileNotFoundError(
                f"Source directory was not found after checkout: {source_dir}"
            )

        candidates: list[KernelMetadata] = []

        for path in sorted(source_dir.glob("*.npy")):
            metadata = parse_kernel_filename(path)

            if metadata is None:
                continue

            if metadata.split == "test" and not args.include_test:
                continue

            candidates.append(metadata)

        if not candidates:
            raise RuntimeError("No matching quantum kernel .npy files were found.")

        copied: list[tuple[KernelMetadata, int, float, float]] = []

        for metadata in candidates:
            destination = args.output_dir / metadata.path.name

            if destination.exists() and not args.overwrite:
                print(f"Keeping existing: {destination.name}")
            else:
                shutil.copy2(
                    metadata.path,
                    destination,
                )
                print(f"Copied: {destination.name}")

            n_samples, minimum, maximum = validate_kernel(destination)

            copied.append(
                (
                    metadata,
                    n_samples,
                    minimum,
                    maximum,
                )
            )

    print()
    print("=== Download summary ===")
    print(f"Repository: {args.repository}")
    print(f"Revision:   {revision}")
    print(f"Source:     {args.source_path}")
    print(f"Output:     {args.output_dir}")
    print(f"Files:      {len(copied)}")

    folds = sorted({metadata.fold for metadata, *_ in copied})
    qubits = sorted({metadata.qubits for metadata, *_ in copied})
    class_pairs = sorted(
        {
            (
                metadata.class_1,
                metadata.class_2,
            )
            for metadata, *_ in copied
        }
    )
    sample_sizes = sorted({n_samples for _, n_samples, _, _ in copied})

    print(f"Folds:      {folds}")
    print(f"Qubits:     {qubits}")
    print(f"Classes:    {class_pairs}")
    print(f"Sizes:      {sample_sizes}")

    print()
    print("Kernel files:")
    for (
        metadata,
        n_samples,
        minimum,
        maximum,
    ) in copied:
        print(
            f"- fold={metadata.fold}, "
            f"qubits={metadata.qubits:3d}, "
            f"classes={metadata.class_1:02d}/{metadata.class_2:02d}, "
            f"n={n_samples:3d}, "
            f"range=[{minimum:.6g}, {maximum:.6g}] "
            f"{metadata.path.name}"
        )

    revision_file = args.output_dir / "SOURCE_REVISION.txt"
    revision_file.write_text(
        "\n".join(
            [
                f"repository={args.repository}",
                f"revision={revision}",
                f"source_path={args.source_path}",
                "",
            ]
        )
    )

    print()
    print(f"Source provenance written to: {revision_file}")


if __name__ == "__main__":
    main()
