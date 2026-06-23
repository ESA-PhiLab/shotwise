"""
One-time setup for IBM Quantum authentication.

Run this script once to store your API token locally.
"""

from qiskit_ibm_runtime import QiskitRuntimeService


def main():
    print("=== IBM Quantum account setup ===")
    print()

    token = input("Enter your IBM Quantum API token: ").strip()

    if not token:
        raise ValueError("Token cannot be empty.")

    # Save account locally (~/.qiskit/)
    QiskitRuntimeService.save_account(
        token=token,
        overwrite=True,  # overwrite if already exists
    )

    print("\n✅ Account saved successfully.")

    # Test connection
    service = QiskitRuntimeService()
    backends = service.backends()

    print(f"\n✅ Connected! Found {len(backends)} backends.")
    print("Example backends:")
    for b in backends[:5]:
        print(" -", b.name)


if __name__ == "__main__":
    main()
