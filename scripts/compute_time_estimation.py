from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

jobs = service.jobs(limit=50)  # adjust if needed

total_compute_seconds = 0.0

for job in jobs:
    try:
        props = job.properties()  # metadata
        metrics = job.metrics()

        # This is the most relevant field if available
        if metrics and "usage" in metrics and "quantum_seconds" in metrics["usage"]:
            qs = metrics["usage"]["quantum_seconds"]
        else:
            qs = 0.0

        print(f"{job.job_id()} | {job.status()} | compute_seconds={qs:.3f}")
        total_compute_seconds += qs

    except Exception as e:
        print(f"{job.job_id()} failed to fetch metadata: {e}")

print("\nEstimated total compute time used:", total_compute_seconds, "seconds")
print("≈", total_compute_seconds / 60, "minutes")
