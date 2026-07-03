from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

job_id = "d92g6b357qjs73b7uf2g"
job = service.job(job_id)

print("job_id:", job.job_id())
print("status:", job.status())

try:
    result = job.result()
    print("Result retrieved successfully")
    print("Number of PUB results:", len(result))
except Exception as exc:
    print("Could not retrieve result:")
    print(repr(exc))
