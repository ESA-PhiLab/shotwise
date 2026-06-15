import numpy as np

from shotwise.backends.synthetic import SyntheticBernoulliBackend

K_true = np.array(
    [
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.5],
        [0.2, 0.5, 1.0],
    ]
)

backend = SyntheticBernoulliBackend(
    K_true,
    noise_sigma_global=0.01,
    noise_sigma_local=0.02,
    random_state=0,
)

shot_matrix = np.array(
    [
        [0, 10, 20],
        [0, 0, 15],
        [0, 0, 0],
    ]
)

counts_upper = backend.sample_upper_triangle_counts(shot_matrix)
print(counts_upper)

estimates = backend.estimate_entries([(0, 1), (1, 2)], [50, 30])
for est in estimates:
    print(est)
