#!/usr/bin/env python3
"""Assert-based tests for the from-scratch k-means implementation.

Run:  python3 tests/test_kmeans.py
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from palette.kmeans import kmeans, kmeans_plusplus_init  # noqa: E402

rng = np.random.default_rng(3)

# Two well-separated Gaussian blobs with known means.
blob_a = rng.normal(loc=(0.0, 0.0), scale=0.5, size=(150, 2))
blob_b = rng.normal(loc=(10.0, 10.0), scale=0.5, size=(150, 2))
X = np.vstack([blob_a, blob_b])

result = kmeans(X, k=2, seed=0)
centers = result.centers[np.argsort(result.centers[:, 0])]
assert np.allclose(centers[0], blob_a.mean(axis=0), atol=0.25), centers
assert np.allclose(centers[1], blob_b.mean(axis=0), atol=0.25), centers

# Inertia must beat the trivial one-cluster objective, and match a
# hand recomputation from the returned labels/centres.
one_cluster = float(((X - X.mean(axis=0)) ** 2).sum())
assert result.inertia < 0.05 * one_cluster, (result.inertia, one_cluster)
d2 = ((X[:, None, :] - result.centers[None, :, :]) ** 2).sum(axis=2)
recomputed = float(d2[np.arange(len(X)), result.labels].sum())
assert abs(recomputed - result.inertia) < 1e-6 * max(1.0, recomputed)

# Reproducibility: same seed -> identical run; k-means++ centres are
# data points.
r1 = kmeans(X, k=2, seed=42)
r2 = kmeans(X, k=2, seed=42)
assert np.array_equal(r1.centers, r2.centers)
assert r1.inertia == r2.inertia and r1.n_iter == r2.n_iter
seeds = kmeans_plusplus_init(X, 4, np.random.default_rng(0))
for row in seeds:
    assert any(np.array_equal(row, x) for x in X)

# k=1 returns the global mean with zero iterations of movement after
# the first update... (centre equals mean exactly).
r = kmeans(X, k=1, seed=0)
assert np.allclose(r.centers[0], X.mean(axis=0), atol=1e-8), r.centers

# Invalid inputs raise.
for bad in (lambda: kmeans(X, k=0), lambda: kmeans(X, k=len(X) + 1),
            lambda: kmeans(X, k=2, init="bogus")):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

print("All tests passed.")
