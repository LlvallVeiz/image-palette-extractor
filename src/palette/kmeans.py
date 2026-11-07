"""K-means clustering from scratch (numpy): Lloyd's algorithm with
k-means++ seeding.

The objective minimised is the *inertia* (within-cluster sum of squared
Euclidean distances)

    J = Σᵢ ‖xᵢ − μ_{c(i)}‖²

exactly as in Lloyd (1982) and in canonical implementations such as
scikit-learn's ``KMeans`` (whose ``inertia_`` attribute is this same
quantity).  See docs/RESEARCH.md.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class KMeansResult:
    centers: np.ndarray   # (k, d) final centroids
    labels: np.ndarray    # (n,) cluster index per point
    inertia: float        # final objective value J
    n_iter: int           # Lloyd iterations performed


def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape (n, k).

    Uses ‖x−c‖² = ‖x‖² − 2x·c + ‖c‖²; tiny negative values from float
    cancellation are clipped to 0.
    """
    x2 = np.einsum("ij,ij->i", X, X)[:, None]
    c2 = np.einsum("ij,ij->i", centers, centers)[None, :]
    d2 = x2 - 2.0 * (X @ centers.T) + c2
    return np.maximum(d2, 0.0)


def kmeans_plusplus_init(X: np.ndarray, k: int,
                         rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (Arthur & Vassilvitskii, 2007).

    The first centre is a uniformly random data point; each subsequent
    centre is a data point sampled with probability proportional to
    D(x)², its squared distance to the nearest centre already chosen.
    """
    n = len(X)
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    first = int(rng.integers(n))
    centers[0] = X[first]
    closest = _squared_distances(X, centers[0:1])[:, 0]
    for j in range(1, k):
        total = closest.sum()
        if total <= 0.0:
            # All points coincide with chosen centres: pick any point.
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=closest / total))
        centers[j] = X[idx]
        closest = np.minimum(
            closest, _squared_distances(X, centers[j:j + 1])[:, 0]
        )
    return centers


def _random_init(X: np.ndarray, k: int,
                 rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].astype(np.float64)


def kmeans(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 100,
           tol: float = 1e-4, init: str = "k-means++") -> KMeansResult:
    """Run Lloyd's algorithm on the rows of X.

    Parameters mirror the canonical API: ``init`` is ``"k-means++"``
    (default) or ``"random"``; iteration stops when the largest centre
    movement falls below ``tol`` or after ``max_iter`` iterations.
    Unlike scikit-learn there is no ``n_init`` best-of-many-runs stage:
    one seeded run keeps results exactly reproducible (documented
    simplification — see docs/RESEARCH.md).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), "
                         f"got shape {X.shape}")
    if not 1 <= k <= len(X):
        raise ValueError(f"k must be between 1 and {len(X)}, got {k}")
    rng = np.random.default_rng(seed)
    if init == "k-means++":
        centers = kmeans_plusplus_init(X, k, rng)
    elif init == "random":
        centers = _random_init(X, k, rng)
    else:
        raise ValueError(f"init must be 'k-means++' or 'random', "
                         f"got {init!r}")

    labels = np.zeros(len(X), dtype=int)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        d2 = _squared_distances(X, centers)
        labels = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members):
                new_centers[j] = members.mean(axis=0)
            else:
                # Empty cluster: restart its centre at the point worst
                # served by the current centres (largest min-distance).
                worst = int(np.argmax(d2[np.arange(len(X)), labels]))
                new_centers[j] = X[worst]
        shift = float(np.max(np.linalg.norm(new_centers - centers,
                                            axis=1)))
        centers = new_centers
        if shift < tol:
            break

    # Final assignment so labels/inertia are consistent with centers.
    d2 = _squared_distances(X, centers)
    labels = np.argmin(d2, axis=1)
    inertia = float(d2[np.arange(len(X)), labels].sum())
    return KMeansResult(centers=centers, labels=labels, inertia=inertia,
                        n_iter=n_iter)
