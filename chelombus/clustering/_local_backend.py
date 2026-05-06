"""Local PQk-means backend stub.

Used on platforms where ``pqkmeans`` (the C++ Matsui et al. library) is not
available — most notably Apple Silicon, where the upstream wheel does not
build. ``PQKMeans`` instantiates this stub instead so the public surface
(``cluster_centers_`` getter / setter, save/load) still works while the
heavy lifting happens through the MLX path (``_fit_mlx`` / ``predict_mlx``).

The stub intentionally does not implement ``fit`` / ``fit_predict`` /
``predict``: those CPU paths require ``pqkmeans`` and should never be reached
when this backend is in use.
"""

import numpy as np


class _LocalImpl:
    """Mimics ``pqkmeans.clustering.PQKMeans._impl`` for set_cluster_centers."""

    def __init__(self) -> None:
        self._centers: list = []

    def set_cluster_centers(self, centers: list) -> None:
        self._centers = centers


class LocalPQKMeansBackend:
    """Drop-in replacement for ``pqkmeans.clustering.PQKMeans`` (centers only).

    Stores cluster centers as a Python list (matching the C++ backend's
    accessor shape) and exposes the same ``_impl.set_cluster_centers`` and
    ``cluster_centers_`` surface used by ``PQKMeans``.
    """

    def __init__(self, encoder, k: int, iteration: int = 20, verbose: bool = False) -> None:
        self.encoder = encoder
        self.k = k
        self.iteration = iteration
        self.verbose = verbose
        self._impl = _LocalImpl()

    @property
    def cluster_centers_(self) -> np.ndarray:
        return np.asarray(self._impl._centers, dtype=np.uint8)

    def fit(self, X_train: np.ndarray) -> None:
        raise RuntimeError(
            "LocalPQKMeansBackend.fit() is not implemented. "
            "Use device='mlx' or 'gpu' on a CUDA host."
        )

    def fit_predict(self, X_train: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "LocalPQKMeansBackend.fit_predict() is not implemented. "
            "Use device='mlx' or 'gpu' on a CUDA host."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "LocalPQKMeansBackend.predict() is not implemented. "
            "Use device='mlx' or 'gpu' on a CUDA host."
        )
