"""MLX-accelerated PQ encoder training and transform for Apple Silicon.

Mirrors the CUDA paths in ``encoder.py``:

    _fit_gpu       -> fit_mlx        (batched GEMM-based KMeans per subvector)
    _transform_gpu -> transform_mlx  (batched argmin per subvector)

The GEMM identity ``||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.cT`` is preserved.
For accumulation we use MLX's ``at[].add(...)`` scatter API in place of
PyTorch's ``index_add_``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import mlx.core as mx


def _mlx_batch_size(
    N: int,
    k: int,
    min_batch: int = 250_000,
    max_batch: int = 2_000_000,
) -> int:
    """Pick a batch size for the assignment step.

    Apple Silicon uses unified memory, so we don't have a discrete VRAM budget
    to size against. We use available system memory minus a 1 GiB reserve.
    Per-point scratch is dominated by the (B, k) distance row and the int32
    label.
    """
    if N <= 0:
        return 0
    floor = min(min_batch, N)
    ceiling = min(max_batch, N)
    bytes_per_point = k * 4 + 8 + 4
    fallback_budget = 1 * 1024**3

    try:
        import psutil

        free = psutil.virtual_memory().available
        usable = max(free - 1 * 1024**3, free // 2)
        scratch_budget = usable if usable > 0 else max(free // 2, floor * bytes_per_point)
    except Exception:
        scratch_budget = fallback_budget

    batch = max(scratch_budget // bytes_per_point, floor)
    return int(min(max(batch, floor), ceiling))


def fit_mlx(encoder, X_train: NDArray, verbose: int = 1) -> None:
    """MLX-accelerated KMeans per subvector. Populates ``encoder.codewords``.

    Args:
        encoder: a ``PQEncoder`` instance with ``m``, ``k``, ``D_subvector``,
            ``iterations``, and ``codewords`` already initialised.
        X_train: (N, D) input matrix, D divisible by ``encoder.m``.
        verbose: passed through to tqdm.
    """
    N = X_train.shape[0]
    subvector_dim = encoder.D_subvector
    X_f32 = np.ascontiguousarray(X_train, dtype=np.float32)
    rng = np.random.default_rng()

    iterable = range(encoder.m)
    if verbose > 0:
        iterable = tqdm(iterable, desc="Training PQ-encoder (MLX)", total=encoder.m)

    for subvector_idx in iterable:
        sub_slice = X_f32[:, subvector_dim * subvector_idx : subvector_dim * (subvector_idx + 1)]
        X_mx = mx.array(np.ascontiguousarray(sub_slice))
        x_sq = mx.sum(X_mx * X_mx, axis=1)  # (N,)
        B = _mlx_batch_size(N, encoder.k)

        # Random init: sample k points
        idx = rng.choice(N, size=encoder.k, replace=(N < encoder.k))
        idx_mx = mx.array(idx.astype(np.int32))
        centroids = mx.take(X_mx, idx_mx, axis=0)
        mx.eval(centroids, x_sq)

        for _ in range(encoder.iterations):
            c_sq = mx.sum(centroids * centroids, axis=1)  # (k,)
            counts = mx.zeros((encoder.k,), dtype=mx.float32)
            sums = mx.zeros_like(centroids)

            for start in range(0, N, B):
                end = min(start + B, N)
                x_batch = X_mx[start:end]
                # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.cT
                dist = (
                    x_sq[start:end][:, None]
                    + c_sq[None, :]
                    - 2.0 * (x_batch @ centroids.T)
                )
                dist = mx.maximum(dist, 0.0)
                labels = mx.argmin(dist, axis=1).astype(mx.int32)

                # Scatter-add into sums and counts.
                ones = mx.ones((end - start,), dtype=mx.float32)
                counts = counts.at[labels].add(ones)
                sums = sums.at[labels].add(x_batch)

                # Force eval to bound memory between batches.
                mx.eval(counts, sums)

            empty = counts == 0
            counts_safe = mx.maximum(counts, 1.0)
            new_centroids = sums / counts_safe[:, None]
            new_centroids = mx.where(empty[:, None], centroids, new_centroids)
            centroids = new_centroids
            mx.eval(centroids)

        encoder.codewords[subvector_idx] = np.asarray(centroids).astype(np.float32)


def transform_mlx(encoder, X: NDArray) -> NDArray:
    """MLX-accelerated transform: return PQ codes (N, m) via per-subvector argmin.

    Uses the same GEMM identity as ``fit_mlx`` to avoid materialising the full
    distance matrix outside the per-batch scratch.
    """
    N, D = X.shape
    subvector_dim = int(D / encoder.m)
    pq_codes = np.zeros((N, encoder.m), dtype=encoder.codebook_dtype)

    cw_mx = [
        mx.array(np.ascontiguousarray(encoder.codewords[sub], dtype=np.float32))
        for sub in range(encoder.m)
    ]
    cw_sq = [mx.sum(cw * cw, axis=1) for cw in cw_mx]
    mx.eval(*cw_mx, *cw_sq)

    # Bound batch by both per-point feature size and codebook scratch.
    bytes_per_point = (D + encoder.k) * 4
    max_batch = max((1 * 1024**3) // bytes_per_point, 1024)
    max_batch = min(max_batch, N)
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)

    for start in range(0, N, max_batch):
        end = min(start + max_batch, N)
        batch_mx = mx.array(X_f32[start:end])
        for sub in range(encoder.m):
            chunk = batch_mx[:, subvector_dim * sub : subvector_dim * (sub + 1)]
            x_sq = mx.sum(chunk * chunk, axis=1)
            dist = x_sq[:, None] + cw_sq[sub][None, :] - 2.0 * (chunk @ cw_mx[sub].T)
            labels = mx.argmin(dist, axis=1)
            pq_codes[start:end, sub] = np.asarray(labels).astype(encoder.codebook_dtype)

    return pq_codes
