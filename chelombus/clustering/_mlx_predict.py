"""MLX-accelerated PQ assignment for Apple Silicon.

Drop-in replacement for ``_predict_numba`` (and the CUDA/Triton ``predict_gpu``)
that runs on Apple Silicon GPUs via the MLX framework.

Two paths are available:

1. **Metal kernel** (``_mlx_metal.predict_metal``): a fused custom Metal kernel
   that loops over centers in-thread with online argmin. Used by default when
   ``mx.fast.metal_kernel`` is available — typically much faster than the pure
   vectorised path on real workloads.
2. **Pure MLX vectorised**: tile-over-points and tile-over-centers with online
   argmin via ``mx.take`` / ``mx.where``. Used as a fallback when the Metal
   kernel can't be imported, e.g. on older MLX builds.

Set ``CHELOMBUS_MLX_NO_METAL=1`` to force the vectorised path (debugging).
"""

from __future__ import annotations

import os
import time as _time

import numpy as np

import mlx.core as mx

_METAL_AVAILABLE = False
if os.environ.get("CHELOMBUS_MLX_NO_METAL", "0") not in {"1", "true", "True"}:
    try:
        from chelombus.clustering._mlx_metal import predict_metal
        _METAL_AVAILABLE = True
    except ImportError:
        pass


# Tile sizes. Mirror Triton choices:
#   BLOCK_N: rows per batch (memory-bound by BLOCK_N * BLOCK_K floats)
#   BLOCK_K: centers per inner tile
# Apple Silicon GPUs have plenty of unified memory, so we run with larger
# tiles than the Triton defaults (32 / 32-128).
_BLOCK_N = 4096
_BLOCK_K_BY_M = ((8, 1024), (32, 512), (float("inf"), 256))
# Fall back to a fixed tile size when system memory telemetry is unavailable
_DEFAULT_BATCH = 1_000_000


# Cache centers + dtables across calls (mirrors _gpu_predict._gpu_cache).
_mlx_cache: dict = {}


def _block_k_for_m(M: int) -> int:
    for threshold, block in _BLOCK_K_BY_M:
        if M <= threshold:
            return block
    return 256  # unreachable; keeps type-checkers happy


def _get_or_upload(key: str, array: np.ndarray, dtype) -> mx.array:
    """Upload numpy array to the MLX device, caching by content.

    Mirrors ``_gpu_predict._get_or_upload``: stores ``(reference_numpy, mx_array)``
    so cache hits validate by content, never by Python id (which is unsafe
    after a free + realloc).
    """
    arr = np.ascontiguousarray(array)
    entry = _mlx_cache.get(key)
    if entry is not None:
        ref, tensor = entry
        if ref.shape == arr.shape and np.array_equal(ref, arr):
            return tensor
    tensor = mx.array(arr).astype(dtype)
    _mlx_cache[key] = (arr.copy(), tensor)
    return tensor


def _auto_batch_size(N: int, M: int) -> int:
    """Pick a batch size from available system memory.

    Apple Silicon has unified memory — there is no separate VRAM telemetry.
    Per point we materialise:

        codes_mx:  M bytes (uint8)
        labels:    4 bytes (int32)
        scratch:   ~ BLOCK_K floats per BLOCK_N rows of points

    The dominant per-batch cost in absolute terms is the BLOCK_N x BLOCK_K
    scratch, which is fixed regardless of batch size. We therefore pick
    ``batch`` to bound code-tensor + label-tensor footprint, with a healthy
    cap so we don't try to upload billions of rows in a single shot.
    """
    try:
        import psutil

        free = psutil.virtual_memory().available
        # Reserve 1 GiB for OS / other processes / MLX cached resident state.
        usable = max(free - 1 * 1024**3, free // 2)
        bytes_per_point = M + 4
        max_batch = max(usable // bytes_per_point, 1024)
        return int(min(max_batch, N, 50_000_000))
    except Exception:
        return min(_DEFAULT_BATCH, N)


def _assign_chunk_mlx(
    codes_mx: mx.array,       # (n_chunk, M) uint8
    centers_mx: mx.array,     # (K, M) uint8
    dtables_mx: mx.array,     # (M, 256, 256) float32
    K: int,
    M: int,
    block_n: int,
    block_k: int,
) -> mx.array:
    """Run the tile-over-(points, centers) online argmin for a single batch.

    Tiles ``codes_mx`` rows in groups of ``block_n`` and ``centers_mx`` rows
    in groups of ``block_k``. Per inner iteration the working scratch is
    bounded by ``(block_n, block_k)`` floats, so the function never blows
    memory regardless of the input size.

    Returns int32 labels of shape ``(n_chunk,)``.
    """
    n_chunk = codes_mx.shape[0]
    centers_i = centers_mx.astype(mx.int32)
    out_chunks = []

    for n_start in range(0, n_chunk, block_n):
        n_end = min(n_start + block_n, n_chunk)
        cur_n = n_end - n_start
        codes_i = codes_mx[n_start:n_end].astype(mx.int32)

        best_dist = mx.full((cur_n,), float("inf"), dtype=mx.float32)
        best_label = mx.zeros((cur_n,), dtype=mx.int32)

        for c_start in range(0, K, block_k):
            c_end = min(c_start + block_k, K)
            cur_block = c_end - c_start
            c_codes = centers_i[c_start:c_end]

            dist = mx.zeros((cur_n, cur_block), dtype=mx.float32)
            for m in range(M):
                # dtables[m] (256, 256) -> gather rows by point codes -> (cur_n, 256)
                row = mx.take(dtables_mx[m], codes_i[:, m], axis=0)
                # gather columns by center codes -> (cur_n, cur_block)
                contrib = mx.take(row, c_codes[:, m], axis=1)
                dist = dist + contrib

            tile_min_dist = mx.min(dist, axis=1)
            tile_min_idx = mx.argmin(dist, axis=1).astype(mx.int32)
            tile_min_label = tile_min_idx + c_start

            update = tile_min_dist < best_dist
            best_dist = mx.where(update, tile_min_dist, best_dist)
            best_label = mx.where(update, tile_min_label, best_label)

        # Force eval per outer N tile so memory doesn't grow unbounded.
        mx.eval(best_label)
        out_chunks.append(best_label)

    if len(out_chunks) == 1:
        return out_chunks[0]
    return mx.concatenate(out_chunks, axis=0)


def predict_mlx(
    pq_codes: np.ndarray,
    centers: np.ndarray,
    dtables: np.ndarray,
    batch_size: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """MLX-accelerated PQ assignment.

    Routes to the Metal kernel (fast path) when ``mx.fast.metal_kernel`` is
    available; falls back to pure-vectorised MLX otherwise. Set
    ``CHELOMBUS_MLX_NO_METAL=1`` to force the vectorised path.

    Args:
        pq_codes: (N, M) uint8 PQ codes.
        centers: (K, M) uint8 cluster center codes.
        dtables: (M, k_cb, k_cb) float32 distance lookup tables.
        batch_size: Max points per dispatch. 0 (default) auto-detects.
        verbose: Print per-batch progress (useful for billion-scale runs).

    Returns:
        (N,) int32 cluster labels.
    """
    if _METAL_AVAILABLE:
        return predict_metal(
            pq_codes, centers, dtables,
            batch_size=batch_size, verbose=verbose,
        )

    N, M = pq_codes.shape
    K = centers.shape[0]

    # Pad dtables to (M, 256, 256) — same shape contract as the Triton kernel.
    if dtables.shape[1] != 256 or dtables.shape[2] != 256:
        padded = np.zeros((M, 256, 256), dtype=np.float32)
        k_cb = dtables.shape[1]
        padded[:, :k_cb, :k_cb] = dtables
        dtables = padded

    centers_mx = _get_or_upload("centers", centers, mx.uint8)
    dtables_mx = _get_or_upload("dtables", dtables, mx.float32)

    if batch_size <= 0:
        batch_size = _auto_batch_size(N, M)

    block_k = _block_k_for_m(M)
    labels_out = np.empty(N, dtype=np.int32)

    n_batches = (N + batch_size - 1) // batch_size
    t0 = _time.time()

    for batch_idx, start in enumerate(range(0, N, batch_size)):
        end = min(start + batch_size, N)
        chunk = np.ascontiguousarray(pq_codes[start:end], dtype=np.uint8)
        codes_mx = mx.array(chunk)

        labels_mx = _assign_chunk_mlx(
            codes_mx, centers_mx, dtables_mx, K, M, _BLOCK_N, block_k,
        )
        labels_out[start:end] = np.asarray(labels_mx)

        if verbose and n_batches > 1:
            elapsed = _time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0.0
            eta = (N - end) / rate if rate > 0 else 0
            print(
                f"    batch {batch_idx + 1}/{n_batches}  "
                f"{end:,}/{N:,} ({end/N*100:.0f}%)  "
                f"rate={rate:,.0f} pts/s  "
                f"ETA={int(eta // 60)}m{int(eta % 60)}s",
                flush=True,
            )

    return labels_out
