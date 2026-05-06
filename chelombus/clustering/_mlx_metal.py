"""Custom Metal kernel for PQ assignment via ``mx.fast.metal_kernel``.

Direct translation of the Triton ``_pq_assign_kernel`` to Metal Shading
Language. Same algorithm — for each data point, loop over all K cluster
centers and accumulate symmetric squared-distance via the precomputed
``dtables`` lookup, tracking the running argmin in registers.

This is the fast path on Apple Silicon. The pure-vectorised path in
``_mlx_predict.py`` has too much per-op overhead (each ``mx.take`` /
``mx.where`` is a separate Metal dispatch); a single fused kernel removes
that overhead and lets the inner loop fan out across the GPU's compute units.

Public API mirrors ``predict_gpu`` and ``predict_mlx`` exactly, so it's a
drop-in replacement.
"""

from __future__ import annotations

import time as _time

import numpy as np

import mlx.core as mx


# Cache compiled kernels keyed by M. M is baked into the kernel source so
# the inner loop unrolls; K and N stay runtime.
_kernel_cache: dict[int, object] = {}

# Resident MLX tensor cache for centers + dtables (mirrors the pure-MLX path).
_resident_cache: dict = {}


def _make_kernel(M: int, block_n: int = 64, block_k: int = 256):
    """Build (or fetch cached) Metal kernel for given (M, block_n, block_k).

    M is baked into source so the inner-most M-loop unrolls. ``block_n`` and
    ``block_k`` size threadgroup-cached center tiles: each threadgroup loads
    ``block_k * M`` bytes of centers into ``threadgroup`` memory once per
    tile, then ``block_n`` threads stream through the tile against their own
    point. This amortises the global center reads across ``block_n`` points
    and is the difference between memory-bound and compute-bound on M-series
    GPUs at K >= 10k.
    """
    key = (M, block_n, block_k)
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    source = f"""
        const uint M = {M};
        const uint BLOCK_N = {block_n};
        const uint BLOCK_K = {block_k};
        const uint TABLE = 256u * 256u;

        uint tid = thread_position_in_threadgroup.x;
        uint gid = threadgroup_position_in_grid.x * BLOCK_N + tid;
        uint N = codes_shape[0];
        uint K = centers_shape[0];

        // Threadgroup-shared centers tile.
        threadgroup uchar centers_tg[BLOCK_K * M];

        // Per-thread state. Threads where gid >= N still participate in the
        // cooperative loads (otherwise we'd diverge before the barrier) but
        // skip the actual distance accumulation.
        bool active = gid < N;

        uchar pc[M];
        if (active) {{
            #pragma unroll
            for (uint m = 0; m < M; ++m) {{
                pc[m] = codes[gid * M + m];
            }}
        }}

        float best_dist = INFINITY;
        int best_label = 0;

        for (uint c_start = 0; c_start < K; c_start += BLOCK_K) {{
            uint cur_block = min(BLOCK_K, K - c_start);

            // Cooperative load: BLOCK_N threads read cur_block * M bytes.
            for (uint i = tid; i < cur_block * M; i += BLOCK_N) {{
                centers_tg[i] = centers[(c_start + i / M) * M + (i % M)];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active) {{
                for (uint c_local = 0; c_local < cur_block; ++c_local) {{
                    float dist = 0.0f;
                    #pragma unroll
                    for (uint m = 0; m < M; ++m) {{
                        uint cc = (uint)centers_tg[c_local * M + m];
                        uint pcm = (uint)pc[m];
                        dist += dtables[(m * 256u + pcm) * 256u + cc];
                    }}
                    if (dist < best_dist) {{
                        best_dist = dist;
                        best_label = (int)(c_start + c_local);
                    }}
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (active) {{
            labels[gid] = best_label;
        }}
    """

    kernel = mx.fast.metal_kernel(
        name=f"pq_assign_m{M}_n{block_n}_k{block_k}",
        input_names=["codes", "centers", "dtables"],
        output_names=["labels"],
        source=source,
        ensure_row_contiguous=True,
    )
    _kernel_cache[key] = kernel
    return kernel


def _pick_block_sizes(M: int) -> tuple[int, int]:
    """Choose (block_n, block_k) for the threadgroup-tiled kernel.

    Threadgroup memory budget: ~32KB usable per group on M-series GPUs. The
    cached tile is ``block_k * M`` bytes for centers; the rest of the budget
    is spent on register spill and per-thread state.

    block_n=256 uses 8 SIMD groups per threadgroup (Apple GPUs have 32-wide
    SIMD), which gives the scheduler enough latency-hiding parallelism. A
    larger block would not fit comfortably alongside the centers tile.
    """
    block_n = 256
    # Aim for ~8KB of TG memory for centers tile.
    target_centers_bytes = 8 * 1024
    block_k = max(64, target_centers_bytes // M)
    block_k = 1 << (block_k.bit_length() - 1)
    block_k = min(block_k, 1024)
    return block_n, block_k


def _get_or_upload(key: str, array: np.ndarray, dtype) -> mx.array:
    arr = np.ascontiguousarray(array)
    entry = _resident_cache.get(key)
    if entry is not None:
        ref, tensor = entry
        if ref.shape == arr.shape and np.array_equal(ref, arr):
            return tensor
    tensor = mx.array(arr).astype(dtype)
    _resident_cache[key] = (arr.copy(), tensor)
    return tensor


def _auto_batch_size(N: int, M: int) -> int:
    """How many points to process per kernel dispatch.

    Per-batch device memory dominated by codes (M bytes) + labels (4 bytes)
    per point. With 16-64 GB of unified memory we can comfortably handle
    100M+ points per dispatch, but going too big lengthens single-call
    latency and delays user-visible progress reporting.
    """
    try:
        import psutil

        free = psutil.virtual_memory().available
        usable = max(free - 2 * 1024**3, free // 2)
        bytes_per_point = M + 4
        return int(min(max(usable // bytes_per_point, 1024), 50_000_000, N))
    except Exception:
        return min(5_000_000, N)


def predict_metal(
    pq_codes: np.ndarray,
    centers: np.ndarray,
    dtables: np.ndarray,
    batch_size: int = 0,
    verbose: bool = False,
    threadgroup: int = 256,
) -> np.ndarray:
    """Metal-accelerated PQ assignment.

    Args:
        pq_codes: (N, M) uint8 PQ codes.
        centers: (K, M) uint8 cluster center codes.
        dtables: (M, k_cb, k_cb) float32 distance lookup tables.
        batch_size: Max points per kernel dispatch. 0 = auto from RAM.
        verbose: Print per-batch progress.
        threadgroup: Threads per threadgroup (must divide grid size or be
            handled by the in-kernel ``gid >= N`` guard).

    Returns:
        (N,) int32 cluster labels.
    """
    N, M = pq_codes.shape
    K = centers.shape[0]

    # Same shape contract as the Triton kernel: dtables padded to (M, 256, 256).
    if dtables.shape[1] != 256 or dtables.shape[2] != 256:
        padded = np.zeros((M, 256, 256), dtype=np.float32)
        k_cb = dtables.shape[1]
        padded[:, :k_cb, :k_cb] = dtables
        dtables = padded

    centers_mx = _get_or_upload("centers", centers, mx.uint8)
    dtables_mx = _get_or_upload("dtables", dtables, mx.float32)

    if batch_size <= 0:
        batch_size = _auto_batch_size(N, M)

    block_n, block_k = _pick_block_sizes(M)
    kernel = _make_kernel(M, block_n=block_n, block_k=block_k)
    labels_out = np.empty(N, dtype=np.int32)

    n_batches = (N + batch_size - 1) // batch_size
    t0 = _time.time()

    for batch_idx, start in enumerate(range(0, N, batch_size)):
        end = min(start + batch_size, N)
        chunk = np.ascontiguousarray(pq_codes[start:end], dtype=np.uint8)
        codes_mx = mx.array(chunk)
        n_chunk = end - start

        # Round grid up to a multiple of block_n; in-kernel guard handles
        # the tail.
        grid_x = ((n_chunk + block_n - 1) // block_n) * block_n
        outputs = kernel(
            inputs=[codes_mx, centers_mx, dtables_mx],
            grid=(grid_x, 1, 1),
            threadgroup=(block_n, 1, 1),
            output_shapes=[(n_chunk,)],
            output_dtypes=[mx.int32],
        )
        labels_mx = outputs[0]
        mx.eval(labels_mx)
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
