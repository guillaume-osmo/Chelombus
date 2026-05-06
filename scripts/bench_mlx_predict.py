"""Benchmark PQ assignment backends.

Compares ``predict_mlx`` (pure MLX, vectorised) and ``_predict_numba``
(Numba CPU baseline) on synthetic PQ codes.

Usage:
    python scripts/bench_mlx_predict.py [--n N] [--k K] [--m M] [--reps R]

The metal-kernel path (``predict_metal``) is also benchmarked when present
on the ``mlx-addons`` branch.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from chelombus.clustering._mlx_predict import predict_mlx
from chelombus.clustering.PyQKmeans import _build_distance_tables, _predict_numba

try:
    from chelombus.clustering._mlx_metal import predict_metal  # type: ignore
    _METAL_AVAILABLE = True
except ImportError:
    _METAL_AVAILABLE = False


def _make_problem(n: int, k_clusters: int, m: int, k_codebook: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    pq_codes = rng.integers(0, k_codebook, size=(n, m), dtype=np.uint8)
    centers = rng.integers(0, k_codebook, size=(k_clusters, m), dtype=np.uint8)
    # Build dtables from a synthetic codebook so distances vary smoothly.
    codewords = rng.standard_normal((m, k_codebook, 4), dtype=np.float32)
    dtables = _build_distance_tables(codewords)
    return pq_codes, centers, dtables


def _time_call(fn, *args, reps: int = 3, warmup: int = 1, **kwargs):
    for _ in range(warmup):
        out = fn(*args, **kwargs)
    elapsed = []
    for _ in range(reps):
        gc.collect()
        t0 = time.time()
        out = fn(*args, **kwargs)
        elapsed.append(time.time() - t0)
    return out, min(elapsed), np.mean(elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--k", type=int, default=10_000)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU baseline")
    args = ap.parse_args()

    n, k, m = args.n, args.k, args.m
    print(f"=== PQ assignment benchmark ===")
    print(f"  N={n:,}  K={k:,}  M={m}  reps={args.reps}  warmup={args.warmup}")

    pq_codes, centers, dtables = _make_problem(n, k, m)

    print()
    if not args.no_cpu:
        labels_cpu, t_min_cpu, t_avg_cpu = _time_call(
            _predict_numba, pq_codes, centers, dtables,
            reps=args.reps, warmup=args.warmup,
        )
        print(f"  CPU (Numba):  min={t_min_cpu:.2f}s  avg={t_avg_cpu:.2f}s  "
              f"rate={n / t_min_cpu:,.0f} pts/s")
    else:
        labels_cpu = None

    labels_mlx, t_min_mlx, t_avg_mlx = _time_call(
        predict_mlx, pq_codes, centers, dtables,
        reps=args.reps, warmup=args.warmup,
    )
    print(f"  MLX vectorised:  min={t_min_mlx:.2f}s  avg={t_avg_mlx:.2f}s  "
          f"rate={n / t_min_mlx:,.0f} pts/s")

    if _METAL_AVAILABLE:
        labels_metal, t_min_metal, t_avg_metal = _time_call(
            predict_metal, pq_codes, centers, dtables,
            reps=args.reps, warmup=args.warmup,
        )
        print(f"  MLX Metal kernel:  min={t_min_metal:.2f}s  avg={t_avg_metal:.2f}s  "
              f"rate={n / t_min_metal:,.0f} pts/s")
        if labels_cpu is not None:
            assert np.array_equal(labels_metal, labels_cpu), \
                "Metal kernel disagrees with CPU baseline"
    else:
        print("  (Metal kernel not available on this branch)")

    if labels_cpu is not None:
        assert np.array_equal(labels_mlx, labels_cpu), \
            "MLX vectorised disagrees with CPU baseline"

    print()
    if labels_cpu is not None:
        print(f"  Speedups vs CPU:  MLX={t_min_cpu / t_min_mlx:.2f}x"
              + (f"  Metal={t_min_cpu / t_min_metal:.2f}x" if _METAL_AVAILABLE else ""))


if __name__ == "__main__":
    main()
