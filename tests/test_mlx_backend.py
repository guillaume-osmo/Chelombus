"""Tests for the MLX (Apple Silicon) backend.

Skipped on platforms without MLX. The tests assert exact-match parity with
the Numba CPU path on small synthetic data: PQ assignment is fully discrete
(integer codes -> float32 distance via lookup table), so when the codebook
has no ties the argmin is deterministic and identical across backends.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="mlx not installed")
if not mx.metal.is_available():
    pytest.skip("MLX Metal device not available", allow_module_level=True)

from chelombus.encoder.encoder import PQEncoder
from chelombus.clustering._mlx_predict import predict_mlx
from chelombus.clustering.PyQKmeans import (
    PQKMeans,
    _build_distance_tables,
    _predict_numba,
)

try:
    from chelombus.clustering._mlx_metal import predict_metal
    _METAL_AVAILABLE = True
except ImportError:
    _METAL_AVAILABLE = False

requires_metal = pytest.mark.skipif(
    not _METAL_AVAILABLE,
    reason="mx.fast.metal_kernel not available",
)


@pytest.fixture
def trained_encoder():
    """Train a PQ encoder on CPU; same fixture for all tests."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((500, 48), dtype=np.float32)
    encoder = PQEncoder(k=16, m=6, iterations=10)
    encoder.fit(X_train, verbose=0, device="cpu")
    return encoder


def test_predict_mlx_matches_numba(trained_encoder):
    """predict_mlx and _predict_numba must produce identical labels."""
    encoder = trained_encoder
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 48), dtype=np.float32)

    pq_codes = encoder.transform(X, verbose=0, device="cpu").astype(np.uint8)
    dtables = _build_distance_tables(encoder.codewords)

    # Use random codes for "centers" so we exercise a non-trivial assignment.
    centers = pq_codes[rng.choice(len(pq_codes), size=32, replace=False)].copy()

    labels_cpu = _predict_numba(pq_codes, centers, dtables)
    labels_mlx = predict_mlx(pq_codes, centers, dtables)

    np.testing.assert_array_equal(labels_mlx, labels_cpu)


def test_predict_mlx_with_padded_dtables(trained_encoder):
    """Even when k_codebook < 256, predict_mlx must pad and match Numba."""
    encoder = trained_encoder
    rng = np.random.default_rng(9)
    X = rng.standard_normal((150, 48), dtype=np.float32)
    pq_codes = encoder.transform(X, verbose=0, device="cpu").astype(np.uint8)
    dtables = _build_distance_tables(encoder.codewords)
    assert dtables.shape == (6, 16, 16), "fixture invariant"

    centers = pq_codes[rng.choice(len(pq_codes), size=8, replace=False)].copy()

    labels_cpu = _predict_numba(pq_codes, centers, dtables)
    labels_mlx = predict_mlx(pq_codes, centers, dtables)
    np.testing.assert_array_equal(labels_mlx, labels_cpu)


def test_predict_mlx_batched(trained_encoder):
    """A batch_size smaller than N must produce the same labels as one shot."""
    encoder = trained_encoder
    rng = np.random.default_rng(11)
    X = rng.standard_normal((1000, 48), dtype=np.float32)
    pq_codes = encoder.transform(X, verbose=0, device="cpu").astype(np.uint8)
    dtables = _build_distance_tables(encoder.codewords)
    centers = pq_codes[rng.choice(len(pq_codes), size=20, replace=False)].copy()

    labels_one_shot = predict_mlx(pq_codes, centers, dtables, batch_size=0)
    labels_batched = predict_mlx(pq_codes, centers, dtables, batch_size=128)
    np.testing.assert_array_equal(labels_one_shot, labels_batched)


def test_encoder_mlx_fit_runs(trained_encoder):
    """The MLX encoder fit path runs and produces a usable codebook."""
    rng = np.random.default_rng(3)
    X_train = rng.standard_normal((400, 48), dtype=np.float32)
    encoder = PQEncoder(k=8, m=6, iterations=5)
    encoder.fit(X_train, verbose=0, device="mlx")

    assert encoder.encoder_is_trained
    assert encoder.codewords.shape == (6, 8, 8)

    # Transform a fresh batch on MLX and on CPU; codes should be valid uint8.
    X = rng.standard_normal((100, 48), dtype=np.float32)
    codes_mlx = encoder.transform(X, verbose=0, device="mlx")
    codes_cpu = encoder.transform(X, verbose=0, device="cpu")
    assert codes_mlx.shape == (100, 6)
    assert codes_cpu.shape == (100, 6)
    # Codes are integer indices into the codebook.
    assert codes_mlx.max() < 8
    assert codes_cpu.max() < 8


def test_encoder_transform_mlx_matches_cpu(trained_encoder):
    """Transform on MLX matches CPU transform when codewords are identical."""
    encoder = trained_encoder  # already trained on CPU
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 48), dtype=np.float32)

    codes_cpu = encoder.transform(X, verbose=0, device="cpu")
    codes_mlx = encoder.transform(X, verbose=0, device="mlx")

    # Argmin on float32 distances: equality holds when there are no ties.
    # On standard normal data with 16 codebook entries this is essentially
    # always the case at float32 precision; assert equality directly.
    np.testing.assert_array_equal(codes_mlx, codes_cpu)


def test_pqkmeans_fit_predict_mlx(trained_encoder):
    """End-to-end: train an encoder on CPU, cluster via MLX path."""
    encoder = trained_encoder
    rng = np.random.default_rng(13)
    X = rng.standard_normal((1000, 48), dtype=np.float32)
    pq_codes = encoder.transform(X, verbose=0, device="cpu").astype(np.uint8)

    clusterer = PQKMeans(encoder, k=12, iteration=5, verbose=False)
    labels = clusterer.fit_predict(pq_codes, device="mlx")

    assert labels.shape == (1000,)
    assert labels.dtype == np.int32
    assert labels.min() >= 0
    assert labels.max() < 12
    # Predict re-uses the trained centers and should match the labels just
    # returned by fit_predict.
    labels_again = clusterer.predict(pq_codes, device="mlx")
    np.testing.assert_array_equal(labels, labels_again)


@requires_metal
def test_predict_metal_matches_numba(trained_encoder):
    """The Metal kernel produces identical labels to the Numba CPU path."""
    encoder = trained_encoder
    rng = np.random.default_rng(101)
    X = rng.standard_normal((400, 48), dtype=np.float32)
    pq_codes = encoder.transform(X, verbose=0, device="cpu").astype(np.uint8)
    dtables = _build_distance_tables(encoder.codewords)
    centers = pq_codes[rng.choice(len(pq_codes), size=24, replace=False)].copy()

    labels_cpu = _predict_numba(pq_codes, centers, dtables)
    labels_metal = predict_metal(pq_codes, centers, dtables)
    np.testing.assert_array_equal(labels_metal, labels_cpu)


@requires_metal
def test_predict_metal_varies_M(trained_encoder):
    """Kernel cache builds correct kernels for different M values."""
    rng = np.random.default_rng(202)
    for m in (4, 6, 8):
        X = rng.standard_normal((300, m * 8), dtype=np.float32)
        encoder = PQEncoder(k=16, m=m, iterations=5)
        encoder.fit(X, verbose=0, device="cpu")
        pq_codes = encoder.transform(X, verbose=0, device="cpu").astype(np.uint8)
        dtables = _build_distance_tables(encoder.codewords)
        centers = pq_codes[rng.choice(len(pq_codes), size=10, replace=False)].copy()

        labels_cpu = _predict_numba(pq_codes, centers, dtables)
        labels_metal = predict_metal(pq_codes, centers, dtables)
        np.testing.assert_array_equal(
            labels_metal, labels_cpu,
            err_msg=f"Metal/CPU mismatch at M={m}",
        )


def test_pqkmeans_predict_cuda_path_unavailable_raises(trained_encoder):
    """Asking for 'gpu' on a non-CUDA host should raise, not silently degrade."""
    import chelombus.clustering.PyQKmeans as mod

    if mod._GPU_AVAILABLE:
        pytest.skip("CUDA host: this test only runs on Apple Silicon")

    encoder = trained_encoder
    clusterer = PQKMeans(encoder, k=4, iteration=2)
    with pytest.raises(RuntimeError, match="GPU/CUDA requested"):
        clusterer._resolve_backend("gpu")
