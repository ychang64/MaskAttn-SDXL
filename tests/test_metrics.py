from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from maskattn_sdxl.metrics import (
    _knn_radii,
    _manifold_coverage,
    compute_fid_and_precision_recall,
    frechet_distance_from_features,
    precision_recall_from_features,
)


def _line(*values: float) -> np.ndarray:
    return np.array([[value, 0.0] for value in values])


def test_coverage_counts_every_reference_ball_not_only_the_nearest() -> None:
    reference = _line(0.0, 0.1, 0.2, 0.3, 5.0, 10.0, 15.0, 20.0)
    radii = _knn_radii(reference, 3)
    query = _line(0.8)
    nearest = int(np.argmin(np.linalg.norm(reference - query, axis=1)))
    assert np.linalg.norm(reference[nearest] - query) > radii[nearest]
    assert _manifold_coverage(query, reference, radii) == 1.0


def test_precision_and_recall_bounds() -> None:
    points = np.random.default_rng(0).normal(size=(64, 4))
    assert precision_recall_from_features(points, points.copy()) == (1.0, 1.0)
    assert precision_recall_from_features(points, points + 1000.0) == (0.0, 0.0)


def test_precision_recall_needs_more_samples_than_k() -> None:
    with pytest.raises(ValueError, match="more than 3"):
        precision_recall_from_features(np.zeros((3, 2)), np.zeros((3, 2)))


def test_fid_of_identical_features_is_zero_and_grows_with_a_shift() -> None:
    points = np.random.default_rng(0).normal(size=(256, 4))
    assert frechet_distance_from_features(points, points) == pytest.approx(0.0, abs=1e-6)
    assert frechet_distance_from_features(points, points + 3.0) == pytest.approx(36.0, abs=1e-6)


def test_fid_matches_the_matrix_square_root_formula() -> None:
    from scipy import linalg

    rng = np.random.default_rng(0)
    first = rng.normal(size=(512, 6)) @ rng.normal(size=(6, 6))
    second = rng.normal(size=(512, 6)) @ rng.normal(size=(6, 6)) + 0.5
    covariance_first, covariance_second = np.cov(first, rowvar=False), np.cov(second, rowvar=False)
    difference = first.mean(axis=0) - second.mean(axis=0)
    root = np.real(linalg.sqrtm(covariance_first @ covariance_second))
    expected = difference @ difference + np.trace(covariance_first + covariance_second - 2.0 * root)
    assert frechet_distance_from_features(first, second) == pytest.approx(expected, rel=1e-8)


def test_fid_stays_finite_below_the_feature_dimension() -> None:
    rng = np.random.default_rng(0)
    assert np.isfinite(frechet_distance_from_features(rng.normal(size=(32, 128)), rng.normal(size=(32, 128))))


def test_wrapper_runs_clean_fid_folder_extraction(monkeypatch, tmp_path) -> None:
    """Only the Inception network is replaced, so Clean-FID's real file loading and resizing run offline."""
    pytest.importorskip("cleanfid")
    from cleanfid import fid
    from PIL import Image

    rng = np.random.default_rng(0)
    for name in ("generated", "real"):
        (tmp_path / name).mkdir()
        for index in range(6):
            color = tuple(int(value) for value in rng.integers(0, 256, size=3))
            Image.new("RGB", (8, 8), color).save(tmp_path / name / f"{index}.png")

    monkeypatch.setattr(fid, "build_feature_extractor", lambda mode, device: lambda batch: batch.mean(dim=(2, 3)))
    results = compute_fid_and_precision_recall(tmp_path / "generated", tmp_path / "real", device="cpu")
    assert set(results) == {"fid", "precision", "recall", "num_generated", "num_real"}
    assert np.isfinite(results["fid"])
    assert results["num_generated"] == results["num_real"] == 6
