"""Tests for field_site_sfm.satellite_fusion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from field_site_sfm.satellite_fusion import (
    FusionControlPoint,
    apply_affine_to_pixels,
    estimate_affine_transform,
)


# ---------------------------------------------------------------------------
# FusionControlPoint
# ---------------------------------------------------------------------------


class TestFusionControlPoint:
    def test_construction(self) -> None:
        cp = FusionControlPoint(
            sfm_pixel_x=100.0,
            sfm_pixel_y=200.0,
            satellite_pixel_x=300.0,
            satellite_pixel_y=400.0,
            world_x=10.5,
            world_y=20.5,
            label="GCP_NW",
        )
        assert cp.sfm_pixel_x == 100.0
        assert cp.world_x == 10.5
        assert cp.label == "GCP_NW"

    def test_repr_contains_label(self) -> None:
        cp = FusionControlPoint(0, 0, 0, 0, 0, 0, label="MYPOINT")
        assert "MYPOINT" in repr(cp)


# ---------------------------------------------------------------------------
# estimate_affine_transform
# ---------------------------------------------------------------------------


class TestEstimateAffineTransform:
    def _make_control_points(
        self,
        n: int = 4,
        *,
        scale_x: float = 0.001,
        scale_y: float = 0.001,
        offset_x: float = -73.9,
        offset_y: float = 40.7,
    ) -> list[FusionControlPoint]:
        """Create synthetic control points with a known affine relationship."""
        points = []
        for i in range(n):
            px = float(i * 100)
            py = float(i * 50)
            wx = offset_x + px * scale_x
            wy = offset_y + py * scale_y
            points.append(
                FusionControlPoint(
                    sfm_pixel_x=px,
                    sfm_pixel_y=py,
                    satellite_pixel_x=0,
                    satellite_pixel_y=0,
                    world_x=wx,
                    world_y=wy,
                )
            )
        return points

    def test_requires_at_least_3_points(self) -> None:
        with pytest.raises(ValueError, match="3"):
            estimate_affine_transform([])
        with pytest.raises(ValueError, match="3"):
            estimate_affine_transform(
                [FusionControlPoint(0, 0, 0, 0, 0, 0)] * 2
            )

    def test_exact_transform_3_points(self) -> None:
        """With exactly 3 non-collinear points the fit should be near-exact."""
        cps = self._make_control_points(n=3)
        T = estimate_affine_transform(cps)
        assert T.shape == (3, 3)

        # Verify the transform maps SfM pixels to world coords correctly.
        for cp in cps:
            src = np.array([[cp.sfm_pixel_x, cp.sfm_pixel_y, 1.0]])
            pred = (T @ src.T).T[0]
            assert pred[0] == pytest.approx(cp.world_x, abs=1e-6)
            assert pred[1] == pytest.approx(cp.world_y, abs=1e-6)

    def test_returns_3x3_matrix(self) -> None:
        cps = self._make_control_points(n=5)
        T = estimate_affine_transform(cps)
        assert T.shape == (3, 3)

    def test_more_points_overdetermined(self) -> None:
        """More points than 3 should still work (least-squares)."""
        cps = self._make_control_points(n=10)
        T = estimate_affine_transform(cps)
        assert T.shape == (3, 3)


# ---------------------------------------------------------------------------
# apply_affine_to_pixels
# ---------------------------------------------------------------------------


class TestApplyAffineToPixels:
    def test_identity(self) -> None:
        T = np.eye(3)
        pixels = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = apply_affine_to_pixels(pixels, T)
        # Identity: world == sfm pixel (the third coord collapses).
        np.testing.assert_allclose(result[:, 0], pixels[:, 0])
        np.testing.assert_allclose(result[:, 1], pixels[:, 1])

    def test_shape(self) -> None:
        T = np.eye(3)
        pixels = np.random.rand(7, 2)
        result = apply_affine_to_pixels(pixels, T)
        assert result.shape == (7, 2)

    def test_known_transform(self) -> None:
        """Apply a pure translation transform and verify output."""
        # world_x = sfm_x + 5,  world_y = sfm_y + 10
        T = np.array([
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 10.0],
            [0.0, 0.0, 1.0],
        ])
        pixels = np.array([[0.0, 0.0], [100.0, 50.0]])
        result = apply_affine_to_pixels(pixels, T)
        np.testing.assert_allclose(result[0], [5.0, 10.0])
        np.testing.assert_allclose(result[1], [105.0, 60.0])

    def test_roundtrip_with_estimate(self) -> None:
        """estimate_affine_transform + apply_affine_to_pixels roundtrip."""
        scale = 0.0005
        offset_x = -73.5
        offset_y = 40.5

        cps = [
            FusionControlPoint(
                sfm_pixel_x=float(i * 200),
                sfm_pixel_y=float(i * 100),
                satellite_pixel_x=0,
                satellite_pixel_y=0,
                world_x=offset_x + float(i * 200) * scale,
                world_y=offset_y + float(i * 100) * scale,
            )
            for i in range(4)
        ]

        T = estimate_affine_transform(cps)
        test_pixels = np.array([[100.0, 50.0], [500.0, 250.0]])
        world = apply_affine_to_pixels(test_pixels, T)

        # Verify against the expected linear mapping.
        expected_wx = offset_x + test_pixels[:, 0] * scale
        expected_wy = offset_y + test_pixels[:, 1] * scale
        np.testing.assert_allclose(world[:, 0], expected_wx, atol=1e-5)
        np.testing.assert_allclose(world[:, 1], expected_wy, atol=1e-5)


# ---------------------------------------------------------------------------
# georeference_orthophoto (requires rasterio – tested with mocks)
# ---------------------------------------------------------------------------


class TestGeoreferenceOrthophoto:
    def test_raises_without_rasterio(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "field_site_sfm.satellite_fusion._RASTERIO_AVAILABLE", False
        )
        from field_site_sfm.satellite_fusion import georeference_orthophoto

        with pytest.raises(ImportError, match="rasterio"):
            georeference_orthophoto(
                ortho_path=tmp_path / "ortho.tif",
                control_points=[
                    FusionControlPoint(0, 0, 0, 0, 0, 0),
                    FusionControlPoint(1, 0, 0, 0, 1, 0),
                    FusionControlPoint(0, 1, 0, 0, 0, 1),
                ],
                output_path=tmp_path / "out.tif",
            )

    def test_raises_with_too_few_control_points(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "field_site_sfm.satellite_fusion._RASTERIO_AVAILABLE", True
        )
        from field_site_sfm.satellite_fusion import georeference_orthophoto

        with pytest.raises(ValueError, match="3"):
            georeference_orthophoto(
                ortho_path=tmp_path / "ortho.tif",
                control_points=[FusionControlPoint(0, 0, 0, 0, 0, 0)],
                output_path=tmp_path / "out.tif",
            )
