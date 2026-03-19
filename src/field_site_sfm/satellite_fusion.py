"""Fuse a reconstructed orthophoto with a satellite (reference) image.

**Context**

The SfM pipeline produces a local orthophoto that has excellent spatial
resolution (~5–10 cm/pixel) but initially no geographic coordinate
system.  A satellite image of the same area has accurate geographic
coordinates but poor resolution (typically 30–60 cm/pixel for freely
available imagery).

This module geo-references the SfM orthophoto by:

1. Identifying at least 3 control points that appear in both images
   (the same GCPs that were marked in the field frames can act as
   tie-points if their satellite pixel coordinates are also recorded).
2. Computing an affine or projective transformation from SfM pixel
   space → satellite image CRS.
3. Warping the SfM orthophoto into the satellite image's CRS so that
   both layers can be composited and compared.

The output is a GeoTIFF that uses the satellite image's coordinate
reference system, making it directly loadable in QGIS, GDAL, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_gcps as rasterio_from_gcps
    from rasterio.control import GroundControlPoint as RasterioGCP

    _RASTERIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Control-point dataclass
# ---------------------------------------------------------------------------


class FusionControlPoint:
    """A tie-point with coordinates in both the SfM orthophoto and the
    satellite image.

    Parameters
    ----------
    sfm_pixel_x:
        Column in the SfM orthophoto (pixels from left).
    sfm_pixel_y:
        Row in the SfM orthophoto (pixels from top).
    satellite_pixel_x:
        Column in the satellite image (pixels from left).
    satellite_pixel_y:
        Row in the satellite image (pixels from top).
    world_x:
        Easting / longitude in the satellite image's CRS.
    world_y:
        Northing / latitude in the satellite image's CRS.
    label:
        Optional human-readable label (e.g. ``"GCP_NW"``).
    """

    def __init__(
        self,
        sfm_pixel_x: float,
        sfm_pixel_y: float,
        satellite_pixel_x: float,
        satellite_pixel_y: float,
        world_x: float,
        world_y: float,
        label: str = "",
    ) -> None:
        self.sfm_pixel_x = sfm_pixel_x
        self.sfm_pixel_y = sfm_pixel_y
        self.satellite_pixel_x = satellite_pixel_x
        self.satellite_pixel_y = satellite_pixel_y
        self.world_x = world_x
        self.world_y = world_y
        self.label = label

    def __repr__(self) -> str:
        return (
            f"FusionControlPoint(label={self.label!r}, "
            f"sfm=({self.sfm_pixel_x:.1f}, {self.sfm_pixel_y:.1f}), "
            f"world=({self.world_x:.6f}, {self.world_y:.6f}))"
        )


# ---------------------------------------------------------------------------
# Affine estimation
# ---------------------------------------------------------------------------


def estimate_affine_transform(
    control_points: Sequence[FusionControlPoint],
) -> np.ndarray:
    """Estimate a 3 × 3 affine transform from SfM orthophoto pixels to
    world coordinates using least-squares regression.

    The returned matrix *T* satisfies::

        [world_x, world_y, 1]ᵀ = T @ [sfm_x, sfm_y, 1]ᵀ

    Parameters
    ----------
    control_points:
        At least 3 :class:`FusionControlPoint` instances.

    Returns
    -------
    numpy.ndarray
        Shape ``(3, 3)`` affine transform matrix.

    Raises
    ------
    ValueError
        If fewer than 3 control points are provided.
    """
    if len(control_points) < 3:
        raise ValueError(
            f"At least 3 control points are required for affine estimation "
            f"(got {len(control_points)})."
        )

    src = np.array(
        [[cp.sfm_pixel_x, cp.sfm_pixel_y, 1.0] for cp in control_points],
        dtype=np.float64,
    )
    dst_x = np.array([cp.world_x for cp in control_points], dtype=np.float64)
    dst_y = np.array([cp.world_y for cp in control_points], dtype=np.float64)

    # Solve Ax = b for each output coordinate dimension.
    coeff_x, *_ = np.linalg.lstsq(src, dst_x, rcond=None)
    coeff_y, *_ = np.linalg.lstsq(src, dst_y, rcond=None)

    transform = np.eye(3, dtype=np.float64)
    transform[0, :] = coeff_x  # world_x row
    transform[1, :] = coeff_y  # world_y row

    residuals_x = src @ coeff_x - dst_x
    residuals_y = src @ coeff_y - dst_y
    rmse = float(np.sqrt(np.mean(residuals_x**2 + residuals_y**2)))
    logger.info("Affine fit RMSE = %.4f world-coordinate units", rmse)

    return transform


def apply_affine_to_pixels(
    pixel_coords: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply an affine transform to an array of pixel coordinates.

    Parameters
    ----------
    pixel_coords:
        Shape ``(N, 2)`` array of ``[x, y]`` pixel coordinates.
    transform:
        Shape ``(3, 3)`` affine transform matrix from
        :func:`estimate_affine_transform`.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 2)`` array of world coordinates ``[world_x, world_y]``.
    """
    n = pixel_coords.shape[0]
    homogeneous = np.hstack([pixel_coords, np.ones((n, 1), dtype=np.float64)])
    world = (transform @ homogeneous.T).T
    return world[:, :2]


# ---------------------------------------------------------------------------
# GeoTIFF georeferencing
# ---------------------------------------------------------------------------


def georeference_orthophoto(
    ortho_path: str | Path,
    control_points: Sequence[FusionControlPoint],
    output_path: str | Path,
    crs: str = "EPSG:4326",
) -> Path:
    """Georeference an SfM orthophoto and write a GeoTIFF.

    Uses *rasterio* to attach a georeferenced transform to the orthophoto
    based on the supplied control points.

    Parameters
    ----------
    ortho_path:
        Path to the input orthophoto (any format readable by rasterio).
    control_points:
        At least 3 :class:`FusionControlPoint` instances whose
        ``world_x`` / ``world_y`` coordinates are in *crs*.
    output_path:
        Destination GeoTIFF path.
    crs:
        Target coordinate reference system (e.g. ``"EPSG:4326"`` for
        WGS-84 geographic, ``"EPSG:32633"`` for UTM zone 33 N).

    Returns
    -------
    Path
        Path to the written GeoTIFF.

    Raises
    ------
    ImportError
        If *rasterio* is not installed.
    ValueError
        If fewer than 3 control points are provided.
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio is required for georeferencing.  "
            "Install it with: pip install rasterio"
        )

    if len(control_points) < 3:
        raise ValueError(
            f"At least 3 control points required; got {len(control_points)}."
        )

    ortho_path = Path(ortho_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(ortho_path)) as src:
        profile = src.profile.copy()
        data = src.read()

    # Build rasterio GCPs from our control points.
    rasterio_gcps = [
        RasterioGCP(
            col=cp.sfm_pixel_x,
            row=cp.sfm_pixel_y,
            x=cp.world_x,
            y=cp.world_y,
            id=cp.label or str(i),
        )
        for i, cp in enumerate(control_points)
    ]

    gcp_transform, _ = rasterio_from_gcps(rasterio_gcps, data.shape[-2], data.shape[-1])
    # rasterio_from_gcps returns (Affine transform, residuals); residuals are unused here.

    profile.update(
        driver="GTiff",
        crs=crs,
        transform=gcp_transform,
    )

    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(data)

    logger.info("Wrote georeferenced orthophoto to %s (CRS: %s)", output_path, crs)
    return output_path


# ---------------------------------------------------------------------------
# Satellite image loading
# ---------------------------------------------------------------------------


def load_satellite_image(
    satellite_path: str | Path,
) -> tuple["rasterio.DatasetReader", np.ndarray]:
    """Open a satellite image and return the dataset and pixel array.

    Parameters
    ----------
    satellite_path:
        Path to the satellite GeoTIFF (or any rasterio-readable format).

    Returns
    -------
    tuple[rasterio.DatasetReader, numpy.ndarray]
        The open rasterio dataset and the pixel data array
        ``(bands, height, width)``.

    Raises
    ------
    ImportError
        If *rasterio* is not installed.
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required.  pip install rasterio")

    satellite_path = Path(satellite_path)
    dataset = rasterio.open(str(satellite_path))
    data = dataset.read()
    logger.info(
        "Loaded satellite image %s  shape=%s  CRS=%s",
        satellite_path.name,
        data.shape,
        dataset.crs,
    )
    return dataset, data


def composite_ortho_on_satellite(
    satellite_path: str | Path,
    ortho_georef_path: str | Path,
    output_path: str | Path,
    blend_alpha: float = 0.6,
) -> Path:
    """Composite a georeferenced orthophoto on top of a satellite image.

    The orthophoto is resampled to the satellite image's grid and
    blended using a weighted average.

    Parameters
    ----------
    satellite_path:
        Path to the reference satellite GeoTIFF.
    ortho_georef_path:
        Path to the georeferenced orthophoto GeoTIFF (output of
        :func:`georeference_orthophoto`).
    output_path:
        Destination composite GeoTIFF.
    blend_alpha:
        Weight of the orthophoto layer (0 = satellite only,
        1 = orthophoto only).

    Returns
    -------
    Path
        Path to the written composite GeoTIFF.
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required.  pip install rasterio")

    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    satellite_path = Path(satellite_path)
    ortho_georef_path = Path(ortho_georef_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(satellite_path)) as sat_ds:
        sat_data = sat_ds.read().astype(np.float32)
        sat_profile = sat_ds.profile.copy()
        sat_transform = sat_ds.transform
        sat_crs = sat_ds.crs
        sat_shape = sat_data.shape[-2:]

    with rasterio.open(str(ortho_georef_path)) as ortho_ds:
        ortho_reprojected = np.zeros_like(sat_data)
        reproject(
            source=ortho_ds.read().astype(np.float32),
            destination=ortho_reprojected,
            src_transform=ortho_ds.transform,
            src_crs=ortho_ds.crs,
            dst_transform=sat_transform,
            dst_crs=sat_crs,
            dst_shape=sat_shape,
            resampling=Resampling.bilinear,
        )

    # Where the ortho has data (non-zero), blend; elsewhere keep satellite.
    mask = ortho_reprojected.max(axis=0) > 0
    composite = sat_data.copy()
    composite[:, mask] = (
        (1.0 - blend_alpha) * sat_data[:, mask]
        + blend_alpha * ortho_reprojected[:, mask]
    )

    sat_profile.update(dtype="uint8")
    composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)

    with rasterio.open(str(output_path), "w", **sat_profile) as dst:
        dst.write(composite_uint8)

    logger.info("Wrote composite image to %s", output_path)
    return output_path
