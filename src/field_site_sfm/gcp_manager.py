"""Ground Control Point (GCP) management for the field-site SfM pipeline.

GCPs are physical markers placed in the field (bright-orange discs at
the four corners and center of the 20 × 20 m plot) whose 3-D world
coordinates are known — either from GPS measurements, from a total
station, or by digitising the marker centres on a referenced satellite
image.

This module handles:

* Loading GCPs from a simple CSV file.
* Writing the ``gcp_list.txt`` file expected by OpenSfM.
* A lightweight helper that records which images each GCP appears in
  (required for triangulation).

OpenSfM ``gcp_list.txt`` format (one observation per line)::

    <x_world> <y_world> <z_world> <x_pixel> <y_pixel> <image_name>

The header line must be ``WGS84`` or a custom CRS EPSG code.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GCPObservation:
    """A single observation of a GCP in one image frame.

    Attributes
    ----------
    image_name:
        Filename of the image (e.g. ``"frame_000042.jpg"``).
    pixel_x:
        Column coordinate of the GCP centre in the image (pixels).
    pixel_y:
        Row coordinate of the GCP centre in the image (pixels).
    """

    image_name: str
    pixel_x: float
    pixel_y: float


@dataclass
class GCPPoint:
    """A single Ground Control Point.

    Attributes
    ----------
    gcp_id:
        Human-readable identifier, e.g. ``"GCP_NW"`` or ``"1"``.
    world_x:
        Easting / longitude in the reference CRS (metres or degrees).
    world_y:
        Northing / latitude in the reference CRS (metres or degrees).
    world_z:
        Elevation in the reference CRS (metres above datum).
    observations:
        List of :class:`GCPObservation` instances – one per image in
        which this GCP was manually marked.
    """

    gcp_id: str
    world_x: float
    world_y: float
    world_z: float
    observations: list[GCPObservation] = field(default_factory=list)

    def add_observation(self, image_name: str, pixel_x: float, pixel_y: float) -> None:
        """Append a new pixel-space observation."""
        self.observations.append(GCPObservation(image_name, pixel_x, pixel_y))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_REQUIRED_WORLD_COLUMNS = {"gcp_id", "world_x", "world_y", "world_z"}


def load_gcps_from_csv(csv_path: str | Path) -> list[GCPPoint]:
    """Load GCP world coordinates from a CSV file.

    Expected columns (in any order, case-insensitive):

    * ``gcp_id``  – unique identifier for each marker
    * ``world_x`` – easting or longitude
    * ``world_y`` – northing or latitude
    * ``world_z`` – elevation

    Observation columns (``image_name``, ``pixel_x``, ``pixel_y``) are
    optional: if present, observations are loaded too; otherwise the
    GCPs are returned with empty :attr:`GCPPoint.observations` lists.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.

    Returns
    -------
    list[GCPPoint]
        One :class:`GCPPoint` per row.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    csv_path = Path(csv_path)
    points: list[GCPPoint] = []

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty or has no header: {csv_path}")

        columns = {c.strip().lower() for c in reader.fieldnames}
        missing = _REQUIRED_WORLD_COLUMNS - columns
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found: {columns}"
            )

        has_obs = {"image_name", "pixel_x", "pixel_y"}.issubset(columns)

        for row in reader:
            row_lower = {k.strip().lower(): v.strip() for k, v in row.items()}
            gcp = GCPPoint(
                gcp_id=row_lower["gcp_id"],
                world_x=float(row_lower["world_x"]),
                world_y=float(row_lower["world_y"]),
                world_z=float(row_lower["world_z"]),
            )
            if has_obs and row_lower.get("image_name"):
                gcp.add_observation(
                    image_name=row_lower["image_name"],
                    pixel_x=float(row_lower["pixel_x"]),
                    pixel_y=float(row_lower["pixel_y"]),
                )
            points.append(gcp)

    logger.info("Loaded %d GCPs from %s", len(points), csv_path)
    return points


def merge_gcp_observations(
    gcps: list[GCPPoint],
    observation_csv: str | Path,
) -> list[GCPPoint]:
    """Merge per-image pixel observations from a separate CSV into *gcps*.

    The observation CSV must have columns:
    ``gcp_id``, ``image_name``, ``pixel_x``, ``pixel_y``.

    Parameters
    ----------
    gcps:
        List of :class:`GCPPoint` objects (already holding world coordinates).
    observation_csv:
        Path to the CSV file containing pixel observations.

    Returns
    -------
    list[GCPPoint]
        The same list, mutated in place.
    """
    obs_path = Path(observation_csv)
    gcp_map = {g.gcp_id: g for g in gcps}

    with obs_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gid = row["gcp_id"].strip()
            if gid not in gcp_map:
                logger.warning("Observation references unknown GCP id '%s'; skipping", gid)
                continue
            gcp_map[gid].add_observation(
                image_name=row["image_name"].strip(),
                pixel_x=float(row["pixel_x"]),
                pixel_y=float(row["pixel_y"]),
            )

    logger.info("Merged observations from %s", obs_path)
    return gcps


def write_opensfm_gcp_list(
    gcps: Sequence[GCPPoint],
    output_path: str | Path,
    crs: str = "WGS84",
) -> Path:
    """Write a ``gcp_list.txt`` file in the format expected by OpenSfM.

    Format::

        <CRS>
        <x_world> <y_world> <z_world> <x_pixel> <y_pixel> <image_name>
        ...

    Only GCPs that have at least one observation are written, as
    un-observed GCPs cannot be used in bundle adjustment.

    Parameters
    ----------
    gcps:
        Sequence of :class:`GCPPoint` objects.
    output_path:
        Destination file path.
    crs:
        Coordinate Reference System string written as the header line.
        Use ``"WGS84"`` for geographic (lon/lat) coordinates or an
        EPSG code such as ``"EPSG:32633"`` for a projected CRS.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = Path(output_path)
    total_obs = 0

    with output_path.open("w") as fh:
        fh.write(f"{crs}\n")
        for gcp in gcps:
            if not gcp.observations:
                logger.warning(
                    "GCP '%s' has no observations; skipping in gcp_list.txt",
                    gcp.gcp_id,
                )
                continue
            for obs in gcp.observations:
                fh.write(
                    f"{gcp.world_x} {gcp.world_y} {gcp.world_z} "
                    f"{obs.pixel_x} {obs.pixel_y} {obs.image_name}\n"
                )
                total_obs += 1

    logger.info(
        "Wrote %d GCP observations (%d points) to %s",
        total_obs,
        sum(1 for g in gcps if g.observations),
        output_path,
    )
    return output_path


def load_opensfm_gcp_list(gcp_list_path: str | Path) -> tuple[str, list[GCPPoint]]:
    """Parse an existing OpenSfM ``gcp_list.txt`` file.

    Parameters
    ----------
    gcp_list_path:
        Path to the file.

    Returns
    -------
    tuple[str, list[GCPPoint]]
        ``(crs_string, list_of_gcps)`` where each :class:`GCPPoint`
        aggregates all observations found for its world coordinate.
    """
    path = Path(gcp_list_path)
    lines = path.read_text().splitlines()

    if not lines:
        raise ValueError(f"Empty GCP list file: {path}")

    crs = lines[0].strip()
    # Key: (world_x, world_y, world_z) → GCPPoint
    point_map: dict[tuple[float, float, float], GCPPoint] = {}
    counter = 0

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            logger.warning("Skipping malformed GCP line: %r", line)
            continue
        wx, wy, wz = float(parts[0]), float(parts[1]), float(parts[2])
        px, py = float(parts[3]), float(parts[4])
        image_name = parts[5]
        key = (wx, wy, wz)
        if key not in point_map:
            counter += 1
            point_map[key] = GCPPoint(
                gcp_id=str(counter),
                world_x=wx,
                world_y=wy,
                world_z=wz,
            )
        point_map[key].add_observation(image_name, px, py)

    gcps = list(point_map.values())
    logger.info("Loaded %d GCPs from %s", len(gcps), path)
    return crs, gcps
