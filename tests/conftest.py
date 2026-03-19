"""Shared fixtures for the field_site_sfm test suite."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory (alias for pytest's tmp_path)."""
    return tmp_path


@pytest.fixture
def sample_gcp_csv(tmp_path: Path) -> Path:
    """Write a minimal GCP world-coordinate CSV and return its path."""
    csv_path = tmp_path / "gcps.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gcp_id", "world_x", "world_y", "world_z"])
        writer.writerow(["GCP_NW", "-73.9857", "40.7580", "15.0"])
        writer.writerow(["GCP_NE", "-73.9837", "40.7580", "15.1"])
        writer.writerow(["GCP_SE", "-73.9837", "40.7560", "15.0"])
        writer.writerow(["GCP_SW", "-73.9857", "40.7560", "14.9"])
        writer.writerow(["GCP_C",  "-73.9847", "40.7570", "15.0"])
    return csv_path


@pytest.fixture
def sample_gcp_observations_csv(tmp_path: Path) -> Path:
    """Write a minimal GCP observation CSV and return its path."""
    csv_path = tmp_path / "gcp_observations.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gcp_id", "image_name", "pixel_x", "pixel_y"])
        writer.writerow(["GCP_NW", "frame_000001.jpg", "120", "340"])
        writer.writerow(["GCP_NW", "frame_000002.jpg", "125", "335"])
        writer.writerow(["GCP_NE", "frame_000010.jpg", "900", "340"])
        writer.writerow(["GCP_C",  "frame_000020.jpg", "512", "512"])
    return csv_path


@pytest.fixture
def sample_opensfm_gcp_list(tmp_path: Path) -> Path:
    """Write a minimal OpenSfM gcp_list.txt and return its path."""
    gcp_path = tmp_path / "gcp_list.txt"
    lines = [
        "WGS84",
        "-73.9857 40.7580 15.0 120 340 frame_000001.jpg",
        "-73.9857 40.7580 15.0 125 335 frame_000002.jpg",
        "-73.9837 40.7580 15.1 900 340 frame_000010.jpg",
        "-73.9847 40.7570 15.0 512 512 frame_000020.jpg",
    ]
    gcp_path.write_text("\n".join(lines) + "\n")
    return gcp_path
