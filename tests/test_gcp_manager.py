"""Tests for field_site_sfm.gcp_manager."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from field_site_sfm.gcp_manager import (
    GCPObservation,
    GCPPoint,
    load_gcps_from_csv,
    load_opensfm_gcp_list,
    merge_gcp_observations,
    write_opensfm_gcp_list,
)


# ---------------------------------------------------------------------------
# GCPPoint
# ---------------------------------------------------------------------------


class TestGCPPoint:
    def test_add_observation(self) -> None:
        gcp = GCPPoint("G1", 1.0, 2.0, 3.0)
        gcp.add_observation("frame_000001.jpg", 100.0, 200.0)
        assert len(gcp.observations) == 1
        obs = gcp.observations[0]
        assert obs.image_name == "frame_000001.jpg"
        assert obs.pixel_x == 100.0
        assert obs.pixel_y == 200.0

    def test_default_empty_observations(self) -> None:
        gcp = GCPPoint("G1", 0.0, 0.0, 0.0)
        assert gcp.observations == []


# ---------------------------------------------------------------------------
# load_gcps_from_csv
# ---------------------------------------------------------------------------


class TestLoadGcpsFromCsv:
    def test_loads_world_coords(self, sample_gcp_csv: Path) -> None:
        gcps = load_gcps_from_csv(sample_gcp_csv)
        assert len(gcps) == 5
        ids = [g.gcp_id for g in gcps]
        assert "GCP_NW" in ids
        assert "GCP_C" in ids

    def test_correct_coordinates(self, sample_gcp_csv: Path) -> None:
        gcps = load_gcps_from_csv(sample_gcp_csv)
        nw = next(g for g in gcps if g.gcp_id == "GCP_NW")
        assert nw.world_x == pytest.approx(-73.9857)
        assert nw.world_y == pytest.approx(40.7580)
        assert nw.world_z == pytest.approx(15.0)

    def test_loads_observations_when_present(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "gcps_with_obs.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["gcp_id", "world_x", "world_y", "world_z",
                             "image_name", "pixel_x", "pixel_y"])
            writer.writerow(["G1", "10.0", "20.0", "5.0",
                             "frame_000001.jpg", "100", "200"])
            writer.writerow(["G1", "10.0", "20.0", "5.0",
                             "frame_000002.jpg", "110", "205"])

        gcps = load_gcps_from_csv(csv_path)
        # Both rows have same gcp_id → two GCPPoint objects (one per CSV row).
        assert sum(len(g.observations) for g in gcps) == 2

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        bad_csv = tmp_path / "bad.csv"
        with bad_csv.open("w", newline="") as fh:
            csv.writer(fh).writerows([["name", "x"], ["G1", "1.0"]])

        with pytest.raises(ValueError, match="missing required columns"):
            load_gcps_from_csv(bad_csv)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_gcps_from_csv("does_not_exist.csv")

    def test_empty_csv_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises(ValueError):
            load_gcps_from_csv(empty)


# ---------------------------------------------------------------------------
# merge_gcp_observations
# ---------------------------------------------------------------------------


class TestMergeGcpObservations:
    def test_merges_correctly(
        self,
        sample_gcp_csv: Path,
        sample_gcp_observations_csv: Path,
    ) -> None:
        gcps = load_gcps_from_csv(sample_gcp_csv)
        gcps = merge_gcp_observations(gcps, sample_gcp_observations_csv)

        nw = next(g for g in gcps if g.gcp_id == "GCP_NW")
        assert len(nw.observations) == 2

    def test_unknown_gcp_id_skipped(
        self, sample_gcp_csv: Path, tmp_path: Path
    ) -> None:
        bad_obs = tmp_path / "bad_obs.csv"
        with bad_obs.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["gcp_id", "image_name", "pixel_x", "pixel_y"])
            writer.writerow(["NONEXISTENT", "frame.jpg", "0", "0"])

        gcps = load_gcps_from_csv(sample_gcp_csv)
        # Should not raise, just log a warning.
        gcps = merge_gcp_observations(gcps, bad_obs)
        total_obs = sum(len(g.observations) for g in gcps)
        assert total_obs == 0


# ---------------------------------------------------------------------------
# write_opensfm_gcp_list
# ---------------------------------------------------------------------------


class TestWriteOpensfmGcpList:
    def _make_gcp_with_obs(self, gcp_id: str, wx: float, wy: float, wz: float) -> GCPPoint:
        gcp = GCPPoint(gcp_id, wx, wy, wz)
        gcp.add_observation("frame_000001.jpg", 100.0, 200.0)
        gcp.add_observation("frame_000002.jpg", 110.0, 205.0)
        return gcp

    def test_header_written(self, tmp_path: Path) -> None:
        gcp = self._make_gcp_with_obs("G1", 10.0, 20.0, 5.0)
        out = write_opensfm_gcp_list([gcp], tmp_path / "gcp_list.txt", crs="WGS84")
        lines = out.read_text().splitlines()
        assert lines[0] == "WGS84"

    def test_correct_format(self, tmp_path: Path) -> None:
        gcp = self._make_gcp_with_obs("G1", 10.0, 20.0, 5.0)
        out = write_opensfm_gcp_list([gcp], tmp_path / "gcp_list.txt")
        lines = out.read_text().splitlines()
        # 1 header + 2 observations
        assert len(lines) == 3
        parts = lines[1].split()
        assert parts[0] == "10.0"   # world_x
        assert parts[1] == "20.0"   # world_y
        assert parts[2] == "5.0"    # world_z
        assert parts[3] == "100.0"  # pixel_x
        assert parts[4] == "200.0"  # pixel_y
        assert parts[5] == "frame_000001.jpg"

    def test_gcp_without_observations_skipped(self, tmp_path: Path) -> None:
        gcp_no_obs = GCPPoint("G1", 10.0, 20.0, 5.0)  # no observations
        gcp_with_obs = self._make_gcp_with_obs("G2", 11.0, 21.0, 5.0)
        out = write_opensfm_gcp_list(
            [gcp_no_obs, gcp_with_obs], tmp_path / "gcp_list.txt"
        )
        lines = out.read_text().splitlines()
        # Only G2's observations should appear.
        for line in lines[1:]:
            assert "11.0" in line

    def test_custom_crs(self, tmp_path: Path) -> None:
        gcp = self._make_gcp_with_obs("G1", 10.0, 20.0, 5.0)
        out = write_opensfm_gcp_list([gcp], tmp_path / "gcp_list.txt", crs="EPSG:32633")
        assert out.read_text().splitlines()[0] == "EPSG:32633"

    def test_returns_path_object(self, tmp_path: Path) -> None:
        out = write_opensfm_gcp_list([], tmp_path / "gcp_list.txt")
        assert isinstance(out, Path)


# ---------------------------------------------------------------------------
# load_opensfm_gcp_list
# ---------------------------------------------------------------------------


class TestLoadOpensfmGcpList:
    def test_parses_crs(self, sample_opensfm_gcp_list: Path) -> None:
        crs, _ = load_opensfm_gcp_list(sample_opensfm_gcp_list)
        assert crs == "WGS84"

    def test_parses_gcps(self, sample_opensfm_gcp_list: Path) -> None:
        _, gcps = load_opensfm_gcp_list(sample_opensfm_gcp_list)
        # 3 distinct world coordinate triplets in the fixture.
        assert len(gcps) == 3

    def test_aggregates_observations(self, sample_opensfm_gcp_list: Path) -> None:
        _, gcps = load_opensfm_gcp_list(sample_opensfm_gcp_list)
        # First GCP (-73.9857, 40.7580, 15.0) has 2 observations.
        first = gcps[0]
        assert len(first.observations) == 2

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        with pytest.raises(ValueError, match="Empty"):
            load_opensfm_gcp_list(empty)

    def test_roundtrip(self, tmp_path: Path) -> None:
        """write then load should reproduce the same data."""
        gcp = GCPPoint("G1", 1.234, 5.678, 9.0)
        gcp.add_observation("img_001.jpg", 111.0, 222.0)
        gcp.add_observation("img_002.jpg", 333.0, 444.0)

        list_path = tmp_path / "gcp_list.txt"
        write_opensfm_gcp_list([gcp], list_path, crs="EPSG:4326")
        crs, loaded = load_opensfm_gcp_list(list_path)

        assert crs == "EPSG:4326"
        assert len(loaded) == 1
        assert loaded[0].world_x == pytest.approx(1.234)
        assert loaded[0].world_y == pytest.approx(5.678)
        assert len(loaded[0].observations) == 2
