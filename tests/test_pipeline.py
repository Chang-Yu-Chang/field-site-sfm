"""Tests for field_site_sfm.pipeline (PipelineConfig and Pipeline)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from field_site_sfm.pipeline import Pipeline, PipelineConfig


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_paths_become_path_objects(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        cfg = PipelineConfig(video_path=str(video), output_dir=str(tmp_path / "out"))
        assert isinstance(cfg.video_path, Path)
        assert isinstance(cfg.output_dir, Path)

    def test_optional_paths_none(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        cfg = PipelineConfig(video_path=video, output_dir=tmp_path / "out")
        assert cfg.gcp_csv is None
        assert cfg.satellite_path is None

    def test_optional_paths_converted(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        gcps = tmp_path / "gcps.csv"
        gcps.write_text("gcp_id,world_x,world_y,world_z\n")
        sat = tmp_path / "sat.tif"
        sat.write_bytes(b"tif")

        cfg = PipelineConfig(
            video_path=video,
            output_dir=tmp_path / "out",
            gcp_csv=str(gcps),
            satellite_path=str(sat),
        )
        assert isinstance(cfg.gcp_csv, Path)
        assert isinstance(cfg.satellite_path, Path)

    def test_project_dir_is_subdir_of_output(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        cfg = PipelineConfig(video_path=video, output_dir=tmp_path / "out")
        pipeline = Pipeline(cfg)
        assert pipeline.project_dir.parent == cfg.output_dir


# ---------------------------------------------------------------------------
# Pipeline._write_opensfm_config
# ---------------------------------------------------------------------------


class TestPipelineWriteOpensfmConfig:
    def test_config_file_created(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        cfg = PipelineConfig(video_path=video, output_dir=tmp_path / "out")
        pipeline = Pipeline(cfg)
        pipeline.project_dir.mkdir(parents=True, exist_ok=True)
        pipeline._write_opensfm_config()

        config_path = pipeline.project_dir / "config.yaml"
        assert config_path.exists()

    def test_processes_overridden(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        cfg = PipelineConfig(
            video_path=video,
            output_dir=tmp_path / "out",
            processes=8,
        )
        pipeline = Pipeline(cfg)
        pipeline.project_dir.mkdir(parents=True, exist_ok=True)
        pipeline._write_opensfm_config()

        with (pipeline.project_dir / "config.yaml").open() as fh:
            loaded = yaml.safe_load(fh)

        assert loaded["processes"] == 8

    def test_extra_config_applied(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        cfg = PipelineConfig(
            video_path=video,
            output_dir=tmp_path / "out",
            extra_opensfm_config={"feature_type": "SIFT", "custom_key": "custom_val"},
        )
        pipeline = Pipeline(cfg)
        pipeline.project_dir.mkdir(parents=True, exist_ok=True)
        pipeline._write_opensfm_config()

        with (pipeline.project_dir / "config.yaml").open() as fh:
            loaded = yaml.safe_load(fh)

        assert loaded["feature_type"] == "SIFT"
        assert loaded["custom_key"] == "custom_val"


# ---------------------------------------------------------------------------
# Pipeline._run_opensfm_step
# ---------------------------------------------------------------------------


class TestPipelineRunOpensfmStep:
    def _make_pipeline(self, tmp_path: Path) -> Pipeline:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        cfg = PipelineConfig(video_path=video, output_dir=tmp_path / "out")
        return Pipeline(cfg)

    def test_calls_subprocess(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline(tmp_path)
        with patch("field_site_sfm.pipeline.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok")
            pipeline._run_opensfm_step("detect_features")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "opensfm" in cmd
            assert "detect_features" in cmd

    def test_raises_on_nonzero_exit(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline(tmp_path)
        with patch("field_site_sfm.pipeline.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="error output")
            with pytest.raises(RuntimeError, match="detect_features"):
                pipeline._run_opensfm_step("detect_features")
