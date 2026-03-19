"""Tests for the field_site_sfm CLI entry-points."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from field_site_sfm.cli.extract_frames import main as extract_frames_main
from field_site_sfm.cli.run_pipeline import main as run_pipeline_main


# ---------------------------------------------------------------------------
# extract_frames CLI
# ---------------------------------------------------------------------------


class TestExtractFramesCli:
    def test_missing_video_shows_error(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            extract_frames_main,
            ["--video", str(tmp_path / "nofile.mp4"), "--output", str(tmp_path / "out")],
        )
        assert result.exit_code != 0

    def test_successful_extraction(self, tmp_path: Path) -> None:
        fake_mp4 = tmp_path / "fake.mp4"
        fake_mp4.write_bytes(b"fake")

        import numpy as np

        def _fake_extract(*args, **kwargs):
            return [tmp_path / "frame_000001.jpg", tmp_path / "frame_000002.jpg"]

        runner = CliRunner()
        with patch("field_site_sfm.cli.extract_frames.extract_frames", side_effect=_fake_extract):
            result = runner.invoke(
                extract_frames_main,
                ["--video", str(fake_mp4), "--output", str(tmp_path / "out")],
            )
        assert result.exit_code == 0
        assert "2" in result.output

    def test_image_list_flag(self, tmp_path: Path) -> None:
        fake_mp4 = tmp_path / "fake.mp4"
        fake_mp4.write_bytes(b"fake")

        def _fake_extract(*args, **kwargs):
            return [tmp_path / "frame_000001.jpg"]

        runner = CliRunner()
        with (
            patch("field_site_sfm.cli.extract_frames.extract_frames", side_effect=_fake_extract),
            patch("field_site_sfm.cli.extract_frames.write_image_list") as mock_wil,
        ):
            result = runner.invoke(
                extract_frames_main,
                [
                    "--video", str(fake_mp4),
                    "--output", str(tmp_path / "out"),
                    "--image-list",
                ],
            )
        assert result.exit_code == 0
        mock_wil.assert_called_once()


# ---------------------------------------------------------------------------
# run_pipeline CLI
# ---------------------------------------------------------------------------


class TestRunPipelineCli:
    def test_missing_video_shows_error(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            run_pipeline_main,
            [
                "--video", str(tmp_path / "nofile.mp4"),
                "--output", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0

    def test_pipeline_invoked(self, tmp_path: Path) -> None:
        fake_mp4 = tmp_path / "fake.mp4"
        fake_mp4.write_bytes(b"fake")

        runner = CliRunner()
        with patch("field_site_sfm.cli.run_pipeline.Pipeline") as MockPipeline:
            instance = MagicMock()
            MockPipeline.return_value = instance
            result = runner.invoke(
                run_pipeline_main,
                ["--video", str(fake_mp4), "--output", str(tmp_path / "out")],
            )

        assert result.exit_code == 0
        instance.run.assert_called_once()

    def test_no_dense_flag(self, tmp_path: Path) -> None:
        fake_mp4 = tmp_path / "fake.mp4"
        fake_mp4.write_bytes(b"fake")

        runner = CliRunner()
        with patch("field_site_sfm.cli.run_pipeline.Pipeline") as MockPipeline:
            instance = MagicMock()
            MockPipeline.return_value = instance
            runner.invoke(
                run_pipeline_main,
                [
                    "--video", str(fake_mp4),
                    "--output", str(tmp_path / "out"),
                    "--no-dense",
                ],
            )
        cfg_passed = MockPipeline.call_args[0][0]
        assert cfg_passed.run_dense is False
