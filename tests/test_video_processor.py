"""Tests for field_site_sfm.video_processor."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from field_site_sfm.video_processor import (
    DEFAULT_EXTRACT_FPS,
    copy_frames_to_opensfm_project,
    extract_frames,
    write_image_list,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_video_cap(
    fps: float = 30.0,
    total_frames: int = 90,
    frame_height: int = 100,
    frame_width: int = 200,
    read_returns_false_after: int | None = None,
) -> MagicMock:
    """Return a mock ``cv2.VideoCapture`` that yields dummy BGR frames."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        # cv2.CAP_PROP_FPS = 5
        5: fps,
        # cv2.CAP_PROP_FRAME_COUNT = 7
        7: float(total_frames),
    }.get(prop, 0.0)

    call_count = [0]

    def _read() -> tuple[bool, np.ndarray]:
        if read_returns_false_after is not None and call_count[0] >= read_returns_false_after:
            return False, None
        call_count[0] += 1
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        return True, frame

    cap.read.side_effect = _read
    return cap


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------


class TestExtractFrames:
    def test_missing_video_raises(self, tmp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_frames("nonexistent.mp4", tmp_dir)

    def test_extract_correct_number_of_frames(
        self, tmp_path: Path
    ) -> None:
        """Extract 1 fps from a 90-frame / 30-fps video (3.0 s).

        step_frames = 30 / 1 = 30.  The loop condition is inclusive
        (frame_index / video_fps <= end_sec), so frames are sampled at
        frame indices 0, 30, 60, 90 → timestamps 0.0, 1.0, 2.0, 3.0 s.
        That gives 4 frames.
        """
        fake_mp4 = tmp_path / "fake.mp4"
        fake_mp4.write_bytes(b"fake")

        with (
            patch("field_site_sfm.video_processor.cv2.VideoCapture") as mock_cap_cls,
            patch("field_site_sfm.video_processor.cv2.imwrite", return_value=True),
        ):
            cap_instance = _make_fake_video_cap(fps=30.0, total_frames=90)
            mock_cap_cls.return_value = cap_instance

            frames = extract_frames(
                fake_mp4,
                tmp_path / "frames",
                fps=1.0,
            )

        # Frames at t = 0, 1, 2, 3 s (inclusive boundary) → 4 frames.
        assert len(frames) == 4

    def test_returns_sorted_paths(self, tmp_path: Path) -> None:
        fake_mp4 = tmp_path / "fake.mp4"
        fake_mp4.write_bytes(b"fake")

        with (
            patch("field_site_sfm.video_processor.cv2.VideoCapture") as mock_cap_cls,
            patch("field_site_sfm.video_processor.cv2.imwrite", return_value=True),
        ):
            cap_instance = _make_fake_video_cap(fps=30.0, total_frames=90)
            mock_cap_cls.return_value = cap_instance

            frames = extract_frames(fake_mp4, tmp_path / "out", fps=1.0)

        names = [f.name for f in frames]
        assert names == sorted(names)

    def test_cannot_open_video_raises(self, tmp_path: Path) -> None:
        fake_mp4 = tmp_path / "fake.mp4"
        fake_mp4.write_bytes(b"fake")

        with patch("field_site_sfm.video_processor.cv2.VideoCapture") as mock_cap_cls:
            cap_instance = MagicMock()
            cap_instance.isOpened.return_value = False
            mock_cap_cls.return_value = cap_instance

            with pytest.raises(RuntimeError, match="could not open"):
                extract_frames(fake_mp4, tmp_path / "out")

    def test_invalid_fps_raises(self, tmp_path: Path) -> None:
        fake_mp4 = tmp_path / "fake.mp4"
        fake_mp4.write_bytes(b"fake")

        with patch("field_site_sfm.video_processor.cv2.VideoCapture") as mock_cap_cls:
            cap_instance = _make_fake_video_cap(fps=0.0, total_frames=90)
            mock_cap_cls.return_value = cap_instance

            with pytest.raises(RuntimeError, match="FPS"):
                extract_frames(fake_mp4, tmp_path / "out")

    def test_output_dir_created(self, tmp_path: Path) -> None:
        fake_mp4 = tmp_path / "fake.mp4"
        fake_mp4.write_bytes(b"fake")
        out_dir = tmp_path / "deeply" / "nested" / "frames"

        with (
            patch("field_site_sfm.video_processor.cv2.VideoCapture") as mock_cap_cls,
            patch("field_site_sfm.video_processor.cv2.imwrite", return_value=True),
        ):
            cap_instance = _make_fake_video_cap()
            mock_cap_cls.return_value = cap_instance
            extract_frames(fake_mp4, out_dir, fps=1.0)

        assert out_dir.exists()


# ---------------------------------------------------------------------------
# write_image_list
# ---------------------------------------------------------------------------


class TestWriteImageList:
    def test_writes_filenames(self, tmp_dir: Path) -> None:
        frames = [
            tmp_dir / "frame_000001.jpg",
            tmp_dir / "frame_000002.jpg",
        ]
        list_path = tmp_dir / "image_list.txt"
        result = write_image_list(frames, list_path)

        assert result == list_path
        lines = list_path.read_text().splitlines()
        assert lines == ["frame_000001.jpg", "frame_000002.jpg"]

    def test_empty_list(self, tmp_dir: Path) -> None:
        list_path = tmp_dir / "empty.txt"
        write_image_list([], list_path)
        assert list_path.read_text() == ""


# ---------------------------------------------------------------------------
# copy_frames_to_opensfm_project
# ---------------------------------------------------------------------------


class TestCopyFramesToOpensfmProject:
    def test_copies_files(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        frames = []
        for i in range(3):
            p = src_dir / f"frame_{i:06d}.jpg"
            p.write_bytes(b"jpeg-data")
            frames.append(p)

        project_dir = tmp_path / "project"
        copy_frames_to_opensfm_project(frames, project_dir)

        images_dir = project_dir / "images"
        assert images_dir.is_dir()
        for p in frames:
            assert (images_dir / p.name).exists()
