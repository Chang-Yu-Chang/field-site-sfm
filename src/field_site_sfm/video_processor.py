"""Extract individual frames from an Insta360 MP4 video file.

The Insta360 records equirectangular (360°) video.  This module uses
*OpenCV* to pull JPEG frames at a configurable rate so that the
resulting image set has sufficient overlap for SfM reconstruction.

For a "lawnmower" walk at ~0.5 m/s and a 3-metre pole height the
recommended capture rate is **2 fps**, which gives roughly 25 cm of
movement between consecutive frames and ensures >70 % overlap between
adjacent lawnmower lanes spaced 2 m apart.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterator

import cv2

logger = logging.getLogger(__name__)

# Default frames-per-second to extract.  At 0.5 m/s walk speed this
# gives one frame every ~0.25 m of camera travel.
DEFAULT_EXTRACT_FPS: float = 2.0

# JPEG quality used when writing extracted frames (0–100).
JPEG_QUALITY: int = 95


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = DEFAULT_EXTRACT_FPS,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    prefix: str = "frame",
) -> list[Path]:
    """Extract frames from *video_path* at *fps* and write them to *output_dir*.

    Parameters
    ----------
    video_path:
        Path to the source Insta360 MP4 file.
    output_dir:
        Directory where JPEG frames are written.  Created if it does
        not exist.
    fps:
        Number of frames to extract per second of video.  Defaults to
        :data:`DEFAULT_EXTRACT_FPS`.
    start_sec:
        Timestamp (seconds) to start extraction.  Defaults to the
        beginning of the video.
    end_sec:
        Timestamp (seconds) to stop extraction.  Defaults to the end
        of the video.  Pass ``None`` to extract until the end.
    prefix:
        Filename prefix for extracted frames (e.g. ``"frame"`` →
        ``frame_000001.jpg``).

    Returns
    -------
    list[Path]
        Sorted list of paths to the written JPEG files.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    video_fps: float = cap.get(cv2.CAP_PROP_FPS)
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec: float = total_frames / video_fps if video_fps > 0 else 0.0

    if video_fps <= 0:
        raise RuntimeError(
            f"Cannot determine FPS of video (got {video_fps}).  "
            "Is the file a valid MP4?"
        )

    end_sec_actual = min(end_sec, duration_sec) if end_sec is not None else duration_sec
    step_frames: float = video_fps / fps  # original frames between each extracted frame

    logger.info(
        "Video: %.1f fps, %.1f s duration (%.0f frames)",
        video_fps,
        duration_sec,
        total_frames,
    )
    logger.info(
        "Extracting %.1f fps from %.1f s to %.1f s",
        fps,
        start_sec,
        end_sec_actual,
    )

    written: list[Path] = []
    frame_index: float = start_sec * video_fps
    output_counter: int = 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

    while frame_index / video_fps <= end_sec_actual:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()
        if not ret:
            break

        out_path = output_dir / f"{prefix}_{output_counter:06d}.jpg"
        cv2.imwrite(
            str(out_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        written.append(out_path)
        logger.debug("Wrote %s", out_path.name)

        frame_index += step_frames
        output_counter += 1

    cap.release()
    logger.info("Extracted %d frames to %s", len(written), output_dir)
    return written


def write_image_list(frames: list[Path], output_path: str | Path) -> Path:
    """Write a plain-text list of frame filenames (one per line).

    OpenSfM's ``opensfm detect_features`` command can optionally read
    such a list to restrict processing to a subset of images.

    Parameters
    ----------
    frames:
        Ordered list of frame paths as returned by :func:`extract_frames`.
    output_path:
        Destination text file.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = Path(output_path)
    with output_path.open("w") as fh:
        for p in frames:
            fh.write(p.name + "\n")
    logger.info("Wrote image list to %s (%d entries)", output_path, len(frames))
    return output_path


def iter_video_frames(
    video_path: str | Path,
) -> Iterator[tuple[int, float, "cv2.typing.MatLike"]]:
    """Yield ``(frame_number, timestamp_sec, frame_bgr)`` for every frame.

    Useful for inspection or thumbnail generation without writing every
    frame to disk.

    Parameters
    ----------
    video_path:
        Path to the source MP4 file.

    Yields
    ------
    tuple[int, float, numpy.ndarray]
        Frame number (0-based), timestamp in seconds, and the BGR image
        array.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_no = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = frame_no / video_fps if video_fps > 0 else 0.0
            yield frame_no, ts, frame
            frame_no += 1
    finally:
        cap.release()


def copy_frames_to_opensfm_project(frames: list[Path], project_dir: str | Path) -> None:
    """Copy extracted frames into an OpenSfM project's ``images/`` sub-directory.

    Parameters
    ----------
    frames:
        List of frame paths as returned by :func:`extract_frames`.
    project_dir:
        Root of the OpenSfM project directory.
    """
    images_dir = Path(project_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for src in frames:
        dst = images_dir / src.name
        shutil.copy2(src, dst)
    logger.info("Copied %d frames to %s", len(frames), images_dir)
