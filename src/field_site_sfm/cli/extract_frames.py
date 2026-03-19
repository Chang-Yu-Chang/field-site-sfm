"""CLI: extract frames from an Insta360 MP4 video.

Usage::

    sfm-extract-frames --video survey.mp4 --output frames/ --fps 2
"""

from __future__ import annotations

import logging
import sys

import click

from ..video_processor import extract_frames, write_image_list


@click.command()
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the Insta360 MP4 file.",
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(file_okay=False),
    help="Output directory for JPEG frames.",
)
@click.option(
    "--fps", "-r",
    default=2.0,
    show_default=True,
    type=float,
    help="Frames to extract per second of video.",
)
@click.option(
    "--start", "-s",
    default=0.0,
    show_default=True,
    type=float,
    help="Start time in seconds.",
)
@click.option(
    "--end", "-e",
    default=None,
    type=float,
    help="End time in seconds (default: end of video).",
)
@click.option(
    "--prefix", "-p",
    default="frame",
    show_default=True,
    help="Filename prefix for extracted frames.",
)
@click.option(
    "--image-list",
    is_flag=True,
    default=False,
    help="Write an image_list.txt alongside the frames.",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable debug logging.")
def main(
    video: str,
    output: str,
    fps: float,
    start: float,
    end: float | None,
    prefix: str,
    image_list: bool,
    verbose: bool,
) -> None:
    """Extract frames from an Insta360 360° video file."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    frames = extract_frames(
        video_path=video,
        output_dir=output,
        fps=fps,
        start_sec=start,
        end_sec=end,
        prefix=prefix,
    )

    if image_list:
        from pathlib import Path  # noqa: PLC0415

        list_path = Path(output) / "image_list.txt"
        write_image_list(frames, list_path)
        click.echo(f"Image list written to {list_path}")

    click.echo(f"Extracted {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
