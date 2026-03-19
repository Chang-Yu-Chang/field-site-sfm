"""CLI: run the full field-site SfM pipeline.

Usage::

    sfm-run-pipeline \\
        --video survey.mp4 \\
        --output output/run_001 \\
        --gcp-csv gcps/gcp_world.csv \\
        --gcp-obs gcps/gcp_observations.csv \\
        --satellite satellite/google_site.tif \\
        --fps 2 \\
        --processes 4
"""

from __future__ import annotations

import logging
import sys

import click

from ..pipeline import Pipeline, PipelineConfig


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
    help="Root output directory.",
)
@click.option(
    "--fps", "-r",
    default=2.0,
    show_default=True,
    type=float,
    help="Frames to extract per second of video.",
)
@click.option(
    "--start",
    default=0.0,
    show_default=True,
    type=float,
    help="Video start time in seconds.",
)
@click.option(
    "--end",
    default=None,
    type=float,
    help="Video end time in seconds (default: end of video).",
)
@click.option(
    "--gcp-csv",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV file with GCP world coordinates.",
)
@click.option(
    "--gcp-obs",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV file with per-image GCP pixel observations.",
)
@click.option(
    "--gcp-crs",
    default="WGS84",
    show_default=True,
    help="CRS string for GCP world coordinates.",
)
@click.option(
    "--satellite",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Satellite reference GeoTIFF for fusion.",
)
@click.option(
    "--blend-alpha",
    default=0.6,
    show_default=True,
    type=float,
    help="Blending weight for satellite fusion (0=satellite, 1=ortho).",
)
@click.option(
    "--opensfm-config",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Custom OpenSfM config.yaml (uses bundled default if omitted).",
)
@click.option(
    "--opensfm-bin",
    default="opensfm",
    show_default=True,
    help="Name or path of the opensfm executable.",
)
@click.option(
    "--no-dense",
    is_flag=True,
    default=False,
    help="Skip dense reconstruction (faster, sparse-only).",
)
@click.option(
    "--processes",
    default=4,
    show_default=True,
    type=int,
    help="Number of parallel worker processes for OpenSfM.",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable debug logging.")
def main(
    video: str,
    output: str,
    fps: float,
    start: float,
    end: float | None,
    gcp_csv: str | None,
    gcp_obs: str | None,
    gcp_crs: str,
    satellite: str | None,
    blend_alpha: float,
    opensfm_config: str | None,
    opensfm_bin: str,
    no_dense: bool,
    processes: int,
    verbose: bool,
) -> None:
    """Run the full field-site SfM pipeline on an Insta360 video."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    cfg = PipelineConfig(
        video_path=video,
        output_dir=output,
        extract_fps=fps,
        video_start_sec=start,
        video_end_sec=end,
        gcp_csv=gcp_csv,
        gcp_observations_csv=gcp_obs,
        gcp_crs=gcp_crs,
        satellite_path=satellite,
        satellite_blend_alpha=blend_alpha,
        opensfm_config_path=opensfm_config,
        opensfm_executable=opensfm_bin,
        run_dense=not no_dense,
        processes=processes,
    )

    pipeline = Pipeline(cfg)
    try:
        pipeline.run()
    except Exception as exc:
        click.echo(f"Pipeline failed: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Pipeline finished.  Results in: {output}")


if __name__ == "__main__":
    main()
