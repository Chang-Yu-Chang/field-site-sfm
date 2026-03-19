"""Main SfM pipeline orchestrator.

This module ties together the three processing stages:

1. **Frame extraction** – pull JPEG frames from the Insta360 MP4.
2. **OpenSfM reconstruction** – run the full OpenSfM pipeline inside a
   project directory.
3. **GCP bundle adjustment & georeferencing** – apply Ground Control
   Points to scale and orient the model, then produce a georeferenced
   orthophoto.
4. **Satellite fusion** – composite the high-resolution orthophoto
   onto the satellite reference image.

Usage example::

    from field_site_sfm.pipeline import Pipeline, PipelineConfig

    cfg = PipelineConfig(
        video_path="raw/survey_20240601.mp4",
        gcp_csv="gcps/gcp_world_coords.csv",
        gcp_observations_csv="gcps/gcp_pixel_obs.csv",
        satellite_path="satellite/google_20m_site.tif",
        output_dir="output/run_001",
    )
    pipeline = Pipeline(cfg)
    pipeline.run()
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .gcp_manager import (
    GCPPoint,
    load_gcps_from_csv,
    merge_gcp_observations,
    write_opensfm_gcp_list,
)
from .video_processor import (
    copy_frames_to_opensfm_project,
    extract_frames,
    write_image_list,
)

logger = logging.getLogger(__name__)

# Path to the bundled default OpenSfM config template.
_DEFAULT_CONFIG_TEMPLATE = Path(__file__).parent.parent.parent / "config" / "opensfm_config.yaml"

# OpenSfM sub-commands to run in order for a full (sparse + dense) reconstruction.
OPENSFM_STEPS = [
    "extract_metadata",
    "detect_features",
    "match_features",
    "create_tracks",
    "reconstruct",
    "bundle",
    "undistort",
    "compute_depthmaps",
    "merge_depthmaps",
    "compute_statistics",
    "export_ply",
    "export_openmvs",
]

# Sub-set of OPENSFM_STEPS for a sparse-only reconstruction (no dense depthmaps).
OPENSFM_SPARSE_STEPS = [
    "extract_metadata",
    "detect_features",
    "match_features",
    "create_tracks",
    "reconstruct",
    "bundle",
    "undistort",
    "compute_statistics",
    "export_ply",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """All parameters required to run the full pipeline.

    Attributes
    ----------
    video_path:
        Path to the source Insta360 MP4 file.
    output_dir:
        Root directory for all pipeline outputs.
    extract_fps:
        Number of frames to extract per second of video.  Default 2 fps
        gives ~25 cm spacing at 0.5 m/s walk speed.
    video_start_sec:
        Seconds from the start of the video to begin frame extraction.
    video_end_sec:
        Seconds at which to stop extraction (``None`` = end of video).
    gcp_csv:
        Path to a CSV file with GCP world coordinates.  Columns:
        ``gcp_id``, ``world_x``, ``world_y``, ``world_z``.
        Pass ``None`` to skip GCP registration.
    gcp_observations_csv:
        Path to a CSV file with per-image GCP pixel observations.
        Columns: ``gcp_id``, ``image_name``, ``pixel_x``, ``pixel_y``.
        Required when *gcp_csv* is provided and world-coord CSV does
        not already include observation columns.
    gcp_crs:
        CRS string for the GCP world coordinates, e.g. ``"WGS84"`` or
        ``"EPSG:32633"``.
    satellite_path:
        Path to the satellite reference GeoTIFF.  Pass ``None`` to
        skip the satellite fusion step.
    satellite_blend_alpha:
        Blending weight for the composite (0 = satellite only,
        1 = ortho only).
    opensfm_config_path:
        Path to a custom ``config.yaml`` for OpenSfM.  If ``None`` the
        bundled default configuration is used.
    opensfm_executable:
        Name or path of the OpenSfM runner.  Defaults to ``"opensfm"``.
    run_dense:
        Whether to run dense reconstruction (depthmap + merge steps).
        Set ``False`` for a quick sparse-only pass.
    processes:
        Number of parallel worker processes for OpenSfM.
    """

    video_path: str | Path
    output_dir: str | Path

    extract_fps: float = 2.0
    video_start_sec: float = 0.0
    video_end_sec: float | None = None

    gcp_csv: str | Path | None = None
    gcp_observations_csv: str | Path | None = None
    gcp_crs: str = "WGS84"

    satellite_path: str | Path | None = None
    satellite_blend_alpha: float = 0.6

    opensfm_config_path: str | Path | None = None
    opensfm_executable: str = "opensfm"

    run_dense: bool = True
    processes: int = 4

    extra_opensfm_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)
        self.output_dir = Path(self.output_dir)
        if self.gcp_csv is not None:
            self.gcp_csv = Path(self.gcp_csv)
        if self.gcp_observations_csv is not None:
            self.gcp_observations_csv = Path(self.gcp_observations_csv)
        if self.satellite_path is not None:
            self.satellite_path = Path(self.satellite_path)
        if self.opensfm_config_path is not None:
            self.opensfm_config_path = Path(self.opensfm_config_path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Orchestrates the full field-site SfM pipeline.

    Parameters
    ----------
    config:
        A :class:`PipelineConfig` instance.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.project_dir = config.output_dir / "opensfm_project"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all pipeline stages in order."""
        logger.info("=== Field-site SfM pipeline starting ===")
        logger.info("Video   : %s", self.cfg.video_path)
        logger.info("Output  : %s", self.cfg.output_dir)

        self._stage_extract_frames()
        self._stage_setup_opensfm_project()
        self._stage_run_opensfm()

        if self.cfg.gcp_csv is not None:
            self._stage_apply_gcps()

        if self.cfg.satellite_path is not None:
            self._stage_fuse_satellite()

        logger.info("=== Pipeline complete ===")

    # ------------------------------------------------------------------
    # Stage 1: Frame extraction
    # ------------------------------------------------------------------

    def _stage_extract_frames(self) -> list[Path]:
        frames_dir = self.cfg.output_dir / "frames"
        logger.info("[1/4] Extracting frames to %s", frames_dir)

        frames = extract_frames(
            video_path=self.cfg.video_path,
            output_dir=frames_dir,
            fps=self.cfg.extract_fps,
            start_sec=self.cfg.video_start_sec,
            end_sec=self.cfg.video_end_sec,
        )
        write_image_list(frames, self.cfg.output_dir / "image_list.txt")

        copy_frames_to_opensfm_project(frames, self.project_dir)
        self._frames = frames
        return frames

    # ------------------------------------------------------------------
    # Stage 2: OpenSfM project setup
    # ------------------------------------------------------------------

    def _stage_setup_opensfm_project(self) -> None:
        logger.info("[2/4] Setting up OpenSfM project at %s", self.project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._write_opensfm_config()

    def _write_opensfm_config(self) -> None:
        """Merge template + user overrides and write config.yaml."""
        if (
            self.cfg.opensfm_config_path is not None
            and Path(self.cfg.opensfm_config_path).is_file()
        ):
            template_path = Path(self.cfg.opensfm_config_path)
        elif _DEFAULT_CONFIG_TEMPLATE.is_file():
            template_path = _DEFAULT_CONFIG_TEMPLATE
        else:
            template_path = None

        if template_path is not None:
            with template_path.open() as fh:
                cfg_dict: dict = yaml.safe_load(fh) or {}
        else:
            cfg_dict = {}

        # Apply per-run overrides.
        cfg_dict["processes"] = self.cfg.processes
        cfg_dict.update(self.cfg.extra_opensfm_config)

        out_path = self.project_dir / "config.yaml"
        with out_path.open("w") as fh:
            yaml.dump(cfg_dict, fh, default_flow_style=False)
        logger.debug("Wrote OpenSfM config to %s", out_path)

    # ------------------------------------------------------------------
    # Stage 3: Run OpenSfM
    # ------------------------------------------------------------------

    def _stage_run_opensfm(self) -> None:
        steps = OPENSFM_STEPS if self.cfg.run_dense else OPENSFM_SPARSE_STEPS
        logger.info("[3/4] Running OpenSfM (%d steps)", len(steps))

        for step in steps:
            self._run_opensfm_step(step)

    def _run_opensfm_step(self, step: str) -> None:
        cmd = [self.cfg.opensfm_executable, step, str(self.project_dir)]
        logger.info("  opensfm %s", step)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"OpenSfM step '{step}' failed (exit {result.returncode}).\n"
                f"Output:\n{result.stdout}"
            )
        logger.debug("  %s: OK", step)

    # ------------------------------------------------------------------
    # Stage 4: GCP bundle adjustment
    # ------------------------------------------------------------------

    def _stage_apply_gcps(self) -> None:
        logger.info("[4a] Applying GCPs from %s", self.cfg.gcp_csv)

        gcps = load_gcps_from_csv(self.cfg.gcp_csv)  # type: ignore[arg-type]

        if self.cfg.gcp_observations_csv is not None:
            gcps = merge_gcp_observations(gcps, self.cfg.gcp_observations_csv)

        gcp_list_path = self.project_dir / "gcp_list.txt"
        write_opensfm_gcp_list(gcps, gcp_list_path, crs=self.cfg.gcp_crs)

        # Re-run bundle with GCPs.
        self._run_opensfm_step("bundle")
        logger.info("GCP bundle adjustment complete")

    # ------------------------------------------------------------------
    # Stage 5: Satellite fusion
    # ------------------------------------------------------------------

    def _stage_fuse_satellite(self) -> None:
        logger.info("[4b] Fusing with satellite image: %s", self.cfg.satellite_path)
        # The satellite fusion module is imported lazily so that the rest
        # of the pipeline works even when rasterio is not installed.
        try:
            from .satellite_fusion import (  # noqa: PLC0415
                composite_ortho_on_satellite,
            )
        except ImportError as exc:
            logger.warning("Satellite fusion skipped: %s", exc)
            return

        ortho_path = self.project_dir / "undistorted" / "mosaic" / "mosaic.tif"
        if not ortho_path.exists():
            logger.warning(
                "Orthophoto not found at expected path %s; skipping fusion.",
                ortho_path,
            )
            return

        composite_path = self.cfg.output_dir / "composite.tif"
        composite_ortho_on_satellite(
            satellite_path=self.cfg.satellite_path,  # type: ignore[arg-type]
            ortho_georef_path=ortho_path,
            output_path=composite_path,
            blend_alpha=self.cfg.satellite_blend_alpha,
        )
        logger.info("Composite saved to %s", composite_path)
