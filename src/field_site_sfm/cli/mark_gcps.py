"""CLI: interactively mark GCPs on extracted frames.

Displays each extracted frame and lets the user click on the pixel
location of each known GCP.  The resulting observations are saved to a
CSV file that can be fed into the pipeline.

Usage::

    sfm-mark-gcps --frames frames/ --gcps gcp_world.csv \\
                  --output gcp_observations.csv

Keyboard shortcuts while the viewer is open:

* **Left-click**  – record the click location as the current GCP observation
* **n**           – next GCP
* **s**           – skip this image for the current GCP
* **q**           – quit and save progress

.. note::
    This tool requires an interactive display (X11/Wayland).  On
    head-less servers, mark GCPs manually and create the observation
    CSV directly.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import click

from ..gcp_manager import GCPPoint, load_gcps_from_csv, write_opensfm_gcp_list

logger = logging.getLogger(__name__)


def _interactive_mark(
    frames: list[Path],
    gcps: list[GCPPoint],
    output_csv: Path,
) -> None:
    """Run the interactive OpenCV window to collect pixel observations."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        click.echo("ERROR: opencv-python is required for interactive marking.", err=True)
        sys.exit(1)

    observations: list[dict] = []
    last_click: list[tuple[int, int]] = []

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            last_click.clear()
            last_click.append((x, y))

    for gcp in gcps:
        click.echo(f"\nMarking GCP: {gcp.gcp_id} "
                   f"(world: {gcp.world_x:.4f}, {gcp.world_y:.4f}, {gcp.world_z:.4f})")

        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            # Resize for display if very large.
            h, w = img.shape[:2]
            scale = min(1.0, 1280 / max(w, 1))
            if scale < 1.0:
                display = cv2.resize(img, (int(w * scale), int(h * scale)))
            else:
                display = img.copy()

            win = f"Mark GCP {gcp.gcp_id} – {frame_path.name}"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(win, _on_mouse)

            # Draw prompt text.
            cv2.putText(
                display,
                f"Click on GCP {gcp.gcp_id} | n=next | s=skip | q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.imshow(win, display)

            while True:
                key = cv2.waitKey(50) & 0xFF
                if key == ord("s"):
                    break  # skip this image
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    _save_observations(observations, output_csv)
                    click.echo(f"Saved {len(observations)} observations to {output_csv}")
                    return
                if key == ord("n") or last_click:
                    if last_click:
                        cx, cy = last_click[0]
                        # Convert back to original image coordinates.
                        orig_x = int(cx / scale)
                        orig_y = int(cy / scale)
                        observations.append(
                            {
                                "gcp_id": gcp.gcp_id,
                                "image_name": frame_path.name,
                                "pixel_x": orig_x,
                                "pixel_y": orig_y,
                            }
                        )
                        logger.debug(
                            "Recorded %s in %s at (%d, %d)",
                            gcp.gcp_id,
                            frame_path.name,
                            orig_x,
                            orig_y,
                        )
                        last_click.clear()
                    break

            cv2.destroyWindow(win)

    _save_observations(observations, output_csv)
    click.echo(f"Saved {len(observations)} observations to {output_csv}")


def _save_observations(observations: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["gcp_id", "image_name", "pixel_x", "pixel_y"],
        )
        writer.writeheader()
        writer.writerows(observations)


@click.command()
@click.option(
    "--frames", "-f",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory of extracted JPEG frames.",
)
@click.option(
    "--gcps", "-g",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV file with GCP world coordinates (gcp_id, world_x, world_y, world_z).",
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output CSV for pixel observations.",
)
@click.option(
    "--gcp-list", "-l",
    default=None,
    type=click.Path(dir_okay=False),
    help="Also write an OpenSfM gcp_list.txt to this path.",
)
@click.option(
    "--crs",
    default="WGS84",
    show_default=True,
    help="CRS string for the gcp_list.txt header.",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable debug logging.")
def main(
    frames: str,
    gcps: str,
    output: str,
    gcp_list: str | None,
    crs: str,
    verbose: bool,
) -> None:
    """Interactively mark GCP positions on extracted video frames."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    frames_dir = Path(frames)
    output_csv = Path(output)

    frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.jpeg"))
    if not frame_paths:
        click.echo("ERROR: no JPEG files found in frames directory.", err=True)
        sys.exit(1)

    gcp_points = load_gcps_from_csv(gcps)
    click.echo(f"Loaded {len(gcp_points)} GCPs from {gcps}")
    click.echo(f"Found {len(frame_paths)} frames in {frames}")

    _interactive_mark(frame_paths, gcp_points, output_csv)

    if gcp_list is not None:
        # Reload the freshly-written observations and merge.
        from ..gcp_manager import merge_gcp_observations  # noqa: PLC0415

        gcp_points = merge_gcp_observations(gcp_points, output_csv)
        write_opensfm_gcp_list(gcp_points, Path(gcp_list), crs=crs)
        click.echo(f"OpenSfM gcp_list.txt written to {gcp_list}")


if __name__ == "__main__":
    main()
