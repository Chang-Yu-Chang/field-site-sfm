# field-site-sfm

Process Insta360 360° video and satellite imagery into a georeferenced,
high-resolution 3-D reconstruction of a field site using
[OpenSfM](https://opensfm.org/).

## Overview

The pipeline combines two complementary data sources:

| Source | Strengths | Weaknesses |
|---|---|---|
| Satellite imagery | Accurate geographic coordinates | Low spatial resolution (30–60 cm/px) |
| Insta360 video | High local resolution (~5–10 cm/px) | No inherent coordinate system |

Ground Control Points (GCPs) placed in the field act as the bridge: their
pixel locations in the video frames and their known world coordinates
(measured by GPS or digitised from the satellite image) allow OpenSfM to
georegister the reconstruction.

---

## Field Protocol

### 1 · Site Preparation (5–10 min)

* Place **5 GCP markers** (bright-orange 10 cm discs) in a dice-5 pattern:
  four corners and the centre of the 20 × 20 m plot.
* Lay a **1-metre scale bar** flat on the ground near the centre.
* Identify at least **3 satellite anchors** (static features visible in
  both the ground video and the satellite image, e.g. a manhole cover or
  a distinctive tree trunk base).

### 2 · Hardware configuration

| Setting | Value |
|---|---|
| Mode | 5.7 K / 30 fps video |
| Shutter | Manual > 1/500 s |
| Stabilisation | FlowState enabled |
| Mount | 3-metre telescopic selfie stick (nadir) |

### 3 · The "Lawnmower" Walk (10–15 min)

1. Walk parallel lanes across the 20 × 20 m area, spaced ≤ 2 m apart
   (> 70 % overlap).
2. Walk the full perimeter once, pole tilted ~10° inward.
3. Walk a full 360° circle around each tree / large obstacle.
4. Pace: ~0.5 m/s, smooth and steady.

### 4 · Environmental rules

* **Lighting**: overcast preferred; avoid harsh direct sunlight.
* **Wind**: do not survey when foliage is swaying noticeably.
* **Clothing**: dark, non-reflective; stand under the pole.

---

## Project structure

```
field-site-sfm/
├── config/
│   └── opensfm_config.yaml      # OpenSfM settings for equirectangular video
├── src/
│   └── field_site_sfm/
│       ├── video_processor.py   # Extract JPEG frames from Insta360 MP4
│       ├── gcp_manager.py       # Load, mark, and export GCPs
│       ├── satellite_fusion.py  # Georeference ortho + composite on satellite
│       ├── pipeline.py          # Full pipeline orchestrator
│       └── cli/
│           ├── extract_frames.py
│           ├── mark_gcps.py
│           └── run_pipeline.py
└── tests/
    ├── conftest.py
    ├── test_video_processor.py
    ├── test_gcp_manager.py
    ├── test_satellite_fusion.py
    ├── test_pipeline.py
    └── test_cli.py
```

---

## Installation

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install OpenSfM (follow https://opensfm.org/docs/building.html)
#    then install this package and its Python dependencies
pip install -e ".[dev]"
```

> **Note**: `rasterio` (required for the satellite fusion step) may need
> system libraries.  See the
> [rasterio install guide](https://rasterio.readthedocs.io/en/stable/installation.html).

---

## Usage

### Step 1 – Extract frames

```bash
sfm-extract-frames \
    --video  raw/survey_20240601.mp4 \
    --output frames/ \
    --fps    2 \
    --image-list
```

At 0.5 m/s walk speed, `--fps 2` gives one frame every ~25 cm — enough
for > 70 % overlap between adjacent lawnmower lanes.

### Step 2 – Mark GCPs interactively

```bash
sfm-mark-gcps \
    --frames  frames/ \
    --gcps    gcps/gcp_world_coords.csv \
    --output  gcps/gcp_observations.csv \
    --gcp-list gcps/gcp_list.txt
```

The GCP world-coordinate CSV must have columns:
`gcp_id`, `world_x`, `world_y`, `world_z`.

### Step 3 – Run the full pipeline

```bash
sfm-run-pipeline \
    --video      raw/survey_20240601.mp4 \
    --output     output/run_001 \
    --gcp-csv    gcps/gcp_world_coords.csv \
    --gcp-obs    gcps/gcp_observations.csv \
    --satellite  satellite/google_site.tif \
    --fps        2 \
    --processes  4
```

The pipeline:

1. Extracts frames at the requested FPS.
2. Sets up an OpenSfM project with the bundled equirectangular config.
3. Runs the full OpenSfM reconstruction (feature detection → matching →
   SfM → dense depth maps → mosaic export).
4. Re-runs bundle adjustment with the GCPs for georeferencing and
   metric scale.
5. Composites the georeferenced orthophoto onto the satellite image.

#### Key output files

| Path | Description |
|---|---|
| `output/run_001/frames/` | Extracted JPEG frames |
| `output/run_001/opensfm_project/reconstruction.json` | Sparse reconstruction |
| `output/run_001/opensfm_project/undistorted/mosaic/mosaic.tif` | High-res orthophoto |
| `output/run_001/composite.tif` | Orthophoto blended onto satellite image |

---

## Python API

```python
from field_site_sfm.pipeline import Pipeline, PipelineConfig

cfg = PipelineConfig(
    video_path="raw/survey.mp4",
    output_dir="output/run_001",
    gcp_csv="gcps/gcp_world.csv",
    gcp_observations_csv="gcps/gcp_obs.csv",
    satellite_path="satellite/google_site.tif",
    extract_fps=2.0,
    processes=4,
)
Pipeline(cfg).run()
```

Lower-level helpers are also importable directly:

```python
from field_site_sfm.video_processor import extract_frames
from field_site_sfm.gcp_manager import load_gcps_from_csv, write_opensfm_gcp_list
from field_site_sfm.satellite_fusion import estimate_affine_transform, georeference_orthophoto
```

---

## Running the tests

```bash
pytest
```

---

## License

MIT – see [LICENSE](LICENSE).
