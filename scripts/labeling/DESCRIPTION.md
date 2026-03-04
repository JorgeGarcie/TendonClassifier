# Labeling Pipeline - Project Description

## Overview

Data processing pipeline that takes raw multi-sensor recordings (camera, force/torque, TCP pose) and produces a labeled ground-truth dataset for tendon classification. Handles sensor discovery, contact window detection, frame sampling, coordinate transforms, and GT grid lookup.

## Pipeline Stages

```
1. discover_and_index.py    → run_manifest.json
2. extract_valid_windows.py → valid_frames.json
3. gt_labeler.py            → output/gt_grids/*.npz  (bbox GT grids per phantom/config)
4. generate_gt.py           → gt_manifest.csv + images/
```

> **Note**: `annotate_centerlines.py` and `generate_masks.py` are legacy scripts — not part of the active pipeline.

### Stage 1: Discover and Index (`discover_and_index.py`)
- Scans `DATA_ROOT` for folders matching `{phantom}_{motion}_{force}` pattern
- Validates each run has required files: `frames/`, `camera_frames.csv`, `tcp_pose.csv`, `wrench_data.csv`
- Resolves STL file and rotation from phantom configs
- Outputs `run_manifest.json`

### Stage 2: Extract Valid Windows (`extract_valid_windows.py`)
- Detects contact windows using force magnitude thresholding with hysteresis
- Samples frames within contact windows (every-N or uniform-M sampling)
- Matches TCP pose to each frame via nearest-neighbor timestamp lookup
- Resamples force/torque data to camera rate via linear interpolation (see Data Synchronization)
- Outputs `valid_frames.json`

### Stage 3: Generate GT Grids (`gt_labeler.py`)
- Reads `phantom_configs.json` for each phantom/config (STL file, rotation_deg, pattern, tendon bounds)
- Constructs a 2D grid in STL coordinates and labels each cell: `type_id` (none/single/crossed/double), `hit`, `depth_mm`
- **Bounding-box only** — no raycasting. `flip_y = (rotation_deg == 180)` controls which half of the y-axis maps to which tendon type for crossed/double phantoms
- Outputs `.npz` files to `output/gt_grids/` (e.g. `p4_s2m.npz`)
- Run per phantom/config: `python gt_labeler.py --phantom p4 --config s2m`
- Run all: `python gt_labeler.py`

### Stage 4: Generate GT Dataset (`generate_gt.py`)
- Loads precomputed GT grids (`.npz`) for each phantom/motion combo
- For each frame: transforms TCP world position to STL grid coordinates, looks up tendon presence/depth/type
- Resamples full 6-axis wrench (fx, fy, fz, tx, ty, tz) to camera rate via linear interpolation
- **Center crops** images to `CROP_SIZE x CROP_SIZE` (currently 1080x1080) from the raw 1920x1080 frames before saving. Cropping preserves the sensor region without distorting aspect ratio.
- Outputs `gt_manifest.csv` with columns: run_id, phantom_type, frame_idx, timestamp, image_path, presence, depth_mm, tendon_type, gx, gy, force_magnitude, fx, fy, fz, tx, ty, tz

## Data Synchronization

### Sensor Rates
- **Camera**: ~15-30 Hz (variable, depends on recording)
- **Force/Torque (coinft)**: ~100-200 Hz
- **TCP Pose**: ~100 Hz (from robot controller)

### Resampling Strategy
The force/torque sensor operates at a much higher frequency than the camera. To produce paired (image, force) samples, **force data is resampled to camera rate via linear interpolation** (`np.interp`). For each camera frame timestamp, the surrounding F/T readings are linearly interpolated to estimate the force at the exact moment of image capture.

- **Sensor selection**: Only the `coinft` sensor is used (configured via `config.WRENCH_SENSOR`). The raw `wrench_data.csv` contains interleaved readings from both the coinft and flexiv (robot internal) sensors; flexiv readings are filtered out before interpolation.
- **TCP pose**: Matched via nearest-neighbor timestamp lookup (not interpolated, as pose changes slowly relative to sensor rate).

## Raw Data Format

Each run folder contains:
```
{phantom}_{motion}_{force}-{timestamp}/
├── frames/              # Camera images (frame_XXXXXX.jpg)
├── camera_frames.csv    # time, frame_number, image_path
├── tcp_pose.csv         # time, x, y, z, qx, qy, qz, qw
└── wrench_data.csv      # time, sensor, fx, fy, fz, tx, ty, tz
```

## File Structure

```
scripts/labeling/
├── config.py                    # Pipeline configuration
├── configs/
│   ├── phantom_configs.json     # Phantom STL, rotation, tendon bounds per config
│   ├── run_manifest.json        # [generated] discovered runs + split labels
│   └── valid_frames.json        # [generated] contact-window frames
├── discover_and_index.py        # Stage 1: scan and validate runs
├── extract_valid_windows.py     # Stage 2: contact detection + frame sampling
├── gt_labeler.py                # Stage 3: bbox GT grid generation → output/gt_grids/
├── generate_gt.py               # Stage 4: GT dataset generation → gt_manifest.csv
├── visualize_bounds.py          # TCP trajectory + tendon bounds calibration tool
├── visualize_gt_spatial.py      # GT label grid: thumbnails sorted by gy position
├── inspect_frames.py            # Frame inspection utility
├── rawdata/                     # Raw recordings + STL files
└── output/
    ├── gt_grids/                # Precomputed .npz GT grids (per phantom/config)
    └── gt_dataset/              # Final output (images/ + gt_manifest.csv)
```

## Configuration (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FORCE_THRESHOLD_N` | 12.0 | Contact detection threshold (N) |
| `CROP_SIZE` | 1080 | Center crop size (pixels) applied to images in generate_gt.py |
| `WRENCH_SENSOR` | `"coinft"` | F/T sensor used for resampling |
| `FRAME_SAMPLING.mode` | `"every_n"` | Frame sampling strategy |
| `FRAME_SAMPLING.n` | 1 | Take every Nth frame |

## Natural Arc (nat) Runs — Test Set Only, No GT

Runs with `approach_label="nat"` use a natural arc palpation trajectory where the probe sweeps both x and y simultaneously. At high scanning speeds, the TCP position at force peak does **not** land cleanly at x=0 (the tendon center), so bounding-box GT labels are misaligned for these runs.

**Decision**: nat runs are designated `"split": "test_nat"` in `run_manifest.json` and **excluded from training** via `exclude_run_regex: "_nat-"` in all training YAML configs. They can be used as a qualitative test set but **must not be evaluated with standard GT metrics** derived from the bounding-box labeling pipeline.

Nat run IDs (10 total):
- p1_t_0_nat, p1_t_90_nat
- p2_s2d_0_nat, p2_s2d_90_nat
- p3_thicker2thinner_0_nat, p3_thicker2thinner_90_nat
- p4_s2m_0_nat, p4_s2m_90_nat
- p5_s2m_0_nat, p5_s2m_90_nat

## Tendon Bounds Calibration

The `tendon_bounds` in `phantom_configs.json` define the x/y region where the GT grid considers a tendon present. At the boundary, the probe can visually see the tendon even if it's slightly outside the box — so these bounds should be slightly larger than the exact STL geometry.

Use `visualize_bounds.py` to calibrate:

```bash
# See TCP trajectory + current bounds box for a run
python visualize_bounds.py --run-id p4_m2s_35N-2025-11-25_22.46.29

# Also show a sample center-cropped frame
python visualize_bounds.py --run-id p4_m2s_35N-... --show-frames

# List all available runs
python visualize_bounds.py --list
```

The scatter plot shows all TCP positions in STL grid coordinates colored by force magnitude. The red box is the current `tendon_bounds`. Adjust the bounds in `phantom_configs.json` based on what you observe, then re-run `generate_gt.py` to regenerate the dataset with updated GT.

**Important**: Always run `generate_gt.py` (with center crop) before using `visualize_bounds.py` with `--show-frames`, so the displayed crops match what the model sees.

## Dependencies

```
numpy
pandas
opencv-python
```
