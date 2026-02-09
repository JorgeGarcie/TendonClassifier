# TendonClassifier

Classifies tendon type (none, single, crossed, double) beneath silicone phantoms using a robot arm with a tactile camera and force/torque sensor.

## Setup

```
pip install -r requirements.txt
```

## Project Structure

- `scripts/labeling/` - data pipeline: contact detection, ground truth generation from STL meshes
- `scripts/classification/` - models and training (spatial + temporal, with ablation configs)

## Labeling Pipeline

1. `discover_and_index.py` - indexes ROS bag data
2. `extract_valid_windows.py` - finds contact windows using force threshold
3. `generate_gt.py` - maps TCP position to tendon label via raycasted STL grids

## Training

```
cd scripts/classification
python train_v2.py --config configs/spatial_combined.yaml
```

See `scripts/classification/configs/` for all experiment configs (force-only, image-only, combined, temporal variants).
