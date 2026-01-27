"""
Configuration file for tendon labeling and mask generation pipeline.
"""
# Paths (relative to labeling directory)
DATA_ROOT = "rawdata"
CONFIGS_ROOT = "configs"
OUTPUT_ROOT = "output"
STL_DIR = "rawdata"

# Phantom configuration file (maps phantom + motion type → STL file + rotation)
PHANTOM_CONFIGS_FILE = "phantom_configs.json"

# Camera intrinsics
CAMERA = {
    "hfov_deg": 120,
    "vfov_deg": 66,
    "cam_z": 0.0127 + 0.006594 + 0.005,  # from TendonLabeler
}

# Processing params
FORCE_THRESHOLD_N = 12.0  # hysteresis threshold
KEYFRAME_INTERVAL = 10    # annotate centerline every N frames
FRAME_SAMPLING = {
    "mode": "every_n",    # "every_n" or "uniform_m"
    "n": 1,               # for every_n: take every nth frame
    "m": 50,              # for uniform_m: sample m frames per window
}

# CSV column mappings (so agent knows your file formats)
CAMERA_FRAMES_COLS = ["time", "frame_number", "image_path"]
TCP_POSE_COLS = ["time", "x", "y", "z", "qx", "qy", "qz", "qw"]
WRENCH_DATA_COLS = ["time", "sensor", "fx", "fy", "fz", "tx", "ty", "tz"]  # adjust as needed

# Column name mapping for compatibility
TIME_COL = "time"
FRAME_IDX_COL = "frame_number"
IMAGE_PATH_COL = "image_path"

# GT grid generation 
GT_GRID_DIR = "output/gt_grids"
GT_GRID_PARAMS = {
    "x_half": 0.0381,
    "y_half": 0.0635,
    "dx": 0.0004,
    "dy": 0.0004,
    "ray_z0": 0.025,
}
