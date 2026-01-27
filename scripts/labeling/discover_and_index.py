"""
Discover and index data runs.

Recursively scans DATA_ROOT for folders matching pattern {phantom}_{motion}_{force}.
Validates each folder and outputs run_manifest.json.
"""

import json
import os
import re
from pathlib import Path
import logging

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected files in each run folder
REQUIRED_FILES = ["frames", "camera_frames.csv", "tcp_pose.csv", "wrench_data.csv"]
RUN_PATTERN = re.compile(r"^(p\d+)_([a-z0-9]+)_(\d+N)(?:-.+)?$")

# Load phantom configurations
PHANTOM_CONFIGS = {}
def load_phantom_configs():
    """Load phantom configurations from JSON file."""
    global PHANTOM_CONFIGS
    config_path = Path(config.CONFIGS_ROOT) / config.PHANTOM_CONFIGS_FILE
    if config_path.exists():
        with open(config_path, "r") as f:
            PHANTOM_CONFIGS = json.load(f)
        logger.info(f"Loaded phantom configs from {config_path}")
    else:
        logger.warning(f"Phantom configs not found at {config_path}")

def get_stl_config(phantom_type, motion_type):
    """
    Resolve STL file and rotation for a phantom + motion type.

    Returns: {"stl_file": str, "rotation_deg": int} or None
    """
    if not PHANTOM_CONFIGS:
        return None

    phantom_cfg = PHANTOM_CONFIGS.get(phantom_type)
    if not phantom_cfg:
        logger.warning(f"Phantom type {phantom_type} not in configs")
        return None

    # Configs may be nested under "configs" key or flat at phantom level
    configs = phantom_cfg.get("configs", phantom_cfg)

    # Try exact motion type match first, fallback to "default"
    stl_cfg = configs.get(motion_type) or configs.get("default")
    if not stl_cfg:
        logger.warning(f"No STL config for {phantom_type}/{motion_type}")
        return None

    return stl_cfg


def discover_runs(data_root):
    """
    Recursively scan data_root for run folders.

    Returns list of dicts with: run_id, path, phantom_type, motion_type, force_label, valid, missing_files
    """
    runs = []
    data_root = Path(data_root)

    if not data_root.exists():
        logger.warning(f"DATA_ROOT does not exist: {data_root}")
        return runs

    for item in data_root.iterdir():
        if not item.is_dir():
            continue

        folder_name = item.name
        match = RUN_PATTERN.match(folder_name)

        if not match:
            logger.debug(f"Skipping non-matching folder: {folder_name}")
            continue

        phantom_type, motion_type, force_label = match.groups()

        # Validate required files
        missing_files = []
        for required in REQUIRED_FILES:
            full_path = item / required
            if not full_path.exists():
                missing_files.append(required)

        valid = len(missing_files) == 0

        # Resolve STL configuration
        stl_cfg = get_stl_config(phantom_type, motion_type)

        run_entry = {
            "run_id": folder_name,
            "path": str(item.absolute()),
            "phantom_type": phantom_type,
            "motion_type": motion_type,
            "force_label": force_label,
            "stl_file": stl_cfg.get("stl_file") if stl_cfg else None,
            "rotation_deg": stl_cfg.get("rotation_deg") if stl_cfg else 0,
            "valid": valid,
            "missing_files": missing_files,
        }

        runs.append(run_entry)

        if valid:
            logger.info(f"✓ Valid run: {folder_name}")
        else:
            logger.warning(f"✗ Invalid run: {folder_name} (missing: {missing_files})")

    return runs


def main():
    logger.info(f"Scanning DATA_ROOT: {config.DATA_ROOT}")

    load_phantom_configs()
    runs = discover_runs(config.DATA_ROOT)

    valid_count = sum(1 for r in runs if r["valid"])
    invalid_count = len(runs) - valid_count

    manifest = {
        "runs": runs,
        "summary": {
            "total": len(runs),
            "valid": valid_count,
            "invalid": invalid_count,
        }
    }

    output_path = Path(config.CONFIGS_ROOT) / "run_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved to: {output_path}")
    logger.info(f"Summary: {valid_count} valid, {invalid_count} invalid out of {len(runs)} total")


if __name__ == "__main__":
    main()
