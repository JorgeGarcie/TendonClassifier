"""
Visual spot check of forced-label runs.
Shows center-cropped frames as a grid so you can verify labels look correct.

Usage:
    python inspect_forced_runs.py --save /tmp/forced_check
    python inspect_forced_runs.py --run none_traverse-2026-02-24 --save /tmp/forced_check
    python inspect_forced_runs.py --n-frames 30 --save /tmp/forced_check
"""
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

MANIFEST = Path("output/gt_dataset/gt_manifest.csv")
IMAGES_ROOT = Path("output/gt_dataset")

FORCED_RUNS = [
    "none_traverse-2026-02-24",
    "p1_single_traverse-2026-02-24",
    "p4_crossed_traverse-2026-02-24",
    "p5_double_traverse-2026-02-24",
]

LABEL_NAMES = {0: "none", 1: "single", 2: "crossed", 3: "double"}
LABEL_COLORS = {
    0: (180, 180, 180),  # grey
    1: (80, 200, 80),    # green
    2: (80, 80, 220),    # blue
    3: (220, 80, 80),    # red
}

THUMB = 200      # thumbnail size in the grid
BORDER = 6       # border thickness
N_COLS = 10


def center_crop(img, size):
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    y1, y2 = max(0, cy - half), min(h, cy + half)
    x1, x2 = max(0, cx - half), min(w, cx + half)
    return img[y1:y2, x1:x2]


def make_grid(frames_info, crop_size, title):
    """frames_info: list of (img_path, label, frame_idx)"""
    thumbs = []
    for img_path, label, frame_idx in frames_info:
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
        else:
            img = center_crop(img, crop_size)
            img = cv2.resize(img, (THUMB, THUMB))

        # colored border
        color = LABEL_COLORS.get(label, (200, 200, 200))
        bordered = cv2.copyMakeBorder(img, BORDER, BORDER, BORDER, BORDER,
                                      cv2.BORDER_CONSTANT, value=color)
        # label text
        txt = f"{frame_idx} [{LABEL_NAMES.get(label, label)}]"
        cv2.putText(bordered, txt, (4, THUMB + BORDER - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(bordered)

    cell = THUMB + 2 * BORDER
    n_rows = (len(thumbs) + N_COLS - 1) // N_COLS
    canvas = np.zeros((n_rows * cell, N_COLS * cell, 3), dtype=np.uint8)

    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, N_COLS)
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = thumb

    # title bar
    title_bar = np.zeros((40, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([title_bar, canvas])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="specific run_id (default: all forced runs)")
    parser.add_argument("--n-frames", type=int, default=40,
                        help="number of evenly-spaced frames to sample per run (default: 40)")
    parser.add_argument("--crop", type=int, default=1000,
                        help="center crop size to display (default: 1000)")
    parser.add_argument("--save", required=True, help="output directory")
    args = parser.parse_args()

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)

    runs = [args.run] if args.run else FORCED_RUNS

    for run_id in runs:
        run_df = df[df["run_id"] == run_id].reset_index(drop=True)
        if run_df.empty:
            print(f"Run {run_id} not found in manifest, skipping")
            continue

        # evenly sample n_frames
        indices = np.linspace(0, len(run_df) - 1, args.n_frames, dtype=int)
        sampled = run_df.iloc[indices]

        frames_info = []
        for _, row in sampled.iterrows():
            img_path = IMAGES_ROOT / row["image_path"]
            frames_info.append((img_path, int(row["tendon_type"]), int(row["frame_idx"])))

        label = int(run_df["tendon_type"].iloc[0])
        title = f"{run_id}  |  label={label} ({LABEL_NAMES.get(label,'?')})  |  {len(run_df)} total frames  |  crop={args.crop}px"
        grid = make_grid(frames_info, args.crop, title)

        out_path = save_dir / f"{run_id}.jpg"
        cv2.imwrite(str(out_path), grid)
        print(f"Saved: {out_path}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
