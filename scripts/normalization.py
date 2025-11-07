import cv2
import numpy as np
from tqdm import tqdm

video_path = "/home/apasco/cs230/cs230-proj/data/video_20250910_214944.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {video_path}")

means = []
stds = []

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {frame_count} frames...")
idx = 0
while True:
    print(f"Processing frame {idx+1}/{frame_count}", end="\r")
    ret, frame = cap.read()
    if not ret:
        break
    # Convert BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0  # normalize to [0,1]

    # Compute mean and std per channel
    means.append(frame.mean(axis=(0, 1)))
    stds.append(frame.std(axis=(0, 1)))

    idx += 1

cap.release()

# Average across all frames
mean_rgb = np.mean(means, axis=0)
std_rgb = np.mean(stds, axis=0)

print("Mean RGB:", mean_rgb)
print("Std RGB:", std_rgb)
