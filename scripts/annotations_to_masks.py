import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

# --- Frame ranges to include ---
frames = [
    [0, 268],
    [286, 295],
    [354, 440],
    [460, 597],
    [641, 692],
    [703, 756],
    [764, 812],
]

# --- Paths ---
video_path = "/home/apasco/cs230/cs230-proj/data/data-2025-10-09.mp4"
xml_path = "/home/apasco/cs230/cs230-proj/data/annotations-2025-10-09-v2.xml"
images_dir = "/home/apasco/cs230/cs230-proj/data/images"
masks_dir = "/home/apasco/cs230/cs230-proj/data/masks"

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# --- Load XML ---
tree = ET.parse(xml_path)
root = tree.getroot()

# Build mapping: image_name -> list of annotations
frame_annotations = {}

for image_tag in root.findall("image"):
    img_name = image_tag.attrib["name"]
    annos = []

    # Ellipses
    for ellipse in image_tag.findall("ellipse"):
        cx = float(ellipse.attrib["cx"])
        cy = float(ellipse.attrib["cy"])
        rx = float(ellipse.attrib["rx"])
        ry = float(ellipse.attrib["ry"])
        label = ellipse.attrib["label"]
        annos.append(
            {"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry, "label": label}
        )

    # Polygons
    for poly in image_tag.findall("polygon"):
        label = poly.attrib["label"]
        points_str = poly.attrib["points"]
        points = np.array(
            [
                [int(float(x)), int(float(y))]
                for x, y in (p.split(",") for p in points_str.split(";"))
            ],
            np.int32,
        )
        annos.append({"type": "polygon", "points": points, "label": label})

    if annos:
        frame_annotations[img_name] = annos

print(f"Found {len(frame_annotations)} annotated frames.")

# --- Label mapping ---
class_labels = sorted(
    {anno["label"] for annos in frame_annotations.values() for anno in annos}
)
class_map = {label: i + 1 for i, label in enumerate(class_labels)}  # background=0
print("Class map:", class_map)

# --- Precompute allowed frame indices ---
allowed_frames = set()
for start, end in frames:
    allowed_frames.update(range(start, end + 1))

# --- Open video ---
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx not in allowed_frames:
        frame_idx += 1
        continue

    frame_name = f"frame_{frame_idx:06d}"

    if frame_name in frame_annotations:
        rgb_path = os.path.join(images_dir, f"{frame_name}.png")
        cv2.imwrite(rgb_path, frame)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        ellipses = [a for a in frame_annotations[frame_name] if a["type"] == "ellipse"]
        polys = [a for a in frame_annotations[frame_name] if a["type"] == "polygon"]

        # If there are two ellipses, remove the smaller (inner) one
        if len(ellipses) == 2:
            areas = [e["rx"] * e["ry"] for e in ellipses]
            outer_idx = int(np.argmax(areas))
            inner_idx = 1 - outer_idx
            outer = ellipses[outer_idx]
            inner = ellipses[inner_idx]
            label = outer["label"]

            # Draw outer ellipse
            cv2.ellipse(
                mask,
                (int(round(outer["cx"])), int(round(outer["cy"]))),
                (int(round(outer["rx"])), int(round(outer["ry"]))),
                0,
                0,
                360,
                class_map[label],
                -1,
            )
            # Cut out the inner ellipse (set to background)
            cv2.ellipse(
                mask,
                (int(round(inner["cx"])), int(round(inner["cy"]))),
                (int(round(inner["rx"])), int(round(inner["ry"]))),
                0,
                0,
                360,
                0,
                -1,
            )
        else:
            # Draw single ellipse (normal case)
            for e in ellipses:
                cv2.ellipse(
                    mask,
                    (int(round(e["cx"])), int(round(e["cy"]))),
                    (int(round(e["rx"])), int(round(e["ry"]))),
                    0,
                    0,
                    360,
                    class_map[e["label"]],
                    -1,
                )

        # Draw any polygons
        for p in polys:
            cv2.fillPoly(mask, [p["points"]], class_map[p["label"]])

        mask_path = os.path.join(masks_dir, f"{frame_name}.png")
        Image.fromarray(mask).save(mask_path)

        print(f"[{frame_idx}] Saved {rgb_path} and {mask_path}")

    frame_idx += 1

cap.release()
print("Done!")
