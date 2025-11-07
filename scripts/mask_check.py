import cv2
import matplotlib.pyplot as plt
import numpy as np

mask_path = (
    "/home/apasco/cs230/cs230-proj/data/masks/frame_000056.png"  # pick any GT mask
)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # keep original values

# Convert pixels=1 to 255 for visibility
display_mask = (mask > 0).astype(np.uint8) * 255

plt.figure(figsize=(8, 6))
plt.imshow(display_mask, cmap="gray")
plt.title("Ground Truth Mask (scaled)")
plt.axis("off")
plt.show()
