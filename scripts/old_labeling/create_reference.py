import numpy as np
from PIL import Image
import os
from pathlib import Path


def build_reference(image_dir, num_frames=10, save_path="I_ref.png"):
    """
    Build a reference image by averaging multiple pre-recorded frames.

    Args:
        image_dir: Path to directory containing pre-recorded images
        num_frames: Number of frames to use (default 10)
        save_path: Path to save the reference image (default "I_ref.png")

    Returns:
        I_ref: Reference image as numpy array (H x W x 3, float in [0,1])
    """
    # Get list of image files from directory (sorted by name)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not image_files:
        raise ValueError(f"No image files found in directory: {image_dir}")

    # Use only the first num_frames
    image_files = image_files[:num_frames]
    images = []

    for img_path in image_files:
        frame = np.array(Image.open(img_path))              # H x W x 3, uint8
        frame_float = frame / 255.0           # Convert to [0, 1]
        images.append(frame_float)

    # Calculate mean across all frames
    I_ref = np.mean(images, axis=0)           # H x W x 3, float

    # Save the reference image
    # Convert back to [0, 255] for image file
    I_ref_uint8 = (I_ref * 255).astype(np.uint8)
    Image.fromarray(I_ref_uint8).save(save_path)

    # Alternatively, you can save as numpy array:
    # np.save(save_path.replace('.png', '.npy'), I_ref)

    print(f"Reference image created from {len(image_files)} frames and saved to {save_path}")
    return I_ref
