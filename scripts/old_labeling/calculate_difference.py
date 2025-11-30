import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def resize_to_target(img_array, target_shape=(1080, 1920, 3)):
    """
    Resize image to target shape if needed.

    Args:
        img_array: Input image as numpy array (H x W x C)
        target_shape: Target shape (H, W, C), default (1080, 1920, 3)

    Returns:
        Resized image array with target shape
    """
    current_shape = img_array.shape

    # If already the correct shape, return as-is
    if current_shape == target_shape:
        return img_array

    # Convert to PIL Image for resizing
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)

    pil_img = Image.fromarray(img_uint8)

    # Resize to target (W, H) format for PIL
    target_w, target_h = target_shape[1], target_shape[0]
    resized_pil = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Convert back to numpy array and original dtype
    resized_array = np.array(resized_pil)

    # Ensure it has 3 channels
    if len(resized_array.shape) == 2:  # Grayscale
        resized_array = np.stack([resized_array] * 3, axis=-1)

    # Convert back to original dtype
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        resized_array = resized_array.astype(np.float32) / 255.0

    print(f"Resized image from {current_shape} to {resized_array.shape}")
    return resized_array


def visualize_delta(contact_image_path, ref_path="I_ref.png"):
    """
    Visualize the difference between a reference image and a contact image.

    Args:
        contact_image_path: Path to the contact/current image
        ref_path: Path to the reference image (default "I_ref.png")
    """
    # Load reference image
    I_ref = np.array(Image.open(ref_path)).astype(np.float32) / 255.0  # H x W x 3, float [0,1]
    I_ref = resize_to_target(I_ref, target_shape=(1080, 1920, 3))

    # Load contact image
    I_contact = np.array(Image.open(contact_image_path)).astype(np.float32) / 255.0  # H x W x 3, float [0,1]
    I_contact = resize_to_target(I_contact, target_shape=(1080, 1920, 3))

    # Calculate difference
    delta_I = I_contact - I_ref                # H x W x 3, float

    # Apply low pass filter (Gaussian blur with sigma=40)
    sigma = 100
    delta_I_smooth = gaussian_filter(delta_I, sigma=sigma)  # Low frequency components
    delta_I_clean = delta_I - delta_I_smooth  # High frequency components (cleaned)

    # Calculate grayscale magnitude of the difference
    delta_mag = np.sqrt(delta_I[:, :, 0]**2 + delta_I[:, :, 1]**2 + delta_I[:, :, 2]**2)  # H x W
    delta_mag_clean = np.sqrt(delta_I_clean[:, :, 0]**2 + delta_I_clean[:, :, 1]**2 + delta_I_clean[:, :, 2]**2)  # H x W

    # Visualize magnitude as grayscale heatmap
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(I_contact)
    plt.title('Original')
    plt.colorbar()

    # Optionally show ΔI as RGB (shift to [0,1] for display)
    plt.subplot(1, 2, 2)
    plt.imshow(delta_I_clean * 0.5 + 0.5)           # Shift to [0,1] for display
    plt.title('RGB Difference (shifted)')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
