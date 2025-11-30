"""
Batch annotation script for labeling tendon normals across multiple images.

This script:
1. Loads a reference image
2. Iterates through contact images
3. Computes difference images
4. Allows user to annotate each image's tendon axis
5. Skips images if data is missing or user chooses to skip
6. Saves all normal maps and metadata
7. Generates a summary report
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from pathlib import Path
from scipy.ndimage import gaussian_filter
from label_tendon_normals import label_tendon_normals, normalize_vector
from calculate_difference import resize_to_target


class BatchAnnotator:
    def __init__(self,
                 images_dir="images_to_label",
                 ref_image_path="I_ref.png",
                 output_dir="normal_maps",
                 target_shape=(1080, 1920, 3),
                 gaussian_sigma=None):
        """
        Initialize the batch annotator.

        Args:
            images_dir: Directory containing contact images
            ref_image_path: Path to reference image
            output_dir: Directory to save normal maps and metadata
            target_shape: Target image shape (H, W, C)
            gaussian_sigma: Gaussian filter sigma (None=skip for speed, 0=raw diff, num=apply filter)
        """
        self.images_dir = images_dir
        self.ref_image_path = ref_image_path
        self.output_dir = output_dir
        self.target_shape = target_shape
        self.gaussian_sigma = gaussian_sigma

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Load reference image
        self.I_ref = self._load_and_resize_image(ref_image_path)
        if self.I_ref is None:
            raise FileNotFoundError(f"Reference image not found: {ref_image_path}")

        print(f"Reference image loaded: {ref_image_path}")
        print(f"Reference image shape: {self.I_ref.shape}")

        # Initialize tracking
        self.processed = []
        self.skipped = []
        self.failed = []

    def _load_and_resize_image(self, image_path):
        """Load and resize a single image."""
        try:
            if not os.path.exists(image_path):
                return None
            img = np.array(Image.open(image_path)).astype(np.float32) / 255.0
            return resize_to_target(img, target_shape=self.target_shape)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _compute_delta_I_clean(self, I_contact, sigma=None):
        """
        Compute difference image (optionally with Gaussian smoothing).

        Args:
            I_contact: Contact image array
            sigma: Gaussian filter sigma for smoothing. If None, skip smoothing.
                   Set sigma=0 to use raw difference only.

        Returns:
            delta_I_clean: Difference image (with or without smoothing)
        """
        delta_I = I_contact - self.I_ref

        # Skip Gaussian filtering if sigma is None (for speed)
        if sigma is None or sigma == 0:
            return delta_I

        # Apply Gaussian filter to separate low and high frequency
        delta_I_smooth = gaussian_filter(delta_I, sigma=sigma)
        delta_I_clean = delta_I - delta_I_smooth

        return delta_I_clean

    def get_image_list(self):
        """Get list of images to process from the images directory."""
        if not os.path.isdir(self.images_dir):
            print(f"Warning: Images directory not found: {self.images_dir}")
            return []

        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = []

        for file in sorted(os.listdir(self.images_dir)):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(file)

        print(f"Found {len(image_files)} images in {self.images_dir}")
        return image_files

    def process_image(self, image_filename, skip_existing=True):
        """
        Process a single image.

        Args:
            image_filename: Filename of the contact image
            skip_existing: If True, skip if normal map already exists

        Returns:
            (success, image_id, normals, metadata)
        """
        image_path = os.path.join(self.images_dir, image_filename)
        image_id = Path(image_filename).stem

        # Check if already processed
        npy_path = os.path.join(self.output_dir, f"{image_id}_normals.npy")
        if skip_existing and os.path.exists(npy_path):
            print(f"\n⊘ Skipping {image_id} (already processed)")
            self.skipped.append(image_id)
            return False, image_id, None, None

        # Load contact image
        I_contact = self._load_and_resize_image(image_path)
        if I_contact is None:
            print(f"\n✗ Failed to load {image_id}")
            self.failed.append((image_id, "Failed to load image"))
            return False, image_id, None, None

        print(f"\n{'='*60}")
        print(f"Processing: {image_id}")
        print(f"{'='*60}")

        # Compute difference
        try:
            if self.gaussian_sigma is None:
                print(f"Computing difference (no Gaussian filter for speed)")
            else:
                print(f"Computing difference (Gaussian sigma={self.gaussian_sigma})")
            delta_I_clean = self._compute_delta_I_clean(I_contact, sigma=self.gaussian_sigma)
        except Exception as e:
            print(f"✗ Failed to compute difference for {image_id}: {e}")
            self.failed.append((image_id, f"Difference computation failed: {e}"))
            return False, image_id, None, None

        # Annotate (visualize=False for speed)
        try:
            normals, metadata = label_tendon_normals(image_id, delta_I_clean, visualize=False)
            print(f"✓ Successfully processed {image_id}")
            self.processed.append(image_id)
            return True, image_id, normals, metadata
        except KeyboardInterrupt:
            print(f"\n⊘ Skipped by user: {image_id}")
            self.skipped.append(image_id)
            return False, image_id, None, None
        except Exception as e:
            print(f"\n✗ Error processing {image_id}: {e}")
            self.failed.append((image_id, str(e)))
            return False, image_id, None, None

    def ask_user_continue(self, current_idx, total):
        """
        Prompt user to continue (default is to continue).
        Just press Enter to continue to next image, or type 'stop' to stop.
        """
        response = input(
            f"[{current_idx}/{total}] Press Enter for next, or 'stop': "
        ).strip().lower()

        if response in ['stop', 'n', 'no']:
            return 'stop'
        else:
            return 'continue'

    def run(self, skip_existing=True, interactive=True):
        """
        Run batch annotation on all images.

        Args:
            skip_existing: Skip images that already have normal maps
            interactive: Ask user after each newly processed image whether to continue
        """
        image_files = self.get_image_list()

        if not image_files:
            print("No images found to process!")
            return

        for idx, image_filename in enumerate(image_files, 1):
            # Check if already processed
            image_id = Path(image_filename).stem
            npy_path = os.path.join(self.output_dir, f"{image_id}_normals.npy")
            already_processed = os.path.exists(npy_path)

            # Process the image
            success, image_id, normals, metadata = self.process_image(
                image_filename, skip_existing=skip_existing
            )

            # Only ask to continue if we just processed a NEW image (not skipped)
            # Skip silently for already-processed images
            if interactive and success and idx < len(image_files):
                response = self.ask_user_continue(idx, len(image_files))
                if response == 'stop':
                    print("\nBatch annotation stopped by user.")
                    break

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print summary of batch processing."""
        total = len(self.processed) + len(self.skipped) + len(self.failed)

        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total images:     {total}")
        print(f"Processed:        {len(self.processed)} ✓")
        print(f"Skipped:          {len(self.skipped)} ⊘")
        print(f"Failed:           {len(self.failed)} ✗")
        print(f"{'='*60}\n")

        if self.processed:
            print("Processed:")
            for image_id in self.processed:
                print(f"  ✓ {image_id}")

        if self.skipped:
            print(f"\nSkipped:")
            for image_id in self.skipped:
                print(f"  ⊘ {image_id}")

        if self.failed:
            print(f"\nFailed:")
            for image_id, error in self.failed:
                print(f"  ✗ {image_id}: {error}")

        # Save summary report
        summary = {
            'total': total,
            'processed': self.processed,
            'skipped': self.skipped,
            'failed': self.failed
        }

        summary_path = os.path.join(self.output_dir, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary report saved to: {summary_path}")


def main():
    """Main entry point for batch annotation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch annotate tendon normals for multiple images"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images_to_label",
        help="Directory containing contact images (default: images)"
    )
    parser.add_argument(
        "--ref-image",
        type=str,
        default="I_ref.png",
        help="Path to reference image (default: I_ref.png)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="normal_maps",
        help="Directory to save normal maps (default: normal_maps)"
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=1080,
        help="Target image height (default: 1080)"
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=1920,
        help="Target image width (default: 1920)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip images that already have normal maps (default: True)"
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-process all images even if normal maps exist"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Process all images without asking user confirmation"
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=None,
        help="Gaussian filter sigma (default: None=skip for speed). Use 100 for smoothing."
    )

    args = parser.parse_args()

    # Create annotator
    try:
        annotator = BatchAnnotator(
            images_dir=args.images_dir,
            ref_image_path=args.ref_image,
            output_dir=args.output_dir,
            target_shape=(args.target_height, args.target_width, 3),
            gaussian_sigma=args.gaussian_sigma
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Run batch annotation
    annotator.run(
        skip_existing=args.skip_existing,
        interactive=not args.non_interactive
    )


if __name__ == "__main__":
    main()
