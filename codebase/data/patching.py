import os
import cv2
import numpy as np
import re
from typing import Optional, Tuple, List
from codebase.data.cropping import crop_fabric_region

class NODefectPatchCreator:
    def __init__(
            self,
            input_folder: str,
            output_folder: str,
            patch_size: Tuple[int, int] = (256, 256),
            prefix: str = "patch",
            exclude_fabric_codes: Optional[List[str]] = None,
    ):
        """
        Initializes the PatchCreator with paths, patch size, filename prefix, and optional exclusions.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.patch_size = patch_size
        self.prefix = prefix
        self.exclude_fabric_codes = exclude_fabric_codes or []

    def list_image_files_recursively(self, folder: str) -> List[str]:
        """
        Recursively searches for all image files in the specified folder and subfolders.
        """
        image_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        return image_files

    def extract_fabric_code(self, filename: str) -> str:
        """
        Extracts the fabric code from filenames in the format: nnnn_ddd_ff.png
        """
        match = re.match(r"^\d+_\d+_(\d+)\.png$", filename)
        if match:
            return match.group(1).zfill(5)
        else:
            return "xxxxx"

    def extract_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Splits an image into non-overlapping patches of the specified size.
        """
        patch_h, patch_w = self.patch_size
        img_h, img_w = image.shape
        patches = []
        for y in range(0, img_h - patch_h + 1, patch_h):
            for x in range(0, img_w - patch_w + 1, patch_w):
                patch = image[y:y + patch_h, x:x + patch_w]
                patches.append(patch)
        return patches

    def process_and_save_patches(self):
        """
        Processes all images in the input folder:
        - Crops the relevant fabric region
        - Skips excluded fabric codes
        - Extracts and saves non-overlapping patches
        """
        os.makedirs(self.output_folder, exist_ok=True)
        image_paths = self.list_image_files_recursively(self.input_folder)
        patch_counter = 0

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            fabric_code = self.extract_fabric_code(filename)

            if fabric_code in self.exclude_fabric_codes:
                print(f"[Skipped] Excluding fabric code: {fabric_code} ({filename})")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping unreadable file: {img_path}")
                continue

            relative_path = os.path.relpath(img_path, self.input_folder)
            relative_folder = os.path.dirname(relative_path)
            full_output_folder = os.path.join(self.output_folder, relative_folder)
            os.makedirs(full_output_folder, exist_ok=True)

            cropped = crop_fabric_region(img, crop_offset=100)
            patches = self.extract_patches(cropped)

            for patch in patches:
                patch_filename = f"{self.prefix}_{fabric_code}_{patch_counter:05d}.png"
                save_path = os.path.join(full_output_folder, patch_filename)
                cv2.imwrite(save_path, patch)
                patch_counter += 1

        print(f"Saved {patch_counter} patches to {self.output_folder}")


def extract_patches(image: np.ndarray, patch_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    Splits image into non-overlapping patches of given size. 
    Skips any patches that don't fit fully.

    Args:
        image (np.ndarray): Cropped grayscale image
        patch_size (Tuple[int, int]): (height, width) of patches

    Returns:
        List[np.ndarray]: List of patches
    """
    patch_h, patch_w = patch_size
    img_h, img_w = image.shape

    patches = []
    for y in range(0, img_h - patch_h + 1, patch_h):
        for x in range(0, img_w - patch_w + 1, patch_w):
            patch = image[y:y + patch_h, x:x + patch_w]
            patches.append(patch)
    return patches

def list_image_files_recursively(folder: str) -> List[str]:
    """
    Recursively finds all image files in the given folder and subfolders.

    Args:
        folder (str): Root folder path.

    Returns:
        List[str]: List of full file paths.
    """
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

import os
import cv2
from typing import Tuple
from pathlib import Path
import re

def extract_fabric_code(filename: str) -> str:
    """
    Extracts the fabric code (ff) from filenames in the format: nnnn_ddd_ff.png
    """
    match = re.match(r"^\d+_\d+_(\d+)\.png$", filename)
    if match:
        return match.group(1).zfill(5)
    else:
        return "xxxxx"  # fallback if pattern doesn't match

def process_and_save_patches(
        input_folder: str,
        output_folder: str,
        patch_size: Tuple[int, int] = (256, 256),
        prefix: str = "patch"
):
    """
    Processes all images in a folder (including subfolders): crops and tiles them into patches,
    saves them into a mirrored folder structure, and preserves fabric codes in filenames.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = list_image_files_recursively(input_folder)

    patch_counter = 0
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        # Get relative subfolder path
        relative_path = os.path.relpath(img_path, input_folder)
        relative_folder = os.path.dirname(relative_path)
        full_output_folder = os.path.join(output_folder, relative_folder)
        os.makedirs(full_output_folder, exist_ok=True)

        # Extract fabric code from filename
        filename = os.path.basename(img_path)
        fabric_code = extract_fabric_code(filename)

        # Crop and patch
        cropped = crop_fabric_region(img, crop_offset=100)
        patches = extract_patches(cropped, patch_size)

        for i, patch in enumerate(patches):
            patch_filename = f"{prefix}_{fabric_code}_{patch_counter:05d}.png"
            save_path = os.path.join(full_output_folder, patch_filename)
            cv2.imwrite(save_path, patch)
            patch_counter += 1

    print(f"Saved {patch_counter} patches to {output_folder}")
