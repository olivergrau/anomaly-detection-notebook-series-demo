from codebase.data.cropping import detect_crop_boundary
from codebase.data.patching import extract_patches, list_image_files_recursively
import cv2
import numpy as np
from typing import Tuple, List, Optional

class DefectPatchCreator:
    def __init__(
            self,
            defect_folder: str,
            mask_folder: str,
            image_output_folder: str,
            mask_output_folder: str,
            patch_size: Tuple[int, int] = (256, 256),
            prefix: str = "defect",
            fabric_codes_intact_image_path: Optional[str] = None,
            intact_fabric_codes: Optional[List[str]] = None,
    ):
        """
        Initializes the DefectPatchCreator with the corresponding folders, patch size, filename prefix,
        and additional parameters for handling intact patches.
        
        Existing behavior:
         - Processes defect images and their masks.
         - Saves defective patch pairs (image + mask) in the output directories.
        
        Additional functionality:
         - If a patch is intact (mask is all zeros) and its fabric code is in intact_fabric_codes,
           the image patch is additionally saved to fabric_codes_intact_image_path.
        """
        self.defect_folder = defect_folder
        self.mask_folder = mask_folder
        self.image_output_folder = image_output_folder
        self.mask_output_folder = mask_output_folder
        self.patch_size = patch_size
        self.prefix = prefix

        # New parameters for intact patches:
        self.fabric_codes_intact_image_path = fabric_codes_intact_image_path
        self.intact_fabric_codes = intact_fabric_codes or []

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
        Extracts the fabric code from filenames in the format: nnnn_ddd_ff.png.
        """
        match = re.match(r"^\d+_\d+_(\d+)\.png$", filename)
        if match:
            return match.group(1).zfill(5)
        else:
            return "xxxxx"  # Fallback code

    def find_corresponding_mask(self, image_path: str) -> str:
        """
        Finds the corresponding mask path for a defect image.
        Assumes the mask filename follows the pattern <basename>_mask.png.
        """
        rel_path = os.path.relpath(image_path, self.defect_folder)
        dir_part, file_part = os.path.split(rel_path)
        base_name, _ = os.path.splitext(file_part)
        mask_filename = f"{base_name}_mask.png"
        return os.path.join(self.mask_folder, dir_part, mask_filename)

    def process_and_save_patches(self):
        """
        Processes all defect images and their masks:
         - Loads the image and the corresponding mask.
         - Extracts the fabric code.
         - Crops both using detect_crop_boundary.
         - Splits the cropped images into patches.
         - For each patch:
            - If the patch is defective (mask contains any nonzero pixel), it is saved to the defect output folders.
            - If the patch is intact (mask is all zeros) and its fabric code is in intact_fabric_codes,
              then the patch image is additionally saved to fabric_codes_intact_image_path.
         - Saves patch pairs in a mirrored folder structure.
        """
        os.makedirs(self.image_output_folder, exist_ok=True)
        os.makedirs(self.mask_output_folder, exist_ok=True)

        # Create the additional intact images folder if provided.
        if self.fabric_codes_intact_image_path:
            os.makedirs(self.fabric_codes_intact_image_path, exist_ok=True)

        image_paths = self.list_image_files_recursively(self.defect_folder)
        patch_counter = 0

        for img_path in image_paths:
            mask_path = self.find_corresponding_mask(img_path)
            if not os.path.exists(mask_path):
                print(f"[!] No corresponding mask found for: {img_path}")
                continue

            # Load image and mask as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"[!] Skipping file because image or mask is unreadable: {img_path} or {mask_path}")
                continue

            # Extract fabric code from the filename
            filename = os.path.basename(img_path)
            fabric_code = self.extract_fabric_code(filename)

            # Crop image and mask using the detected boundary
            start_col = detect_crop_boundary(img)
            cropped_img = img[:, start_col + 100:]
            cropped_mask = mask[:, start_col + 100:]
            assert cropped_img.shape == cropped_mask.shape, f"Shape mismatch after crop: {img_path}"

            # Create patches from the cropped image and mask
            img_patches = extract_patches(cropped_img, self.patch_size)
            mask_patches = extract_patches(cropped_mask, self.patch_size)

            for i, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):                
                # Save defect patch pair as usual
                patch_id = f"{self.prefix}_{fabric_code}_{patch_counter:05d}.png"
                img_save_path = os.path.join(self.image_output_folder, patch_id)
                mask_save_path = os.path.join(self.mask_output_folder, patch_id)
                cv2.imwrite(img_save_path, img_patch)
                cv2.imwrite(mask_save_path, mask_patch)
                patch_counter += 1

                if (np.max(mask_patch) == 0):                
                    # Patch is intact. If the fabric code is in intact_fabric_codes and an intact output path is provided,
                    # save the intact image patch to that folder.
                    if (fabric_code in self.intact_fabric_codes) and self.fabric_codes_intact_image_path:
                        patch_id = f"normal_{fabric_code}_{patch_counter+2156:05d}.png"
                        intact_save_path = os.path.join(self.fabric_codes_intact_image_path, patch_id)
                        cv2.imwrite(intact_save_path, img_patch)
                        patch_counter += 1
                    # Otherwise, intact patches are already processed.

        print(f"Saved {patch_counter} patches (defect and intact) to:")
        print(f"  Defect Images: {self.image_output_folder}")
        print(f"  Defect Masks:  {self.mask_output_folder}")
        
        if self.fabric_codes_intact_image_path:
            print(f"  Intact Images for fabrics {self.intact_fabric_codes}: {self.fabric_codes_intact_image_path}")



def find_corresponding_mask(image_path: str, defect_root: str, mask_root: str) -> str:
    """
    Given a defect image path, find the corresponding mask path.
    The mask filename format is assumed to be: <basename>_mask.png

    Args:
        image_path (str): Full path to defect image
        defect_root (str): Root path of defect images
        mask_root (str): Root path of mask images

    Returns:
        str: Full path to corresponding mask image
    """
    rel_path = os.path.relpath(image_path, defect_root)
    dir_part, file_part = os.path.split(rel_path)
    base_name, _ = os.path.splitext(file_part)
    mask_filename = f"{base_name}_mask.png"

    return os.path.join(mask_root, dir_part, mask_filename)


import os
import cv2
from typing import Tuple
import re

def extract_fabric_code(filename: str) -> str:
    """
    Extracts the fabric code (ff) from filenames in the format: nnnn_ddd_ff.png
    """
    match = re.match(r"^\d+_\d+_(\d+)\.png$", filename)
    if match:
        return match.group(1).zfill(5)
    else:
        return "xxxxx"  # fallback code

def process_defect_images_with_masks(
        defect_folder: str,
        mask_folder: str,
        image_output_folder: str,
        mask_output_folder: str,
        patch_size: Tuple[int, int] = (256, 256),
        prefix: str = "defect"
):
    """
    Process defect images and their masks: crop both, tile both, and save aligned patches,
    preserving the fabric code in filenames.
    """
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)

    image_paths = list_image_files_recursively(defect_folder)
    patch_counter = 0

    for img_path in image_paths:
        mask_path = find_corresponding_mask(img_path, defect_folder, mask_folder)

        if not os.path.exists(mask_path):
            print(f"[!] No corresponding mask for: {img_path}")
            continue

        # Load image and mask
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"[!] Skipping unreadable file: {img_path} or {mask_path}")
            continue

        # Extract fabric code
        filename = os.path.basename(img_path)
        fabric_code = extract_fabric_code(filename)

        # Crop both image and mask
        start_col = detect_crop_boundary(img)
        cropped_img = img[:, start_col + 100:]
        cropped_mask = mask[:, start_col + 100:]

        assert cropped_img.shape == cropped_mask.shape, f"Shape mismatch after crop: {img_path}"

        # Extract and save patches
        img_patches = extract_patches(cropped_img, patch_size)
        mask_patches = extract_patches(cropped_mask, patch_size)

        for i, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
            patch_id = f"{prefix}_{fabric_code}_{patch_counter:05d}.png"
            img_save_path = os.path.join(image_output_folder, patch_id)
            mask_save_path = os.path.join(mask_output_folder, patch_id)

            cv2.imwrite(img_save_path, img_patch)
            cv2.imwrite(mask_save_path, mask_patch)
            patch_counter += 1

    print(f"Saved {patch_counter} image+mask patch pairs to:")
    print(f"- Images: {image_output_folder}")
    print(f"- Masks:  {mask_output_folder}")

