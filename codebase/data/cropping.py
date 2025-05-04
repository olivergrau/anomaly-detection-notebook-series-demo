import numpy as np
import cv2

def detect_crop_boundary(img_array: np.ndarray, window: int = 10) -> int:
    """
    Detects the starting column index where fabric texture begins
    based on the largest intensity change (gradient) in column means.

    Args:
        img_array (np.ndarray): Grayscale image array (H x W)
        window (int): Smoothing window size

    Returns:
        int: Detected start column index for cropping
    """
    column_means = img_array.mean(axis=0)  # average per column (i.e., along height)
    smoothed = np.convolve(column_means, np.ones(window) / window, mode='valid')
    gradient = np.abs(np.gradient(smoothed))

    # Find position of the maximum change
    max_index = np.argmax(gradient)

    # Add window offset due to convolution shrinkage
    return max_index + window // 2


def crop_fabric_region(img_array: np.ndarray, crop_offset=100) -> np.ndarray:
    """
    Crop the left border of the fabric image dynamically using gradient-based detection.

    Args:
        img_array (np.ndarray): Input grayscale fabric image array

    Returns:
        np.ndarray: Cropped image array starting from fabric region
    """
    start_col = detect_crop_boundary(img_array)
    return img_array[:, start_col + crop_offset:]


def add_border(image: np.ndarray, border_size: int = 2, color: int = 0) -> np.ndarray:
    """
    Adds a border around the image.
    
    This is only for visualization purposes and does not affect the actual image data.
    """
    return cv2.copyMakeBorder(
        image, top=border_size, bottom=border_size, left=border_size, right=border_size,
        borderType=cv2.BORDER_CONSTANT, value=color
    )