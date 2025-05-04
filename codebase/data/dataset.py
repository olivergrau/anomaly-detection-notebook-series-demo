import os
from typing import Optional, Literal
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps  # Import ImageOps for histogram equalization
import numpy as np
import torch

class FabricPatchDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            mask_dir: Optional[str] = None,
            mode: Literal["train", "test"] = "train",
            normalize: Literal["zero-one", "z-norm"] = "zero-one",
            augment: bool = False,
            image_size: Optional[int] = None,
            fabric_code: Optional[str] = None,
            use_histogram_equalization: bool = False,
            repeat_3_channels: bool = False,
            use_fft_transform: bool = False,
            use_log1p_for_fft: bool = False,
    ):
        """
        PyTorch Dataset for anomaly detection patch images with fabric code filtering.
        
        Args:
            image_dir (str): Directory containing image patches.
            mask_dir (Optional[str]): Directory containing mask patches (required for test mode).
            mode (str): 'train' or 'test'.
            normalize (str): 'zero-one' or 'z-norm'.
            augment (bool): Apply augmentations in training.
            image_size (Optional[int]): Resize patches (assumes square).
            fabric_code (Optional[str]): Fabric code filter (e.g., "00042").
            use_histogram_equalization (bool): Whether to apply CLAHE.
            repeat_3_channels (bool): Repeat grayscale channels to get 3 channels.
            use_fft_transform (bool): If True, return an additional FFT magnitude spectrum of the image.
        """
        self.mode = mode
        self.normalize = normalize
        self.augment = augment
        self.image_size = image_size
        self.fabric_code = fabric_code
        self.use_histogram_equalization = use_histogram_equalization
        self.repeat_3_channels = repeat_3_channels
        self.use_fft_transform = use_fft_transform  # store FFT flag
        self.use_log1p_for_fft = use_log1p_for_fft  # store log1p flag
        
        print(f"Using FFT transform: {self.use_fft_transform}")

        # Load and sort all image files
        all_image_files = sorted([
            os.path.join(root, fname)
            for root, _, files in os.walk(image_dir)
            for fname in files
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Filter images based on fabric_code if provided
        if self.fabric_code is not None:
            self.image_paths = [
                p for p in all_image_files if f"_{self.fabric_code}_" in os.path.basename(p)
            ]
        else:
            self.image_paths = all_image_files

        if self.mode == 'test':
            assert mask_dir is not None, "mask_dir is required in test mode"
            all_mask_files = sorted([
                os.path.join(root, fname)
                for root, _, files in os.walk(mask_dir)
                for fname in files
                if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if self.fabric_code is not None:
                self.mask_paths = [
                    p for p in all_mask_files if f"_{self.fabric_code}_" in os.path.basename(p)
                ]
            else:
                self.mask_paths = all_mask_files
            assert len(self.image_paths) == len(self.mask_paths), "Mismatch in number of image and mask patches"

        self.transform_img = self._build_image_transforms()  # call the transform builder
        self.transform_mask = transforms.ToTensor()  # Only convert mask to tensor

    def _build_image_transforms(self):
        import numpy as np
        import cv2
        
        print(f"Using histogram equalization: {self.use_histogram_equalization}")
        print(f"Using image size: {self.image_size}")
        print(f"Using channel repeating: {self.repeat_3_channels}")
        print(f"Using normalization: {self.normalize}")
        print(f"Using augmentations: {self.augment}")

        def apply_clahe(pil_img):
            """Apply CLAHE to a PIL grayscale image using OpenCV"""
            img_np = np.array(pil_img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized_np = clahe.apply(img_np)
            return Image.fromarray(equalized_np)
    
        tfms = []
        
        if self.image_size:
            tfms.append(transforms.Resize((self.image_size, self.image_size)))

        tfms.append(transforms.Grayscale(num_output_channels=1))
        
        # Apply CLAHE if enabled
        if self.use_histogram_equalization:
            tfms.append(transforms.Lambda(apply_clahe))
            
        # Convert image to grayscale; if repeat_3_channels is true, repeat to get 3 channels.
        if self.repeat_3_channels:
            tfms.append(transforms.Grayscale(num_output_channels=3))
        
        # Convert image to tensor
        tfms.append(transforms.ToTensor())
        
        if self.normalize == 'z-norm':
            tfms.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    
        if self.augment and self.mode == 'train':
            aug = transforms.RandomApply([
                transforms.RandomHorizontalFlip(p=0.5),
                # Add other augmentations here if needed
            ], p=0.7)
            tfms = [aug] + tfms
    
        return transforms.Compose(tfms)

    def _fft_transform(self, tensor_img: torch.Tensor, apply_log: bool = True) -> torch.Tensor:
        """
        Computes the FFT magnitude spectrum from a tensor image.

        Args:
            tensor_img (Tensor): Image tensor of shape [C, H, W]. Supports grayscale ([1, H, W]) or RGB ([3, H, W]).
            apply_log (bool): If True, apply log1p scaling to enhance small magnitude details (useful for visualization).
                            If False, return raw magnitude spectrum normalized to [0, 1].

        Returns:
            Tensor: FFT magnitude spectrum of shape [C, H, W], normalized to [0, 1].
        """
        assert tensor_img.ndim == 3, f"Expected shape [C, H, W], got {tensor_img.shape}"
        C, H, W = tensor_img.shape
        fft_channels = []

        for c in range(C):
            channel = tensor_img[c]

            # Compute 2D FFT and shift zero-frequency component to center
            fft = torch.fft.fft2(channel)
            fft_shifted = torch.fft.fftshift(fft)

            # Compute magnitude
            magnitude = torch.abs(fft_shifted)

            if apply_log:
                magnitude = torch.log1p(magnitude)

            # Normalize to [0, 1]
            norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
            fft_channels.append(norm)

        return torch.stack(fft_channels, dim=0)  # shape: [C, H, W]



    # def _fft_transform(self, tensor_img: torch.Tensor) -> torch.Tensor:
    #     """
    #     Computes the FFT magnitude spectrum from a tensor image.
        
    #     Args:
    #         tensor_img (Tensor): Image tensor of shape [C, H, W]. Expects a grayscale image or 3-channel
    #                                image where the first channel is used.
                                   
    #     Returns:
    #         Tensor: FFT magnitude spectrum of shape [1, H, W], normalized to [0, 1].
    #     """
    #     # Take the first channel as the grayscale representation
    #     gray = tensor_img[0]  # shape: [H, W]

    #     # Compute the 2D FFT
    #     fft = torch.fft.fft2(gray)
        
    #     # Shift the zero-frequency component to the center
    #     fft_shifted = torch.fft.fftshift(fft)
        
    #     # Compute magnitude and apply log scaling
    #     magnitude = torch.abs(fft_shifted)
    #     magnitude = torch.log1p(magnitude)
        
    #     # Normalize to [0, 1]
    #     magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
    #     return magnitude.unsqueeze(0)  # [1, H, W]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image from file
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        
        # Apply image transformations
        x = self.transform_img(img)   # x will be a tensor, typically shape [C, H, W]
        
        # Optionally compute FFT spectra of the image after transform
        fft_spec = self._fft_transform(x, self.use_log1p_for_fft) if self.use_fft_transform else None

        if self.mode == 'train':
            # For training, return (image, fft) if enabled, else just image.
            if self.use_fft_transform:
                return fft_spec
            else:
                return x

        # For test mode: also load the mask and create a binary label.
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        y_mask = self.transform_mask(mask)  # stays in [0, 1]
        y_patch = int(y_mask.max() > 0)  # defect if any pixel > 0

        # Return tuple: image, mask, binary label, filename, index, and optionally fft_spec.
        if self.use_fft_transform:
            return fft_spec, y_mask, y_patch, os.path.basename(img_path), idx 
        else:
            return x, y_mask, y_patch, os.path.basename(img_path), idx