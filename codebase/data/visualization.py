import torchvision.utils as vutils

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.filters import threshold_otsu
import numpy as np

def visualize_anomalies_sorted_by_coverage(
        model,
        dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_visuals: int = 5,
        blur_kernel: int = 0
):
    """
    Visualizes anomalous samples sorted by ground-truth anomaly area coverage (descending).
    Applies Otsu's method to determine a per-image threshold on the normalized reconstruction error map.

    Args:
        model: Trained VAE
        dataloader: DataLoader with batches of (images, masks, labels, ...)
        device: 'cuda' or 'cpu'
        max_visuals: Max number of examples to display
        blur_kernel: If >0, apply average pooling to smooth error map
    """
    model.eval()
    model.to(device)
    collected = []

    with torch.no_grad():
        for images, masks, labels, _, _ in tqdm(dataloader, desc="Collecting Anomalies"):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            recon, _, _ = model(images)
            error_map = (recon - images).pow(2)  # L2 error

            if blur_kernel > 0:
                error_map = F.avg_pool2d(error_map, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)

            # Normalize error maps per image
            norm_error_map = (error_map - error_map.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1))
            norm_error_map /= norm_error_map.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8

            for i in range(images.size(0)):
                if labels[i].item() != 1:
                    continue  # only show anomalous samples

                # Otsu thresholding
                err_np = norm_error_map[i, 0].cpu().numpy()
                try:
                    otsu_thresh = threshold_otsu(err_np)
                except ValueError:
                    otsu_thresh = 0.5  # fallback for constant images

                pred_mask = (err_np > otsu_thresh).astype(np.float32)

                anomaly_area = masks[i].sum().item()

                collected.append({
                    "original": images[i].detach().cpu(),
                    "reconstruction": recon[i].detach().cpu(),
                    "error_map": err_np,  # already on CPU
                    "pred_mask": pred_mask,
                    "true_mask": masks[i, 0].detach().cpu(),
                    "area": anomaly_area,
                    "otsu": otsu_thresh
                })

    # Sort by anomaly area (descending)
    collected.sort(key=lambda x: x["area"], reverse=True)

    for i, sample in enumerate(collected[:max_visuals]):
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))

        axes[0].imshow(sample["original"][0], cmap="gray")
        axes[0].set_title("Original")

        axes[1].imshow(sample["reconstruction"][0], cmap="gray")
        axes[1].set_title("Reconstruction")

        im = axes[2].imshow(sample["error_map"], cmap="hot", vmin=0, vmax=1)
        axes[2].set_title(f"Error Map\n(Otsu={sample['otsu']:.3f})")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(sample["pred_mask"], cmap="gray")
        axes[3].set_title("Predicted Mask")

        axes[4].imshow(sample["true_mask"], cmap="gray")
        axes[4].set_title("Ground Truth Mask")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

def visualize_anomalies_only(
        model,
        dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_visuals: int = 5,
        blur_kernel: int = 0  # Set >0 to apply average pooling
):
    """
    Visualizes only anomalous samples (label == 1) from a VAE using pixel-wise error maps.
    Uses Otsu's method for dynamic per-image thresholding.

    Args:
        model: Trained VAE
        dataloader: DataLoader with (images, masks, labels, ...)
        device: 'cuda' or 'cpu'
        max_visuals: Maximum number of samples to display
        blur_kernel: If >0, applies average pooling with this kernel size to smooth error maps
    """
    model.eval()
    model.to(device)
    shown = 0

    with torch.no_grad():
        for images, masks, labels, _, _ in tqdm(dataloader, desc="Filtering Anomalies"):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            recon, _, _ = model(images)
            error_map = (recon - images).pow(2)  # Use L2 error

            if blur_kernel > 0:
                error_map = F.avg_pool2d(error_map, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)

            # Normalize per-sample error maps to [0,1]
            norm_error_map = (error_map - error_map.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1))
            norm_error_map /= norm_error_map.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8

            for i in range(images.shape[0]):
                if labels[i].item() != 1:
                    continue  # skip normal samples

                # --- Otsu thresholding (on CPU) ---
                error_np = norm_error_map[i, 0].cpu().numpy()
                try:
                    otsu_thresh = threshold_otsu(error_np)
                except ValueError:
                    # Fallback in case Otsu fails (e.g., constant image)
                    otsu_thresh = 0.5
                
                pred_mask = (error_np > otsu_thresh).astype(np.float32)

                fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                axes[0].imshow(images[i, 0].cpu(), cmap="gray")
                axes[0].set_title("Original")

                axes[1].imshow(recon[i, 0].cpu(), cmap="gray")
                axes[1].set_title("Reconstruction")

                im = axes[2].imshow(error_np, cmap="hot", vmin=0, vmax=1)
                axes[2].set_title(f"Error Map\n(Otsu={otsu_thresh:.3f})")
                plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

                axes[3].imshow(pred_mask, cmap="gray")
                axes[3].set_title("Predicted Mask (Otsu)")

                axes[4].imshow(masks[i, 0].cpu(), cmap="gray")
                axes[4].set_title("Ground Truth Mask")

                for ax in axes:
                    ax.axis("off")
                plt.tight_layout()
                plt.show()

                shown += 1
                if shown >= max_visuals:
                    return

def visualize_error_overlay(
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        n: int = 4,
        alpha: float = 0.5
):
    """
    Visualizes original images with overlaid error heatmaps.

    Args:
        inputs (Tensor): Original images, shape (B, 1, H, W)
        reconstructions (Tensor): Reconstructed images, shape (B, 1, H, W)
        n (int): Number of examples to show
        alpha (float): Transparency of the heatmap overlay
    """
    inputs = inputs[:n].cpu()
    reconstructions = reconstructions[:n].cpu()
    errors = torch.abs(inputs - reconstructions)

    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axs = [axs]

    for i in range(n):
        base_img = inputs[i].squeeze().numpy()
        heatmap = errors[i].squeeze().numpy()

        axs[i].imshow(base_img, cmap='gray')
        axs[i].imshow(heatmap, cmap='hot', alpha=alpha)
        axs[i].set_title(f"Overlay {i+1}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_reconstruction_errors(
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        n: int = 4
):
    """
    Visualizes original images, reconstructions, and error heatmaps side by side.

    Args:
        inputs (Tensor): Original images, shape (B, 1, H, W)
        reconstructions (Tensor): Reconstructed images, shape (B, 1, H, W)
        n (int): Number of examples to show
    """
    inputs = inputs[:n].cpu()
    reconstructions = reconstructions[:n].cpu()
    errors = torch.abs(inputs - reconstructions)

    fig, axs = plt.subplots(n, 3, figsize=(10, 3 * n))

    for i in range(n):
        axs[i, 0].imshow(inputs[i].squeeze(), cmap='gray')
        axs[i, 0].set_title("Original")
        axs[i, 1].imshow(reconstructions[i].squeeze(), cmap='gray')
        axs[i, 1].set_title("Reconstruction")
        axs[i, 2].imshow(errors[i].squeeze(), cmap='hot')
        axs[i, 2].set_title("Error Heatmap")

        for ax in axs[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def show_test_batch(images, masks, labels, n=4, grayscale=True):
    """
    Visualize a batch of test images with their masks and binary labels.
    
    Args:
        images (Tensor): Batch of images (B, 1, H, W)
        masks (Tensor): Batch of masks (B, 1, H, W)
        labels (Tensor): Binary labels (B,)
        n (int): Number of examples to show
    """
    images = images[:n]
    masks = masks[:n]
    labels = labels[:n]

    fig, axs = plt.subplots(n, 2, figsize=(6, 3 * n))
    for i in range(n):
        if grayscale:
            axs[i, 0].imshow(images[i].squeeze().numpy(), cmap='gray')
            axs[i, 0].set_title(f"Image Patch [{labels[i].item()}]")
            axs[i, 1].imshow(masks[i].squeeze().numpy(), cmap='gray')
            axs[i, 1].set_title("Ground Truth Mask")
            for ax in axs[i]:
                ax.axis('off')
        else:
            img = images[i][0].cpu().numpy()  # take channel 0 (all 3 are the same if repeated)
            mask = masks[i][0].cpu().numpy()
    
            axs[i, 0].imshow(img, cmap='gray')
            axs[i, 0].set_title(f"Image Patch [{labels[i].item()}]")
            axs[i, 1].imshow(mask, cmap='gray')
            axs[i, 1].set_title("Ground Truth Mask")

        for ax in axs[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_train_batch(images, n=6):
    """
    Visualize a batch of training images.

    Args:
        images (Tensor): Batch of images (B, 1, H, W)
        n (int): Number of examples to show
    """
    images = images[:n]
    grid = vutils.make_grid(images, nrow=n, padding=2, normalize=True)
    npimg = grid.numpy()
    plt.figure(figsize=(n * 2, 2))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.title("Training Patches")
    plt.axis('off')
    plt.show()