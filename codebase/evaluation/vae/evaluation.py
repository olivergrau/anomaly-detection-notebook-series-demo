import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import jaccard_score

def visualize_anomalies_only(
        model,
        dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.4,
        max_visuals: int = 5,
        blur_kernel: int = 0  # Set >0 to apply average pooling
):
    """
    Visualizes only anomalous samples (label == 1) from a VAE using pixel-wise error maps.

    Args:
        model: Trained VAE
        dataloader: DataLoader with (images, masks, labels, ...)
        device: 'cuda' or 'cpu'
        threshold: Threshold for predicted binary mask
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
            #error_map = torch.abs(recon - images)
            error_map = (recon - images).pow(2)

            if blur_kernel > 0:
                error_map = F.avg_pool2d(error_map, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)

            # Normalize per-sample error maps to [0,1]
            norm_error_map = (error_map - error_map.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1))
            norm_error_map /= norm_error_map.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8

            pred_masks = (norm_error_map > threshold).float()

            for i in range(images.shape[0]):
                if labels[i].item() != 1:
                    continue  # skip normal samples

                fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                axes[0].imshow(images[i, 0].cpu(), cmap="gray")
                axes[0].set_title("Original")

                axes[1].imshow(recon[i, 0].cpu(), cmap="gray")
                axes[1].set_title("Reconstruction")

                axes[2].imshow(norm_error_map[i, 0].cpu(), cmap="hot")
                axes[2].set_title("Error Map")

                axes[3].imshow(pred_masks[i, 0].cpu(), cmap="gray")
                axes[3].set_title("Predicted Mask")

                axes[4].imshow(masks[i, 0].cpu(), cmap="gray")
                axes[4].set_title("Ground Truth Mask")

                for ax in axes:
                    ax.axis("off")
                plt.tight_layout()
                plt.show()

                shown += 1
                if shown >= max_visuals:
                    return

def evaluate_pixelwise_iou(
        model,
        dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5,
        show_visuals: bool = False,
        max_visuals: int = 5
) -> Tuple[float, List[float]]:
    """
    Evaluate a VAE using pixel-wise error maps, thresholding, and IoU with ground-truth masks.

    Args:
        model: Trained VAE
        dataloader: test DataLoader providing (images, masks, labels, ...)
        threshold: Threshold for binary mask generation from error map
        show_visuals: Whether to visualize outputs
        max_visuals: How many examples to visualize

    Returns:
        mean_iou: Mean IoU across all samples
        all_ious: List of individual IoU scores
    """
    model.eval()
    model.to(device)

    all_ious = []
    visualized = 0

    with torch.no_grad():
        for images, masks, _, _, _ in tqdm(dataloader, desc="Pixel-wise Evaluation"):
            images = images.to(device)
            masks = masks.to(device)

            recon, _, _ = model(images)

            # Error map: per-pixel absolute error
            error_map = torch.abs(recon - images)  # shape [B, 1, H, W]

            # Normalize each error map to [0, 1]
            norm_error_map = (error_map - error_map.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0])
            norm_error_map /= norm_error_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8

            # Threshold to get binary predicted masks
            pred_masks = (norm_error_map > threshold).float()

            # Compute IoU per sample
            for i in range(images.shape[0]):
                pred = pred_masks[i, 0].cpu().numpy().astype(np.uint8).flatten()
                true = masks[i, 0].cpu().numpy().astype(np.uint8).flatten()

                iou = jaccard_score(true, pred, zero_division=0)
                all_ious.append(iou)

                if show_visuals and visualized < max_visuals:
                    visualized += 1
                    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                    axes[0].imshow(images[i, 0].cpu(), cmap="gray")
                    axes[0].set_title("Original")
                    axes[1].imshow(recon[i, 0].cpu(), cmap="gray")
                    axes[1].set_title("Reconstruction")
                    axes[2].imshow(error_map[i, 0].cpu(), cmap="hot")
                    axes[2].set_title("Error Map")
                    axes[3].imshow(pred_masks[i, 0].cpu(), cmap="gray")
                    axes[3].set_title("Predicted Mask")
                    axes[4].imshow(masks[i, 0].cpu(), cmap="gray")
                    axes[4].set_title("GT Mask")
                    for ax in axes:
                        ax.axis("off")
                    plt.tight_layout()
                    plt.show()

    mean_iou = np.mean(all_ious)
    print(f"✅ Mean IoU: {mean_iou:.4f}")
    return mean_iou, all_ious

def evaluate_model(
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        postprocess: bool = False,
        show_progress: bool = True
) -> Tuple[List[float], List[int]]:
    """
    Evaluates a trained autoencoder on a test set with ground-truth masks.
    Returns patch-wise anomaly scores and ground-truth binary labels.
    """
    model.eval()
    model.to(device)

    errors = []
    labels = []

    data_iter = tqdm(dataloader, desc="Evaluating") if show_progress else dataloader

    with torch.no_grad():
        for images, masks, patch_labels, _, _ in data_iter:
            images = images.to(device)
            recon, _, _ = model(images)

            if postprocess:
                error_map = torch.abs(recon - images)
                if error_map.shape[1] > 1:
                    error_map = error_map.mean(dim=1, keepdim=True)

                error_map_blurred = F.avg_pool2d(error_map, kernel_size=7, stride=1, padding=3)

                patch_scores = []
                for i in range(error_map_blurred.shape[0]):
                    single_map = error_map_blurred[i, 0]
                    min_val = single_map.min()
                    max_val = single_map.max()
                    normalized_map = (single_map - min_val) / (max_val - min_val + 1e-8)
                    patch_scores.append(normalized_map.mean().item())

                errors.extend(patch_scores)
            else:
                batch_errors = torch.mean((recon - images) ** 2, dim=[1, 2, 3])
                errors.extend(batch_errors.cpu().numpy())

            labels.extend(patch_labels.cpu().numpy())

    model.train()
    return errors, labels

def evaluate_scores(errors: List[float], labels: List[int], threshold: float = None):
    """
    Computes and prints ROC AUC, precision, recall, F1 score, and optionally plots the ROC curve.
    If no threshold is provided, one is chosen based on the 95th percentile of normal errors.

    Args:
        errors (List[float]): Anomaly scores per patch
        labels (List[int]): Ground truth binary labels (0 = normal, 1 = defective)
        threshold (float, optional): Decision threshold. If None, determined automatically.
    """
    errors = np.array(errors)
    labels = np.array(labels)

    # ROC AUC
    roc_auc = roc_auc_score(labels, errors)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Determine threshold if not given (95th percentile of normal scores)
    if threshold is None:
        threshold = np.percentile(errors[labels == 0], 95)
        print(f"Auto-selected threshold (95th percentile of normal scores): {threshold:.4f}")
    else:
        print(f"Using provided threshold: {threshold:.4f}")

    # Predict anomalies using threshold
    preds = (errors > threshold).astype(int)

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(labels, errors)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
