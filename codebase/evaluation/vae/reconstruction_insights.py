import torch
from matplotlib import pyplot as plt
import imageio
import os
import shutil

def clear_snapshot_folder(folder_path):
    """
    Clears all contents of the snapshot folder before training.
    If the folder doesn't exist, it is created.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # deletes entire folder and contents
    os.makedirs(folder_path, exist_ok=True)  # recreates the empty folder

def snapshot_inputs(dataloader, device, num_images=5):
    model_inputs = next(iter(dataloader))[:num_images].to(device)
    return model_inputs

# Call this after each epoch with the fixed input and model
def save_reconstructions_per_epoch(model, epoch, fixed_inputs, output_dir, device):
    import os
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(fixed_inputs.to(device))

    for i in range(fixed_inputs.size(0)):
        orig = fixed_inputs[i].cpu().squeeze().numpy()
        recon_img = recon[i].cpu().squeeze().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(orig, cmap='gray')
        axs[0].set_title("Original")
        axs[1].imshow(recon_img, cmap='gray')
        axs[1].set_title(f"Recon (Epoch {epoch})")
        for ax in axs: ax.axis('off')

        plt.tight_layout()
        filename = os.path.join(output_dir, f"epoch_{epoch:03d}_img_{i:02d}.png")
        plt.savefig(filename)
        plt.close()

def create_gif_from_reconstructions(output_dir, img_index=0, save_path="recon_animation.gif", duration=0.8):
    files = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f"_recon_{img_index:02d}.png" in f
    ])
    images = [imageio.imread(f) for f in files]
    imageio.mimsave(save_path, images, duration=duration)
