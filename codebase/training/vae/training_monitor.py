from typing import Optional
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

class VAETrainingMonitor:
    def __init__(
            self,
            snapshot_dir: str,
            fixed_inputs: torch.Tensor,
            device: torch.device,
            log_fft: bool = True,
            save_every: int = 1,
            number_recons: int = 5,
            number_ffts: int = 5
    ):
        """
        Initializes the monitor.

        Args:
            snapshot_dir (str): Directory to save visual outputs.
            fixed_inputs (torch.Tensor): A batch of fixed input images (N, 1, H, W).
            device (torch.device): Device to run inference.
            log_fft (bool): Whether to save FFT visualizations.
            save_every (int): Save every N epochs.
        """
        self.snapshot_dir = snapshot_dir
        self.fixed_inputs = fixed_inputs.to(device)
        self.device = device
        self.log_fft = log_fft
        self.save_every = save_every
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.loss_log = []
        self.number_recons = number_recons
        self.number_ffts = number_ffts

    def after_epoch(self, model, epoch: int, loss_value: float):
        """
        Call this at the end of each epoch.

        Args:
            model: Trained VAE model.
            epoch (int): Current epoch number.
            loss_value (float): Average loss for this epoch.
        """
        self.loss_log.append(loss_value)

        if epoch % self.save_every != 0:
            return

        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(self.fixed_inputs)

        # Save reconstructions
        for i in range(min(len(self.fixed_inputs), self.number_recons)):
            orig = self.fixed_inputs[i].cpu().squeeze().numpy()
            recon_img = recon[i].cpu().squeeze().numpy()
            diff = np.abs(orig - recon_img)

            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            axs[0].imshow(orig, cmap='gray')
            axs[0].set_title("Original")
            axs[1].imshow(recon_img, cmap='gray')
            axs[1].set_title("Recon")
            axs[2].imshow(diff, cmap='hot')
            axs[2].set_title("Error")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            path = os.path.join(self.snapshot_dir, f"epoch_{epoch:03d}_recon_{i}.png")
            plt.savefig(path)
            plt.close()

        # Plot latent distributions
        mu_np = mu.cpu().numpy().flatten()
        sigma_np = np.exp(0.5 * logvar.cpu().numpy()).flatten()
        plt.hist(mu_np, bins=100, alpha=0.6, label='mu')
        plt.hist(sigma_np, bins=100, alpha=0.6, label='sigma')
        plt.legend()
        plt.title("Latent μ and σ Distributions")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.snapshot_dir, f"epoch_{epoch:03d}_latent.png"))
        plt.close()

        # FFT difference
        if self.log_fft:
            for i in range(min(len(self.fixed_inputs), self.number_ffts)):
                orig = self.fixed_inputs[i].cpu().squeeze()
                recon_img = recon[i].cpu().squeeze()

                fft_orig = torch.fft.fftshift(torch.fft.fft2(orig))
                fft_recon = torch.fft.fftshift(torch.fft.fft2(recon_img))

                mag_orig = torch.log1p(torch.abs(fft_orig)).numpy()
                mag_recon = torch.log1p(torch.abs(fft_recon)).numpy()
                diff = np.abs(mag_orig - mag_recon)

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(mag_orig, cmap='gray')
                axs[0].set_title("FFT Input")
                axs[1].imshow(mag_recon, cmap='gray')
                axs[1].set_title("FFT Recon")
                axs[2].imshow(diff, cmap='hot')
                axs[2].set_title("FFT Error")
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                fft_path = os.path.join(self.snapshot_dir, f"epoch_{epoch:03d}_fft_{i}.png")
                plt.savefig(fft_path)
                plt.close()
            
        model.train()

    def finalize(self):
        """
        Call after training is complete to save loss curve.
        """
        if self.loss_log:
            plt.plot(self.loss_log)
            plt.title("Training Loss Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.snapshot_dir, "loss_curve.png"))
            plt.close()
