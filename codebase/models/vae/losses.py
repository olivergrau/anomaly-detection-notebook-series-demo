import torch
import torch.nn.functional as F
from pytorch_msssim import ssim  # Ensure this package is installed

# ----------------------------------------------------------------
# Classic spatial-domain VAE loss
# ----------------------------------------------------------------
def vae_loss(recon_x, x, mu, logvar, kl_weight=0.0001):
    mse_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return mse_loss + kl_weight * kl_loss, mse_loss, kl_loss

# ----------------------------------------------------------------
# Frequency-domain VAE loss (real part only)
# ----------------------------------------------------------------
def frequency_vae_loss(recon_x, x, mu, logvar, kl_weight=1.0):
    # Compute real part of the 2D FFT
    freq_x = torch.fft.fftn(x, dim=(2, 3)).real
    freq_recon = torch.fft.fftn(recon_x, dim=(2, 3)).real
    
    # Take absolute value of the real parts
    mag_x = freq_x.abs()
    mag_recon = freq_recon.abs()

    # MSE on real-valued magnitudes
    freq_loss = F.mse_loss(mag_recon, mag_x, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    total_loss = freq_loss + kl_weight * kl_loss
    return total_loss, freq_loss, kl_loss

# ----------------------------------------------------------------
# Log-scaled frequency-domain VAE loss (real part only)
# ----------------------------------------------------------------
def log_scaled_frequency_vae_loss(recon_x, x, mu, logvar, kl_weight=0.1):
    # Real part of FFT
    freq_x = torch.fft.fftn(x, dim=(2, 3)).real
    freq_recon = torch.fft.fftn(recon_x, dim=(2, 3)).real

    # Log-scaled magnitudes
    log_mag_x = torch.log1p(freq_x.abs())
    log_mag_recon = torch.log1p(freq_recon.abs())

    freq_loss = F.mse_loss(log_mag_recon, log_mag_x, reduction='mean')

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    total_loss = freq_loss + kl_weight * kl_loss
    return total_loss, freq_loss, kl_loss

# ----------------------------------------------------------------
# Log-scaled hybrid VAE loss (real part only)
# ----------------------------------------------------------------
def log_scaled_hybrid_vae_loss(recon_x, x, mu, logvar, kl_weight=1.0, alpha=0.5):
    # Real part of FFT
    freq_x = torch.fft.fftn(x, dim=(2, 3)).real
    freq_recon = torch.fft.fftn(recon_x, dim=(2, 3)).real
    
    # Log magnitude (real part only)
    log_mag_x = torch.log1p(freq_x.abs())
    log_mag_recon = torch.log1p(freq_recon.abs())
    freq_loss = F.mse_loss(log_mag_recon, log_mag_x, reduction='mean')

    # Spatial MSE
    mse_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Weighted combination
    total = alpha * freq_loss + (1 - alpha) * mse_loss + kl_weight * kl_loss
    return total, freq_loss, mse_loss, kl_loss

# ----------------------------------------------------------------
# Hybrid VAE loss (real part only)
# ----------------------------------------------------------------
def hybrid_vae_loss(recon_x, x, mu, logvar, kl_weight=1.0, alpha=0.5):
    # Real part of FFT
    freq_x = torch.fft.fftn(x, dim=(2, 3)).real
    freq_recon = torch.fft.fftn(recon_x, dim=(2, 3)).real
    mag_x = freq_x.abs()
    mag_recon = freq_recon.abs()
    
    # Frequency domain MSE
    freq_loss = F.mse_loss(mag_recon, mag_x, reduction='mean')

    # Spatial domain
    mse_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Combined
    total = alpha * freq_loss + (1 - alpha) * mse_loss + kl_weight * kl_loss
    return total, freq_loss, mse_loss, kl_loss

# ----------------------------------------------------------------
# Log Scaled Hybrid Spatial VAE loss (real part only)
# ----------------------------------------------------------------
def log_scaled_hybrid_spatial_vae_loss(recon_x, x, mu, logvar, kl_weight=1.0, alpha=0.5, beta=0.5):
    """
    Hybrid VAE loss combining:
      - Log-scaled frequency domain loss (MSE on log-magnitude of real FFT)
      - Spatial domain loss (weighted combination of MSE and SSIM)
      - KL divergence loss
    """
    # Real part of FFT
    freq_x = torch.fft.fftn(x, dim=(2, 3)).real
    freq_recon = torch.fft.fftn(recon_x, dim=(2, 3)).real

    # Log-scaled magnitudes
    log_mag_x = torch.log1p(freq_x.abs())  # log(1 + |Re(FFT(x))|)
    log_mag_recon = torch.log1p(freq_recon.abs())

    freq_loss = F.mse_loss(log_mag_recon, log_mag_x, reduction='mean')

    # Spatial domain losses
    mse_loss = F.mse_loss(recon_x, x, reduction='mean')
    ssim_loss = 1 - ssim(recon_x, x, data_range=x.max() - x.min(), size_average=True)
    spatial_loss = beta * mse_loss + (1 - beta) * ssim_loss

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Weighted hybrid loss
    total = alpha * freq_loss + (1 - alpha) * spatial_loss + kl_weight * kl_loss
    return total, freq_loss, spatial_loss, kl_loss

# ----------------------------------------------------------------
# Hybrid Spatial VAE loss (real part only)
# ----------------------------------------------------------------
def hybrid_spatial_vae_loss(recon_x, x, mu, logvar, kl_weight=1.0, alpha=0.5, beta=0.5):
    """
    Hybrid VAE loss combining:
      - Frequency domain loss (MSE on real-part FFT)
      - Spatial domain loss (weighted combination of MSE and SSIM)
      - KL divergence loss
    """
    # Real part of FFT
    freq_x = torch.fft.fftn(x, dim=(2, 3)).real
    freq_recon = torch.fft.fftn(recon_x, dim=(2, 3)).real
    
    mag_x = freq_x.abs()
    mag_recon = freq_recon.abs()
    freq_loss = F.mse_loss(mag_recon, mag_x, reduction='mean')

    # Spatial domain
    mse_loss = F.mse_loss(recon_x, x, reduction='mean')
    ssim_loss = 1 - ssim(recon_x, x, data_range=x.max() - x.min(), size_average=True)
    spatial_loss = beta * mse_loss + (1 - beta) * ssim_loss

    # KL
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Weighted combination
    total = alpha * freq_loss + (1 - alpha) * spatial_loss + kl_weight * kl_loss
    return total, freq_loss, mse_loss, ssim_loss, kl_loss