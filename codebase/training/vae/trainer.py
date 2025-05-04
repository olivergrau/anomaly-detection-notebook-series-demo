import torch
from tqdm.notebook import tqdm
from codebase.evaluation.vae.evaluation import evaluate_model, evaluate_scores

from codebase.models.vae.losses import (
    frequency_vae_loss, 
    vae_loss, 
    log_scaled_frequency_vae_loss, 
    log_scaled_hybrid_vae_loss, 
    hybrid_vae_loss, 
    hybrid_spatial_vae_loss,
    log_scaled_hybrid_spatial_vae_loss
)

def train_vae(
        model, 
        train_loader, 
        test_loader, 
        optimizer,
        device,
        epochs=10,          
        eval_every=10, 
        loss_type='hybrid_vae_loss', 
        alpha=0.5,
        beta=0.5,
        kl_weight=0.01,
        warmup_epochs=20, 
        monitor=None,
        post_process=False,
        saved_weights_path="saved_weights/aitex_vae8_weights.pth",
        writer=None,
        # Early Stopping Args
        early_stopping=True,
        patience=3,
        min_delta=1e-4
    ):
    """
    Train the VAE model with evaluation every `eval_every` epochs.

    Early stopping is based on training loss alone, which is less ideal 
    in an unsupervised setting. However, it can still help terminate 
    training if the training loss does not improve further.

    Args:
        ...
        early_stopping (bool): If True, enable early stopping based on training loss.
        patience (int): Number of epochs (in a row) with no improvement allowed before stopping.
        min_delta (float): Minimum change in loss to qualify as improvement.
    """

    model.to(device)
    model.train()
    
    # Tracking for Early Stopping
    best_train_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_kl = 0
        epoch_mse = 0
        epoch_ssim = 0
        epoch_freq = 0
        
        current_kl_weight = min(kl_weight, kl_weight * (epoch / warmup_epochs))
        
        # ---------------------------
        # Train for one epoch
        # ---------------------------
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs = batch.to(device)
            recon, mu, logvar = model(inputs)

            if loss_type == 'vae_loss':
                loss, mse_loss_val, kl_loss_val = vae_loss(recon, inputs, mu, logvar, kl_weight=current_kl_weight)
                freq_loss_val = torch.tensor(0.0, device=device)
            elif loss_type == 'frequency_vae_loss':
                loss, freq_loss_val, kl_loss_val = frequency_vae_loss(recon, inputs, mu, logvar, kl_weight=current_kl_weight)
                mse_loss_val = torch.tensor(0.0, device=device)
            elif loss_type == 'log_scaled_frequency_vae_loss':
                loss, freq_loss_val, kl_loss_val = log_scaled_frequency_vae_loss(recon, inputs, mu, logvar, kl_weight=current_kl_weight)
                mse_loss_val = torch.tensor(0.0, device=device)
            elif loss_type == 'hybrid_vae_loss':
                loss, freq_loss_val, mse_loss_val, kl_loss_val = hybrid_vae_loss(
                    recon, inputs, mu, logvar,
                    kl_weight=current_kl_weight, alpha=alpha
                )
            elif loss_type == 'hybrid_spatial_vae_loss':
                loss, freq_loss_val, mse_loss_val, ssim_loss_val, kl_loss_val = hybrid_spatial_vae_loss(
                    recon, inputs, mu, logvar,
                    kl_weight=current_kl_weight, alpha=alpha, beta=beta
                )
            elif loss_type == 'loc_scaled_hybrid_spatial_vae_loss':
                loss, freq_loss_val, mse_loss_val, kl_loss_val = log_scaled_hybrid_spatial_vae_loss(
                    recon, inputs, mu, logvar,
                    kl_weight=current_kl_weight, alpha=alpha, beta=beta
                )
            elif loss_type == 'log_scaled_hybrid_vae_loss':
                loss, freq_loss_val, mse_loss_val, kl_loss_val = log_scaled_hybrid_vae_loss(
                    recon, inputs, mu, logvar,
                    kl_weight=current_kl_weight, alpha=alpha
                )
            else:
                raise ValueError(f"Invalid loss_type: {loss_type}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_kl += kl_loss_val.item() * batch_size
            epoch_mse += mse_loss_val.item() * batch_size
            epoch_ssim += ssim_loss_val.item() * batch_size
            epoch_freq += freq_loss_val.item() * batch_size

            # Log per batch
            if writer:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Loss/Batch_Total", loss.item(), global_step)
                writer.add_scalar("Loss/Batch_KL", kl_loss_val.item(), global_step)

                if mse_loss_val.item() != 0.0:
                    writer.add_scalar("Loss/Batch_MSE", mse_loss_val.item(), global_step)

                if ssim_loss_val is not None and ssim_loss_val.item() != 0.0:
                    writer.add_scalar("Loss/Batch_SSIM", ssim_loss_val.item(), global_step)
    
                if freq_loss_val.item() != 0.0:
                    writer.add_scalar("Loss/Batch_Freq", freq_loss_val.item(), global_step)

        # ---------------------------
        # Epoch-end stats & logging
        # ---------------------------
        data_count = len(train_loader.dataset)
        avg_loss = epoch_loss / data_count
        avg_kl = epoch_kl / data_count
        avg_mse = epoch_mse / data_count
        avg_ssim = epoch_ssim / data_count
        avg_freq = epoch_freq / data_count

        log_line = f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.6f}"
        
        # If you want to display sub-losses only if they're non-zero
        if mse_loss_val.item() != 0.0:
            log_line += f" | MSE: {avg_mse:.6f}"
        
        if ssim_loss_val is not None and ssim_loss_val.item() != 0.0:
            log_line += f" | SSIM: {avg_ssim:.6f}"
        
        if freq_loss_val.item() != 0.0:
            log_line += f" | Freq: {avg_freq:.6f}"
        
        log_line += f" | KL: {avg_kl:.6f}"
        
        print(log_line)

        # Save model each epoch
        torch.save(model.state_dict(), f"{saved_weights_path}/aitex_vae8_weights_{epoch}.pth")

        # TensorBoard Logging
        if writer:
            writer.add_scalar("Loss/Epoch_Total", avg_loss, epoch + 1)
            writer.add_scalar("Loss/Epoch_KL", avg_kl, epoch + 1)
            if avg_mse != 0.0:
                writer.add_scalar("Loss/Epoch_MSE", avg_mse, epoch + 1)
            if avg_ssim is not None and avg_ssim != 0.0:
                writer.add_scalar("Loss/Epoch_MSE", avg_ssim, epoch + 1)
            if avg_freq != 0.0:
                writer.add_scalar("Loss/Epoch_Freq", avg_freq, epoch + 1)

        # Monitor callback (custom)
        if monitor is not None:
            monitor.after_epoch(model, epoch + 1, avg_loss)

        # ---------------------------
        # Early Stopping on Training Loss
        # ---------------------------
        if early_stopping:
            if avg_loss < (best_train_loss - min_delta):
                # There's a new best training loss
                best_train_loss = avg_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
                
                # Save the best model so far
                best_path = f"{saved_weights_path}/best_vae_model_by_train_loss.pth"
                torch.save(model.state_dict(), best_path)
                print(f"  -> New best training loss. Model saved to {best_path}")
            else:
                epochs_no_improve += 1
                print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No improvement for {patience} consecutive epochs.")
                print(f"Best training loss was {best_train_loss:.6f} at epoch {best_epoch}.")
                break

        # ---------------------------
        # Periodic evaluation (optional)
        # ---------------------------
        if (epoch + 1) % eval_every == 0:
            print("Evaluating model ...")
            errors, labels = evaluate_model(model, test_loader, postprocess=post_process)
            evaluate_scores(errors, labels)

    print("✅ VAE Training Complete.")
    if early_stopping and best_epoch > 0:
        print(f"Best training loss {best_train_loss:.6f} achieved at epoch {best_epoch}.")
