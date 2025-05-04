import os
import streamlit as st
from PIL import Image
import imageio

def vae_training_dashboard(base_dir: str):
    st.title("🧠 VAE Training Monitor Dashboard")

    # 🔽 Step 1: Select training run
    run_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    selected_run = st.selectbox("Select Training Run", run_dirs)
    snapshot_dir = os.path.join(base_dir, selected_run)

    # 🔎 Collect files
    recon_files = sorted([f for f in os.listdir(snapshot_dir) if "recon" in f and f.endswith(".png")])
    latent_files = sorted([f for f in os.listdir(snapshot_dir) if "latent" in f])
    fft_files = sorted([f for f in os.listdir(snapshot_dir) if "fft" in f])

    if not recon_files:
        st.warning("No reconstruction files found.")
        return

    # 🔽 Step 2: Select epoch and image
    epochs = sorted(list(set(int(f.split("_")[1]) for f in recon_files)))
    img_indices = sorted(list(set(int(f.split("_")[-1].split(".")[0]) for f in recon_files)))

    selected_epoch = st.selectbox("Select Epoch", epochs)
    selected_img = st.selectbox("Select Image Index", img_indices)

    if selected_epoch is not None and selected_img is not None:
        # 📸 Recon
        recon_path = os.path.join(snapshot_dir, f"epoch_{selected_epoch:03d}_recon_{selected_img}.png")
        st.subheader("📸 Reconstruction")
        st.image(Image.open(recon_path), caption=f"Reconstruction - Epoch {selected_epoch}", use_container_width=True)

        # 🌌 FFT
        fft_path = os.path.join(snapshot_dir, f"epoch_{selected_epoch:03d}_fft_{selected_img}.png")
        if os.path.exists(fft_path):
            st.subheader("🌌 FFT Comparison")
            st.image(Image.open(fft_path), caption=f"FFT - Epoch {selected_epoch}", use_container_width=True)

        # 📊 Latent
        latent_path = os.path.join(snapshot_dir, f"epoch_{selected_epoch:03d}_latent.png")
        if os.path.exists(latent_path):
            st.subheader("📊 Latent Distribution")
            st.image(Image.open(latent_path), caption=f"Latent μ and σ - Epoch {selected_epoch}", use_container_width=True)

        # 📉 Loss Curve
        loss_path = os.path.join(snapshot_dir, "loss_curve.png")
        if os.path.exists(loss_path):
            st.subheader("📉 Training Loss Curve")
            st.image(Image.open(loss_path), caption="Loss over Epochs", use_container_width=True)

    # 🎞️ Animation Mode: Reconstructions
    st.markdown("---")
    st.subheader("🎞️ Reconstruction Animation")
    enable_anim = st.checkbox("Enable Animation for Reconstructions")

    if enable_anim:
        anim_img_index = st.selectbox("Select Image Index for Animation", img_indices, index=0, key="recon_anim_img")
        recon_anim_files = sorted([
            f for f in recon_files if f"recon_{anim_img_index}" in f
        ])
        recon_images = [Image.open(os.path.join(snapshot_dir, f)) for f in recon_anim_files]
        if recon_images:
            recon_frame = st.slider("Epoch (Recon)", 0, len(recon_images) - 1, 0)
            st.image(recon_images[recon_frame], use_container_width=True,
                     caption=f"Epoch: {recon_anim_files[recon_frame]}")
        else:
            st.warning("No recon images found for animation.")

    # 🎞️ Animation Mode: FFTs
    st.markdown("---")
    st.subheader("🎞️ FFT Animation")
    enable_fft_anim = st.checkbox("Enable Animation for FFTs")

    if enable_fft_anim:
        fft_img_index = st.selectbox("Select Image Index for FFT Animation", img_indices, index=0, key="fft_anim_img")
        fft_anim_files = sorted([
            f for f in fft_files if f"fft_{fft_img_index}" in f
        ])
        fft_images = [Image.open(os.path.join(snapshot_dir, f)) for f in fft_anim_files]
        if fft_images:
            fft_frame = st.slider("Epoch (FFT)", 0, len(fft_images) - 1, 0)
            st.image(fft_images[fft_frame], use_container_width=True,
                     caption=f"Epoch: {fft_anim_files[fft_frame]}")
        else:
            st.warning("No FFT images found for animation.")

    # 🎞️ Animation Mode: Latent Distribution
    st.markdown("---")
    st.subheader("🎞️ Latent Distribution Animation")
    enable_latent_anim = st.checkbox("Enable Animation for μ and σ")
    
    if enable_latent_anim:
        latent_anim_files = sorted([
            f for f in latent_files if "latent" in f
        ])
        if latent_anim_files:
            latent_images = [Image.open(os.path.join(snapshot_dir, f)) for f in latent_anim_files]
            latent_frame = st.slider("Epoch (Latent)", 0, len(latent_images) - 1, 0)
            st.image(latent_images[latent_frame], use_container_width=True,
                     caption=f"{latent_anim_files[latent_frame]}")
    else:
        st.warning("No latent histograms found for animation.")


# 🚀 Launch dashboard
if __name__ == "__main__":
    vae_training_dashboard(base_dir="../../../artifacts/vae/monitor_snapshots")
