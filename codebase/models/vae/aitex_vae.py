import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = False

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Reduces channel-wise redundancy by adaptively recalibrating feature responses.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(b, c)
        
        # Excitation: fully connected bottleneck and expansion
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# --- ChannelGate: Learnable gating mechanism for skip connections
class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=4):
        """
        A channel-wise gating module.
        
        Args:
            channels (int): Number of channels in the feature map.
            reduction (int): Reduction ratio for the intermediate channel size.
        """
        super(ChannelGate, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # Global average pooling
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()                       # Outputs a value in [0,1] per channel
        )

    def forward(self, enc_feat, dec_feat):
        """
        Forward pass for gating.
        
        Args:
            enc_feat: Encoder features from the skip connection.
            dec_feat: Decoder features (already upsampled to same spatial size as enc_feat).
            
        Returns:
            Merged feature: decoder output + gated encoder features.
        """
        gate = self.fc(enc_feat)   # shape: (B, channels, 1, 1)
        gated_enc = enc_feat * gate  # Gate applied per channel
        
        # Merge gated encoder features into the decoder (using addition)
        return dec_feat + gated_enc

# --- AitexVAEv8 with Gated Skip Connections using ChannelGate
class AitexVAEv8(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, img_size=256, dropout_p=0.2, use_attention=False):
        """
        A VAE variant for anomaly detection on textile images, now using gated skip connections.
        
        Changes:
          - The encoder is split into separate layers that save intermediate features.
          - The decoder upsamples the latent vector and at each stage, a gated skip connection (ChannelGate)
            is applied to modulate the contribution from the encoder.
          - Optionally applies an SE attention block after latent expansion.
        
        Args:
            in_channels (int): Number of image channels.
            latent_dim (int): Dimensionality of the latent space.
            img_size (int): Input image size (assumed square).
            dropout_p (float): Dropout probability for encoder layers.
            use_attention (bool): If True, adds a SEBlock after latent expansion.
        """
        super(AitexVAEv8, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.use_attention = use_attention
        self.forward_count = 0

        # --- Encoder Blocks (for skip connections) ---
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(dropout_p)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d(dropout_p)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout2d(dropout_p)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout2d(dropout_p)

        # Spatial dimensions reduce by factor of 16 (e.g., 256 -> 16)
        self.feature_map_size = img_size // 16
        self.feature_map_dim = 256 * (self.feature_map_size ** 2)  # e.g., 256 * 16 * 16

        # --- Bottleneck ---
        self.fc_intermediate = nn.Linear(self.feature_map_dim, 1024)
        self.bn_intermediate = nn.BatchNorm1d(1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim, self.feature_map_dim)
        self.seblock = SEBlock(256) if use_attention else nn.Identity()

        # Upsampling blocks via transposed convolutions:
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 32x32 -> 64x64
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 64x64 -> 128x128
        self.up4 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1)  # 128x128 -> 256x256

        # --- Gated Skip Connections using ChannelGate ---
        self.gate4 = ChannelGate(256)  # Applies to skip connection from conv4 (lowest resolution)
        self.gate3 = ChannelGate(128)  # For conv3's skip connection (32x32 resolution)
        self.gate2 = ChannelGate(64)   # For conv2 (64x64 resolution)
        self.gate1 = ChannelGate(32)   # For conv1 (128x128 resolution)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # --- Encoder ---
        skip1 = self.drop1(self.relu1(self.conv1(x)))           # (B, 32, 128, 128)
        skip2 = self.drop2(self.relu2(self.conv2(skip1)))         # (B, 64, 64, 64)
        skip3 = self.drop3(self.relu3(self.conv3(skip2)))         # (B, 128, 32, 32)
        conv4_out = self.drop4(self.relu4(self.conv4(skip3)))     # (B, 256, 16, 16)
        skip4 = conv4_out  # store for skip connection

        encoded = conv4_out.view(x.size(0), -1)
        intermediate = F.relu(self.bn_intermediate(self.fc_intermediate(encoded)))
        mu = self.fc_mu(intermediate)
        logvar = self.fc_logvar(intermediate)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)

        # --- Decoder ---
        dec_input = self.fc_dec(z)
        x_dec = dec_input.view(x.size(0), 256, self.feature_map_size, self.feature_map_size)
        x_dec = self.seblock(x_dec)  # Optionally refine with SE attention

        # Stage 0: Gated skip connection from conv4
        x_dec = self.gate4(skip4, x_dec)

        # Stage 1: Upsample to 32x32 then apply gated skip from conv3
        x_dec = self.up1(x_dec)           # (B, 128, 32, 32)
        x_dec = self.gate3(skip3, x_dec)

        # Stage 2: Upsample to 64x64 then gated skip from conv2
        x_dec = self.up2(x_dec)           # (B, 64, 64, 64)
        x_dec = self.gate2(skip2, x_dec)

        # Stage 3: Upsample to 128x128 then gated skip from conv1
        x_dec = self.up3(x_dec)           # (B, 32, 128, 128)
        x_dec = self.gate1(skip1, x_dec)

        # Final upsample to 256x256
        x_recon = self.up4(x_dec)         # (B, in_channels, 256, 256)

        self.forward_count += 1
        return x_recon, mu, logvar

class AitexVAEv6(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, img_size=256, dropout_p=0.2, use_attention=False):
        """
        A VAE variant for AITEX anomaly detection with concat-based skip connections.

        Key modifications compared to the original:
          - Encoder is split into explicit convolutional blocks which store intermediate feature maps.
          - Decoder uses transposed convolutions. At each upsampling stage, the decoder's activation is 
            concatenated with the corresponding encoder feature map.
          - A 1x1 convolution (conv_catX) fuses the concatenated features, reducing the channel count.
          - An optional SE attention block is applied after projecting the latent vector.

        Args:
            in_channels (int): Number of image channels.
            latent_dim (int): Dimensionality of the latent space.
            img_size (int): Input image size (assumed square).
            dropout_p (float): Dropout probability for the encoder layers.
            use_attention (bool): If True, add an SE attention block after the latent expansion.
        """
        super(AitexVAEv6, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.use_attention = use_attention
        self.forward_count = 0

        # --- Encoder: Define individual convolutional blocks ---
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(dropout_p)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d(dropout_p)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout2d(dropout_p)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout2d(dropout_p)

        # After 4 downsampling layers, spatial dimensions shrink by 16.
        self.feature_map_size = img_size // 16  # e.g., 256 // 16 = 16
        self.feature_map_dim = 256 * (self.feature_map_size ** 2)  # 256 * 16 * 16

        # --- Intermediate Bottleneck ---
        self.fc_intermediate = nn.Linear(self.feature_map_dim, 1024)
        self.bn_intermediate = nn.BatchNorm1d(1024)

        # Latent space mapping: produce mean and log variance.
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # --- Decoder ---
        # Map latent vector back to a feature map.
        self.fc_dec = nn.Linear(latent_dim, self.feature_map_dim)
        
        # Optional attention block after latent expansion.
        self.seblock = SEBlock(256) if use_attention else nn.Identity()

        # Upsampling stages (using transposed convolutions)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 32x32 -> 64x64
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 64x64 -> 128x128
        self.up4 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1)  # 128x128 -> 256x256

        # --- Concat fusion layers (1x1 convolutions) after concatenation for skip connections ---
        # After merging decoder and encoder features, we reduce the channel count back to the expected number.
        self.conv_cat4 = nn.Conv2d(256 + 256, 256, kernel_size=1)  # For 16x16 resolution: latent & conv4
        self.conv_cat3 = nn.Conv2d(128 + 128, 128, kernel_size=1)  # For 32x32: from up1 & conv3
        self.conv_cat2 = nn.Conv2d(64 + 64, 64, kernel_size=1)     # For 64x64: from up2 & conv2
        self.conv_cat1 = nn.Conv2d(32 + 32, 32, kernel_size=1)     # For 128x128: from up3 & conv1

    def reparameterize(self, mu, logvar):
        """Apply the reparameterization trick: sample from N(mu, sigma^2)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # ----- Encoder -----
        skip1 = self.drop1(self.relu1(self.conv1(x)))   # (B, 32, 128, 128)
        skip2 = self.drop2(self.relu2(self.conv2(skip1))) # (B, 64, 64, 64)
        skip3 = self.drop3(self.relu3(self.conv3(skip2))) # (B, 128, 32, 32)
        conv4_out = self.drop4(self.relu4(self.conv4(skip3))) # (B, 256, 16, 16)
        skip4 = conv4_out  # Save as skip connection at the lowest resolution.
        
        # Flatten and pass through intermediate bottleneck.
        encoded = conv4_out.view(x.size(0), -1)
        intermediate = F.relu(self.bn_intermediate(self.fc_intermediate(encoded)))
        mu = self.fc_mu(intermediate)
        logvar = self.fc_logvar(intermediate)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # Reparameterize to obtain latent vector.
        z = self.reparameterize(mu, logvar)

        # ----- Decoder -----
        dec_input = self.fc_dec(z) 
        x_dec = dec_input.view(x.size(0), 256, self.feature_map_size, self.feature_map_size)  # (B, 256, 16, 16)
        x_dec = self.seblock(x_dec)  # Optional attention

        # --- Stage 0: Merge latent expansion with skip4 ---
        x_dec = torch.cat([x_dec, skip4], dim=1)  # Now (B, 512, 16, 16)
        x_dec = self.conv_cat4(x_dec)              # Reduce to (B, 256, 16, 16)

        # --- Stage 1: Upsample to 32x32 and merge with skip3 ---
        x_dec = self.up1(x_dec)                    # (B, 128, 32, 32)
        x_dec = torch.cat([x_dec, skip3], dim=1)     # (B, 128+128=256, 32, 32)
        x_dec = self.conv_cat3(x_dec)              # Reduce to (B, 128, 32, 32)

        # --- Stage 2: Upsample to 64x64 and merge with skip2 ---
        x_dec = self.up2(x_dec)                    # (B, 64, 64, 64)
        x_dec = torch.cat([x_dec, skip2], dim=1)     # (B, 64+64=128, 64, 64)
        x_dec = self.conv_cat2(x_dec)              # Reduce to (B, 64, 64, 64)

        # --- Stage 3: Upsample to 128x128 and merge with skip1 ---
        x_dec = self.up3(x_dec)                    # (B, 32, 128, 128)
        x_dec = torch.cat([x_dec, skip1], dim=1)     # (B, 32+32=64, 128, 128)
        x_dec = self.conv_cat1(x_dec)              # Reduce to (B, 32, 128, 128)

        # --- Stage 4: Final upsampling to 256x256 ---
        x_recon = self.up4(x_dec)                  # (B, in_channels, 256, 256)

        self.forward_count += 1

        return x_recon, mu, logvar

class AitexVAEv5(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, img_size=256, dropout_p=0.2, use_attention=False):
        """
        A VAE variant for AITEX anomaly detection with skip connections.

        Changes from your original:
          - The encoder is split into distinct convolutional blocks storing feature maps for skip connections.
          - The decoder upsamples using transposed convolutions with additive skip connections from the encoder.
          - The rest of the structure (e.g., intermediate fully connected bottleneck, latent mapping) is maintained.

        Args:
            in_channels (int): Number of image channels.
            latent_dim (int): Dimensionality of the latent space.
            img_size (int): Input image size (assumed square).
            dropout_p (float): Dropout probability for the encoder layers.
            use_attention (bool): If True, add an SE attention block after latent expansion.
        """
        super(AitexVAEv5, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.use_attention = use_attention
        self.forward_count = 0

        # Encoder: define each conv block explicitly so that skip connections can be saved.
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(dropout_p)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d(dropout_p)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout2d(dropout_p)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout2d(dropout_p)

        # After 4 downsampling layers, the spatial dimensions shrink by 16.
        self.feature_map_size = img_size // 16  # e.g., 256 // 16 = 16
        
        # Flattened feature map dimension: 256 * (feature_map_size^2)
        self.feature_map_dim = 256 * (self.feature_map_size ** 2)

        # Intermediate Bottleneck: reduce flattened feature map to a smaller representation.
        self.fc_intermediate = nn.Linear(self.feature_map_dim, 1024)
        self.bn_intermediate = nn.BatchNorm1d(1024)

        # Latent space mapping: produce mean and log variance.
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # Decoder: map latent vector back to a feature map.
        self.fc_dec = nn.Linear(latent_dim, self.feature_map_dim)
        
        # Optional attention block at the bottleneck.
        self.seblock = SEBlock(256) if use_attention else nn.Identity()

        # Decoder upsampling blocks.
        # Stage 1: Upsample from 16x16 to 32x32, channels: 256 -> 128.
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        
        # Stage 2: Upsample from 32x32 to 64x64, channels: 128 -> 64.
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        
        # Stage 3: Upsample from 64x64 to 128x128, channels: 64 -> 32.
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        
        # Stage 4: Upsample from 128x128 to 256x256, channels: 32 -> in_channels.
        self.up4 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # --- Encoder with skip connections ---
        skip1 = self.drop1(self.relu1(self.conv1(x)))      # Shape: (batch, 32, 128, 128)
        skip2 = self.drop2(self.relu2(self.conv2(skip1)))    # Shape: (batch, 64, 64, 64)
        skip3 = self.drop3(self.relu3(self.conv3(skip2)))    # Shape: (batch, 128, 32, 32)
        conv4_out = self.drop4(self.relu4(self.conv4(skip3)))  # Shape: (batch, 256, 16, 16)
        skip4 = conv4_out  # Save as skip connection.
        
        # Flatten the last encoder feature map.
        encoded = conv4_out.view(x.size(0), -1)  # Shape: (batch, feature_map_dim)
        
        # Intermediate bottleneck with BatchNorm.
        intermediate = F.relu(self.bn_intermediate(self.fc_intermediate(encoded)))
        mu = self.fc_mu(intermediate)
        logvar = self.fc_logvar(intermediate)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # (Optional) Debug prints.
        if hasattr(self, 'DEBUG') and self.DEBUG and self.forward_count % 20 == 0:
            print("Forward pass count: {}".format(self.forward_count))
            print(f"shape of mu and logvar: {mu.shape}, {logvar.shape}")
            print("Intermediate stats: mean={:.3f}, std={:.3f}".format(intermediate.mean().item(), intermediate.std().item()))
            print("mu stats: mean={:.3f}, std={:.3f}".format(mu.mean().item(), mu.std().item()))
            print("logvar stats: mean={:.3f}, std={:.3f}".format(logvar.mean().item(), logvar.std().item()))
            print("Latent vector stats: mean={:.3f}, std={:.3f}".format(mu.mean().item(), torch.exp(0.5 * logvar).std().item()))

        # Reparameterize to sample the latent vector.
        z = self.reparameterize(mu, logvar)

        # --- Decoder ---
        dec_input = self.fc_dec(z)  # Shape: (batch, feature_map_dim)
        dec_input = dec_input.view(x.size(0), 256, self.feature_map_size, self.feature_map_size)  # (batch, 256, 16, 16)
        
        # Optionally apply the attention block.
        dec_input = self.seblock(dec_input)
        
        # Add skip connection from the encoder's conv4.
        x_dec = dec_input + skip4

        # Upsample stage 1: 16x16 -> 32x32; then add skip from conv3.
        x_dec = self.up1(x_dec)         # Shape: (batch, 128, 32, 32)
        x_dec = x_dec + skip3           # (Addition: both have 128 channels and 32x32 spatial size)
        
        # Upsample stage 2: 32x32 -> 64x64; then add skip from conv2.
        x_dec = self.up2(x_dec)         # Shape: (batch, 64, 64, 64)
        x_dec = x_dec + skip2           # (Addition: both have 64 channels and 64x64 spatial size)
        
        # Upsample stage 3: 64x64 -> 128x128; then add skip from conv1.
        x_dec = self.up3(x_dec)         # Shape: (batch, 32, 128, 128)
        x_dec = x_dec + skip1           # (Addition: both have 32 channels and 128x128 spatial size)
        
        # Upsample stage 4: 128x128 -> 256x256.
        x_recon = self.up4(x_dec)       # Final reconstruction: (batch, in_channels, 256, 256)

        self.forward_count += 1
        return x_recon, mu, logvar


class AitexVAEv4(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, img_size=256, dropout_p=0.2, use_attention=False):
        """
        A VAE variant for AITEX anomaly detection.
        
        Changes from v3:
          - Strengthened decoder: maps the latent vector to a 256-channel feature map, then upsamples in 4 stages.
          - Adds an intermediate bottleneck with BatchNorm between the flattened encoder output and the latent space,
            which helps scale down the large activation values and stabilizes the KL loss.
          - Optionally adds an SE attention block at the bottleneck before decoding.
        
        Args:
            in_channels (int): Number of image channels.
            latent_dim (int): Dimensionality of the latent space.
            img_size (int): Input image size (assumed square).
            dropout_p (float): Dropout probability for the encoder layers.
            use_attention (bool): If True, add an SE attention block at the bottleneck.
        """
        super(AitexVAEv4, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.use_attention = use_attention
        self.forward_count = 0

        # --- Build Encoder ---
        # Fixed architecture: in_channels -> 32 -> 64 -> 128 -> 256
        self.encoder = nn.Sequential(
            # Stage 1: in_channels -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Stage 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Stage 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Stage 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            nn.Flatten()  # Flatten for fully connected layers
        )

        # With 4 conv layers (stride=2 each), the spatial dimensions shrink by 16.
        self.feature_map_size = img_size // 16  # e.g., 256//16 = 16
        
        # The flattened feature map dimension: 256 * (feature_map_size^2)
        self.feature_map_dim = 256 * (self.feature_map_size ** 2)  # 256*16*16 = 65536

        # --- Intermediate Bottleneck ---
        # Reduce the huge flattened vector to a smaller representation.
        self.fc_intermediate = nn.Linear(self.feature_map_dim, 1024)
        self.bn_intermediate = nn.BatchNorm1d(1024)

        # Latent space mapping: produce mean and log variance from the 1024-dim vector.
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # --- Build Decoder ---
        # Map latent vector to a feature map with more channels (256) to strengthen the decoder.
        self.fc_dec = nn.Linear(latent_dim, 256 * (self.feature_map_size ** 2))

        # Decoder: 4 upsampling stages using transposed convolutions.
        # Each stage doubles the spatial dimensions.
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, self.feature_map_size, self.feature_map_size)),
            # Optionally insert an attention block at the bottleneck:
            SEBlock(256) if self.use_attention else nn.Identity(),

            # Upsample: 256 -> 128 (16x16 -> 32x32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(128), # batch norm destabilizes the KL loss
            nn.ReLU(inplace=True),

            # Upsample: 128 -> 64 (32x32 -> 64x64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(64), # batch norm destabilizes the KL loss
            nn.ReLU(inplace=True),

            # Upsample: 64 -> 32 (64x64 -> 128x128)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(32), # batch norm destabilizes the KL loss
            nn.ReLU(inplace=True),

            # Upsample: 32 -> in_channels (128x128 -> 256x256)
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1)
            # No final activation (like Sigmoid) here because you're using MSE/Freq losses.
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std

    def forward(self, x):
        # Encode the input image.
        encoded = self.encoder(x)
        # Pass through the intermediate bottleneck.
        intermediate = F.relu(self.bn_intermediate(self.fc_intermediate(encoded)))
        mu = self.fc_mu(intermediate)
        logvar = self.fc_logvar(intermediate)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        if DEBUG and self.forward_count % 20 == 0:
            # Print stats every 20 forward passes.
            print("Forward pass count: {}".format(self.forward_count))
            print(f"shape of mu and logvar: {mu.shape}, {logvar.shape}")
            print("Intermediate bottleneck stats: mean={:.3f}, std={:.3f}".format(intermediate.mean().item(), intermediate.std().item()))
            print("mu stats: mean={:.3f}, std={:.3f}".format(mu.mean().item(), mu.std().item()))
            print("logvar stats: mean={:.3f}, std={:.3f}".format(logvar.mean().item(), logvar.std().item()))
            print("Latent vector stats: mean={:.3f}, std={:.3f}".format(mu.mean().item(), torch.exp(0.5 * logvar).std().item()))
        
        # Sample latent vector using the reparameterization trick.
        z = self.reparameterize(mu, logvar)

        # Decode the latent vector back to image space.
        dec_input = self.fc_dec(z)
        x_recon = self.decoder(dec_input)
        
        self.forward_count += 1
        
        return x_recon, mu, logvar


class AitexVAEv3(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, img_size=256, dropout_p=0.2):
        """
        A VAE for AITEX anomaly detection with fixed encoder and decoder configurations.
        
        Args:
            in_channels (int): Number of image channels.
            latent_dim (int): Dimensionality of the latent space.
            img_size (int): Input image size (assumed square).
            dropout_p (float): Dropout probability for the encoder layers.
        """
        super(AitexVAEv3, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size

        # --- Build Encoder ---
        # Fixed architecture: in_channels -> 32 -> 64 -> 128 -> 256
        self.encoder = nn.Sequential(
            # Stage 1: in_channels -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Stage 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Stage 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Stage 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            nn.Flatten()  # Flatten for fully connected layers
        )

        # Since we downsample 4 times (stride=2 each), the feature map size is:
        self.feature_map_size = img_size // 16 # this must be adapted if we change the encoder (img_size // (2 ** num_enc_layers))
        
        # The flattened feature map dimension: 256 * (feature_map_size^2)
        self.feature_map_dim = 256 * (self.feature_map_size ** 2) # the 256 is the final out_channels number of the last stage of the encoder

        # Latent space mapping: produce mean and log variance
        self.fc_mu = nn.Linear(self.feature_map_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_map_dim, latent_dim)

        # --- Build Decoder ---
        # Fully connected layer: latent vector -> feature map with 64 channels
        self.fc_dec = nn.Linear(latent_dim, 64 * (self.feature_map_size ** 2))

        # Fixed decoder architecture:
        # Unflatten to (64, feature_map_size, feature_map_size), then
        # Upsample using transposed convolutions: 64 -> 32 -> in_channels
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, self.feature_map_size, self.feature_map_size)), # the 64 is the first element of the decoder_channels list

            # Stage 1: Transposed convolution: 64 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),

            # Stage 2: Transposed convolution: 32 -> in_channels
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=4, padding=0),
            #nn.Sigmoid() # Do not use it here, it is for BCE losses and we are using MSE+Freq or Freq only
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std

    def forward(self, x):
        # Encode the input image
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Sample latent vector using the reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode the latent vector back to image space
        dec_input = self.fc_dec(z)
        x_recon = self.decoder(dec_input)
        
        return x_recon, mu, logvar


class AitexVAEv2(nn.Module):
    def __init__(self,
                 in_channels=1,
                 latent_dim=32,
                 img_size=256,
                 dropout_p=0.2,
                 encoder_channels=None,
                 decoder_channels=None):
        """
        A VAE for AITEX anomaly detection with configurable encoder and decoder channels.

        Args:
            in_channels (int): Number of image channels (1 for grayscale).
            latent_dim (int): Dimensionality of the latent space.
            img_size (int): Input image height/width (assumed square).
            dropout_p (float): Dropout probability in the encoder.
            encoder_channels (list): List of output channels for each encoder conv layer.
                                     Default: [32, 64, 128, 256]
            decoder_channels (list): List of channels for the decoder upsampling.
                                     The first element is the channel count after fc_dec,
                                     the intermediate layers, and the final layer must match in_channels.
                                     Default: [64, 32, in_channels]
        """
        super(AitexVAEv2, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size

        # Default channel configs
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        if decoder_channels is None:
            decoder_channels = [64, 32, in_channels]

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        # Number of downsampling layers = len(encoder_channels).
        self.num_enc_layers = len(encoder_channels)
        # Each conv layer downsamples by a factor of 2:
        self.feature_map_size = img_size // (2 ** self.num_enc_layers)

        # --- Build Encoder ---
        encoder_layers = []
        prev_ch = in_channels
        for out_ch in encoder_channels:
            encoder_layers.append(nn.Conv2d(prev_ch, out_ch, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout2d(dropout_p))
            prev_ch = out_ch
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        # Flattened feature-map dimension
        self.feature_map_dim = encoder_channels[-1] * (self.feature_map_size ** 2)

        # Latent space
        self.fc_mu = nn.Linear(self.feature_map_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_map_dim, latent_dim)

        # --- Build Decoder ---
        # 1) Fully connected layer: latent -> feature-map
        #    The first element in decoder_channels is the # of channels after unflatten
        self.fc_dec = nn.Linear(latent_dim,
                                decoder_channels[0] * (self.feature_map_size ** 2))

        # 2) Transposed Convs
        # We'll build up a list of layers that:
        #   - Unflatten -> (decoder_channels[0], feature_map_size, feature_map_size)
        #   - For each pair of consecutive elements in decoder_channels, do ConvT
        #   - The final out_channels must match in_channels
        dec_layers = []
        dec_layers.append(
            nn.Unflatten(1, (decoder_channels[0], self.feature_map_size, self.feature_map_size))
        )

        # Example: If decoder_channels = [64, 32, 1], we want:
        #   ConvT(64->32), ReLU, ConvT(32->1), Sigmoid
        # And each ConvT does stride=2 or stride=4 to go from feature_map_size->img_size
        # You can adapt stride/padding to your preference.

        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]

            # If this is not the final layer, do a ReLU after it
            is_last = (i == len(decoder_channels) - 2)

            dec_layers.append(nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=4, padding=0
            ))
            if not is_last:
                dec_layers.append(nn.ReLU())

        dec_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        # Decode
        dec_input = self.fc_dec(z)
        x_recon = self.decoder(dec_input)

        return x_recon, mu, logvar


class AitexVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, img_size=256, dropout_p=0.2):
        """
        A VAE tailored for anomaly detection on grid-like AITEX patches.
        
        Parameters:
          in_channels (int): Number of image channels.
          latent_dim (int): Dimensionality of the latent space (recommended 32).
          img_size (int): Height/width of the square input image (e.g., 256).
          dropout_p (float): Dropout probability for encoder regularization.
        """
        super(AitexVAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size

        # --- Encoder: Deep & Regularized ---
        # Four convolutional layers with stride 2 for downsampling:
        # Input: 256x256 -> Output: 16x16 (256/2^4)
        self.encoder = nn.Sequential(
            # Stage 1: 256 -> 128
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Stage 2: 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Stage 3: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            # Stage 4: 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),

            nn.Flatten()  # Flatten for the fully-connected layers
        )

        # Calculate the spatial dimensions after the encoder.
        # For img_size=256 with 4 downsamplings, the spatial dimension is 256/16 = 16.
        self.feature_map_size = img_size // 16
        self.feature_map_dim = 256 * (self.feature_map_size ** 2)  # 256 * 16 * 16

        # --- Latent Space ---
        # Compress the high-dimensional feature map to a compact latent vector.
        self.fc_mu = nn.Linear(self.feature_map_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_map_dim, latent_dim)

        # --- Decoder: Shallow and Limited ---
        # Instead of mirroring the encoder, we map the latent vector to a lower-capacity feature map.
        # We choose to project to a feature map of shape (64, 16, 16) and then perform two upsampling stages.
        self.fc_dec = nn.Linear(latent_dim, 64 * self.feature_map_size * self.feature_map_size)

        self.decoder = nn.Sequential(
            # Reshape the linear projection to (64, 16, 16)
            nn.Unflatten(1, (64, self.feature_map_size, self.feature_map_size)),
            # Upsample from 16x16 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            # Upsample from 64x64 -> 256x256
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=4, padding=0),
            nn.Sigmoid()  # Normalize outputs to [0, 1]
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        # Decode
        dec_input = self.fc_dec(z)
        x_recon = self.decoder(dec_input)
        return x_recon, mu, logvar
