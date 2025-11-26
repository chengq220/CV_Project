import torch
import torch.nn as nn

def _get_latent_from_out(out):
    """Extract latent tensor from VAE output dict."""
    for k in ("quantize", "quantized", "encoded", "z"):
        if k in out:
            return out[k]
    raise KeyError(f"No latent key found in VAE output. Available keys: {list(out.keys())}")

class CompressionLayer(nn.Module):
    """
    Compression layer using encoder from decomposed VAE.
    Takes concatenated latents and compresses spatially.
    """
    def __init__(self, encoder, in_ch, out_ch):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Project to match encoder input
        x = self.proj(x)
        # Compress spatially
        x = self.pool(x)
        return x

class DecompressionLayer(nn.Module):
    """
    Decompression layer using decoder from decomposed VAE.
    Takes compressed latents and decompresses spatially with skip connections.
    """
    def __init__(self, decoder, in_ch, skip_ch, out_ch):
        super().__init__()
        self.decoder = decoder
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        self.proj_skip = nn.Conv2d(skip_ch, in_ch, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip):
        # Upsample
        x = self.up(x)
        # Align spatial dims
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
        # Project and concatenate skip
        skip = self.proj_skip(skip)
        x = torch.cat([x, skip], dim=1)
        # Refine
        x = self.conv(x)
        return x

class ResidualLatentUNet(nn.Module):
    """
    U-Net architecture built from decomposed VAE encoders/decoders.
    Each layer in compression side uses VAE encoder, each in decompression side uses VAE decoder.
    Handles all spatial sizes by progressive compression/decompression.
    
    Args:
        model_container: DecomposeVAE instance
        device: device to run on
        num_layers: number of compression/decompression layers (depth of U)
    """
    def __init__(self, model_container, device="cpu", num_layers=3):
        super().__init__()
        self.device = torch.device(device)
        self.num_layers = num_layers
        self.fullvae = model_container.getFullVAE().to(self.device)
        self.encoder_vae = model_container.getEncoder().to(self.device)
        self.decoder_vae = model_container.getDecoder().to(self.device)
        self.fullvae.eval()
        self.encoder_vae.eval()
        self.decoder_vae.eval()
        
        # Infer latent dimensions from dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128, device=self.device)
            out = self.fullvae(dummy)
            z = _get_latent_from_out(out)
            _, latent_ch, _, _ = z.shape
        
        # Concatenated latent channels (image + residual)
        in_ch = latent_ch * 2
        
        # Build channel progression
        base_ch = latent_ch
        channels = [in_ch]
        for i in range(num_layers):
            channels.append(base_ch * (2 ** (i + 1)))
        
        # Compression layers (encoder side)
        self.comp_layers = nn.ModuleList()
        for i in range(num_layers):
            self.comp_layers.append(
                CompressionLayer(
                    self.encoder_vae,
                    in_ch=channels[i],
                    out_ch=channels[i + 1]
                )
            )
        
        # Bottleneck
        bottleneck_ch = channels[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], bottleneck_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decompression layers (decoder side, mirrored)
        self.decomp_layers = nn.ModuleList()
        for i in range(num_layers):
            decomp_in = bottleneck_ch if i == 0 else channels[num_layers - i + 1]
            skip_ch = channels[num_layers - i]
            decomp_out = channels[num_layers - i]
            self.decomp_layers.append(
                DecompressionLayer(
                    self.decoder_vae,
                    in_ch=decomp_in,
                    skip_ch=skip_ch,
                    out_ch=decomp_out
                )
            )
        
        # Final projection back to concatenated latent channels
        self.final_conv = nn.Conv2d(channels[1], in_ch, kernel_size=1)
    
    def forward(self, img_tensor):
        """
        Args:
            img_tensor: [B, C, H, W] raw image (any size)
        
        Returns:
            dict with keys:
                z_image: original image latent
                z_residual: original residual latent
                z_concat: concatenated (z_image + z_residual)
                z_refined: refined concatenated latent from U-Net
                recon: reconstruction from fullvae
                residual: computed residual
                compression_levels: list of compressed feature maps
        """
        img_tensor = img_tensor.to(self.device)
        
        # Get latents from fullvae
        with torch.no_grad():
            out = self.fullvae(img_tensor)
            recon = out["x_recon"]
            residual = img_tensor - recon
            
            out_r = self.fullvae(residual)
            
            z_image = _get_latent_from_out(out)
            z_residual = _get_latent_from_out(out_r)
        
        # Align spatial dims
        if z_image.shape[-2:] != z_residual.shape[-2:]:
            z_residual = nn.functional.interpolate(
                z_residual, size=z_image.shape[-2:], mode="nearest"
            )
        
        # Concatenate
        x = torch.cat([z_image, z_residual], dim=1)
        
        # Compression path: progressively compress
        compression_levels = []
        for comp in self.comp_layers:
            compression_levels.append(x)
            x = comp(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decompression path: progressively decompress with skip connections
        for decomp, skip in zip(self.decomp_layers, reversed(compression_levels)):
            x = decomp(x, skip)
        
        # Final projection
        z_refined = self.final_conv(x)
        
        return {
            "z_image": z_image,
            "z_residual": z_residual,
            "z_concat": torch.cat([z_image, z_residual], dim=1),
            "z_refined": z_refined,
            "recon": recon,
            "residual": residual,
            "compression_levels": compression_levels
        }
