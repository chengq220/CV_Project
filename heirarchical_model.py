import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualLatentUNet(nn.Module):
    """
    2-layer hierarchical U-Net with proper spatial compression.
    Fixed version with correct skip connection handling.
    """
    def __init__(self, model_container, device="cpu", num_layers=2):
        super().__init__()
        self.device = torch.device(device)
        self.fullvae = model_container.getFullVAE().to(self.device)
        self.fullvae.eval()
        
        # Get latent dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28, device=self.device)
            zq, *_ = self.fullvae.quantize(dummy)
            _, latent_ch, latent_h, latent_w = zq.shape
        
        print(f"  Detected latent: {latent_ch} channels, {latent_h}×{latent_w} spatial")
        
        in_ch = latent_ch * 2  # concatenated (image + residual)
        
        # Fixed 2-layer architecture
        # Level 0: 7x7, 128 channels (in_ch)
        # Level 1: 3x3, 256 channels (after encoder 1)
        # Level 2: 1x1, 512 channels (after encoder 2)
        # Then reverse back
        
        print(f"  Building 2-layer hierarchy...")
        print(f"  Input: {latent_h}×{latent_w}, {in_ch} channels")
        
        # Encoder 1: 7x7, 128ch -> 3x3, 256ch
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Encoder 2: 3x3, 256ch -> 1x1, 512ch
        self.enc2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Bottleneck: 1x1, 512ch
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder 2: 1x1, 512ch -> 3x3, 256ch (with skip from enc2)
        self.up2 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, padding=1),  # 512 upsampled + 512 skip
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder 1: 3x3, 256ch -> 7x7, 128ch (with skip from enc1)
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1),  # 256 upsampled + 256 skip
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Final output
        self.final = nn.Conv2d(128, in_ch, 1)
        
        print(f"  Architecture:")
        print(f"    Input:      {latent_h}×{latent_w}, {in_ch}ch")
        print(f"    Enc1 out:   ~{latent_h//2}×{latent_w//2}, 256ch")
        print(f"    Enc2 out:   ~{latent_h//4}×{latent_w//4}, 512ch")
        print(f"    Bottleneck: ~{latent_h//4}×{latent_w//4}, 512ch")
        print(f"    Dec2 out:   ~{latent_h//2}×{latent_w//2}, 256ch")
        print(f"    Dec1 out:   {latent_h}×{latent_w}, 128ch")
        print(f"    Final:      {latent_h}×{latent_w}, {in_ch}ch")
    
    def forward(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            # Get reconstruction
            out_full = self.fullvae(img_tensor)
            recon = out_full["x_recon"]
            
            # Get latents
            zq_img, _, _, _ = self.fullvae.quantize(img_tensor)
            residual = img_tensor - recon
            zq_res, _, _, _ = self.fullvae.quantize(residual)
            
            # Align spatial dims
            if zq_img.shape[-2:] != zq_res.shape[-2:]:
                zq_res = F.interpolate(zq_res, size=zq_img.shape[-2:], mode='nearest')
        
        # Concatenate
        z_concat = torch.cat([zq_img, zq_res], dim=1)
        
        # Encoder path
        e1 = self.enc1(z_concat)  # 7x7 -> 3x3, 256ch
        e2 = self.enc2(e1)        # 3x3 -> 1x1, 512ch
        
        # Bottleneck
        b = self.bottleneck(e2)   # 1x1, 512ch
        
        # Decoder path with skip connections
        # Decoder 2: upsample and add skip from enc2
        d2_up = self.up2(b)       # 1x1 -> 3x3, 512ch
        if d2_up.shape[2:] != e2.shape[2:]:
            d2_up = F.interpolate(d2_up, size=e2.shape[2:], mode='nearest')
        d2 = self.dec2(torch.cat([d2_up, e2], dim=1))  # concat and refine to 256ch
        
        # Decoder 1: upsample and add skip from enc1
        d1_up = self.up1(d2)      # 3x3 -> 7x7, 256ch
        if d1_up.shape[2:] != e1.shape[2:]:
            d1_up = F.interpolate(d1_up, size=e1.shape[2:], mode='nearest')
        d1 = self.dec1(torch.cat([d1_up, e1], dim=1))  # concat and refine to 128ch
        
        # Ensure spatial dimensions match input before final projection
        if d1.shape[2:] != z_concat.shape[2:]:
            d1 = F.interpolate(d1, size=z_concat.shape[2:], mode='nearest')
        
        # Final projection
        z_refined = self.final(d1)
        
        return {
            "z_image": zq_img,
            "z_residual": zq_res,
            "z_concat": z_concat,
            "z_refined": z_refined,
            "recon": recon,
            "residual": residual,
            "compression_levels": [e1, e2]  # Show compressed levels
        }
