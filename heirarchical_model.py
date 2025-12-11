import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import math

# Assuming decompose.py, vqvae.py, and utils.py are in the current path.
# This code will now correctly load the VQ-VAE and train the U-Net.
from decompose import DecomposeVAE

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    # --- Paths ---
    WEIGHT_PATH = "checkpoints/vqvae_best.pth" # CORRECTED PATH
    DATA_DIR = "./data"
    OUTPUT_DIR = "./hierarchical_unet_checkpoints"
    
    # --- Training Hyperparameters ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    LOG_INTERVAL = 100 
    
    # --- Model Configuration ---
    NUM_HIERARCHY_LAYERS = 2
    
    # --- Loss Weights (Adjust to tune training) ---
    LOSS_WEIGHT_SMOOTHNESS = 1.0 
    LOSS_WEIGHT_RESIDUAL = 0.1  

# Ensure output directory exists
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"Using device: {Config.DEVICE}")


# ==============================================================================
# 2. FIXED RESIDUALLATENTUNET MODEL (Sizing Issues Resolved)
# ==============================================================================
class ResidualLatentUNet(nn.Module):
    def __init__(self, model_container, device="cpu", num_layers=2):
        super().__init__()
        self.device = torch.device(device)
        self.fullvae = model_container.getFullVAE().to(self.device)
        self.fullvae.eval()
        
        # Freeze VAE parameters
        for param in self.fullvae.parameters():
            param.requires_grad = False
        
        # Get latent dimensions from VAE instance
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28, device=self.device)
            zq, *_ = self.fullvae.quantize(dummy) 
            _, latent_ch, latent_h, latent_w = zq.shape
        
        in_ch = latent_ch * 2  # concatenated (image + residual)
        print(f"  U-Net Input Latent Size: {latent_h}x{latent_w}, {in_ch} channels")
        
        # --- Encoder Path (Compression) ---
        # Enc1: 7x7 -> 4x4 (Skip 1)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Enc2: 4x4 -> 2x2 (Skip 2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # --- Decoder Path (Decompression & Skip) ---
        
        # Dec2 (Innermost): Upsample (512ch) + Skip (512ch) -> 256ch
        self.up2 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Dec1 (Outermost): Upsample (256ch) + Skip (256ch) -> 128ch
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Final projection: 128ch -> 128ch (in_ch)
        self.final = nn.Conv2d(128, in_ch, 1)
        
    def forward(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            # Get VAE latents and reconstruction
            zq_img, *_ = self.fullvae.quantize(img_tensor)
            recon = self.fullvae.decoder(zq_img)
            residual = img_tensor - recon
            
            # Get residual latent
            zq_res, *_ = self.fullvae.quantize(residual)
            
            # Align spatial dims (Safety check, should match)
            if zq_img.shape[-2:] != zq_res.shape[-2:]:
                zq_res = F.interpolate(zq_res, size=zq_img.shape[-2:], mode='nearest')
        
        # U-Net Input (7x7, 128ch)
        z_concat = torch.cat([zq_img, zq_res], dim=1)
        
        # Encoder path
        e1 = self.enc1(z_concat)    # Skip 1 (e1, expected 4x4)
        e2 = self.enc2(e1)          # Skip 2 (e2, expected 2x2)
        
        # Bottleneck
        b = self.bottleneck(e2)     # (expected 2x2)
        
        # Decoder 2 (Skip: e2)
        d2_up = self.up2(b)         # Upconvolution (expected 4x4)
        
        # Interpolation check for Dec 2: Target size is e2.shape (expected 2x2)
        if d2_up.shape[2:] != e2.shape[2:]:
            # Resize upsampled feature to match the skip connection
            d2_up = F.interpolate(d2_up, size=e2.shape[2:], mode='nearest')
        d2 = self.dec2(torch.cat([d2_up, e2], dim=1)) 
        
        # Decoder 1 (Skip: e1)
        d1_up = self.up1(d2)        # Upconvolution (expected 8x8 or 6x6)
        
        # Interpolation check for Dec 1: Target size is z_concat.shape (expected 7x7)
        if d1_up.shape[2:] != z_concat.shape[2:]: 
            d1_up = F.interpolate(d1_up, size=z_concat.shape[2:], mode='nearest') # This forces d1_up to 7x7

        # --- FIX: SPATIAL MISMATCH RESOLUTION ---
        # The error occurs because e1 (e.g., 4x4 or 3x3) does not match d1_up (7x7).
        # We must resize e1 to match the target size of d1_up (which is 7x7).
        if d1_up.shape[2:] != e1.shape[2:]:
            e1_resized = F.interpolate(e1, size=d1_up.shape[2:], mode='nearest')
        else:
            e1_resized = e1
        # ----------------------------------------

        d1 = self.dec1(torch.cat([d1_up, e1_resized], dim=1)) 
        
        # Final projection
        z_refined = self.final(d1)
        
        return {
            "z_image": zq_img,
            "z_residual": zq_res,
            "z_concat": z_concat,
            "z_refined": z_refined,
            "recon": recon,
            "residual": residual,
        }


# ==============================================================================
# 3. HIERARCHICAL LOSS FUNCTION
# ==============================================================================
class HierarchicalLoss(nn.Module):
    def __init__(self, smoothness_weight, residual_weight):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.residual_weight = residual_weight

    def forward(self, unet_output, original_image):
        z_refined = unet_output["z_refined"]
        z_concat = unet_output["z_concat"]
        smoothness_loss = F.mse_loss(z_refined, z_concat)

        z_residual = unet_output["z_residual"]
        residual_energy_loss = torch.mean(z_residual ** 2) 

        total_loss = (self.smoothness_weight * smoothness_loss) + \
                     (self.residual_weight * residual_energy_loss)
        
        return total_loss, {
            "total": total_loss.item(),
            "smoothness": smoothness_loss.item(),
            "residual": residual_energy_loss.item(),
        }


# ==============================================================================
# 4. TRAINING AND UTILITY FUNCTIONS
# ==============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, total_smoothness, total_residual = 0, 0, 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS}')
    
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss, loss_dict = criterion(output, data)
        loss.backward()
        optimizer.step()
        
        total_loss += loss_dict['total']
        total_smoothness += loss_dict['smoothness']
        total_residual += loss_dict['residual']
        
        if batch_idx % Config.LOG_INTERVAL == 0:
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'smooth': f"{loss_dict['smoothness']:.4f}",
                'res_e': f"{loss_dict['residual']:.4f}"
            })
    
    num_batches = len(dataloader)
    return {
        'total': total_loss / num_batches,
        'smoothness': total_smoothness / num_batches,
        'residual': total_residual / num_batches
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_smoothness, total_residual = 0, 0, 0
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc='Validating', leave=False): 
            data = data.to(device)
            output = model(data)
            loss, loss_dict = criterion(output, data)
            
            total_loss += loss_dict['total']
            total_smoothness += loss_dict['smoothness']
            total_residual += loss_dict['residual']
    
    num_batches = len(dataloader)
    return {
        'total': total_loss / num_batches,
        'smoothness': total_smoothness / num_batches,
        'residual': total_residual / num_batches
    }


def visualize_results(model, dataloader, device, epoch, save_dir):
    model.eval()
    data, _ = next(iter(dataloader))
    data = data[:8].to(device) 
    
    with torch.no_grad():
        output = model(data)
    
    fig, axes = plt.subplots(4, data.shape[0], figsize=(data.shape[0]*2, 8))
    
    titles = ['Original', 'VAE Recon', 'Residual', 'Refined Latent']
    for i in range(data.shape[0]):
        # Original Image
        axes[0, i].imshow(data[i, 0].cpu().numpy(), cmap='gray')
        # VAE Reconstruction
        axes[1, i].imshow(output['recon'][i, 0].cpu().numpy(), cmap='gray')
        # Initial Residual
        axes[2, i].imshow(output['residual'][i, 0].cpu().numpy(), cmap='RdBu_r', vmin=-0.5, vmax=0.5) 
        # Refined Latent (First Channel)
        # Note: z_refined is a latent tensor, visualizing channel 0 as a proxy
        axes[3, i].imshow(output['z_refined'][i, 0].cpu().numpy(), cmap='viridis')
        
        
