import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Assuming these files are present in the environment
from decompose import DecomposeVAE 

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    WEIGHT_PATH = "checkpoints/vqvae_best.pth" 
    DATA_DIR = "./data"
    OUTPUT_DIR = "./hierarchical_unet_checkpoints" 
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    
    # --- CHECK THIS FILENAME ---
    UNET_WEIGHT_FILENAME = "unet_epoch_10.pth" 
    VISUALIZATION_TITLE = "Input Image vs. Refined Reconstruction Output (2 Epochs)"
    # Note: We will use plt.show() instead of saving the file for notebook display
    # OUTPUT_IMAGE_FILENAME = "unet_visualization_epoch2.png" 

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
class ResidualLatentUNet(nn.Module):
    def __init__(self, model_container, device="cpu", num_layers=2):
        super().__init__()
        self.device = torch.device(device)
        self.fullvae = model_container.getFullVAE().to(self.device).eval()
        for param in self.fullvae.parameters(): param.requires_grad = False
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28, device=self.device)
            zq, *_ = self.fullvae.quantize(dummy)  # you quantized the entire input image, not the latent space... 
            _, latent_ch, _, _ = zq.shape
            self.latent_ch = latent_ch 
        
        in_ch = latent_ch * 2
        
        # U-Net Architecture (must match training code)
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.enc2 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(512 + 512, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(256 + 256, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(128, in_ch, 1)
        
    def forward(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            zq_img, *_ = self.fullvae.quantize(img_tensor)
            residual = img_tensor - self.fullvae.decoder(zq_img)
            zq_res, *_ = self.fullvae.quantize(residual)
            if zq_img.shape[-2:] != zq_res.shape[-2:]:
                zq_res = F.interpolate(zq_res, size=zq_img.shape[-2:], mode='nearest')
        
        z_concat = torch.cat([zq_img, zq_res], dim=1)
        
        # U-Net forward pass (compressed for brevity)
        e1 = self.enc1(z_concat); e2 = self.enc2(e1); b = self.bottleneck(e2)
        d2_up = self.up2(b);         
        if d2_up.shape[2:] != e2.shape[2:]: d2_up = F.interpolate(d2_up, size=e2.shape[2:], mode='nearest')
        d2 = self.dec2(torch.cat([d2_up, e2], dim=1)); 
        d1_up = self.up1(d2);        
        target_size = z_concat.shape[2:]
        if d1_up.shape[2:] != target_size: d1_up = F.interpolate(d1_up, size=target_size, mode='nearest')
        e1_resized = F.interpolate(e1, size=d1_up.shape[2:], mode='nearest') if d1_up.shape[2:] != e1.shape[2:] else e1
        z_refined = self.final(self.dec1(torch.cat([d1_up, e1_resized], dim=1)))
        
        # Final Image Output
        z_refined_img = z_refined.narrow(1, 0, self.latent_ch)
        refined_recon = self.fullvae.decoder(z_refined_img) 

        return {"refined_recon": refined_recon}

# ==============================================================================
# 3. VISUALIZATION FUNCTION (with MSE)
# ==============================================================================
def visualize_input_output(model, dataloader, device, title):
    """Shows Input Image vs. Refined Reconstruction and displays MSE."""
    model.eval()
    data, _ = next(iter(dataloader))
    data = data[:8].to(device)
    
    with torch.no_grad():
        output = model(data)
    refined_recon = output['refined_recon']
    
    # --- Calculate MSE per image ---
    mse_values = F.mse_loss(refined_recon, data, reduction='none')
    mse_per_image = mse_values.mean(dim=[1, 2, 3]).cpu().numpy()
    # -------------------------------
    
    fig, axes = plt.subplots(2, data.shape[0], figsize=(data.shape[0]*2, 4.5))
    
    titles = ['Input Image', 'Refined Reconstruction']
    for i in range(data.shape[0]):
        
        # --- Annotate with MSE ---
        mse_text = f"MSE: {mse_per_image[i]:.4f}"
        axes[0, i].set_title(mse_text, fontsize=8, color='blue', y=1.05)
        # -------------------------
        
        axes[0, i].imshow(data[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].imshow(refined_recon[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        
        for r in range(2):
            axes[r, i].axis('off')
            if i == 0: axes[r, i].set_ylabel(titles[r], rotation=0, labelpad=40, fontsize=10)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show() # Use plt.show() for notebook display

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
print(f"--- Loading Checkpoint: {Config.UNET_WEIGHT_FILENAME} and Visualizing ---")

# 1. Instantiate VQ-VAE and U-Net models
model_container = DecomposeVAE(weight_path=Config.WEIGHT_PATH, device=Config.DEVICE) 
unet_model = ResidualLatentUNet(model_container=model_container, device=Config.DEVICE).to(Config.DEVICE)

# 2. Load U-Net Checkpoint
checkpoint_path = os.path.join(Config.OUTPUT_DIR, Config.UNET_WEIGHT_FILENAME)
try:
    unet_model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    print(f"✓ U-Net checkpoint loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load U-Net checkpoint from {checkpoint_path}. Ensure the file exists and the model architecture is correct.")
    raise e

# 3. Load Validation Data
transform = torchvision.transforms.ToTensor()
val_dataset = datasets.MNIST(root=Config.DATA_DIR, train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# 4. Run Visualization
visualize_input_output(unet_model, val_loader, Config.DEVICE, Config.VISUALIZATION_TITLE)

print("--- Visualization Complete ---")
