import torch
from torch.utils.data import Dataset

class MNISTResidualDataset(Dataset):
    def __init__(self, mnist_dataset, vae_model):
        self.mnist = mnist_dataset
        self.vae = vae_model.eval()

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]        
        img = img.unsqueeze(0) 
        with torch.no_grad():
            recon = self.vae(img)["x_recon"] 
        residual = img - recon
        return residual.squeeze(0), label    
