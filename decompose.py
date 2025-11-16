import torch
from utils import load_weights
from vqvae import VQVAE

"""
Make it easier to get the invidual parts of the model
"""
class DecomposeVAE():
    def __init__(self, weight_path, device):
        _, _, model_w, _, model_args = load_weights(weight_path)
        fullVAE = VQVAE(**model_args).to(device)
        self.model = fullVAE.load_state_dict(model_w)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

    def getEncoder(self):
        return self.encoder
    
    def getDecoder(self):
        return self.decoder
    
    def getFullVAE(self):
        return self.model

