from utils import load_weights
from models.vqvae import VQVAE
import torch.nn as nn 

"""
Make it easier to get the invidual parts of the model
"""
class DecomposeVAE():
    def __init__(self, weight_path, device):
        _, _, model_w, _, model_args = load_weights(weight_path=weight_path, device=device)
        self.model = VQVAE(**model_args).to(device)
        _ = self.model.load_state_dict(model_w)

    def getEncoder(self):
        return self.model.encoder
    
    def getQuantizer(self):
        return nn.Sequential(self.model.pre_vq_conv, self.model.vq)
    
    def getCodeBook(self):
        return self.model.vq.e_i_ts

    def getDecoder(self):
        return self.model.decoder

    def getFullVAE(self):
        return self.model

