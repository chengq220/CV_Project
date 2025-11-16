import torch
import numpy as np
import os
from PIL import Image


"""
Saves a batch of image into grids
"""
def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    (batch_size, img_size) = img_tensors.shape[:2]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * batch_size, ncols * batch_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")
    return 1

"""
Returns the weights and the training status of the model
"""
def load_weights(save_path, device):
    assert os.path.isfile(save_path), "Path is invalid"
    load_dict = torch.load(save_path, map_location=device)
    lr = load_dict['lr']
    epoch = load_dict['epoch']
    model = load_dict['model_state_dict']
    optimizer = load_dict['optimizer_state_dict']
    model_args = load_dict['model_args']
    return lr, epoch, model, optimizer, model_args


"""
Compute the residual between target and reconstruction images
"""
def computeResidual(tg, recon):
    assert tg.shape == recon.shape, "Shape mismatch between images"
    return recon - tg
