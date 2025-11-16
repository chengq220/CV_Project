import torch
import numpy as np
import os
from PIL import Image
import math
import matplotlib.pyplot as plt

"""
Saves a batch of image into grids
"""
def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    
    batch_size, H, W, _ = img_tensors.shape   # get correct H and W
    ncols = math.ceil(batch_size / nrows)     # ceil to fit all images
    img_arr = np.zeros((nrows * H, ncols * W, 3))

    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * W
        row_end = row_start + W
        col_start = col_idx * W
        col_end = col_start + W
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")
    return 1

def plot_image_batch(img_tensors, nrows=None, figname=None):
    N, C, H, W = img_tensors.shape

    if nrows is None:
        nrows = int(math.sqrt(N))
    ncols = math.ceil(N / nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    axes = axes.flatten()

    for i in range(nrows * ncols):
        ax = axes[i]
        ax.axis('off')
        if i < N:
            img = img_tensors[i].detach().cpu()
            if C == 1:
                img = img.squeeze(0)
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                img = img.permute(1, 2, 0)
                ax.imshow(img, vmin=0, vmax=1)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)
    plt.show()

"""
Returns the weights and the training status of the model
"""
def load_weights(weight_path, device):
    assert os.path.isfile(weight_path), "Path is invalid"
    load_dict = torch.load(weight_path, map_location=device)
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
