"""
Taken from https://github.com/jzbontar/pixelcnn-pytorch/blob/master/main.py
"""
import time

import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch import nn, optim, cuda
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class PixelCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PixelCNN, self).__init__()
        fm = hidden_dim
        self.net = nn.Sequential(
            MaskedConv2d('A', input_dim,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            nn.Conv2d(fm, output_dim, 1))
        
    def forward(self, x):
        return self.net(x)
    
def show_as_image(binary_image, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.imshow(binary_image, cmap='gray')
    plt.xticks([]); plt.yticks([])

def batch_images_to_one(batches_images, save):
    n_square_elements = int(np.sqrt(batches_images.shape[0]))
    rows_images = np.split(np.squeeze(batches_images), n_square_elements)
    if save:
        plt.savefig(save)
    return np.vstack([np.hstack(row_images) for row_images in rows_images])


def generate_samples(n_samples, model, starting_point=(0, 0), starting_image=None, IMAGE_WIDTH = 32, IMAGE_HEIGHT=32):
    samples = torch.from_numpy(
        starting_image if starting_image is not None else np.zeros((n_samples * n_samples, 1, IMAGE_WIDTH, IMAGE_HEIGHT))).float()

    model.train(False)

    for i in range(IMAGE_WIDTH):
        for j in range(IMAGE_HEIGHT):
            if i < starting_point[0] or (i == starting_point[0] and j < starting_point[1]):
                continue
            out = model(Variable(samples, volatile=True))
            probs = F.softmax(out[:, :, i, j]).data
            samples[:, :, i, j] = torch.multinomial(probs, 1).float()
    show_as_image(batch_images_to_one(generate_samples(n_samples=10)), figsize=(10, 20))
    return samples.numpy()

if __name__ == "__main__":
    device = "cuda:0"
    net = PixelCNN(input_dim=1,hidden_dim=64,output_dim=512).to(device)
    tr = data.DataLoader(datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor()),
                        batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    te = data.DataLoader(datasets.MNIST('data', train=False, download=False, transform=transforms.ToTensor()),
                        batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    optimizer = optim.Adam(net.parameters())
    for epoch in range(25):
        # train
        err_tr = []
        net.train(True)
        for input, _ in tr:
            input = Variable(input.to(device))
            target = Variable((input.data[:,0] * 255).long())
            loss = F.cross_entropy(net(input), target)
            err_tr.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute error on test set
        err_te = []
        net.train(False)
        for input, _ in te:
            input = Variable(input.to(device))
            target = Variable((input.data[:,0] * 255).long())
            loss = F.cross_entropy(net(input), target)
            err_te.append(loss.data[0])
        cuda.synchronize()
        time_te = time.time() - time_te

        # sample
        # sample.fill_(0)
        # net.train(False)
        # for i in range(28):
        #     for j in range(28):
        #         out = net(Variable(sample, volatile=True))
        #         probs = F.softmax(out[:, :, i, j]).data
        #         sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.

        print('epoch={}; nll_tr={:.7f}; nll_te={:.7f}'.format(
            epoch, np.mean(err_tr), np.mean(err_te)))
