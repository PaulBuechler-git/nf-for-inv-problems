import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


def gaussian_kernel_generator(kernel_size, std):
    """ Method that generates Gaussian Kernel matrices"""
    if kernel_size % 2 == 0:
        raise ValueError(f'Kernel size has to be odd')
    distribution = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2) * std)
    kernel = torch.zeros(kernel_size, kernel_size)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            pos = torch.tensor([i - center, j - center])
            kernel[i, j] = torch.exp(distribution.log_prob(pos))
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


class BlurOperator:
    def __init__(self, kernel, device='cpu'):
        self.device = device
        self.kernel = kernel
        self.kernel_size = kernel.squeeze().size(0)

    def __call__(self, image, padding=True):
        if padding:
            pad_trans = transforms.Pad([self.kernel_size // 2, ], padding_mode="reflect")
            image = pad_trans(image)
        return F.conv2d(image, self.kernel).to(self.device)
