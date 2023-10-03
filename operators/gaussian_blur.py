import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


class GaussianBlur:
    def __init__(self, kernel_size, sigma):
        if kernel_size % 2 == 0:
            raise ValueError(f'Kernel size has to be odd')
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.distribution = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)*(self.sigma**2))
        self.kernel = self.get_gaussian_kernel()

    def get_gaussian_kernel(self):
        kernel = torch.zeros(self.kernel_size, self.kernel_size)
        center = self.kernel_size // 2

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                pos = torch.tensor([i - center, j - center])
                kernel[i, j] = torch.exp(self.distribution.log_prob(pos))
        kernel /= kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def __call__(self, image, padding=True):
        if padding:
            pad_trans = transforms.Pad([self.kernel_size//2, ], padding_mode="reflect")
            image = pad_trans(image)
        return F.conv2d(image, self.kernel)


