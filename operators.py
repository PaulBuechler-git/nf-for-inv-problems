import abc

import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F


class Operator:
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Operator has to be callable')


class NoiseOperator:
    def __init__(self, mean=0, std=1, device='cpu'):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, img):
        noise = torch.tensor(np.random.normal(self.mean, self.std, img.size()), device=self.device,
                             dtype=torch.float)
        return torch.clamp(img + noise * 1 / 255., min=0, max=1)


def append_noise(img, mean=0, std=1, device='cpu'):
    noise = torch.tensor(np.random.normal(mean, std, img.size()), device=device, dtype=torch.float) / 255.
    return torch.clamp(img + noise, min=0, max=1)


class BlurOperator(Operator):
    def __init__(self, kernel, device='cpu'):
        self.device = device
        self.kernel = kernel.to(device)
        self.kernel_size = kernel.squeeze().size(0)

    def __call__(self, image, padding=True):
        if padding:
            pad_trans = transforms.Pad([self.kernel_size // 2, ], padding_mode="reflect")
            image = pad_trans(image)
        return F.conv2d(image, self.kernel).to(self.device)
