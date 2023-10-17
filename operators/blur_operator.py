import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


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
