import torch
import torchvision.transforms as T
import numpy as np


def image_normalization():
    return T.Normalize([0, ], [256., ])


def image_dequantization(device):
    return T.Lambda(lambda img: img + torch.rand(size=img.size(), device=device))
