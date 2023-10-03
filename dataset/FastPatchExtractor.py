import torch.nn as nn
import torch.nn.functional as F

"""
Expects input image of shape N, C, H, W
"""


class FastPatchExtractor:

    def __init__(self, p_dims, pad=True, pad_mode='constant'):
        self.p_dims = p_dims
        self.pad = pad
        self.pad_mode = pad_mode
        self.unfold = nn.Unfold(kernel_size=self.p_dims)

    def get_padded(self, image):
        p_h, p_w = self.p_dims
        pad_w = p_w // 2
        pad_h = p_h // 2
        return F.pad(image, (pad_h, pad_h, pad_w, pad_w), mode=self.pad_mode)

    def extract(self, image, batch_size):
        if self.pad:
            p_h, p_w = self.p_dims
            pad_w = p_w // 2
            pad_h = p_h // 2
            image = F.pad(image, (pad_h, pad_h, pad_w, pad_w), mode=self.pad_mode)
        patches = self.unfold(image)
