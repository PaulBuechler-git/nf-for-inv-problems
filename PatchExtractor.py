import numpy as np
import random
from torch import Tensor
import torch
import torch.nn.functional as F


class PatchExtractor:
    img: Tensor
    p_dims: tuple
    pad: bool
    pad_mode: str

    """
    the input image should have the shape CHW
    """
    def __init__(self, img: Tensor, p_dims: tuple, pad=True, pad_mode='constant'):
        self.img = img
        self.p_dims = p_dims
        self.pad = pad
        self.pad_mode = pad_mode
        pad_img, pad_dims, inner_dims = self._create_base_img()
        self.base_img = pad_img
        self.base_dims = pad_dims
        self.inner_dims = inner_dims

    def extract_patch(self, x, y):
        x, y = self._get_patch_coordinates((x, y))
        x_s, x_e = x
        y_s, y_e = y
        return self.base_img[:, y_s:y_e, x_s:x_e]

    def extract_patch_by_index(self, index):
        c, h, w = self.inner_dims
        x = index % h
        y = index // h
        return self.extract_patch(x, y)

    def __getitem__(self, index):
        return self.extract_patch_by_index(index)

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        c, h, w = self.inner_dims
        if self.pos < w*h:
            patch = self.extract_patch_by_index(self.pos)
            self.pos += 1
            return patch
        else:
            raise StopIteration

    def __len__(self):
        c, h, w = self.inner_dims
        return w*h

    def get_random_patch(self):
        """

        :return: random patch following the
        """
        c, h, w = self.inner_dims
        random_x = int(random.uniform(0, w))
        random_y = int(random.uniform(0, h))
        return self.extract_patch(random_x, random_y)

    def _create_base_img(self) -> tuple:
        """
        method that flattens and pads the input image
        :return: tuple with flattened and padded image
        """
        p_h, p_w = self.p_dims
        if self.pad:
            pad_w = p_w // 2
            pad_h = p_h // 2
            padded = F.pad(self.img, (pad_h, pad_h, pad_w, pad_w), mode=self.pad_mode)
            return padded, padded.shape, self.img.shape
        else:
            c, h, w = self.img.shape
            pad_w = p_w // 2
            pad_h = p_h // 2
            inner_dims = (c, w - (2*pad_w), h-(2*pad_h))
            #inner_dims = (c, w - p_w-1, h - p_h-1)
            return self.img, self.img.shape, inner_dims

    def _get_patch_coordinates(self, img_coordinates: tuple) -> tuple:
        x, y = img_coordinates
        img_c, img_h, img_w = self.img.shape
        # check if the input coordinates are part of the image
        if x >= img_w or x < 0:
            raise ValueError(f'Value {x} out of range 0 < {x} < {img_w}')
        if y >= img_h or y < 0:
            raise ValueError(f'Value {y} out of range 0 < {y} < {img_h}')
        p_w, p_h = self.p_dims
        return (x, x + p_w), (y, y + p_h)


