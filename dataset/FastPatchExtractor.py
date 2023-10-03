import os

import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

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

    def extract(self, image):
        if not len(image.shape) == 4:
            squeezes = 4 - len(image.shape)
            for _ in range(squeezes):
                image = image.unsqueeze(0)
        n, c, w, h = image.shape
        p_h, p_w = self.p_dims
        if self.pad:
            pad_w = p_w // 2
            pad_h = p_h // 2
            image = F.pad(image, (pad_h, pad_h, pad_w, pad_w), mode=self.pad_mode)
        unfolded = self.unfold(image).permute(2, 0, 1)
        n, b, p = unfolded.shape
        return unfolded.reshape(n*b, p)


class FastPatchDataset(Dataset):
    transform_PIL = T.Compose([
        T.Lambda(lambda im: im.convert('RGB')),
        T.PILToTensor(),
        T.Grayscale(num_output_channels=1),
        T.Lambda(lambda i: i.to(torch.float32))])

    def __init__(self, path, p_dims, padding=True, padding_mode='reflect', transforms=None):
        super().__init__()
        self.p_dims = p_dims
        self.path = path
        self.files = self.get_file_list()
        self.transforms = transforms
        self.patch_extractor = FastPatchExtractor(p_dims=p_dims, pad=padding, pad_mode=padding_mode)
        self.patches = self.get_patches()

    def __len__(self):
        b, p_size = self.patches.shape
        return b

    def __getitem__(self, index):
        patch = self.patches[index]
        if self.transforms:
            patch = self.transforms(patch)
        return patch

    def get_patches(self):
        img_patches = None
        for file in self.files:
            image = self.transform_PIL(Image.open(file))
            patches = self.patch_extractor.extract(image)
            if not img_patches:
                img_patches = patches
            else:
                img_patches = torch.stack([img_patches, patches])
        return img_patches

    def get_file_list(self):
        if os.path.isdir(self.path):
            return [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        else:
            if os.path.isfile(self.path):
                file_types = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
                if self.path.endswith(file_types):
                    return [self.path]
                else:
                    raise ValueError(f'Passed path ({self.path}) is not an image file in one of these formats: {file_types}')
            else:
                raise ValueError(f'Passed path ({self.path}) is not a file or directory.')
