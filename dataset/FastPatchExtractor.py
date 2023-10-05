import os

import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms as T


class FastPatchExtractor(nn.Module):
    def __init__(self, p_dim, pad=False, center=False):
        """
        Class that extracts patches of equal size from input images
        :param p_dim: Dimension of the patches. Expects an int
        :param pad: Enables/Disables padding around the edges in order to avoid side effects
        :param center: Enable/Disables if returned patches are normalized or not
        """
        super().__init__()
        self.pad = pad
        self.center = center
        self.p_dim = p_dim
        self.pad_size = p_dim - 1
        self.unfold = nn.Unfold(kernel_size=self.p_dim)

    def extract(self, image, batch_size=None):
        """
        Method that extracts all patches from the given image. If batch_size is set it returns a batch
        of the randomly permuted patches
        :param image: image with dimensions NxCxHxW
        :param batch_size: number of batches to be returned
        :return:
        """
        if self.pad:
            image = torch.cat((image, image[:, :, :self.pad_size, :]), 2)
            image = torch.cat((image, image[:, :, :, :self.pad_size]), 3)
        patches = self.unfold(image).squeeze(0).transpose(1, 0)

        if batch_size:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx, :]
        if self.center:
            patches = patches - torch.mean(patches, -1).unsqueeze(-1)
        return patches


class FastPatchDataset:
    transform_PIL = T.Compose([
        T.Lambda(lambda im: im.convert('RGB')),
        T.PILToTensor(),
        T.Grayscale(num_output_channels=1),
        T.Lambda(lambda i: i.to(torch.float32))])

    def __init__(self, path, p_dims, pad=False, center=False, device='cpu',
                 output_transforms=T.Compose([]), input_transform=T.Compose([])):
        super().__init__()
        self.p_dims = p_dims
        self.path = path
        self.device = device
        self.out_transforms = output_transforms
        self.in_transforms = input_transform
        self.patch_extractor = FastPatchExtractor(p_dims, pad=pad, center=center)
        self.images = self.load_images()

    def get_batch(self, batch_size):
        """
        Method that randomly selects one image of the provided and returns a random batch of the batch size
        :return Patch batch in flat shape NxP with N the batch size and P the squared patch size
        """
        image_count = len(self.images)
        random_index = np.random.randint(image_count)
        selected_img = self.images[random_index].unsqueeze(0)
        return self.patch_extractor.extract(selected_img, batch_size)

    def get_images(self):
        return self.images

    def load_images(self):
        """Method that returns the images in the shape of NxCxHxW"""
        transform_pipeline = T.Compose([
            # convert to RGB if RGBA is given
            T.Lambda(lambda im: im.convert('RGB')),
            # convert to pytorch tensor
            T.PILToTensor(),
            # convert to grayscale
            T.Grayscale(num_output_channels=1),
            # convert to torch float32
            T.Lambda(lambda i: i.to(torch.float32)),
            # normalize
            T.Lambda(lambda i: i / 255.),
        ])
        images = [transform_pipeline(Image.open(file)).to(self.device) for file in self.get_file_list()]
        # stacked = torch.stack(images, dim=0)
        stacked = images
        # normalize the images
        return stacked

    def get_file_list(self):
        if os.path.isdir(self.path):
            return [os.path.join(self.path, f) for f in os.listdir(self.path) if
                    f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        else:
            if os.path.isfile(self.path):
                file_types = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
                if self.path.endswith(file_types):
                    return [self.path]
                else:
                    raise ValueError(
                        f'Passed path ({self.path}) is not an image file in one of these formats: {file_types}')
            else:
                raise ValueError(f'Passed path ({self.path}) is not a file or directory.')
