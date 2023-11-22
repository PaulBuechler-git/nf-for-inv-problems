import os
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from matplotlib import patches, pyplot as plt
from torchvision import transforms as T


class PatchExtractor(nn.Module):
    def __init__(self, p_size, pad=False, center=False, device='cpu'):
        """
        Class that extracts patches of equal size from input images
        :param p_size: Dimension of the patches. Expects an int
        :param pad: Enables/Disables padding around the edges in order to avoid side effects
        :param center: Enable/Disables if returned patches are normalized or not
        """
        super().__init__()
        self.pad = pad
        self.center = center
        self.p_dim = p_size
        self.pad_size = p_size - 1
        self.device = device
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
        patches = self.unfold(image).squeeze(0).transpose(1, 0).to(self.device)

        if batch_size:
            idx = torch.randperm(patches.size(0), device=self.device)[:batch_size]
            patches = patches[idx, :]
        if self.center:
            patches = patches - torch.mean(patches, -1).unsqueeze(-1)
        return patches


class ImageLoader:

    def __init__(self, path, device='cpu', transform=T.Compose([])):
        self.path = path
        self.device = device
        self._images = self.load_images()
        self.transform = transform

    def __len__(self):
        return len(self._images)

    def get_random_image(self):
        idx = np.random.randint(0, len(self._images))
        return self.__getitem__(idx)

    def __getitem__(self, item):
        return self.transform(self._images[item])

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
        ])
        return [transform_pipeline(Image.open(file)).to(self.device) for file in self.get_file_list()]

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


def plot_image(path, img0, selected_patch=(100, 100), p_size=30):
    image = img0.to('cpu')
    out_dim = (p_size, p_size)
    patch_extractor = PatchExtractor(p_size=p_size, pad=True)
    c, x, y = image.size()
    p_x, p_y = selected_patch
    p_pos = p_y*x + p_x
    p_image0 = torch.reshape(patch_extractor.extract(image.unsqueeze(0))[p_pos], out_dim)
    fig, axes = plt.subplots(2, 1, figsize=(2, 4))
    rect1 = patches.Rectangle((p_x, p_y), p_size, p_size, linewidth=1, edgecolor='r',facecolor='none')
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].add_patch(rect1)
    axes[0].axis('off')
    axes[1].imshow(p_image0.squeeze(), cmap='gray')
    axes[1].axis('off')
    fig.tight_layout()
    fig.show()
    fig.savefig(path)