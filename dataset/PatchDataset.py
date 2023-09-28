import os
from torch.utils.data import Dataset
from .PatchExtractor import PatchExtractor
from PIL import Image
from torchvision.transforms import Compose, PILToTensor


class PatchDataset(Dataset):
    d_path: str
    p_dims: tuple
    p_extractors: list[PatchExtractor]
    file_list: list[str]
    p_amount: int
    transform_PIL = Compose([PILToTensor()])
    img_dims = None
    transform: Compose
    pad: bool
    pad_mode: str

    def __init__(self, d_path: str, p_dims: tuple,
                 pad: bool = False, pad_mode: str = 'constant',
                 transform=None,
                 device='cpu'):
        """
        Dataset for patches from images
        :type p_dims: tuple
        :type d_path:
        """
        self.d_path = d_path
        self.file_list = self.get_file_list()
        self.p_dims = p_dims
        self.transform = transform
        self.pad = pad
        self.pad_mode = pad_mode
        self.device = device
        self.p_extractors, self.img_dims = self.get_patch_extractors()
        self.p_amount = sum(map(lambda p_e: len(p_e), self.p_extractors))

    def __len__(self):
        return self.p_amount

    def get_patch_by_index(self, index):
        c, w, h = self.img_dims
        flattened_index_range = w * h
        pe_index = index // flattened_index_range
        p_index = index % flattened_index_range
        patch = self.p_extractors[pe_index][p_index]
        return patch, index, pe_index, p_index

    def __getitem__(self, index):
        patch, _, _, _ = self.get_patch_by_index(index)
        if self.transform:
            patch = self.transform(patch)
        return patch

    def get_patch_extractors(self):
        images = list(map(lambda file: self.transform_PIL(Image.open(file)).to(self.device), self.file_list))
        patch_extractors = list(map(lambda img: PatchExtractor(img, p_dims=self.p_dims, pad=self.pad, pad_mode=self.pad_mode),
                        images))
        return patch_extractors, patch_extractors[0].inner_dims

    def get_file_list(self):
        if os.path.isdir(self.d_path):
            return [os.path.join(self.d_path, f) for f in os.listdir(self.d_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        else:
            if os.path.isfile(self.d_path):
                file_types = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
                if self.d_path.endswith(file_types):
                    return [self.d_path]
                else:
                    raise ValueError(f'Passed path ({self.d_path}) is not an image file in one of these formats: {file_types}')
            else:
                raise ValueError(f'Passed path ({self.d_path}) is not a file or directory.')




