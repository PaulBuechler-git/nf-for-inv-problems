from PatchDataset import PatchDataset
import torch

from PatchExtractor import PatchExtractor
from torchvision.transforms import Compose, PILToTensor
from PIL import Image

p_dims = (7, 7)

# training_data_set = PatchDataset('./data/set12/train/01.png', p_dims, transform=None)


def check_dimensions_match(dataset):
    first_tensor = dataset[0]
    expected_shape = first_tensor.shape
    ilist = []
    for idx in range(len(dataset)):
        tensor = dataset[idx]
        if tensor.shape != expected_shape:
           # print(f"Tensor {idx} has shape {tensor.shape}, expected {expected_shape}")
            ilist.append([idx, tensor.shape])
    return ilist


# img = Image.open('./data/set12/train/01.png')
# transform_PIL = Compose([PILToTensor()])
# img = transform_PIL(img)
# print(img.shape)
# extractor = PatchExtractor(img, p_dims, pad=False)
# print(len(extractor))


dataset = PatchDataset('./data/set12/test', p_dims)
print(len(dataset))
lst = check_dimensions_match(dataset)
print(len(lst))
