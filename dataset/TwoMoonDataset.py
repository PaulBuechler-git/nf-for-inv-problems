import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import Dataset
from torchvision import transforms as tf


class TwoMoonDataset(Dataset):
    def __init__(self, sample_count, noise=0.1, transforms=tf.Compose([])):
        self.sample_count = sample_count
        self.noise = noise
        self.transforms = transforms
        data, _ = datasets.make_moons(n_samples=sample_count, noise=self.noise)
        self.data = torch.from_numpy(np.array(data, dtype=np.float32))

    def __len__(self):
        return self.sample_count

    def __getitem__(self, item):
        return self.transforms(self.data[item])
