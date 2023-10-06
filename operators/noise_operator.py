import numpy as np
import torch.distributions


class NoiseOperator:
    def __init__(self, mean=0, std=1, device='cpu'):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, img):
        noise = torch.tensor(np.random.normal(self.mean, self.std, img.size()), device=self.device, dtype=torch.float)
        return torch.clamp(img + noise*1/255., min=0, max=1)


def append_noise(img, mean=0, std=1, device='cpu'):
    noise = torch.tensor(np.random.normal(mean, std, img.size()), device=device, dtype=torch.float)/255.
    return torch.clamp(img + noise, min=0, max=1)






