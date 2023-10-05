import numpy as np
import torch.distributions


class NoiseOperator:
    def __init__(self, mean=0, std=1):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        noise = torch.tensor(np.random.normal(self.mean, self.std, img.size()), dtype=torch.float)
        return torch.clamp(img + noise*1/255., min=0, max=1)





