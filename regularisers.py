import abc

import torch
from torch import Tensor
import torch.nn.functional as F

from flow_models.FlowModel import FlowModel
from img_utils import PatchExtractor


class Regulariser:
    @abc.abstractmethod
    def evaluate(self, input: Tensor):
        raise NotImplementedError('Evaluate not implemented')

    @abc.abstractmethod
    def get_hparams(self):
        raise NotImplementedError('Regulariser should return the hyperparameters with this function')


class PatchNrRegulariser(Regulariser):

    def __init__(self, flow_model: FlowModel, p_size=6, sample_number=50000, padding=True, padding_size=8, device='cpu'):
        self.padding = padding
        self.padding_size = padding_size
        self.sample_number = sample_number
        self.flow_model = flow_model.to(device)
        self.patch_size = p_size
        self.patch_extractor = PatchExtractor(p_size=p_size, device=device)

    def loss(self, batch):
        self.flow_model.eval()
        z, z_log_det = self.flow_model(batch, rev=True)
        return torch.mean(torch.sum(z**2, dim=1)/2) - torch.mean(z_log_det)

    def get_hparams(self):
        return {
            "p_size": self.patch_size,
            "sample_number": self.sample_number,
            "padding_size": self.padding_size,
            "padding": self.padding
        }

    def evaluate(self, img):
        padded_img = F.pad(img, [self.padding_size, self.padding_size], mode='reflect') if self.padding else img
        batch = self.patch_extractor.extract(padded_img, self.sample_number)
        return self.loss(batch)
