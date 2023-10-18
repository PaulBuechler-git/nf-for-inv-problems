import abc

import torch
from torch import nn


class FlowModel(nn.Module):
    hparams = {}
    model = None

    def __init__(self, hparams=None, path=None, device='cpu'):
        super().__init__()
        if not path is None:
            model_dict = torch.load(path, map_location=device)
            if 'hparams' in model_dict:
                self.hparams = model_dict['hparams']
            if hparams:
                self.hparams = hparams
            else:
                raise Exception('No hyperparameters for model')
            self.model: nn.Module = self._create_model(**hparams)
            self.model.to(device)
            self.model.load_state_dict(model_dict['net_state_dict'])
        else:
            if hparams:
                self.hparams = hparams
            else:
                raise Exception('No hyperparameters for model')
            self.model: nn.Module = self._create_model(**hparams)
            self.model.to(device)

    @classmethod
    @abc.abstractmethod
    def _create_model(cls, **kwargs) -> nn.Module:
        raise NotImplementedError('create_model not implemented')

    @classmethod
    def get_hparams(cls):
        return cls.hparams

    @classmethod
    def get_model(cls):
        return cls.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

