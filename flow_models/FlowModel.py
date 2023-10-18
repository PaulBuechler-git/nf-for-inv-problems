import abc

import torch
from torch import nn


class FlowModel(nn.Module):
    def __init__(self, hparams=None, path=None, device='cpu'):
        super().__init__()
        if not path is None:
            model_dict = torch.load(path, map_location=device)
            print('hparams' in model_dict)
            if 'hparams' in model_dict:
                self.hparams = model_dict['hparams']
                print(self.hparams)
            else:
                if not hparams is None:
                    self.hparams = hparams
                else:
                    raise Exception('No hyperparameters for model')
            created_model = self._create_model(**self.hparams)
            created_model.load_state_dict(model_dict['net_state_dict'])
            self.model = created_model
        else:
            if hparams:
                self.hparams = hparams
            else:
                raise Exception('No hyperparameters for model')
            created_model = self._create_model(**self.hparams)
            self.model = created_model

    @abc.abstractmethod
    def _create_model(self, **kwargs) -> nn.Module:
        raise NotImplementedError('create_model not implemented')

    def get_state(self):
        return {
            "net_state_dict": self.model.state_dict(),
            "hparams": self.hparams
        }

    def get_hparams(self):
        return self.hparams

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

