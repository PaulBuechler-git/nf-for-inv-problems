import abc

import torch
from torch import nn


class FlowModel:

    hparams = {}

    def __init__(self, hparams, state=None):
        if not hparams and not state:
            raise Exception('Either hparams or state has to be set')
        self.hparams = hparams
        self.model: nn.Module = self._create_model(**hparams)
        if state:
            self.model.load_state_dict(state)

    @classmethod
    @abc.abstractmethod
    def _create_model(cls, **kwargs) -> nn.Module:
        raise NotImplementedError('create_model not implemented')

    def get_hparams(self):
        return self.hparams

    def get_model(self):
        return self.model
