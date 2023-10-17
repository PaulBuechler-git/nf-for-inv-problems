import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn as nn
from flow_models.FlowModel import FlowModel


class PatchFlowModel(FlowModel):
    """Simple Flow model with Glow Blocks"""
    default_params = {"num_layers": 10, "sub_net_size": 256, "patch_size": 6}

    def __init__(self, hparams, state=None):
        super().__init__(hparams=self.default_params | hparams, state=state)

    @classmethod
    def _create_model(cls, num_layers, sub_net_size, patch_size):
        # This code belongs to the paper
        #
        # F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl (2023).
        # PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.
        # Inverse Problems, vol. 39, no. 6.

        sub_constructor = lambda c_in, c_out: nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                                                            nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                                                            nn.Linear(sub_net_size, c_out))
        nodes = [Ff.InputNode(patch_size**2, name='input')]
        for k in range(num_layers):
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': sub_constructor, 'clamp': 1.6},
                                 name=F'coupling_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.PermuteRandom,
                                 {'seed': (k + 1)},
                                 name=F'permute_flow_{k}'))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        return Ff.ReversibleGraphNet(nodes, verbose=False)
