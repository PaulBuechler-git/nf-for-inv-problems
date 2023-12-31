import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn as nn
from core.FlowModel import FlowModel


class PatchFlowModel(FlowModel):

    def __init__(self, hparams={"num_layers": 5, "sub_net_size": 512, "dimension": 6 ** 2}, path=None, device='cpu'):
        super().__init__(hparams=hparams, path=path, device=device)

    def _create_model(self, num_layers=5, sub_net_size=512, dimension=36):
        # This code belongs to the paper
        #
        # F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl (2023).
        # PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.
        # Inverse Problems, vol. 39, no. 6.
        # https://github.com/FabianAltekrueger/patchNR

        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                                 nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                                 nn.Linear(sub_net_size, c_out))

        nodes = [Ff.InputNode(dimension, name='input')]
        for k in range(num_layers):
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet_fc, 'clamp': 1.6},
                                 name=F'coupling_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.PermuteRandom,
                                 {'seed': (k + 1)},
                                 name=F'permute_flow_{k}'))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))

        model = Ff.ReversibleGraphNet(nodes, verbose=False)
        return model
