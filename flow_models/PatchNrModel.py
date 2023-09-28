import lightning.pytorch as pl
import torch
import torch.nn as nn
import FrEIA.framework as ff
import FrEIA.modules as fm
import torch.distributions as td


def patch_nr_net(layers, hidden_layer_sub_net, input_dim_flat):
    net_fn = lambda c_in, c_out: nn.Sequential(nn.Linear(c_in, hidden_layer_sub_net), nn.ReLU(),
                             nn.Linear(hidden_layer_sub_net, hidden_layer_sub_net), nn.ReLU(),
                             nn.Linear(hidden_layer_sub_net, c_out))
    graph = [ff.InputNode(input_dim_flat, name='input')]
    for k in range(layers):
        graph.append(ff.Node(graph[-1],
                             fm.GLOWCouplingBlock,
                             {'subnet_constructor': net_fn, 'clamp': 1.6}, name=f'glow block {k}'))
        graph.append(ff.Node(graph[-1], fm.PermuteRandom, {'seed': (k + 1)}, name=f'permute random {k}'))
       # graph.append(ff.Node(graph[-1], fm.ActNorm, {}, name=f'actnorm block K={k}'))
    graph.append(ff.OutputNode(graph[-1], name='output'))

    return ff.GraphINN(graph, verbose=False)


class PatchNrModel(pl.LightningModule):
    def __init__(self, layers, hidden_layer_node_count, input_dimension, learning_rate):
        super().__init__()
        self.lr = learning_rate
        self.prior = td.MultivariateNormal(torch.zeros(input_dimension), torch.eye(input_dimension))
        self.hidden_layer_node_count = hidden_layer_node_count
        self.input_dimension = input_dimension
        self.save_hyperparameters()
        self.patch_nr_model = patch_nr_net(layers, hidden_layer_node_count, input_dimension)

    def sample(self, number):
        sample = self.prior.sample((number, ))
        return self.patch_nr_model(sample, rev=False)

    def training_step(self, batch, batch_id):
        loss = 0
        z, z_jac = self.patch_nr_model(batch, rev=True)
        loss += torch.mean(0.5 * torch.sum(z**2, dim=1) - z_jac)
        # print(loss)
        self.log_dict({"loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.patch_nr_model.parameters(), lr=self.lr)



