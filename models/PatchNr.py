"""import torch
import torch.nn as nn
import FrEIA.framework as ff
import FrEIA.modules as fm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_patch_nr_network(layers, hidden_layer_sub_net, input_dim_flat):
    def net_fn(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, hidden_layer_sub_net), nn.ReLU(),
                             nn.Linear(hidden_layer_sub_net, hidden_layer_sub_net), nn.ReLU(),
                             nn.Linear(hidden_layer_sub_net, c_out))

    graph = [ff.InputNode(input_dim_flat, name='input')]
    for k in range(layers):
        graph.append(ff.Node(graph[-1],
                             fm.GLOWCouplingBlock,
                             {'subnet_constructor': net_fn, 'clamp': 1.6}, name=f'glow block {k}'))
        graph.append(ff.Node(graph[-1], fm.PermuteRandom, {'seed': (k+1)}, name=f'permute random {k}'))
    graph.append(ff.OutputNode(graph[-1], name='output'))

    return ff.ReversibleGraphNet(graph, verbose=False).to(DEVICE)"""
import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_NF(num_layers, sub_net_size, dimension):
    """
    Creates the patchNR network
    """
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size,  c_out))
    nodes = [Ff.InputNode(dimension, name='input')]
    for k in range(num_layers):
        nodes.append(Ff.Node(nodes[-1],
                          Fm.GLOWCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':1.6},
                          name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                          Fm.PermuteRandom,
                          {'seed':(k+1)},
                          name=F'permute_flow_{k}'))
    nodes.append(Ff.OutputNode(nodes[-1], name='output'))

    model = Ff.ReversibleGraphNet(nodes, verbose=False).to(DEVICE)
    return model