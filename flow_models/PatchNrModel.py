import normflows as nf


class PatchNrModel(nf.NormalizingFlow):
    def __init__(self, layers, hidden_layer_node_count, p_dim):
        super().__init__(*self._get_model(layers, hidden_layer_node_count, p_dim**2))
        self.layers = layers
        self.hidden_layer_node_count = hidden_layer_node_count
        self.p_dim = p_dim

    def _get_model(self, layers, hidden_nodes, input_dimensions):
        if input_dimensions % 2:
            raise ValueError('Input dimensions should be even')
        flows = []
        for i in range(layers):
            param_map = nf.nets.MLP([input_dimensions // 2, hidden_nodes, hidden_nodes, input_dimensions], init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map, split_mode='checkerboard'))
            flows.append(nf.flows.Permute(input_dimensions))
        prior = nf.distributions.DiagGaussian(input_dimensions)
        return prior, flows

    def get_model_h_params(self):
        return {
            'layers': self.layers,
            'hidden_layer_node_count': self.hidden_layer_node_count,
            'p_dim': self.p_dim
        }
