import normflows as nf
import torch
import lightning.pytorch as pl


class TwoMoonFlow(pl.LightningModule):
    def __init__(self, input_dims, layers, hidden_nodes, lr=1e-4):
        super().__init__()
        self.input_dims = input_dims
        self.layers = layers
        self.hidden_nodes = hidden_nodes
        self.lr = lr
        self.save_hyperparameters()
        self.model = self.get_model(layers, hidden_nodes)

    @staticmethod
    def get_model(layers, hidden_nodes):
        flows = []
        for i in range(layers):
            param_map = nf.nets.MLP([1, hidden_nodes, hidden_nodes, 2], init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            flows.append(nf.flows.Permute(2, mode='swap'))
        prior = nf.distributions.DiagGaussian(2)
        return nf.NormalizingFlow(prior, flows)

    def set_prior(self, prior):
        self.model.q0 = prior

    def sample(self, n_samples):
        z, log_z = self.model.sample(n_samples)
        return z.detach()

    def training_step(self, batch, batch_id):
        loss = self.model.forward_kld(batch)
        self.log_dict({'forward_kld': loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)
