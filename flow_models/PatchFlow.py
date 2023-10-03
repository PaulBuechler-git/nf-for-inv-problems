import lightning.pytorch as pl
import normflows as nf
import torch

'''Inspired by tutorial of normflows'''


class PatchFlow(pl.LightningModule):

    def __init__(self, input_dims, multiscale_blocks, block_depth, hidden_channels):
        super().__init__()
        self.input_dims = input_dims,
        self.multiscale_blocks = multiscale_blocks
        self.block_depth = block_depth
        self.hidden_channels = hidden_channels
        self.save_hyperparameters()
        self.model = self.get_model(input_dims, L=multiscale_blocks, K=block_depth, hidden_channels=hidden_channels)
        self.cached_loss = None

    @staticmethod
    def get_model(input_dims, L, K, hidden_channels, scale=True):
        channels, width, height = input_dims
        split_mode = 'channel'
        num_classes = 10
        q0 = []
        merges = []
        flows = []
        for i in range(L):
            flows_ = []
            for j in range(K):
                flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                              split_mode=split_mode, scale=scale)]
            flows_ += [nf.flows.Squeeze()]
            flows += [flows_]
            if i > 0:
                merges += [nf.flows.Merge()]
                latent_shape = (channels * 2 ** (L - i), width // 2 ** (L - i),
                                height // 2 ** (L - i))
            else:
                latent_shape = (channels * 2 ** (L + 1), width // 2 ** L,
                                height // 2 ** L)
            q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]
        return nf.MultiscaleFlow(q0, flows, merges)

    def training_step(self, batch, batch_id):
        img_batch, labels = batch
        loss = self.model.forward_kld(img_batch, labels)
        self.log_dict({'forward_kld': loss})
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            self.cached_loss = loss
        else:
            loss = self.cached_loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)
