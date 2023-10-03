from argparse import ArgumentParser
import normflows as nf
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning.pytorch as pl
from dataset.TwoMoonDataset import TwoMoonDataset
from flow_models.TwoMoonFlow import TwoMoonFlow


def main(prop_args):
    batch_size = prop_args.batch_size
    steps = prop_args.steps
    sample_count = batch_size*steps
    layers = prop_args.layers
    hidden_nodes = prop_args.hidden_nodes
    dims = 2
    noise = prop_args.noise
    lr = prop_args.lr
    # Data transforms
    # transform = transforms.Compose([transforms.Lambda(lambda p: 0.20 * (p + 2))])

    # Train Dataset
    train_data_set = TwoMoonDataset(sample_count=sample_count, noise=noise)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = TwoMoonFlow(input_dims=dims, layers=layers, hidden_nodes=hidden_nodes, lr=lr)
    logger = TensorBoardLogger("tb_logs", name="patch_flow_two_moon")
    ckpt_callback = pl.callbacks.ModelCheckpoint(save_top_k=-1)
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", max_steps=steps, logger=logger, callbacks=[ckpt_callback])
    trainer.fit(model, train_data_loader)


if __name__ == "__main__":
    parser = ArgumentParser(description="PatchFlow training ")
    # Model
    parser.add_argument("--layers", type=int, default=32, help="Layers in the parameter networks")
    parser.add_argument("--hidden_nodes", type=int, default=64, help="Hidden nodes in the parameter networks")

    # Data
    parser.add_argument("--noise", type=float, default=0.06, help="Noise level for two moon distribution")
    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for the training")
    parser.add_argument("--steps", type=int, default=8000, help="Epoch for the training")
    main(parser.parse_args())
