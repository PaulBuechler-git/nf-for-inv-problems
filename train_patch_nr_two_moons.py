from argparse import ArgumentParser
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning.pytorch as pl
from dataset.TwoMoonDataset import TwoMoonDataset
from models.PatchNrModel import PatchNrModel


def main(prop_args):
    batch_size = prop_args.batch_size
    sample_count = 100000
    epochs = prop_args.epoc
    layers = prop_args.layers
    hidden_nodes = prop_args.hidden_nodes
    lr = prop_args.lr
    dims = 2
    noise = prop_args.noise
    # Data transforms
    transform = transforms.Compose([transforms.Lambda(lambda p: 0.20*(p + 2))])

    # Train Dataset
    train_data_set = TwoMoonDataset(sample_count=sample_count, noise=noise, transforms=transform)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False, num_workers=1)

    model = PatchNrModel(layers=layers, hidden_layer_node_count=hidden_nodes,
                         input_dimension=dims, learning_rate=lr)
    logger = TensorBoardLogger("tb_logs", name="patch_nr_tm")
    trainer = pl.Trainer(max_epochs=epochs, logger=logger)
    trainer.fit(model, train_data_loader)


if __name__ == "__main__":
    parser = ArgumentParser(description="Glow training")
    # Model
    parser.add_argument("--layers", type=int, default=5, help="Layers in the parameter networks")
    parser.add_argument("--hidden_nodes", type=int, default=32, help="Hidden nodes in the parameter networks")

    #Data
    parser.add_argument("--noise", type=float, default=0.5, help="Noise level for two moon distribution")
    # Training

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the training")
    parser.add_argument("--epoc", type=int, default=10, help="Epoch for the training")
    main(parser.parse_args())
