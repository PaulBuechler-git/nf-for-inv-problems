import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from argparse import ArgumentParser
import lightning.pytorch as pl

from dataset.PatchDataset import PatchDataset
from flow_models.PatchNrModel import PatchNrModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(parsed_args):
    print("Training of PatchNr Flow")
    print(f'Params: {parsed_args}')

    batches_per_epoc = parsed_args.batches_per_epoc
    epochs = parsed_args.epoc

    dims = parsed_args.patch_size**2
    add_noise = 1
    noise_dist = torch.distributions.MultivariateNormal(torch.zeros(dims), torch.eye(dims))

    p_dims = (parsed_args.patch_size, parsed_args.patch_size)
    # Data transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda patch: torch.clip(patch + noise_dist.sample().reshape((1, *p_dims)), 0, 255)),
        transforms.Lambda(lambda patch: patch / 255.0),
        transforms.Lambda(lambda patch: patch.flatten())])


    # Train Dataset
    #train_data_set = PatchDataset('./data/set12/train', p_dims, transform=transform)
    train_data_set = PatchDataset('./data/DIV2K/train', p_dims, output_transform=transform, input_transform=transforms.Compose([transforms.CenterCrop((2000, 1000)),
                                                                                                                                transforms.Grayscale(num_output_channels=1)]))
    train_dataloader = DataLoader(train_data_set, batch_size=parsed_args.batch_size, shuffle=True, num_workers=1)
    needed = parsed_args.batch_size * epochs * batches_per_epoc
    print(f'train data length: {len(train_data_set)}, needed: {epochs * batches_per_epoc * parsed_args.batch_size}')
    if len(train_data_set) < needed:
        raise ValueError(f'Please add data values to the dataloader as the training needs {needed} '
                         f'patches (missing: {len(train_data_set)-needed} patches)')

    model = PatchNrModel(layers=parsed_args.layers, hidden_layer_node_count=parsed_args.hidden_nodes,
                         input_dimension=parsed_args.patch_size**2, learning_rate=parsed_args.lr)
    logger = TensorBoardLogger("tb_logs", name="patch_nr_model")
    trainer = pl.Trainer(limit_train_batches=batches_per_epoc, max_epochs=epochs, logger=logger)
    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser(description="Patch Nr training")
    # Model
    parser.add_argument("--layers", type=int, default=15, help="Layers in the parameter networks")
    parser.add_argument("--hidden_nodes", type=int, default=256, help="Hidden nodes in the parameter networks")

    # Data
    parser.add_argument("--patch_size", type=int, default=7, help="Patch size for the training")
    parser.add_argument("--train_data_path", type=str, default='./data/set12/train',
                        help="Path to the directory with the training images")
    parser.add_argument("--eval_data_path", type=str, default='./data/set12/test',
                        help="Path to the directory with the training images")
    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the training")
    parser.add_argument("--batches_per_epoc", type=int, default=1000, help="Batches per epoc")
    parser.add_argument("--epoc", type=int, default=100, help="Epoch for the training")
    args = parser.parse_args()
    main(args)

