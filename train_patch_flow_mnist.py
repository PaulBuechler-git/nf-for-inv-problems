from argparse import ArgumentParser
import normflows as nf
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning.pytorch as pl
from dataset.TwoMoonDataset import TwoMoonDataset
from flow_models.PatchFlow import PatchFlow
from flow_models.TwoMoonFlow import TwoMoonFlow
import torchvision as tv


def main(prop_args):
    batch_size = prop_args.batch_size
    epoc = prop_args.epoc

    transform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), nf.utils.Scale(255. / 256.), nf.utils.Jitter(1 / 256.)])
    train_data = tv.datasets.MNIST('data/', train=True,
                                     download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               drop_last=True)

    test_data = tv.datasets.MNIST('data/', train=False,
                                    download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    img_dims = (1, 28, 28)
    multiscale_blocks = 2
    block_depth = 16
    hidden_channels = 256

    model = PatchFlow(input_dims=img_dims, multiscale_blocks=multiscale_blocks, block_depth=block_depth, hidden_channels=hidden_channels)
    logger = TensorBoardLogger("tb_logs", name="patch_flow_mnist")
    ckpt_callback = pl.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=1000)
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", max_epochs=epoc, logger=logger, callbacks=[ckpt_callback])
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    parser = ArgumentParser(description="PatchFlow Mnist training ")
    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the training")
    parser.add_argument("--epoc", type=int, default=5, help="Epoch for the training")
    main(parser.parse_args())
