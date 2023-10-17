from argparse import ArgumentParser
import normflows as nf
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T
import lightning.pytorch as pl

from dataset.FastPatchExtractor import FastPatchDataset
from flow_models.PatchFlow import PatchFlow
import torchvision as tv


def main(prop_args):
    batch_size = prop_args.batch_size
    epoc = prop_args.epoc
    channels = 1
    patch_size = 8

    img_dims = (channels, patch_size, patch_size)
    patch_dims = (patch_size, patch_size)
    train_img_path = '../data/material_pt_nr/train.png'
    transform = tv.transforms.Compose([nf.utils.Scale(255. / 256.),
                                       #nf.utils.Jitter(1 / 256.),
                                       T.Lambda(lambda im: im.reshape(*img_dims))])
    train_data = FastPatchDataset(train_img_path, p_dims=patch_dims,  transforms=transform)
    print(train_data[0].shape)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               drop_last=True)

    multiscale_blocks = 2
    block_depth = 16
    hidden_channels = 256*2

    model = PatchFlow(input_dims=img_dims, multiscale_blocks=multiscale_blocks, block_depth=block_depth, hidden_channels=hidden_channels)
    logger = TensorBoardLogger("../tb_logs", name="patch_flow_simple")
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
