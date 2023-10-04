# This code belongs to the paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl (2023).
# PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.
# Inverse Problems, vol. 39, no. 6.
#
# Please cite the paper, if you use the code.
# The script trains the patchNR
import os.path
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm
from dataset.FastPatchExtractor import FastPatchDataset
from flow_models.PatchNrModel import PatchNrModel
from torch.utils.tensorboard import SummaryWriter
import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(passed_args):
    name = 'patch_nr_flow_material'
    patch_size = 6
    num_layers = 5
    subnet_nodes = 512
    lr = 1e-4
    # patchNR = model.create_NF(num_layers, subnet_nodes, dimension=patch_size**2)
    timestamp = datetime.datetime.now().strftime("version_%d-%m-%Y_%H:%M:%S")
    path = os.path.join('./tb_logs', name, timestamp)
    checkpoint_path = os.path.join(path, 'checkpoints')

    # initialize model
    model = PatchNrModel(num_layers, subnet_nodes, patch_size)
    writer = SummaryWriter(log_dir=path)
    batch_size = 32
    optimizer_steps = 750000
    h_params = {
        "patch_size": patch_size,
        "num_layers": num_layers,
        "subnet_nodes": subnet_nodes,
        "lr": lr,
        "batch_size": batch_size
    }

    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = FastPatchDataset('./data/material_pt_nr/train.png', p_dims=patch_size, device=DEVICE)
    eval_data = FastPatchDataset('./data/material_pt_nr/validate.png', p_dims=patch_size, device=DEVICE)
    eval_loss = []
    train_loss = []
    bar = tqdm(range(optimizer_steps))
    for it in bar:
        # extract patches
        patch_batch = train_data.get_batch(batch_size)
        # compute loss
        loss = 0
        invs, jac_inv = model.inverse_and_log_det(patch_batch)
        loss += torch.mean(0.5 * torch.sum(invs ** 2, dim=1) - jac_inv)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 10 == 0:
            # update tqdm description (visible loss)
            bar.set_description(f'T/E:{loss.item()} - {eval_loss[-1][1] if len(eval_loss) > 0 else 0}')
        if it % 100 == 0:
            # save training loss to tensorboard
            writer.add_scalar('loss/training', loss.item(), it)
            train_loss.append([it, loss.item()])

        # validation step
        if it % 1000 == 0:
            with torch.no_grad():
                eval_patch_batch = eval_data.get_batch(batch_size)
                invs, jac_inv = model.inverse_and_log_det(eval_patch_batch)
                val_loss = torch.mean(0.5 * torch.sum(invs ** 2, dim=1) - jac_inv).item()
                eval_loss.append([it, val_loss])
            # save test loss to tensorboard
            writer.add_scalar('loss/test', val_loss, it)
            torch.save({'net_state_dict': model.state_dict(), 'optimizer_dict': optimizer.state_dict(),
                        'hparams': model.get_model_h_params() }, os.path.join(checkpoint_path, f'weights_curr.pth'))
            np.save(os.path.join(path, 'eval_loss.npy'), np.array(eval_loss))
            np.save(os.path.join(path, 'train_loss.npy'), np.array(train_loss))
        # save weights
        if (it + 1) % 50000 == 0:
            it = int((it + 1) / 1000)
            torch.save({'net_state_dict': model.state_dict(), 'optimizer_dict': optimizer.state_dict(),
                        'hparams': model.get_model_h_params()}, os.path.join(checkpoint_path, f'weights_{str(it)}.pth'))
    torch.save({'net_state_dict': model.state_dict(), 'optimizer_dict': optimizer.state_dict(),
                'hparams': model.get_model_h_params()}, 'patchNR_custom/weights_final.pth')


if __name__ == "__main__":
    parser = ArgumentParser(description="PatchFlow training")
    parser.add_argument('--quiet', type=bool, default=False, help="displays progressbar or not")
    # Training
    # parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    # parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the training")
    # parser.add_argument("--epoc", type=int, default=5, help="Epoch for the training")
    main(parser.parse_args())
