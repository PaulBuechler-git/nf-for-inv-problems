#!/usr/bin/python
# This code is related to the paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl (2023).
# PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.
# Inverse Problems, vol. 39, no. 6.
#
# Modifications were made by Paul Büchler


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
    quiet = passed_args.quiet
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
    model = PatchNrModel(num_layers, subnet_nodes, patch_size).to(DEVICE)
    writer = SummaryWriter(log_dir=path)
    batch_size = 32
    optimizer_steps = 750000

    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    print(f'Using Device: {DEVICE}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_data = FastPatchDataset('./data/material_pt_nr/train.png', p_dims=patch_size, device=DEVICE)
    eval_data = FastPatchDataset('./data/material_pt_nr/validate.png', p_dims=patch_size, device=DEVICE)
    eval_loss = []
    train_loss = []
    iterations = tqdm(range(optimizer_steps)) if not quiet else range(optimizer_steps)
    for it in iterations:
        # extract patches
        patch_batch = train_data.get_batch(batch_size)
        # compute loss
        loss = 0
        invs, jac_inv = model.inverse_and_log_det(patch_batch)
        loss += torch.mean(0.5 * torch.sum(invs ** 2, dim=1) - jac_inv)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not quiet:
            if it % 10 == 0:
                # update tqdm description (visible loss)
                iterations.set_description(f'T/E:{loss.item()} - {eval_loss[-1][1] if len(eval_loss) > 0 else 0}')
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
    parser.add_argument('--quiet', type=bool, default=True, help="displays progressbar or not")
    main(parser.parse_args())
