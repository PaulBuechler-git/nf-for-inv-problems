import os.path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T

import utils
from dataset.FastPatchExtractor import FastPatchDataset, FastImageLoader, FastPatchExtractor
from flow_models.PatchFlowModel import PatchFlowModel

import torch
import torch.nn as nn


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path = 'results'
model_name = 'patch_flow'
model_path = os.path.join(path, model_name)
version = 'neu'
version_path = os.path.join(model_path, version)
if not os.path.isdir(model_path):
    os.mkdir(model_path)
if not os.path.isdir(version_path):
    os.mkdir(version_path)

print(f'')
print(f'Using device: {DEVICE}')

hparams = {
    "num_layers": 10,
    "sub_net_size": 512,
    "patch_size": 6
}
torch.save(hparams, os.path.join(version_path, 'hparams.yaml'))
print(f'Created model with hyperparameters: {str(hparams)}')
f_model = PatchFlowModel(hparams)
model = f_model.get_model().to(DEVICE)
batch_size = 64
optimizer_steps = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load images
normalization = T.Compose([T.Normalize([0, ], [255., ])])
eval_image = FastImageLoader('data/material_pt_nr/validate.png', transform=normalization)
images = FastImageLoader('data/material_pt_nr/train.png', transform=normalization)

# Extractor for Patches
patch_extractor = FastPatchExtractor(hparams['patch_size'])
progress = tqdm(range(optimizer_steps))

stepwise_loss = []
loss_list = []
for step in progress:
    image = images.get_random_image()
    batch = patch_extractor.extract(image, batch_size=batch_size)

    loss = 0
    z, log_det = model(batch, rev=True)
    loss += torch.mean(0.5 * torch.sum(z**2, dim=1) - log_det)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    stepwise_loss.append(loss.item())
    if step % 100 == 0:
        with torch.no_grad():
            eval_batch = patch_extractor.extract(eval_image.get_random_image(), batch_size)
            z_eval, z_log_det_eval = model(eval_batch, rev=True)
            val_loss = torch.mean(0.5 * torch.sum(z_eval**2, dim=1) - z_log_det_eval)
        train_loss = np.mean(np.asarray(stepwise_loss))
        stepwise_loss = []
        progress.set_description(f'Train: {train_loss}, Eval:{val_loss}')
        loss_list.append((train_loss, val_loss))
        utils.save_loss('results/loss.npy', loss_list)
        utils.save_checkpoint('results/intermediate_checkpoint.cpt', model, f_model.get_hparams())
utils.save_checkpoint('results/final.cpt', model, f_model.get_hparams())





