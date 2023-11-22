import json
import os
import sqlite3
import torch
from tqdm import tqdm
from core.file_utils import create_versioned_dir
from core.FlowModel import FlowModel
from core.img_utils import PatchExtractor, ImageLoader


def log_likelihood_loss(z, z_log_det):
    return torch.mean(0.5 * torch.sum(z ** 2, dim=1) - z_log_det)


def patch_flow_trainer(name: str, path: str, model: FlowModel, loss_fn, train_images: ImageLoader, validation_images: ImageLoader,
                       patch_size=6, batch_size=64, steps=750000, val_each_steps=1000, loss_log_each_step=100, device='cpu',
                       quiet=False, lr=0.001):

    """Trainer for a patche based normalizing flow.
    It is inspired by the solution algorithm described in the PatchNr paper https://arxiv.org/abs/2205.12021 proposed
    by F. Altekrueger et al.
    This code is more generalized version with the possibility to adjust the parameters
    :param name: name of the trained model
    :param patch: target path to store the weights and loss Tensor containing the degraded input that should be reconstructed.
    :param model: Instance of the generalized FlowModel
    :param loss_fn:  loss function of the training
    :param train_images: ImageLoader instance providing the training images
    :param validation_images: ImageLoader instance providing the training images
    :param patch_size: patch_size for the training. should match the patch size
    :param device to be used by the algorithm. Either 'cpu' or 'gpu'
    :param batch_size: batch_size for the training
    :param steps: amount of training steps
    :param val_each_steps: step size for evaluation steps and checkpoint saving
    :param loss_log_each_step: step size for loss
    :param lr:  learning rate
    """
    if not quiet:
        print(f'Started training for model {name}. \n Will train {steps} steps in device={device}')
    dir = create_versioned_dir(path, name)
    if not quiet:
        print(f'The weights, loss and the parameters will be stored at this location: {dir}')
    hparams = model.get_hparams()
    hparams['patch_size'] = patch_size
    hparams['batch_size'] = batch_size
    hparams['device'] = str(device)
    hparams['train_img_path'] = train_images.path
    hparams['validation_img_path'] = validation_images.path
    hparams['model_name'] = name
    hparams['lr'] = lr
    #dump hparams
    json.dump(hparams, open(os.path.join(dir, 'hparams.yaml'), 'w'))

    # create sqllite3 conection to save the loss values
    connection = sqlite3.connect(os.path.join(dir, 'loss.db'))
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE flow_model_train_loss(step, loss)")
    cursor.execute("CREATE TABLE flow_model_validation_loss(step, loss)")
    connection.commit()

    #load model to device
    model.to(device)

    #init patch extractor
    patch_extractor = PatchExtractor(p_size=patch_size, device=device)
    progress_bar = tqdm(range(steps)) if not quiet else range(steps)

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not quiet:
        print(f'Optimizer is initialized with this parameters: {optimizer.state_dict()}')

    tmp_validation_loss = 0
    tmp_loss =0

    loss_buffer = []

    # perform training
    for step in progress_bar:
        train_image = train_images.get_random_image()
        train_patch_batch = patch_extractor.extract(train_image, batch_size)

        loss = 0
        z, z_log_det = model(train_patch_batch, rev=True)
        loss += loss_fn(z, z_log_det)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % loss_log_each_step == 0:
            tmp_loss = loss.item()
            if not quiet:
                progress_bar.set_description_str(f'T: {tmp_loss}, V:{tmp_validation_loss}')
            loss_buffer.append((step, loss.item()))

        if step % val_each_steps == 0:
            with torch.no_grad():
                val_image = validation_images.get_random_image()
                val_patch_batch = patch_extractor.extract(val_image, batch_size)
                z_val, z_val_log_det = model(val_patch_batch, rev=True)
                val_loss = loss_fn(z_val, z_val_log_det)
                tmp_validation_loss = val_loss.item()
                # write train loss buffer and validation loss to db buffer
                cursor.execute("INSERT INTO flow_model_validation_loss VALUES(?, ?)", (step, val_loss.item()))
                cursor.executemany("INSERT INTO flow_model_train_loss VALUES(?, ?)", loss_buffer)
                # commit buffered sql statements
                connection.commit()
                loss_buffer = []
                # save checkpoint
                torch.save(optimizer.state_dict(), os.path.join(dir, f'optimizer_dict.pth'))
                torch.save(model.get_state(), os.path.join(dir, f'{name}_intermediate.pth'))
                if not quiet:
                    progress_bar.set_description_str(f'T: {tmp_loss}, V:{tmp_validation_loss}')
    connection.close()
    torch.save(model.get_state(), os.path.join(dir, f'{name}_final.pth'))
