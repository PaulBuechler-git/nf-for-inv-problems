import json
import os
import sqlite3

import torch
from tqdm import tqdm

from file_utils import create_versioned_dir
from flow_models.FlowModel import FlowModel
from img_utils import PatchExtractor, ImageLoader


def patch_flow_trainer(name: str, path: str, model: FlowModel, loss_fn, train_images: ImageLoader, validation_images: ImageLoader,
                       patch_size=6, batch_size=64, steps=750000, val_each_steps=1000, loss_log_each_step=100, device='cpu',
                       quiet=False, lr=0.005):
    if not quiet:
        print(f'Started training for model {name}. \n Will train {steps} steps in device={device}')
    dir = create_versioned_dir(path, name)
    if not quiet:
        print(f'The weights, loss and the parameters will be stored at this location: {dir}')
    hparams = model.get_hparams()
    hparams['patch_size'] = patch_size
    hparams['batch_size'] = patch_size
    hparams['device'] = str(device)
    hparams['train_img_path'] = train_images.path
    hparams['validation_img_path'] = validation_images.path
    hparams['model_name'] = name
    hparams['lr'] = lr
    json.dump(hparams, open(os.path.join(dir, 'hparams.yaml'), 'w'))


    # create sqllite3 conection to save the loss values
    connection = sqlite3.connect(os.path.join(dir, 'loss.db'))
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE flow_model_train_loss(step, loss)")
    cursor.execute("CREATE TABLE flow_model_validation_loss(step, loss)")
    connection.commit()



    model.to(device)

    patch_extractor = PatchExtractor(p_size=patch_size, device=device)
    progress_bar = tqdm(range(steps)) if not quiet else range(steps)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not quiet:
        print(f'Optimizer is initialized with this parameters: {optimizer.state_dict()}')

    tmp_validation_loss = 0
    tmp_loss =0

    loss_buffer = []

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
                # write train loss buffer and validation loss to db
                cursor.execute("INSERT INTO flow_model_validation_loss VALUES(?, ?)", (step, val_loss.item()))
                cursor.executemany("INSERT INTO flow_model_train_loss VALUES(?, ?)", loss_buffer)
                connection.commit()
                loss_buffer = []
                # save checkpoint
                torch.save(optimizer.state_dict(), os.path.join(dir, f'optimizer_dict.pth'))
                torch.save(model.get_state(), os.path.join(dir, f'{name}_intermediate.pth'))
                if not quiet:
                    progress_bar.set_description_str(f'T: {tmp_loss}, V:{tmp_validation_loss}')
    connection.close()
    torch.save(model.get_state(), os.path.join(dir, f'{name}_final.pth'))
