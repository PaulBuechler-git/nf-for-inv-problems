import numpy as np
import torch


def save_checkpoint(target_path, model, model_hparams):
    checkpoint = {
        "model_state": model.state_dict(),
        "model_hparams": model_hparams,
    }
    torch.save(checkpoint, target_path)


def load_checkpoint(source_path):
    checkpoint = torch.load(source_path)
    return checkpoint["model_state"], checkpoint["model_hparams"]


def save_loss(target_path, loss):
    np.save(target_path, np.asarray(loss))


def load_loss(target_path):
    return np.load(target_path)
