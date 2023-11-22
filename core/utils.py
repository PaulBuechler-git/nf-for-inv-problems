import numpy as np
import torch
from matplotlib import pyplot as plt, patches

from img_utils import PatchExtractor


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


def plot_image(path, img0, selected_patch=(100, 100), p_size=30):
    image = img0.to('cpu')
    out_dim = (p_size, p_size)
    patch_extractor = PatchExtractor(p_size=p_size, pad=True)
    c, x, y = image.size()
    p_x, p_y = selected_patch
    p_pos = p_y*x + p_x
    p_image0 = torch.reshape(patch_extractor.extract(image.unsqueeze(0))[p_pos], out_dim)
    fig, axes = plt.subplots(2, 1, figsize=(2, 4))
    rect1 = patches.Rectangle((p_x, p_y), p_size, p_size, linewidth=1, edgecolor='r',facecolor='none')
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].add_patch(rect1)
    axes[0].axis('off')
    axes[1].imshow(p_image0.squeeze(), cmap='gray')
    axes[1].axis('off')
    fig.tight_layout()
    fig.show()
    fig.savefig(path)
