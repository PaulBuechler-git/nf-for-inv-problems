#!/opt/conda/bin conda run -n base python
import torch
import torchvision.transforms as T
from flow_models.PatchFlowModel import PatchFlowModel
from img_utils import ImageLoader
from patch_flow_trainer import patch_flow_trainer
from transforms import image_dequantization, image_normalization

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

patch_size = 7

def log_likelihood_loss(z, z_log_det):
    return torch.mean(0.5 * torch.sum(z**2, dim=1) - z_log_det)

model = PatchFlowModel(hparams={"num_layers": 5, "sub_net_size": 512, "dimension": patch_size ** 2})

deq_normalization = T.Compose([
    image_dequantization(device=DEVICE),
    image_normalization()])

train_images = ImageLoader('data/material_pt_nr/train.png', transform=deq_normalization, device=DEVICE)
validation_images = ImageLoader('data/material_pt_nr/validate.png', transform=deq_normalization, device=DEVICE)

patch_flow_trainer('custom_patch_nr', 'results/patch_nr', model, log_likelihood_loss, train_images, validation_images, steps=750000, patch_size=patch_size, device=DEVICE)





