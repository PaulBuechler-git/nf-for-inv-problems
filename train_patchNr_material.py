import torch
import torchvision.transforms as T
from patchNr.PatchFlowModel import PatchFlowModel
from core.Trainer import patch_flow_trainer, log_likelihood_loss
from core.Transforms import image_normalization
from core.img_utils import ImageLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

patch_size = 6

model = PatchFlowModel(hparams={"num_layers": 5, "sub_net_size": 512, "dimension": patch_size ** 2})

deq_normalization = T.Compose([
    image_normalization()])

train_images = ImageLoader('data/material_pt_nr/train.png', transform=deq_normalization, device=DEVICE)
validation_images = ImageLoader('data/material_pt_nr/validate.png', transform=deq_normalization, device=DEVICE)

patch_flow_trainer('custom_patch_nr', 'results/patch_nr', model, log_likelihood_loss, train_images, validation_images,
                   steps=750000, patch_size=patch_size, device=DEVICE)
