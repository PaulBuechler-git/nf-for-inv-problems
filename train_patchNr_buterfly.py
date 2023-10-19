import torch
import torchvision.transforms as T
from flow_models.PatchFlowModel import PatchFlowModel
from img_utils import ImageLoader
from trainer import patch_flow_trainer, log_likelihood_loss
from transforms import image_normalization

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

patch_size = 6

model = PatchFlowModel(hparams={"num_layers": 5, "sub_net_size": 512, "dimension": patch_size ** 2})

deq_normalization = T.Compose([
    image_normalization()])

train_images = ImageLoader('data/DIV2K/buterfly/train.png', transform=deq_normalization, device=DEVICE)
validation_images = ImageLoader('data/DIV2K/buterfly/validate.png', transform=deq_normalization, device=DEVICE)

patch_flow_trainer('butterfly', 'results/patch_nr', model, log_likelihood_loss, train_images, validation_images,
                   steps=1000, patch_size=patch_size, device=DEVICE)
