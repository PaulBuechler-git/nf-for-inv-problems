import matplotlib.pyplot as plt
from torchvision import transforms

from flow_models.PatchFlowModel import PatchFlowModel
from flow_models.PatchNrFlowModel import PatchNrFlowModel
from img_utils import ImageLoader
from operators import BlurOperator
import torch
from dataset.FastPatchExtractor import FastImageLoader
from kernels import gaussian_kernel_generator
from regularisers import PatchNrRegulariser
import numpy as np
from transforms import image_normalization
from utils import plot_image
from variational_model_solver import variational_model_solver

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device: {DEVICE}')

# persistence
# Regularizer params
patch_size = 6
regularizer_samples = 50000
model_path = 'results/patch_nr/custom_patch_nr/version_3/custom_patch_nr_final.pth'
#Target image
image_path = 'data/set12/01.png'
# Blur params
kernel_size = 9
std = 2
#noise params
sigma = 0.1
mu = 0
noise_level = 5.
#variational model parameter
steps = 300
lam = 0.87


patchNrFlowModel = PatchFlowModel(path=model_path)
patch_nr_regulariser = PatchNrRegulariser(patchNrFlowModel, p_size=patch_size, sample_number=regularizer_samples, padding=True, padding_size=8, device=DEVICE)

gaussian_kernel = gaussian_kernel_generator(kernel_size, std).to(DEVICE)
blur_operator = BlurOperator(gaussian_kernel, device=DEVICE)

normalization = transforms.Compose([image_normalization()])
image_loader = ImageLoader(image_path, device=DEVICE, transform=normalization)
ground_truth = image_loader[0].to(DEVICE)

degraded_image = blur_operator(ground_truth)
c, w, h = degraded_image.shape

error_dim = w*h
noise_vector_std1 = torch.reshape(torch.tensor(np.random.normal(mu, sigma, error_dim), device=DEVICE), (1, w, h))
noise_degraded_image = degraded_image + normalization(noise_vector_std1*noise_level)

startpoint = noise_degraded_image.clone()

reconstructed = variational_model_solver(noise_degraded_image, startpoint, blur_operator, regulariser=patch_nr_regulariser, lam=lam, device=DEVICE, steps=300)
img, loss_components = reconstructed