import json
import math
import os.path
from argparse import ArgumentParser

from PIL import Image
from torchvision import transforms

from file_utils import create_versioned_dir, save_normalized
from flow_models.PatchFlowModel import PatchFlowModel
from img_utils import ImageLoader
from operators import BlurOperator
import torch
from kernels import gaussian_kernel_generator
from regularisers import PatchNrRegulariser
import numpy as np
from transforms import image_normalization
from variational_model_solver import variational_model_solver

def main(device, parsed_args):
    name = parsed_args.name
    model_path = parsed_args.model
    image_path = parsed_args.image
    result_path = parsed_args.result_path
    lam = parsed_args.lam
    steps = parsed_args.steps
    kernel_size = parsed_args.kernel_size
    kernel_std = parsed_args.kernel_std
    noise_std = parsed_args.noise_std
    reg_samples = parsed_args.reg_samples

    dir = create_versioned_dir(result_path, name)
    with open(os.path.join(dir, 'params.json'), 'w') as fp:
        json.dump(vars(parsed_args), fp)

    model = PatchFlowModel(path=model_path)
    patch_size = int(math.sqrt(model.hparams['dimension']))
    patch_nr_regulariser = PatchNrRegulariser(model, p_size=patch_size, sample_number=reg_samples, padding=True,
                                              padding_size=8, device=device)

    gaussian_kernel = gaussian_kernel_generator(kernel_size, kernel_std).to(device)
    blur_operator = BlurOperator(gaussian_kernel, device=device)

    normalization = transforms.Compose([image_normalization()])
    image_loader = ImageLoader(image_path, device=device, transform=normalization)
    ground_truth = image_loader[0].to(device)

    degraded_image = blur_operator(ground_truth)
    c, w, h = degraded_image.shape
    error_dim = w * h
    noise_vector_std1 = torch.reshape(torch.tensor(np.random.normal(0, noise_std, error_dim), device=device), (1, w, h))
    noise_degraded_image = degraded_image + normalization(noise_vector_std1 * 5.)

    startpoint = noise_degraded_image.clone()

    reconstructed_image, loss, likelihood, reg = variational_model_solver(noise_degraded_image, startpoint, blur_operator, regulariser=patch_nr_regulariser, lam=lam, device=device, steps=steps)
    rec_img = reconstructed_image.detach().cpu().squeeze().numpy()
    deg_img = degraded_image.detach().cpu().squeeze().numpy()
    gt_img = ground_truth.cpu().squeeze().numpy()
    loss = torch.tensor(loss, device='cpu').detach().numpy()
    likelihood = torch.tensor(likelihood, device='cpu').detach().numpy()
    regulariser = torch.tensor(reg, device='cpu').detach().numpy()

    np.save(os.path.join(dir, 'reconstructed_image.npy'), rec_img)
    save_normalized(os.path.join(dir, 'reconstructed_image.png'), rec_img)
    np.save(os.path.join(dir, 'degraded_image.npy'), deg_img)
    save_normalized(os.path.join(dir, 'degraded_image.png'), deg_img)
    np.save(os.path.join(dir, 'ground_truth_image.npy'), gt_img)
    save_normalized(os.path.join(dir, 'ground_truth_image.png'), gt_img)
    np.save(os.path.join(dir, 'loss.npy'), np.array([loss, likelihood, regulariser]))


if __name__ == "__main__":
    parser = ArgumentParser(description="Patch Flow Image reconstruction ")
    parser.add_argument("--name", type=str, default="material_reconstruction")
    # reconstruction
    parser.add_argument("--model", type=str, default="results/patch_nr/custom_patch_nr/version_3/custom_patch_nr_final.pth", help="model_path")
    parser.add_argument("--image", type=str, default="data/material_pt_nr/test.png", help="image_path")
    parser.add_argument("--result_path", type=str, default="results/deblurring")
    parser.add_argument("--lam", type=float, default=0.15, help="Lambda")
    parser.add_argument("--steps", type=int, default=600, help="reconstruction steps")

    #image degradation
    parser.add_argument("--kernel_size", type=int, default=9, help="Kernel size")
    parser.add_argument("--kernel_std", type=int, default=1, help="Kernel std")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Noise std")

    #regulariser
    parser.add_argument("--reg_samples", type=int, default=50000)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #
    main(DEVICE, parser.parse_args())