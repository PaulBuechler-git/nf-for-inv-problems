import json
import math
import os.path
import sqlite3
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
import skimage.metrics as skim

def main(device, parsed_args):
    name = parsed_args.name
    model_path = parsed_args.model
    image_path = parsed_args.image
    result_path = parsed_args.result_path
    lam_start = parsed_args.lam_start
    lam_end = parsed_args.lam_end
    lam_steps = parsed_args.lam_steps
    steps = parsed_args.steps
    kernel_size = parsed_args.kernel_size
    kernel_std = parsed_args.kernel_std
    noise_std = parsed_args.noise_std
    noise_std_start = parsed_args.noise_std_start
    noise_std_end = parsed_args.noise_std_end
    noise_std_steps = parsed_args.noise_std_steps
    reg_samples = parsed_args.reg_samples

    result_dir = create_versioned_dir(result_path, name)
    dict = vars(parsed_args)
    dict['result_dir'] = result_dir
    with open(os.path.join(result_dir, 'params.json'), 'w') as fp:
        json.dump(dict, fp)

    model = PatchFlowModel(path=model_path)
    patch_size = int(math.sqrt(model.hparams['dimension']))
    patch_nr_regulariser = PatchNrRegulariser(model, p_size=patch_size, sample_number=reg_samples, padding=True,
                                              padding_size=8, device=device)

    # create db to store data
    connection = sqlite3.connect(os.path.join(result_dir, 'values.db'))
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE reconstruction_experiment_inc_noise_loss(loss, likelihood, reg, std_noise, lam)")
    cursor.execute("CREATE TABLE reconstruction_experiment_inc_noise_metrics(psnr, psnr_deg, ssim, ssim_deg, std_noise, lam)")
    connection.commit()

    gaussian_kernel = gaussian_kernel_generator(kernel_size, kernel_std).to(device)
    blur_operator = BlurOperator(gaussian_kernel, device=device)

    normalization = transforms.Compose([image_normalization()])
    image_loader = ImageLoader(image_path, device=device, transform=normalization)
    ground_truth = image_loader[0].to(device)

    degraded_image = blur_operator(ground_truth)

    for std_noise in range(np.linspace(noise_std_start, noise_std_end, noise_std_steps)):
        c, w, h = degraded_image.shape
        error_dim = w * h
        noise_vector_std1 = torch.reshape(torch.tensor(np.random.normal(0, std_noise, error_dim), device=device), (1, w, h))
        noise_degraded_image = degraded_image + normalization(noise_vector_std1 * 5.)
        startpoint = noise_degraded_image.clone()

        for lam in range(np.linspace(lam_start, lam_end, lam_steps)):
            reconstructed_image, loss, likelihood, reg = variational_model_solver(noise_degraded_image, startpoint, blur_operator,
                                                                                  regulariser=patch_nr_regulariser, lam=lam, device=device, steps=steps)
            rec_img = reconstructed_image.detach().cpu().squeeze().numpy()
            deg_img = degraded_image.detach().cpu().squeeze().numpy()
            gt_img = ground_truth.cpu().squeeze().numpy()
            loss = torch.tensor(loss, device='cpu').detach().numpy()
            likelihood = torch.tensor(likelihood, device='cpu').detach().numpy()
            regulariser = torch.tensor(reg, device='cpu').detach().numpy()
            noise_vals = np.repeat(noise_std, len(regulariser))
            lam_vals = np.repeat(lam, len(regulariser))
            joint_losses = zip(loss, likelihood, regulariser, noise_vals, lam_vals)

            image_dir = os.path.join(result_dir, f'lam_{lam}_noise_std_{std_noise}')
            os.mkdir(image_dir)
            np.save(os.path.join(image_dir, 'reconstructed_image.npy'), rec_img)
            save_normalized(os.path.join(image_dir, 'reconstructed_image.png'), rec_img)
            np.save(os.path.join(image_dir, 'degraded_image.npy'), deg_img)
            save_normalized(os.path.join(image_dir, 'degraded_image.png'), deg_img)
            np.save(os.path.join(image_dir, 'ground_truth_image.npy'), gt_img)
            save_normalized(os.path.join(image_dir, 'ground_truth_image.png'), gt_img)
            #np.save(os.path.join(image_dir, 'loss.npy'), np.array([loss, likelihood, regulariser]))
            psnr = skim.peak_signal_noise_ratio(gt_img, rec_img)
            psnr_deg = skim.peak_signal_noise_ratio(gt_img, deg_img)
            mssim, img_gt_rec_ssim  = skim.structural_similarity(gt_img, rec_img, win_size=15, data_range=1., full=True)
            mssim_deg, img_gt_deg_ssim  = skim.structural_similarity(gt_img, deg_img, win_size=15, data_range=1., full=True)
            np.save(os.path.join(image_dir, 'ssim_gt_rec_img.npy'))
            save_normalized(os.path.join(image_dir, 'ssim_gt_rec_img.png'), img_gt_rec_ssim)
            np.save(os.path.join(image_dir, 'ssim_gt_deg_img.npy'))
            save_normalized(os.path.join(image_dir, 'ssim_gt_deg_img.png'), img_gt_deg_ssim)
            cursor.execute("INSERT INTO reconstruction_experiment_inc_noise_loss VALUES(?, ?, ?, ?, ?)", joint_losses)
            cursor.execute("INSERT INTO reconstruction_experiment_inc_noise_metrics VALUES(?, ?, ?, ?, ?)", (psnr, psnr_deg, mssim, mssim_deg, std_noise, lam))
            connection.commit()
    connection.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Patch Flow Image reconstruction ")
    parser.add_argument("--name", type=str, default="material_reconstruction")
    # reconstruction
    parser.add_argument("--model", type=str, default="results/patch_nr/custom_patch_nr/version_3/custom_patch_nr_final.pth", help="model_path")
    parser.add_argument("--image", type=str, default="data/material_pt_nr/test.png", help="image_path")
    parser.add_argument("--result_path", type=str, default="results/reconstruction_experiment_inc_noise")
    parser.add_argument("--lam_start", type=float, default=0., help="Lambda start")
    parser.add_argument("--lam_end", type=float, default=2., help="Lambda end")
    parser.add_argument("--lam_steps", type=float, default=8, help="Lambda steps")
    parser.add_argument("--steps", type=int, default=300, help="reconstruction steps")

    #image degradation
    parser.add_argument("--kernel_size", type=int, default=16, help="Kernel size")
    parser.add_argument("--kernel_std", type=int, default=9, help="Kernel std")
    parser.add_argument("--noise_std_start", type=float, default=0.01, help="Lambda start")
    parser.add_argument("--noise_std_end", type=float, default=4., help="Lambda end")
    parser.add_argument("--noise_std_steps", type=float, default=8, help="Lambda steps")
    #parser.add_argument("--noise_std", type=float, default=0.01, help="Noise std")

    #regulariser
    parser.add_argument("--reg_samples", type=int, default=50000)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #
    main(DEVICE, parser.parse_args())