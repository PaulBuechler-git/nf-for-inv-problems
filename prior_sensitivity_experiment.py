import json
import math
import os.path
import sqlite3
from argparse import ArgumentParser
import time
from torchvision import transforms
from tqdm import tqdm
from core.file_utils import create_versioned_dir
from patchNr.PatchFlowModel import PatchFlowModel
from core.img_utils import ImageLoader
import torch

from core.kernels import gaussian_kernel_generator
from patchNr.Operators import BlurOperator
from patchNr.PatchNrRegulariser import PatchNrRegulariser
import numpy as np


def main(device, parsed_args):
    model_path = parsed_args.model
    image_path = parsed_args.image_path
    name = parsed_args.name
    result_path = parsed_args.result_path
    std_start = parsed_args.std_start
    std_end = parsed_args.std_end
    std_steps = parsed_args.std_steps
    noise_std_start = parsed_args.noise_std_start
    noise_std_end =   parsed_args.noise_std_end
    noise_std_steps = parsed_args.noise_std_steps
    scaling = parsed_args.scaling
    kernel_size = parsed_args.kernel_size
    prior_batch_size = parsed_args.prior_batch_size

    result_dir = create_versioned_dir(result_path, name)

    hparams_dict = vars(parsed_args)
    hparams_dict['result_dir'] = result_dir
    with open(os.path.join(result_dir, 'params.json'), 'w') as fp:
        json.dump(hparams_dict, fp)

    model = PatchFlowModel(path=model_path)
    patch_size = int(math.sqrt(model.hparams['dimension']))

    # create db to store data
    connection = sqlite3.connect(os.path.join(result_dir, 'values.db'))
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE prior_sensitivity_experiment(prior_val, std_blur, std_noise, img_index)")
    connection.commit()

    normalization = transforms.Compose([transforms.Normalize([0, ], [255., ])])
    image_loader = ImageLoader(image_path, device=device, transform=normalization)
    images = list(map(lambda img: img.to(device), [image_loader[i] for i in range(len(image_loader))]))

    prior = PatchNrRegulariser(model, p_size=patch_size, sample_number=prior_batch_size, padding=True,
                               padding_size=16, device=device)
    for image_idx in tqdm(range(len(images))):
        image = images[image_idx]
        c, w, h = image.shape
        for std_blur in np.linspace(std_start, std_end, std_steps):
            gaussian_kernel = gaussian_kernel_generator(kernel_size, std_blur).to(device)
            blur_operator = BlurOperator(gaussian_kernel, device=device)
            for std_noise in np.linspace(noise_std_start, noise_std_end, noise_std_steps):
                noise_vector = torch.reshape(torch.tensor(np.random.normal(0, std_noise, w*h), dtype=torch.float, device=device), image.shape)
                start_time = time.time()
                noisy_image = blur_operator(image + normalization(noise_vector * scaling))
                prior_val = prior.evaluate(noisy_image)
                end_time = time.time()
                cursor.execute("INSERT INTO prior_sensitivity_experiment VALUES(?, ?, ?, ?)",
                               (prior_val.item(), std_blur, std_noise, image_idx))
        connection.commit()


if __name__ == "__main__":
    parser = ArgumentParser(description="Prior sensitivity experiment")
    parser.add_argument("--name", type=str, default="custom_material_prior")
    # reconstruction
    parser.add_argument("--model", type=str, default="results/patch_nr/custom_patch_nr/version_3/custom_patch_nr_final.pth", help="model_path")
    parser.add_argument("--image_path", type=str, default="data/material_pt_nr/testset_superres", help="image_path")
    parser.add_argument("--result_path", type=str, default="results/prior_sensitivity_experiment")

    parser.add_argument("--prior_batch_size", type=int, default=50000)
    parser.add_argument("--std_start", type=int, default=1)
    parser.add_argument("--std_end", type=int, default=20)
    parser.add_argument("--std_steps", type=int, default=5)
    parser.add_argument("--noise_std_start", type=int, default=0)
    parser.add_argument("--noise_std_end", type=int, default=4)
    parser.add_argument("--noise_std_steps", type=int, default=5)
    parser.add_argument("--scaling", type=float, default=2.)
    parser.add_argument("--kernel_size", type=int, default=11)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(DEVICE, parser.parse_args())
