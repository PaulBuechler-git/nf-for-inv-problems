import json
import math
import os.path
import sqlite3
from argparse import ArgumentParser
import time
from torchvision import transforms
from tqdm import tqdm
from file_utils import create_versioned_dir, save_normalized
from flow_models.PatchFlowModel import PatchFlowModel
from img_utils import ImageLoader
import torch

from kernels import gaussian_kernel_generator
from operators import BlurOperator
from regularisers import PatchNrRegulariser
import numpy as np


def main(device, parsed_args):
    model_path = parsed_args.model
    image_path = parsed_args.image_path
    name = parsed_args.name
    result_path = parsed_args.result_path
    std_start = parsed_args.std_start
    std_end = parsed_args.std_end
    std_steps = parsed_args.std_steps
    scaling = parsed_args.scaling
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
    cursor.execute("CREATE TABLE prior_noise_sensitivity_experiment(prior_val, std, img_index, start_time, end_time)")
    connection.commit()

    normalization = transforms.Compose([transforms.Normalize([0, ], [255., ])])
    image_loader = ImageLoader(image_path, device=device, transform=normalization)
    images = list(map(lambda img: img.to(device), [image_loader[i] for i in range(len(image_loader))]))

    prior = PatchNrRegulariser(model, p_size=patch_size, sample_number=prior_batch_size, padding=True,
                               padding_size=16, device=device)

    for std in np.linspace(std_start, std_end, std_steps):
        image = images[0]
        c, w, h = image.shape
        noise_vector = torch.reshape(torch.tensor(np.random.normal(0, std, w*h), dtype=torch.float, device=device), image.shape)
        print(f'Evaluation started for noise with std {std} and scaling {scaling}')
        for image_idx in tqdm(range(len(images))):
            start_time = time.time()
            noisy_image = image + normalization(noise_vector * scaling)
            prior_val = prior.evaluate(noisy_image)
            end_time = time.time()
            cursor.execute("INSERT INTO prior_noise_sensitivity_experiment VALUES(?, ?, ?, ?, ?)",
                           (prior_val.item(), std, image_idx, start_time, end_time))
        connection.commit()


if __name__ == "__main__":
    parser = ArgumentParser(description="Prior Noise sensitivity experiment")
    parser.add_argument("--name", type=str, default="custom_butterfly_prior")
    # reconstruction
    parser.add_argument("--model", type=str, default="results/patch_nr/butterfly/version_1/butterfly_final.pth", help="model_path")
    parser.add_argument("--image_path", type=str, default="data/DIV2K/buterfly/validate.png", help="image_path")
    parser.add_argument("--result_path", type=str, default="results/prior_noise_sensitivity_experiment")

    parser.add_argument("--prior_batch_size", type=int, default=50000)
    parser.add_argument("--std_start", type=int, default=0)
    parser.add_argument("--std_end", type=int, default=6)
    parser.add_argument("--std_steps", type=int, default=20)
    parser.add_argument("--scaling", type=float, default=2.)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(DEVICE, parser.parse_args())
