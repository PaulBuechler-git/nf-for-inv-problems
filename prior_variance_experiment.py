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
from regularisers import PatchNrRegulariser
import numpy as np


def main(device, parsed_args):
    model_path = parsed_args.model
    image_path = parsed_args.image
    name = parsed_args.name
    result_path = parsed_args.result_path
    start = parsed_args.start
    end = parsed_args.end
    step_size = parsed_args.step_size
    evaluations_per_step = parsed_args.evaluations_per_step

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
    cursor.execute("CREATE TABLE prior_variance_experiment(prior_val, prior_samples, start_time, end_time, eval_step)")
    connection.commit()

    normalization = transforms.Compose([transforms.Normalize([0, ], [255., ])])
    image_loader = ImageLoader(image_path, device=device, transform=normalization)
    ground_truth = image_loader[0].to(device)

    for samples_step in range(start, end, step_size):
        prior = PatchNrRegulariser(model, p_size=patch_size, sample_number=samples_step, padding=True,
                                                padding_size=16, device=DEVICE)
        print(f'Evaluation started for: {samples_step}')
        for eval_step in tqdm(range(evaluations_per_step)):
            start_time = time.time()
            prior_val = prior.evaluate(ground_truth)
            end_time = time.time()
            cursor.execute("INSERT INTO prior_variance_experiment VALUES(?, ?, ?, ?, ?)",
                           (prior_val.item(), samples_step, start_time, end_time, eval_step))
        connection.commit()

    image = ground_truth.detach().cpu()
    np.save(os.path.join(result_dir, 'image.npy'), image.numpy())
    save_normalized(os.path.join(result_dir, 'image.png'), image.squeeze().numpy())


if __name__ == "__main__":
    parser = ArgumentParser(description="Prior Variance experiment")
    parser.add_argument("--name", type=str, default="custom_material_prior")
    # reconstruction
    parser.add_argument("--model", type=str, default="results/patch_nr/custom_patch_nr/version_3/custom_patch_nr_final.pth", help="model_path")
    parser.add_argument("--image", type=str, default="data/material_pt_nr/test.png", help="image_path")
    parser.add_argument("--result_path", type=str, default="results/prior_variance_experiment")

    parser.add_argument("--start", type=int, default=1000)
    parser.add_argument("--end", type=int, default=150000)
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--evaluations_per_step", type=int, default=200)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(DEVICE, parser.parse_args())
