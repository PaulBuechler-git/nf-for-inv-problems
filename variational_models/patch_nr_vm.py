import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.PatchExtractor import PatchExtractor


# Following the implementations from this repository
# https://github.com/FabianAltekrueger/patchNR/tree/master
# with the corresponding paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl (2023).
# PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.
# Inverse Problems, vol. 39, no. 6.
class PatchNr_VM:

    def __init__(self, model, patch_size, random_patch_count, max_iter, device):
        self.model = model
        self.patch_size = patch_size
        self.random_patch_count = random_patch_count
        self.max_iter = max_iter
        self.device = device

    def reconstruct_img(self, name, deg_img, operator, lam):
        writer = SummaryWriter(comment=name)
        deg_img = deg_img.to(self.device)
        patch_size = (self.patch_size, self.patch_size)
        rec_img = torch.Tensor(deg_img.clone(), dtype=torch.float, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([rec_img], lr=1e-4)
        for it in tqdm(range(self.max_iter)):
            optimizer.zero_grad()
            patch_extractor = PatchExtractor(rec_img, patch_size)
            random_rec_im_patches = torch.Tensor(
                [patch_extractor.get_random_patch().flatten() for _ in range(self.random_patch_count)])

            pred_inv, log_det_inv = self.model(random_rec_im_patches, rev=True)
            reg = torch.mean(torch.sum(pred_inv ** 2, dim=1) / 2) - torch.mean(log_det_inv)
            lss_data_term = torch.sum((operator(rec_img) - deg_img) ** 2)

            loss = lss_data_term + lam * reg
            writer.add_scalar('loss', loss, it)
            loss.backward()
            optimizer.step()
        writer.close()
        return rec_img
