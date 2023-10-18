import torch
from torch import Tensor
from tqdm import tqdm

from operators import Operator
from regularisers import Regulariser


def variational_model_solver(input_tensor: Tensor, start_tensor: Tensor, operator: Operator, regulariser: Regulariser = None,
                             lam=0.87, steps=600, device='cpu'):
    """Solver for the variational model.
     It is inspired by the solution algorithm described in the PatchNr paper https://arxiv.org/abs/2205.12021 proposed
     by F. Altekrueger et al.
     This code is more generalized version with the possibility to change the main component e.g.
     operators and regularizers.
     :param input_tensor Tensor containing the degraded input that should be reconstructed.
     :param start_tensor Tensor that acts as starting point for the reconstruction procedure.
     :param operator Operator instance that performs the degradation such as blurr, noise, down sampling.
     :param regulariser Regularisation term for the reconstruction.
     :param lam weight parameter for the regulariser.
     :param steps amount of iterations to perform
     :param device to be used by the algorithm. Either 'cpu' or 'gpu'
     :returns a tuple with the reconstructed image at first position and a triple at second position. The triple contains the loss, likelihood and regularisation values.
     """
    degraded_image = input_tensor.clone().to(device)
    start_tensor = start_tensor.clone().to(device)
    reconstructed_image = torch.tensor(start_tensor.clone(), dtype=torch.float, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([reconstructed_image], lr=0.005)

    step_bar = tqdm(range(steps))

    losses = []
    likelihoods = []
    regularisation = []

    for _ in step_bar:
        optimizer.zero_grad()
        reg = regulariser.evaluate(reconstructed_image) if not regulariser is None else 0
        likelihood = torch.sum((operator(reconstructed_image) - degraded_image)**2)
        loss = likelihood + lam*reg
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        likelihoods.append(likelihood.item())
        regularisation.append(reg)
        step_bar.set_description_str(f'Loss: {loss}; Likelihood: {likelihood} R: {reg}')
    return reconstructed_image, (losses, likelihoods, regularisation)