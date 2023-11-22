import torch


def gaussian_kernel_generator(kernel_size, std):
    """ Method that generates Gaussian Kernel matrices"""
    if kernel_size % 2 == 0:
        raise ValueError(f'Kernel size has to be odd')
    distribution = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2) * std)
    kernel = torch.zeros(kernel_size, kernel_size)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            pos = torch.tensor([i - center, j - center])
            kernel[i, j] = torch.exp(distribution.log_prob(pos))
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def mean_kernel_generator(kernel_size):
    """Method that generated a mean filter kernel returns a kernel with shape 1x1xkennel_sizexkernel_size"""
    kernel = torch.ones((kernel_size, kernel_size))
    mean_kernel = (1/(kernel_size**2)) * kernel
    return mean_kernel.unsqueeze(0).unsqueeze(0)
