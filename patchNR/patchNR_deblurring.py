# This code belongs to the paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl (2023).
# PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.
# Inverse Problems, vol. 39, no. 6.
#
# Please cite the paper, if you use the code.
#
# The script reproduces the numerical example Deblurring in the paper.

import torch
from torch import nn
import numpy as np
import model
import scipy.io
from tqdm import tqdm
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

class Blur(nn.Module):
    def __init__(self):
        super().__init__()

        data_dict = scipy.io.loadmat('im05_flit01.mat')

        kernels = data_dict['f']
        self.kernel = np.array(kernels)
        self.blur = nn.Conv2d(1,1, self.kernel.shape[0], bias=False, padding='same' ,padding_mode='reflect')
        self.blur.weight.data = torch.from_numpy(self.kernel).float().unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        return self.blur(x)

def patchNR(img, lam, patch_size, n_patches_out, patchNR, n_iter_max):
    """
    Defines the reconstruction using patchNR as regularizer
    """
                         
    # fixed parameters
    obs = img.to(DEVICE)
    operator = Blur().to(DEVICE)
    center = False 
    init = obs
    pad_size = 4 #pad the image before extracting patches to avoid boundary effects
    pad = [pad_size]*4  
    
    # create patch extractors
    input_im2pat = utils.patch_extractor(patch_size, pad=False, center=center)

    # intialize optimizer for image
    fake_img = torch.tensor(init.clone(),dtype=torch.float,device=DEVICE,requires_grad = True)
    optim_img = torch.optim.Adam([fake_img], lr=0.005)

    # Main loop
    for it in tqdm(range(n_iter_max)):
        optim_img.zero_grad()        
        tmp = nn.functional.pad(fake_img,pad,mode='reflect')
        fake_data = input_im2pat(tmp,n_patches_out)
        
        #patchNR
        pred_inv, log_det_inv = patchNR(fake_data,rev=True)    
        reg = torch.mean(torch.sum(pred_inv**2,dim=1)/2) - torch.mean(log_det_inv)
       
        #data fidelity
        data_fid = torch.sum((operator(fake_img) - obs)**2)
        
        #loss
        loss = data_fid + lam*reg
        loss.backward()
        optim_img.step()    
        
    return fake_img


if __name__ == '__main__':
    #input parameters
    patch_size = 6
    num_layers = 5
    subnet_nodes = 512

    net = model.create_NF(num_layers, subnet_nodes, dimension=patch_size**2)
    weights = torch.load('patchNR_weights/weights_lentils.pth')
    net.load_state_dict(weights['net_state_dict'])
	
    operator = Blur().to(DEVICE)
    gt = utils.imread('input_imgs/img_test_lentils.png')
    with torch.no_grad():
        noisy = operator(gt)
        noisy = noisy + 5/255*torch.randn(noisy.shape,device=DEVICE)

    lam = 0.87
    n_pat = 40000
    iteration = 600
    rec = patchNR(noisy,lam = lam, patch_size = patch_size, n_patches_out = n_pat,
                  patchNR = net, n_iter_max = iteration)
    utils.save_img(rec, 'results/patchNR_lentils')
    
