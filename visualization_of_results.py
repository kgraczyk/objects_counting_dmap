import os
import shutil
import zipfile
from glob import glob
from typing import List, Tuple

import click
import h5py
import wget
import numpy as np

from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from skimage.transform import resize, downscale_local_mean
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from model import UNet, UNet2, UNet2_MC, UNet_MC, FCRN_A, FCRN_A_MC


# only UCSD dataset provides greyscale images instead of RGB
#input_channels = 1 if dataset_name == 'ucsd' else 3


data = h5py.File('nocover/valid.h5','r')

print(list(data.keys()))
#print(data['labels'][10].shape)



def plot_input_and_map(data_,i):
    image = data_['images'][i] + 0.5
    label = data_['labels'][i]
    

    image = np.transpose(image, (1, 2, 0))
    label = np.transpose(label, (1, 2, 0))
    
    #print(image.shape)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow((image * 255).astype(np.uint8))
    axes[1].imshow(label[:,:,0])
    fig.savefig(f'nocover-input-map-i={i}.png')
    
def make_predictions(i, file_path):
    "Works for UNet type for FCRN_A check compatibility"    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    path = file_path.split('/')
    path=path[-1]
    names = path.split('_')


    data_val = h5py.File(f'{names[0]}/valid.h5','r')
    
    input_channels = 1 if names[0] == 'ucsd' else 3
    
    
    network_architecture = names[1]+'_MC' if names[2]=='MC' else  names[1]
    if names[1] == 'FCRN': network_architecture = 'FCRN_A_MC' if names[3]=='MC' else  'FCRN_A'

    p=float(names[-1].split('=')[-1][:3]) 
    
    unet_filters = int(names[-3][3:])
    convolutions = int(names[-2][4:])


    network = {
            'UNet': UNet,
            'UNet_MC': UNet_MC,
            'UNet2': UNet2,
            'UNet2_MC': UNet2_MC,
            'FCRN_A': FCRN_A,
            'FCRN_A_MC': FCRN_A_MC
            }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions,p=p).to(device)

    
    network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load(file_path))

    network.eval()    
    for m in network.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train() 

    image_ = data_val['images'][i]
    label  = data_val['labels'][i]
    
    image = (torch.tensor(image_)).unsqueeze(0)
    image = image.to(device)

    pred = network(image)
    pred = pred.squeeze().to('cpu')
    predf = pred.detach().numpy()
    
    for _ in range(19):
        pred = network(image)
        pred = pred.squeeze().to('cpu')
        predf += pred.detach().numpy()
    
    predf= predf/20.

    image_ = np.transpose(image_, (1, 2, 0))+0.5 #// 0.5 for nocover
    label = np.transpose(label, (1, 2, 0))
    
    #print(image.shape)
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    axes[0].set_title('true')
    axes[0].imshow((image_ * 255).astype(np.uint8))
    axes[1].set_title(f'human annotation, sum={label.sum()/100}')
    axes[1].imshow(label[:,:,0])
    axes[2].set_title(f'network-pred, sum={pred.sum()/100}')
    axes[2].imshow(predf)
    fig.savefig(f'nocover-map-pred-i={i}.png')

make_predictions(10,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(100,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(200,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(300,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(400,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(500,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')

plot_input_and_map(data,10)