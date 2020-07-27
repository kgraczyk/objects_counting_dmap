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
    
def make_predictions(data_,i):
    unet_filters = 64
    convolutions = 4
    p=0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_channels = 3    # 1 if dataset_name == 'ucsd' else 3
    network_architecture = 'UNet2'

    network = {
            'UNet': UNet,
            'UNet2': UNet2,
            'FCRN_A': FCRN_A,
            }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions,p=p).to(device)

    path = f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth'

    network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load(path))
    
    image_ = data_['images'][i]
    label = data_['labels'][i]
    #image.unsqueeze(1)
    
    image =(torch.tensor(image_)).unsqueeze(0)

    pred = network(image.to(device))
    pred = pred.squeeze().to('cpu')

    pred = pred.detach().numpy()
    
    print('print',pred.shape)
    

    image_ = np.transpose(image_, (1, 2, 0))+0.5 #// 0.5 for nocover
    label = np.transpose(label, (1, 2, 0))
    
    #print(image.shape)
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    axes[0].set_title('true')
    axes[0].imshow((image_ * 255).astype(np.uint8))
    axes[1].set_title(f'human annotation, sum={label.sum()/100}')
    axes[1].imshow(label[:,:,0])
    axes[2].set_title(f'network-pred, sum={pred.sum()/100}')
    axes[2].imshow(pred)
    fig.savefig(f'nocover-map-pred-i={i}.png')

make_predictions(data,10)
make_predictions(data,100)
make_predictions(data,200)
make_predictions(data,300)

plot_input_and_map(data,10)