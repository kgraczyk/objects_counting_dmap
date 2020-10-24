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
import pylab as plb
import matplotlib.image as mpimg
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

from os import listdir
from os.path import isfile, join

plb.rcParams['font.size'] = 12

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

    pred  = network(image)
    suma  = pred.sum().item()
    suma2 = suma*suma
    pred  = pred.squeeze().to('cpu')
    predf = pred.detach().numpy()
    predf2 = np.power(predf, 2)

    N = 100
    for _ in range(N-1):
        pred = network(image)
        pred = pred.squeeze().to('cpu')
        suma  += pred.sum().item()
        suma2 += np.power(pred.sum().item(),2)
        predf += pred.detach().numpy()
        predf2 += np.power(pred.detach().numpy(),2)
    
    predf= predf/N
    suma = suma/N
    predf2 = predf2/N - np.power(predf,2)
    predf2 = np.sqrt(predf2)

    suma2  = suma2/N - np.power(suma,2)
    suma2  = np.sqrt(suma2)

    image_ = np.transpose(image_, (1, 2, 0))+0.5 #// 0.5 for nocover
    label = np.transpose(label, (1, 2, 0))
    
    #print(image.shape)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].set_title('true')
    axes[0].imshow((image_ * 255).astype(np.uint8))
    
    axes[1].set_title(f'human annotation, sum={label.sum()/100}')
    z=axes[1].imshow(label[:,:,0],extent=(0,256,0,256),cmap=cm.binary)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(z,cax=cax)
    
    axes[2].set_title(f'drop out, p={p}, sum={suma/100:3.1f}$\pm${suma2/100:1.1f}')
    z=axes[2].imshow(predf,extent=(0,256,0,256),cmap=cm.binary)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(z,cax=cax)
    
    axes[3].set_title(f'uncertainty')
    z=axes[3].imshow(predf2,extent=(0,256,0,256),cmap=cm.binary)
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(z,cax=cax)
    
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    


    fig.savefig(f'nocover-map-pred-i={i}.png')



def make_predictions_boots(folder_,i):
    "Works for UNet type for FCRN_A check compatibility"    
    end = '.pth'
    filest = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.endswith(end)]))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    path = filest[0].split('/')
    path=path[-1]
    names = path.split('_')
    data_val = h5py.File(f'{names[0]}/valid.h5','r')
    input_channels = 1 if names[0] == 'ucsd' else 3
    network_architecture = names[3]+'_MC' if names[4]=='MC' else  names[3]
    if names[3] == 'FCRN': network_architecture = 'FCRN_A_MC' if names[5]=='MC' else  'FCRN_A'

    unet_filters = int(names[-2][3:])
    convolutions = int(names[-1].split('.')[0][4:])

    print( network_architecture)


    network = {
            'UNet': UNet,
            'UNet_MC': UNet_MC,
            'UNet2': UNet2,
            'UNet2_MC': UNet2_MC,
            'FCRN_A': FCRN_A,
            'FCRN_A_MC': FCRN_A_MC
            }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions,p=0.0).to(device)

    
    network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load(filest[0]))

    network.eval()    

    image_ = data_val['images'][i]
    label  = data_val['labels'][i]
    
    image = (torch.tensor(image_)).unsqueeze(0)
    image = image.to(device)

    #pred    = network(image)
    #suma    = pred.sum().item()/100

    #pred    = pred.squeeze().to('cpu')
    #predf   = pred.detach().numpy()
    #predf2  = np.power(predf, 2)

    #suma2   = suma*suma
    

    N = len(filest)
    print(N)
    TAB=[]
    SUM=[]
    for f in filest:
        network.load_state_dict(torch.load(f))
        network.eval() 
        pred = network(image)
        pred = pred.squeeze().to('cpu')
        pred = pred.detach().numpy()
        TAB.append(pred)
        SUM.append(pred.sum()/100)
        
    TAB=np.array(TAB)
    SUM=np.array(SUM)

    predf= np.mean(TAB,axis=0) #predf/N
    suma = np.mean(SUM) #suma/N
    
    predf2 = np.sqrt(np.var(TAB,axis=0))
    suma2  = np.sqrt(np.var(SUM))

    image_ = np.transpose(image_, (1, 2, 0))+0.5 #// 0.5 for nocover
    label = np.transpose(label, (1, 2, 0))
    
    #print(image.shape)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].set_title('true')
    axes[0].imshow((image_ * 255).astype(np.uint8))
    #divider = make_axes_locatable(axes[0])
    #cax = divider.append_axes("right", size="5%", pad=0.05)

    axes[1].set_title(f'human annotation, sum={label.sum()/100:2.1f}')
    z=axes[1].imshow(label[:,:,0],extent=(0,256,0,256),cmap=cm.binary)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(z,cax=cax)

    axes[2].set_title(f'bootstrap, sum={suma:3.1f}$\pm${suma2:2.1f}')
    z=axes[2].imshow(predf,extent=(0,256,0,256),cmap=cm.binary)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(z,cax=cax)
    
    
    axes[3].set_title(f'uncertainty')
    z=axes[3].imshow(predf2,extent=(0,256,0,256),cmap=cm.binary)#cmap='RdGy')cm.hot
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(z,cax=cax)
    
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    
    fig.savefig(f'nocover-boots-map-pred-i={i}.png')
    fig.tight_layout()


make_predictions(10,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(100,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(200,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(300,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(400,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(500,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(150,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(250,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(350,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(450,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')
make_predictions(550,f'resutls-nocover/nocover_UNet2_MC_epochs=100_batch=7_hf=0.0_vf=0.0_uf=64_conv4_p=0.1.pth')


make_predictions_boots('boots_results_nocover',10)
make_predictions_boots('boots_results_nocover',100)
make_predictions_boots('boots_results_nocover',200)
make_predictions_boots('boots_results_nocover',300)
make_predictions_boots('boots_results_nocover',400)
make_predictions_boots('boots_results_nocover',500)
make_predictions_boots('boots_results_nocover',150)
make_predictions_boots('boots_results_nocover',250)
make_predictions_boots('boots_results_nocover',350)
make_predictions_boots('boots_results_nocover',450)
make_predictions_boots('boots_results_nocover',550)

#plot_input_and_map(data,10)