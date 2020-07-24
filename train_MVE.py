"""
Main script used to train networks.
The uncertainties are calculated in Mean Variance Approximation
"""
import os
from typing import Union, Optional, List

import click
import torch
import numpy as np
from matplotlib import pyplot

from data_loader import H5Dataset
from looper_MVE import Looper
from model import FCRN_A_MVE



@click.command()
@click.option('-d', '--dataset_name',
              type=click.Choice(['cell', 'mall', 'ucsd']),
              required=True,
              help='Dataset to train model on (expect proper HDF5 files).')
@click.option('-n', '--network_architecture',
              type=click.Choice(['UNet_MVE', 'FCRN_A_MVE']),
              required=True,
              help='Model to train.')
@click.option('-lr', '--learning_rate', default=1e-2,
              help='Initial learning rate (lr_scheduler is applied).')
@click.option('-e', '--epochs', default=150, help='Number of training epochs.')

@click.option('--batch_size', default=8,
              help='Batch size for both training and validation dataloaders.')
@click.option('-hf', '--horizontal_flip', default=0.0,
              help='The probability of horizontal flip for training dataset.')
@click.option('-vf', '--vertical_flip', default=0.0,
              help='The probability of horizontal flip for validation dataset.')
@click.option('--unet_filters', default=64,
              help='Number of filters for U-Net convolutional layers.')
@click.option('--convolutions', default=2,
              help='Number of layers in a convolutional block.')             
@click.option('--plot', is_flag=True, help="Generate a live plot.")

def train_MVE(dataset_name: str,
          network_architecture: str,
          learning_rate: float,
          epochs: int,
          batch_size: int,
          horizontal_flip: float,
          vertical_flip: float,
          unet_filters: int,
          convolutions: int,
          plot: bool):
    """Train chosen model on selected dataset."""
    # use GPU if avilable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataset = {}     # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders

    for mode in ['train', 'valid']:
        # expected HDF5 files in dataset_name/(train | valid).h5
        data_path = os.path.join(dataset_name, f"{mode}.h5")
        # turn on flips only for training dataset
        dataset[mode] = H5Dataset(data_path,
                                  horizontal_flip if mode == 'train' else 0,
                                  vertical_flip if mode == 'train' else 0)
        dataloader[mode] = torch.utils.data.DataLoader(dataset[mode],
                                                       batch_size=batch_size)

    # only UCSD dataset provides greyscale images instead of RGB
    input_channels = 1 if dataset_name == 'ucsd' else 3

    # initialize a model based on chosen network_architecture
    network = {
#        'UNet': UNet,
        'FCRN_A_MVE': FCRN_A_MVE
    }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions).to(device)
    network = torch.nn.DataParallel(network)

    # initialize loss, optimized and learning rate scheduler
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.1)
    # prob
    epochs__          = 'epochs='+str(epochs)
    batch_size__      = 'batch='+str(batch_size)
    horizontal_flip__ = 'hf='+str(horizontal_flip)
    vertical_flip__   = 'vf=' + str(vertical_flip)
    unet_filters__    = 'uf=' + str(unet_filters)
    convolutions__    = "conv"+str(convolutions)
   

    # if plot flag is on, create a live plot (to be updated by Looper)
    if plot:
        pyplot.ion()
        fig, plots = pyplot.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2

    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(network, device, loss, optimizer,
                          dataloader['train'], len(dataset['train']), plots[0])
 
    valid_looper = Looper(network, device, loss, optimizer,
                          dataloader['valid'], len(dataset['valid']), plots[1],
                          validation=True)
   
    train_looper.LOG=True
    valid_looper.LOG=False

    # current best results (lowest mean absolute error on validation set)
    current_best = np.infty

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n")

        # run training epoch and update learning rate
        train_looper.run()
        lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()

        # update checkpoint if new best is reached
        if result < current_best:
            current_best = result
            torch.save(network.state_dict(),
                       f'{dataset_name}_{network_architecture}_{epochs__}_{batch_size__}_{horizontal_flip__}_{vertical_flip__}_{unet_filters__}_{convolutions__}.pth')
            hist = []
            hist.append(valid_looper.history[-1])
            hist.append(train_looper.history[-1])
            #hist = np.array(hist)
            #print(hist)
            np.savetxt(f'hist_best_{dataset_name}_{network_architecture}_{epochs__}_{batch_size__}_{horizontal_flip__}_{vertical_flip__}_{unet_filters__}_{convolutions__}.csv' 
                        ,hist, delimiter=',')
    

            print(f"\nNew best result: {result}")

        print("\n", "-"*80, "\n", sep='')
        
        if plot:
            fig.savefig(f'status_{dataset_name}_{network_architecture}_{epochs__}_{batch_size__}_{horizontal_flip__}_{vertical_flip__}_{unet_filters__}_{convolutions__}.png')

    print(f"[Training done] Best result: {current_best}")

  

    hist = np.array(train_looper.history)
    np.savetxt(f'hist_train_{dataset_name}_{network_architecture}_{epochs__}_{batch_size__}_{horizontal_flip__}_{vertical_flip__}_{unet_filters__}_{convolutions__}.csv' ,hist,delimiter=',')
    hist = np.array(valid_looper.history)
    np.savetxt(f'hist_test_{dataset_name}_{network_architecture}_{epochs__}_{batch_size__}_{horizontal_flip__}_{vertical_flip__}_{unet_filters__}_{convolutions__}.csv' , hist,delimiter=',')
    

if __name__ == '__main__':
    train_MVE()
