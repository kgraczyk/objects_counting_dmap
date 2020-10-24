"""The implementation of U-Net and FCRN-A models."""
from typing import Tuple

import numpy as np
import torch
from torch import nn


def conv_block(channels: Tuple[int, int],
               size: Tuple[int, int],
               stride: Tuple[int, int]=(1, 1),
               N: int=1):
    """
    Create a block with N convolutional layers with ReLU activation function.
    The first layer is IN x OUT, and all others - OUT x OUT.

    Args:
        channels: (IN, OUT) - no. of input and output channels
        size: kernel size (fixed for all convolution in a block)
        stride: stride (fixed for all convolution in a block)
        N: no. of convolutional layers

    Returns:
        A sequential container of N convolutional layers.
    """
    # a single convolution + batch normalization + ReLU block
    block = lambda in_channels: nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=channels[1],
                  kernel_size=size,
                  stride=stride,
                  bias=False,
                  padding=(size[0] // 2, size[1] // 2)),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU()
    )
    # create and return a sequential container of convolutional layers
    # input size = channels[0] for first block and channels[1] for all others
    # useful feature: if the stride = (1,1), padding = (1,1) and kernel size = (3,3) 
    # than the output is still HxWx channels[1] in the other cases:
    # H(eight) = (H - K + 2 P)//S + 1 similary W(idth) 
    return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])


class ConvCat(nn.Module):
    """Convolution with upsampling + concatenate block."""

    def __init__(self,
                 channels: Tuple[int, int],
                 size: Tuple[int, int],
                 stride: Tuple[int, int]=(1, 1),
                 N: int=1):
        """
        Create a sequential container with convolutional block (see conv_block)
        with N convolutional layers and upsampling by factor 2.
        """
        super(ConvCat, self).__init__()
        self.conv = nn.Sequential(
            conv_block(channels, size, stride, N),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, to_conv: torch.Tensor, to_cat: torch.Tensor):
        """Forward pass.

        Args:
            to_conv: input passed to convolutional block and upsampling
            to_cat: input concatenated with the output of a conv block
        """
        return torch.cat([self.conv(to_conv), to_cat], dim=1)


def conv_block_MC(channels: Tuple[int, int],
               size: Tuple[int, int],
               stride: Tuple[int, int]=(1, 1), 
               p: float=0.0,
               N: int=1):
    """
    Create a block with N convolutional layers with ReLU activation function.
    The first layer is IN x OUT, and all others - OUT x OUT.

    Args:
        channels: (IN, OUT) - no. of input and output channels
        size: kernel size (fixed for all convolution in a block)
        stride: stride (fixed for all convolution in a block)
        N: no. of convolutional layers

    Returns:
        A sequential container of N convolutional layers.
    """
    # a single convolution + batch normalization + ReLU block
    block = lambda in_channels: nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=channels[1],
                  kernel_size=size,
                  stride=stride,
                  bias=False,
                  padding=(size[0] // 2, size[1] // 2)),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.Dropout2d(p),
        nn.ReLU()
    )
    # create and return a sequential container of convolutional layers
    # input size = channels[0] for first block and channels[1] for all others
    return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])

class ConvCat_MC(nn.Module):
    """Convolution with upsampling + concatenate block."""

    def __init__(self,
                 channels: Tuple[int, int],
                 size: Tuple[int, int],
                 stride: Tuple[int, int]=(1, 1),
                 p: float=0.0,
                 N: int=1):
        """
        Create a sequential container with convolutional block (see conv_block)
        with N convolutional layers and upsampling by factor 2.
        """
        super(ConvCat_MC, self).__init__()
        self.conv = nn.Sequential(
            conv_block_MC(channels, size, stride, p, N),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, to_conv: torch.Tensor, to_cat: torch.Tensor):
        """Forward pass.

        Args:
            to_conv: input passed to convolutional block and upsampling
            to_cat: input concatenated with the output of a conv block
        """
        return torch.cat([self.conv(to_conv), to_cat], dim=1)




class FCRN_A(nn.Module):
    """
    Fully Convolutional Regression Network A

    Ref. W. Xie et al. 'Microscopy Cell Counting with Fully Convolutional
    Regression Networks'
    """

    def __init__(self, N: int=1, input_filters: int=3, p: float=0., **kwargs):
        """
        Create FCRN-A model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * no. of filters as defined in an original model:
              input size -> 32 -> 64 -> 128 -> 512 -> 128 -> 64 -> 1

        Args:
            N: no. of convolutional layers per block (see conv_block)
            input_filters: no. of input channels
        """
        super(FCRN_A, self).__init__()
        self.model = nn.Sequential(
            # downsampling
            conv_block(channels=(input_filters, 32), size=(3, 3), N=N),
            nn.MaxPool2d(2),

            conv_block(channels=(32, 64), size=(3, 3), N=N),
            nn.MaxPool2d(2),

            conv_block(channels=(64, 128), size=(3, 3), N=N),
            nn.MaxPool2d(2),

            # "convolutional fully connected"
            conv_block(channels=(128, 512), size=(3, 3), N=N),

            # upsampling
            nn.Upsample(scale_factor=2),
            conv_block(channels=(512, 128), size=(3, 3), N=N),

            nn.Upsample(scale_factor=2),
            conv_block(channels=(128, 64), size=(3, 3), N=N),

            nn.Upsample(scale_factor=2),
            conv_block(channels=(64, 1), size=(3, 3), N=N),
        )

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        return self.model(input)


class FCRN_A_MC(nn.Module):
    """
    Fully Convolutional Regression Network A

    Ref. W. Xie et al. 'Microscopy Cell Counting with Fully Convolutional
    Regression Networks'
    """

    def __init__(self, N: int=1, input_filters: int=3, p: float=0.0, **kwargs):
        """
        Create FCRN-A model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * no. of filters as defined in an original model:
              input size -> 32 -> 64 -> 128 -> 512 -> 128 -> 64 -> 1

        Args:
            N: no. of convolutional layers per block (see conv_block)
            input_filters: no. of input channels
        """
        super(FCRN_A_MC, self).__init__()
        self.model = nn.Sequential(
            # downsampling
            conv_block(channels=(input_filters, 32), size=(3, 3), N=N),
            nn.Dropout2d(p),
            nn.MaxPool2d(2),

            conv_block(channels=(32, 64), size=(3, 3), N=N),
            nn.Dropout2d(p),
            nn.MaxPool2d(2),

            conv_block(channels=(64, 128), size=(3, 3), N=N),
            nn.Dropout2d(p),
            nn.MaxPool2d(2),

            # "convolutional fully connected"
            conv_block(channels=(128, 512), size=(3, 3), N=N),
            nn.Dropout2d(p),

            # upsampling
            nn.Upsample(scale_factor=2),
            conv_block(channels=(512, 128), size=(3, 3), N=N),
            nn.Dropout2d(p),

            nn.Upsample(scale_factor=2),
            conv_block(channels=(128, 64), size=(3, 3), N=N),
            nn.Dropout2d(p),

            nn.Upsample(scale_factor=2),
            conv_block(channels=(64, 1), size=(3, 3), N=N),
        )

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        return self.model(input)


class FCRN_A_MVE(nn.Module):
    """
    Fully Convolutional Regression Network A with modification to account for uncertainty

    Ref. W. Xie et al. 'Microscopy Cell Counting with Fully Convolutional
    Regression Networks'
    """

    def __init__(self, N: int=1, input_filters: int=3,  **kwargs):
        """
        Create FCRN-A model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * no. of filters as defined in an original model:
              input size -> 32 -> 64 -> 128 -> 512 -> 128 -> 64 -> 1

        Args:
            N: no. of convolutional layers per block (see conv_block)
            input_filters: no. of input channels
        """
        super(FCRN_A_MVE, self).__init__()
        self.model_down = nn.Sequential(
            # downsampling
            conv_block(channels=(input_filters, 32), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            # now the output sie is 32 x H//2 x W//2
            conv_block(channels=(32, 64), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            # now the output sie is 64 x H//2//2 x W//2//2
            conv_block(channels=(64, 128), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            # now the output sie is 64 x H//2//2//2 x W//2//2//2
            
            # "convolutional fully connected"
            conv_block(channels=(128, 512), size=(3, 3), N=N)
            # now the output sie is 512 x H//2//2//2 x W//2//2//2//2
        )
            # let us define the vector output contributing to MVE

            # upsampling

        self.model_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv_block(channels=(512, 128), size=(3, 3), N=N),

            nn.Upsample(scale_factor=2),
            conv_block(channels=(128, 64), size=(3, 3), N=N),

            nn.Upsample(scale_factor=2),
            conv_block(channels=(64, 1), size=(3, 3), N=N),
        )
        self.avepool = nn.AdaptiveAvgPool2d((2,2))
        self.dense   = nn.Linear(512*2*2,1)
        self.act     = nn.Tanh()
        self.final   = nn.Linear(1,1,bias=False)
        

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        
        x = self.model_down(input)
        #print("x",x.size())
        y = self.avepool(x)
        #print("y",y.size())
        y = y.view(-1,512*2*2)
        #print("y",y.size())
        y = self.dense(y)
        #print("y",y.size())
        y = self.act(y)
        y = self.final(y)
        y = torch.log(1+torch.exp(y))
       
        return self.model_up(x), y


class UNet(nn.Module):
    """
    U-Net implementation.

    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(self, filters: int=64, input_filters: int=3, p: float=0.0, **kwargs):
        """
        Create U-Net model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * fixed no. of convolutional layers per block = 2 (see conv_block)
            * constant no. of filters for convolutional layers

        Args:
            filters: no. of filters for convolutional layers
            input_filters: no. of input channels
        """
        super(UNet, self).__init__()
        # first block channels size
        initial_filters = (input_filters, filters)
        # channels size for downsampling
        down_filters = (filters, filters)
        # channels size for upsampling (input doubled because of concatenate)
        up_filters = (2 * filters, filters)

        # downsampling
        self.block1 = conv_block(channels=initial_filters, size=(3, 3), N=2)
        self.block2 = conv_block(channels=down_filters, size=(3, 3), N=2)
        self.block3 = conv_block(channels=down_filters, size=(3, 3), N=2)

        # upsampling
        self.block4 = ConvCat(channels=down_filters, size=(3, 3), N=2)
        self.block5 = ConvCat(channels=up_filters, size=(3, 3), N=2)
        self.block6 = ConvCat(channels=up_filters, size=(3, 3), N=2)

        # density prediction
        self.block7 = conv_block(channels=up_filters, size=(3, 3), N=2)
        self.density_pred = nn.Conv2d(in_channels=filters, out_channels=1,
                                      kernel_size=(1, 1), bias=False)

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        # use the same max pooling kernel size (2, 2) across the network
        pool = nn.MaxPool2d(2)
        # downsampling
        block1 = self.block1(input)
        pool1 = pool(block1)
        block2 = self.block2(pool1)
        
        pool2 = pool(block2)
        block3 = self.block3(pool2)
        pool3 = pool(block3)

        # upsampling
        block4 = self.block4(pool3, block3)
        block5 = self.block5(block4, block2)
        block6 = self.block6(block5, block1)

        # density prediction
        block7 = self.block7(block6)
        wynik = self.density_pred(block7)
         
        return wynik

class UNet2(nn.Module):
    """
    U-Net implementation.

    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(self, filters: int=64, input_filters: int=3, p: float=0.0, **kwargs):
        """
        Create U-Net model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * fixed no. of convolutional layers per block = 2 (see conv_block)
            * constant no. of filters for convolutional layers

        Args:
            filters: no. of filters for convolutional layers
            input_filters: no. of input channels
        """
        super(UNet2, self).__init__()
        # first block channels size
        initial_filters = (input_filters, filters)
        # channels size for downsampling
        down_filters = (filters, filters)
        # channels size for upsampling (input doubled because of concatenate)
        up_filters = (2 * filters, filters)

        # downsampling
        self.block1 = conv_block(channels=initial_filters, size=(3, 3), N=2)
        self.block2 = conv_block(channels=down_filters, size=(3, 3), N=2)
        self.block3 = conv_block(channels=down_filters, size=(3, 3), N=2)

        # upsampling
        self.block4 = ConvCat(channels=down_filters, size=(3, 3), N=2)
        self.block5 = ConvCat(channels=up_filters, size=(3, 3), N=2)
        self.block6 = ConvCat(channels=up_filters, size=(3, 3), N=2)

        # density prediction
        self.block7 = conv_block(channels=up_filters, size=(3, 3), N=2)
        self.density_pred = nn.Conv2d(in_channels=filters, out_channels=1,
                                      kernel_size=(1, 1), bias=False)
    

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        # use the same max pooling kernel size (2, 2) across the network
        pool = nn.MaxPool2d(2)
        # downsampling
        block1 = self.block1(input)
        pool1 = pool(block1)
        block2 = self.block2(pool1)
        
        pool2 = pool(block2)
        block3 = self.block3(pool2)
        pool3 = pool(block3)

        # upsampling
        block4 = self.block4(pool3, block3)
        block5 = self.block5(block4, block2)
        block6 = self.block6(block5, block1)

        # density prediction
        block7 = self.block7(block6)
        wynik = self.density_pred(block7)

        wynik = torch.log(1 + torch.exp(wynik))

        return wynik


class UNet_MVE(nn.Module):
    """
    U-Net implementation with modification for accounting for uncertainty

    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(self, filters: int=64, input_filters: int=3, **kwargs):
        """
        Create U-Net model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * fixed no. of convolutional layers per block = 2 (see conv_block)
            * constant no. of filters for convolutional layers

        Args:
            filters: no. of filters for convolutional layers
            input_filters: no. of input channels
        """
        super(UNet, self).__init__()
        # first block channels size
        initial_filters = (input_filters, filters)
        # channels size for downsampling
        down_filters = (filters, filters)
        # channels size for upsampling (input doubled because of concatenate)
        up_filters = (2 * filters, filters)

        # downsampling
        self.block1 = conv_block(channels=initial_filters, size=(3, 3), N=2)
        self.block2 = conv_block(channels=down_filters, size=(3, 3), N=2)
        self.block3 = conv_block(channels=down_filters, size=(3, 3), N=2)

        # upsampling
        self.block4 = ConvCat(channels=down_filters, size=(3, 3), N=2)
        self.block5 = ConvCat(channels=up_filters, size=(3, 3), N=2)
        self.block6 = ConvCat(channels=up_filters, size=(3, 3), N=2)

        # density prediction
        self.block7 = conv_block(channels=up_filters, size=(3, 3), N=2)
        self.density_pred = nn.Conv2d(in_channels=filters, out_channels=1,
                                      kernel_size=(1, 1), bias=False)
        

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        # use the same max pooling kernel size (2, 2) across the network
        pool = nn.MaxPool2d(2)
        # downsampling
        block1 = self.block1(input)
        print('block1 : ',pool.block1)
        pool1 = pool(block1)
        block2 = self.block2(pool1)
        print('block2 : ',pool.block2)
        
        pool2 = pool(block2)
        block3 = self.block3(pool2)
        print('block3 : ',pool.block3)
        
        pool3 = pool(block3)

        # upsampling
        block4 = self.block4(pool3, block3)
        block5 = self.block5(block4, block2)
        block6 = self.block6(block5, block1)

        # density prediction
        block7 = self.block7(block6)
        wynik = self.density_pred(block7)
       # print('wynik : ',wynik)
         
        return wynik


class UNet_MC(nn.Module):
    """
    U-Net implementation.

    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(self, filters: int=64, input_filters: int=3, p: float=0.0, **kwargs):
        """
        Create U-Net model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * fixed no. of convolutional layers per block = 2 (see conv_block)
            * constant no. of filters for convolutional layers

        Args:
            filters: no. of filters for convolutional layers
            input_filters: no. of input channels
        """
        super(UNet_MC, self).__init__()
        # first block channels size
        initial_filters = (input_filters, filters)
        # channels size for downsampling
        down_filters = (filters, filters)
        # channels size for upsampling (input doubled because of concatenate)
        up_filters = (2 * filters, filters)
        self.drop = nn.Dropout2d(p)
        # downsampling
        self.block1 = conv_block_MC(channels=initial_filters, size=(3, 3), p=p, N=2)
        self.block2 = conv_block_MC(channels=down_filters, size=(3, 3), p=p, N=2)
        self.block3 = conv_block_MC(channels=down_filters, size=(3, 3), p=p, N=2)

        # upsampling
        self.block4 = ConvCat_MC(channels=down_filters, size=(3, 3),p=p, N=2)
        self.block5 = ConvCat_MC(channels=up_filters, size=(3, 3),p=p, N=2)
        self.block6 = ConvCat_MC(channels=up_filters, size=(3, 3),p=p, N=2)

        # density prediction
        self.block7 = conv_block_MC(channels=up_filters, size=(3, 3), p=p, N=2)
        self.density_pred = nn.Conv2d(in_channels=filters, out_channels=1,
                                      kernel_size=(1, 1), bias=False)

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        # use the same max pooling kernel size (2, 2) across the network
        pool = nn.MaxPool2d(2)

        # downsampling
        block1 = self.block1(input)
        pool1 = pool(block1)
        block2 = self.block2(pool1)
        pool2 = pool(block2)
        block3 = self.block3(pool2)
        pool3 = pool(block3)

        # upsampling
        block4 = self.block4(pool3, block3)
        block5 = self.block5(block4, block2)
        block6 = self.block6(block5, block1)

        # density prediction
        block7 = self.block7(block6)
        return self.density_pred(block7)




class UNet_MC_old(nn.Module):
    """
    U-Net implementation.

    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(self, filters: int=64, input_filters: int=3, p: float=0.0, **kwargs):
        """
        Create U-Net model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * fixed no. of convolutional layers per block = 2 (see conv_block)
            * constant no. of filters for convolutional layers

        Args:
            filters: no. of filters for convolutional layers
            input_filters: no. of input channels
        """
        super(UNet_MC, self).__init__()
        # first block channels size
        initial_filters = (input_filters, filters)
        # channels size for downsampling
        down_filters = (filters, filters)
        # channels size for upsampling (input doubled because of concatenate)
        up_filters = (2 * filters, filters)
        self.drop = nn.Dropout2d(p)
        # downsampling
        self.block1 = conv_block(channels=initial_filters, size=(3, 3), N=2)
        self.block2 = conv_block(channels=down_filters, size=(3, 3), N=2)
        self.block3 = conv_block(channels=down_filters, size=(3, 3), N=2)

        # upsampling
        self.block4 = ConvCat(channels=down_filters, size=(3, 3), N=2)
        self.block5 = ConvCat(channels=up_filters, size=(3, 3), N=2)
        self.block6 = ConvCat(channels=up_filters, size=(3, 3), N=2)

        # density prediction
        self.block7 = conv_block(channels=up_filters, size=(3, 3), N=2)
        self.density_pred = nn.Conv2d(in_channels=filters, out_channels=1,
                                      kernel_size=(1, 1), bias=False)

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        # use the same max pooling kernel size (2, 2) across the network
        pool = nn.MaxPool2d(2)

        # downsampling
        block1 = self.drop(self.block1(input))
        pool1 = pool(block1)
        block2 = self.drop(self.block2(pool1))
        pool2 = pool(block2)
        block3 = self.drop(self.block3(pool2))
        pool3 = pool(block3)

        # upsampling
        block4 = self.drop(self.block4(pool3, block3))
        block5 = self.drop(self.block5(block4, block2))
        block6 = self.drop(self.block6(block5, block1))

        # density prediction
        block7 = self.drop(self.block7(block6))
        return self.density_pred(block7)



class UNet2_MC(nn.Module):
    """
    U-Net implementation.

    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(self, filters: int=64, input_filters: int=3, p: float=0.0, **kwargs):
        """
        Create U-Net model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * fixed no. of convolutional layers per block = 2 (see conv_block)
            * constant no. of filters for convolutional layers

        Args:
            filters: no. of filters for convolutional layers
            input_filters: no. of input channels
        """
        super(UNet2_MC, self).__init__()
        # first block channels size
        initial_filters = (input_filters, filters)
        # channels size for downsampling
        down_filters = (filters, filters)
        # channels size for upsampling (input doubled because of concatenate)
        up_filters = (2 * filters, filters)
        self.drop = nn.Dropout2d(p)
        # downsampling
        self.block1 = conv_block(channels=initial_filters, size=(3, 3), N=2)
        self.block2 = conv_block(channels=down_filters, size=(3, 3), N=2)
        self.block3 = conv_block(channels=down_filters, size=(3, 3), N=2)

        # upsampling
        self.block4 = ConvCat(channels=down_filters, size=(3, 3), N=2)
        self.block5 = ConvCat(channels=up_filters, size=(3, 3), N=2)
        self.block6 = ConvCat(channels=up_filters, size=(3, 3), N=2)

        # density prediction
        self.block7 = conv_block(channels=up_filters, size=(3, 3), N=2)
        self.density_pred = nn.Conv2d(in_channels=filters, out_channels=1,
                                      kernel_size=(1, 1), bias=False)

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        # use the same max pooling kernel size (2, 2) across the network
        pool = nn.MaxPool2d(2)

        # downsampling
        block1 = self.drop(self.block1(input))
        pool1 = pool(block1)
        block2 = self.drop(self.block2(pool1))
        pool2 = pool(block2)
        block3 = self.drop(self.block3(pool2))
        pool3 = pool(block3)

        # upsampling
        block4 = self.drop(self.block4(pool3, block3))
        block5 = self.drop(self.block5(block4, block2))
        block6 = self.drop(self.block6(block5, block1))

        # density prediction
        block7 = self.drop(self.block7(block6))

        wynik = self.density_pred(block7)

        wynik = torch.log(1 + torch.exp(wynik))

        return wynik



# --- PYTESTS --- #

def run_network(network: nn.Module, input_channels: int):
    """Generate a random image, run through network, and check output size."""
    sample = torch.ones((1, input_channels, 224, 224))
    result = network(input_filters=input_channels)(sample)
    assert result.shape == (1, 1, 224, 224)


def test_UNet_color():
    """Test U-Net on RGB images."""
    run_network(UNet, 3)


def test_UNet_grayscale():
    """Test U-Net on grayscale images."""
    run_network(UNet, 1)


def test_FRCN_color():
    """Test FCRN-A on RGB images."""
    run_network(FCRN_A, 3)


def test_FRCN_grayscale():
    """Test FCRN-A on grayscale images."""
    run_network(FCRN_A, 1)
