"""Looper implementation."""
from typing import Optional, List

import torch
import numpy as np
import matplotlib
from matplotlib import pyplot


class Looper():
    """Looper handles epoch loops, logging, and plotting."""

    def __init__(self,
                 network: torch.nn.Module,
                 device: torch.device,
                 loss: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_size: int,
                 plots: Optional[matplotlib.axes.Axes]=None,
                 MC: bool=False,
                 validation: bool=False):
        """
        Initialize Looper.

        Args:
            network: already initialized model
            device: a device model is working on
            loss: the cost function
            optimizer: already initialized optimizer link to network parameters
            data_loader: already initialized data loader
            dataset_size: no. of samples in dataset
            plot: matplotlib axes
            validation: flag to set train or eval mode

        """
        self.network = network
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.loader = data_loader
        self.size = dataset_size
        self.validation = validation
        self.plots = plots
        self.running_loss = []
        self.MC = MC
        self.history = []
        self.LOG = True

    def run(self):
        """Run a single epoch loop.

        Returns:
            Mean absolute error.
        """
        # reset current results and add next entry for running loss
        self.true_values = []
        self.predicted_values = []
        self.running_loss.append(0)
        #self.history.append(0)

        # set a proper mode: train or eval
         
        self.network.train(not self.validation)
        
   
        #print("echo")

        if self.MC:
            for m in self.network.modules():
                if m.__class__.__name__.startswith('Dropout'): m.train() 
   

   
        for image, label in self.loader:
            
            # move images and labels to given device
            image = image.to(self.device)
            label = label.to(self.device)
        
           
            # clear accumulated gradient if in train mode
            if not self.validation:
                if not self.MC:
                    self.optimizer.zero_grad()
            
            # get model prediction (a density map)
            

            result = self.network(image)
            
        
            # calculate loss and update running loss
            loss = self.loss(result, label)
            self.running_loss[-1] += image.shape[0] * loss.item() / self.size

            # update weights if in train mode
            if not self.validation:
                if not self.MC:
                    loss.backward()
                    self.optimizer.step()

            # loop over batch samples
            for true, predicted in zip(label, result):
                # integrate a density map to get no. of objects
                # note: density maps were normalized to 100 * no. of objects
                #       to make network learn better
                true_counts = torch.sum(true).item() / 100
                predicted_counts = torch.sum(predicted).item() / 100

                # update current epoch results
                self.true_values.append(true_counts)
                self.predicted_values.append(predicted_counts)

        # calculate errors and standard deviation
        self.update_errors()



        # update live plot
        if self.plots is not None:
            self.plot()

        # print epoch summary
        if self.LOG : self.log()

        return self.mean_abs_err

    def update_errors(self):
        """
        Calculate errors and standard deviation based on current
        true and predicted values.
        """
        self.err = [true - predicted for true, predicted in
                    zip(self.true_values, self.predicted_values)]
        self.abs_err = [abs(error) for error in self.err]
        self.mean_err = sum(self.err) / self.size
        self.mean_abs_err = sum(self.abs_err) / self.size
        self.std = np.array(self.err).std()
        temp  = [self.running_loss[-1], self.mean_err, self.mean_abs_err, self.std]
  
        self.history.append(temp)
        

    def plot(self):
        """Plot true vs predicted counts and loss."""
        #print("Plot true vs predicted counts and loss.")
        # true vs predicted counts
        true_line = [[0, max(self.true_values)]] * 2  # y = x
        self.plots[0].cla()
        self.plots[0].set_title('Train' if not self.validation else 'Valid')
        self.plots[0].set_xlabel('True value')
        self.plots[0].set_ylabel('Predicted value')
        self.plots[0].plot(*true_line, 'r-')
        self.plots[0].scatter(self.true_values, self.predicted_values)

        # loss
        epochs = np.arange(1, len(self.running_loss) + 1)
        self.plots[1].cla()
        self.plots[1].set_title('Train' if not self.validation else 'Valid')
        self.plots[1].set_xlabel('Epoch')
        self.plots[1].set_ylabel('Loss')
        self.plots[1].plot(epochs, self.running_loss)

        matplotlib.pyplot.pause(0.01)
        matplotlib.pyplot.tight_layout()

    def log(self):
        """Print current epoch results."""
        print(f"{'Train' if not self.validation else 'Valid'}:\n"
              f"\tAverage loss: {self.running_loss[-1]:3.4f}\n"
              f"\tMean error: {self.mean_err:3.3f}\n"
              f"\tMean absolute error: {self.mean_abs_err:3.3f}\n"
              f"\tError deviation: {self.std:3.3f}")

    def MCdropOut(self,NN, network_architecture, dataset_name):

        self.plots = None
        self.validation = True
        self.LOG = False
        self.MC  = True
        
        self.run()

        Predicted__values   = np.array(self.predicted_values)
        Predicted__values2  = np.power(Predicted__values,2)

        for _ in range(NN):
            #print('hej 0')
            self.run()
            temp   = np.array(self.predicted_values)
            #print("NN = ",NN)
            Predicted__values   += temp
            Predicted__values2  += np.power(temp,2)

        Predicted__values  = Predicted__values/(NN+1)
        Predicted__values2 = Predicted__values2/(NN+1)

        error = np.around(Predicted__values2 - np.power(Predicted__values,2),decimals=6)
    

        error = np.sqrt(error)
        #print(np.sqrt(Predicted__values2[0]))
    
        true_line = [[0, max(self.true_values)]] * 2  # y = x
        figg, axes = pyplot.subplots()

        axes.set_xlabel('True value')
        axes.set_ylabel('Predicted value')
        axes.plot(*true_line, 'r-')
        axes.errorbar(self.true_values,Predicted__values, error,fmt='.k',ecolor='red')

        matplotlib.pyplot.pause(0.01)
        matplotlib.pyplot.tight_layout()

        figg.savefig(f'error_{dataset_name}_{network_architecture}.png')

        result = np.stack((self.true_values,Predicted__values, error),-1)
        np.savetxt(f'error_{dataset_name}_{network_architecture}.csv',result, delimiter=',')
