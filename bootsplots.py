import numpy as np
import matplotlib.pyplot as plt
import pylab as plb
import math
import os
from os import listdir
from os.path import isfile, join

plb.rcParams['font.size'] = 12


def plot_history(folder_,data, net):
    begin_ = 'hist_test_boot_'+data+'_'+net+'_i='
    beginr_ = 'hist_train_boot_'+data+'_'+net+'_i='

    print(begin_)

    filest = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(begin_)]))
    filesr = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(beginr_)]))

    print(filest)    

    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].set_title( net+', '+data+' validation set')
    axes[1].set_title( net+', '+data+' train set')

    for name in filest:
        hist = np.genfromtxt(name, delimiter=",") 
        axes[0].plot(np.arange(len(hist))+1, hist[:,0])
    

    for name in filesr:
        hist = np.genfromtxt(name, delimiter=",") 
        axes[1].plot(np.arange(len(hist))+1, hist[:,0] )

    for i in range(2):
        axes[i].set_yscale('log')
        axes[i].set_xlabel('epochs')
    
    fig.tight_layout()
    fig.savefig(f'history_boost_{data}_{net}.png')


def plot_history2(folder_,data, net):
    begin_ = 'hist_test_boot_i='
    beginr_ = 'hist_train_boot_i='

    print(begin_)

    filest = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(begin_)]))
    filesr = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(beginr_)]))

    print(filest)    

    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].set_title( net+', '+data+' validation set')
    axes[1].set_title( net+', '+data+' train set')

    for name in filest:
        hist = np.genfromtxt(name, delimiter=",") 
        axes[0].plot(np.arange(len(hist))+1, hist[:,0])
    

    for name in filesr:
        hist = np.genfromtxt(name, delimiter=",") 
        axes[1].plot(np.arange(len(hist))+1, hist[:,0] )

    for i in range(2):
        axes[i].set_yscale('log')
        axes[i].set_xlabel('epochs')
    
    fig.tight_layout()
    fig.savefig(f'history_boost_{data}_{net}.png')


def plot_history3(folder_,data, net):
    #begin_ = 'hist_test_boot_i='
    beginr_ = 'hist_train_boot_i='

 
    #filest = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(begin_)]))
    filesr = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(beginr_)]))

    fig, axes = plt.subplots(1,1, figsize=(5,5))
    axes.set_title( net+', '+data+' train set')

    for name in filesr:
        hist = np.genfromtxt(name, delimiter=",") 
        axes.plot(np.arange(len(hist))+1, hist[:,0] )

    axes.set_yscale('log')
    axes.set_xlabel('epochs')
    
    fig.tight_layout()
    fig.savefig(f'history_boost_{data}_{net}.png')




def calculate_prediction_mean_with_1sigma(folder_,filename):
    res = np.genfromtxt(os.path.join(folder_,filename), delimiter=",")
    #print('shape',res[:,0])
    res_temp=res[:,1:]
    MEAN = np.mean(res_temp, axis=1)
    VAR = np.var(res_temp, axis=1)
    VAR = np.sqrt(VAR)
    wyn = np.stack((MEAN,VAR))
    
    wyn =  np.concatenate((wyn,np.array([res[:,0]])))
    
    wyn = wyn.transpose()
    print(wyn.shape)

    ff = os.path.join(folder_,'averaged_'+filename)
    np.savetxt(ff ,wyn, delimiter=',')




  
#calculate_prediction_mean_with_1sigma('boots_results','predicted_train_best_boot_cell_UNet_epochs=70_batch=8_hf=0.0_vf=0.0_uf=64_conv2.csv')
#calculate_prediction_mean_with_1sigma('boots_results','predicted_test_best_boot_cell_UNet_epochs=70_batch=8_hf=0.0_vf=0.0_uf=64_conv2.csv')


def plot_predictions_with_uncertienty(folder_,data,net):

    begin_ = 'averaged_predicted_test_best_boot_'+data+'_'+net
    beginr_ = 'averaged_predicted_train_best_boot_'+data+'_'+net

    filest = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(begin_)]))
    filesr = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(beginr_)]))

    print(filest)    



    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].cla()
    axes[1].cla()


    axes[0].set_title( net+', '+data+' validation set')
    axes[1].set_title( net+', '+data+' train set')

    for name in filest:
        hist = np.genfromtxt(name, delimiter=",") 
        xx = [[0. , 1.1*max(hist[:,2])]] *2        
        # true line
        axes[0].plot(*xx,'r-')
        axes[0].errorbar(hist[:,2],hist[:,0],yerr=hist[:,1],fmt='.k')
    

    for name in filesr:
        hist = np.genfromtxt(name, delimiter=",") 
        xx = [[0. , 1.1*max(hist[:,2])]] *2        
        # true line
        axes[1].plot(*xx,'r-')
        axes[1].errorbar(hist[:,2],hist[:,0],yerr=hist[:,1],fmt='.k')

    for i in range(2):
        axes[i].set_ylabel('predicted')
        axes[i].set_xlabel('true')
    
    fig.tight_layout()
    fig.savefig(f'averaged_boost_{data}_{net}.png')


def plot_hist(folder_,data,net):
    begin_ = 'averaged_predicted_test_best_boot_'+data+'_'+net
    beginr_ = 'averaged_predicted_train_best_boot_'+data+'_'+net

    filest = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(begin_)]))
    filesr = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(beginr_)]))

    #print(filesr," --- ")    
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].cla()
    axes[1].cla()

    fig2, axes2 = plt.subplots(1,2, figsize=(10,5))
    axes2[0].cla()
    axes2[1].cla()


    for name in filest:
        history = np.genfromtxt(name, delimiter=",") 
        h = 100*history[:,1]/history[:,0]
        mean = np.mean(h)
        var  =  np.sqrt(np.var(h))
        axes[0].hist(h,bins=15)
        axes[0].set_title("test set,\n mean="+ str(round(mean,1)) +" $\sqrt{var}=$"+str(round(var,1)))
        axes2[0].set_title("test set")
        axes2[0].scatter(history[:,2],h,marker='.')

    for name in filesr:
        history = np.genfromtxt(name, delimiter=",") 
        h = 100*history[:,1]/history[:,0]
        mean = np.mean(h)
        var  =  np.sqrt(np.var(h))
        
        axes[1].hist(100*history[:,1]/history[:,0],bins=15)
        axes[1].set_title("train set,\n mean="+ str(round(mean,1)) +" $\sqrt{var}=$"+str(round(var,1)))
        axes2[1].set_title("train set")
        axes2[1].scatter(history[:,2],h,marker='.')

        

    for i in range(2):
        axes[i].set_xlabel('uncertainty ($\%$)')
        axes2[i].set_ylabel('uncertainty ($\%$)')
        axes2[i].set_xlabel('true number')

    fig.tight_layout()
    fig.savefig(f'hist_boost_{data}_{net}.png')

    
    fig2.tight_layout()
    fig2.savefig(f'error_vs_number_boost_{data}_{net}.png')

    

#plot_history("boots_results/",'cell', 'UNet')
#plot_history("boots_results/",'ucsd', 'UNet')
plot_history3("boots_results_nocover/",'nocover', 'UNet2')
plot_hist("boots_results_nocover/",'nocover', 'UNet2')

#calculate_prediction_mean_with_1sigma('boots_results','predicted_train_best_boot_cell_UNet_epochs=70_batch=8_hf=0.0_vf=0.0_uf=64_conv2.csv')
#calculate_prediction_mean_with_1sigma('boots_results','predicted_test_best_boot_cell_UNet_epochs=70_batch=8_hf=0.0_vf=0.0_uf=64_conv2.csv')

#calculate_prediction_mean_with_1sigma('boots_results','predicted_test_best_boot_ucsd_UNet_epochs=70_batch=5_hf=0.0_vf=0.0_uf=64_conv8.csv')
#calculate_prediction_mean_with_1sigma('boots_results','predicted_train_best_boot_ucsd_UNet_epochs=70_batch=5_hf=0.0_vf=0.0_uf=64_conv8.csv')

calculate_prediction_mean_with_1sigma('boots_results_nocover/','predicted_test_best_boot_nocover_UNet2_epochs=60_batch=7_hf=0.0_vf=0.0_uf=64_conv4.csv')  
calculate_prediction_mean_with_1sigma('boots_results_nocover/','predicted_train_best_boot_nocover_UNet2_epochs=60_batch=7_hf=0.0_vf=0.0_uf=64_conv4.csv')


#plot_predictions_with_uncertienty('boots_results/','cell','UNet')
#plot_predictions_with_uncertienty('boots_results/','ucsd','UNet')
plot_predictions_with_uncertienty('boots_results_nocover/','nocover','UNet2')

