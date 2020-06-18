import numpy as np
import matplotlib.pyplot as plt
import pylab as plb
import math
import os
from os import listdir
from os.path import isfile, join

plb.rcParams['font.size'] = 12




def make_plots_with_errors(ps,type_,data,model_name,dirr):
    nazwa_beg  = 'error_'+ data +'_'+ type_ +"__"
    end1       = "_MC_p="
    
    os.chdir(dirr)

    pliki = [nazwa_beg+model_name+end1+str(p)+'.csv' for p in ps]

    print(pliki)

    results = [ np.genfromtxt(name, delimiter=",") for name in pliki]
    

    fig, axes  = plt.subplots(1, 2, figsize=(10.5,5))

    N = len(results[0])

    #axes[0].scatter(np.arange(N)+1,(results[0][:,0]-results[0][:,1])/results[0][:,0])
    #axes[0].scatter(np.arange(N)+1,(results[1][:,0]-results[1][:,1])/results[1][:,0])
    
    xx = [[0. , max(results[0][:,0])]] *2
    axes[0].cla()
    # true line
    axes[0].plot(*xx,'r-')

    for i in range(len(ps)):
        axes[0].scatter(results[i][:,0],results[i][:,1],label="p="+str(ps[i]),marker='.') 
        
    axes[0].set_xlabel('true')
    axes[0].set_ylabel('predicted')

    axes[0].legend()

    for i in range(len(ps)-1):
        print(i)
        axes[1].set_xlabel('patterns')
        axes[1].set_ylabel('normalized uncertainty ($\%$)')
        axes[1].scatter(np.arange(N)+1,100*results[i+1][:,2]/results[i+1][:,1],label="p="+str(ps[i+1]),marker='.')
    fig.tight_layout()
    fig.savefig('with_errors_'+ type_+'_'+ data+ '_'+model_name+'.png')
   

    fig2, axes2  = plt.subplots(2, 3, figsize=(16,10))
    axes2 = axes2.flatten()
    for i in range(len(ps)-1):
        print(i)
        axes2[i].set_xlabel('uncertainty ($\%$)')
        mean = np.mean(100*results[i+1][:,2]/results[i+1][:,1])
        var  = np.var((100*results[i+1][:,2]/results[i+1][:,1]))
        var  = math.sqrt(var)
        axes2[i].set_title("p="+str(ps[i+1])+'\n mean='+str(round(mean,1))+', $\sqrt{\mathrm{var}}=$'+str(round(var,1)))
        axes2[i].hist(100*results[i+1][:,2]/results[i+1][:,1], bins=15)
    fig2.tight_layout()
    fig2.savefig('hist_'+ type_+'_'+ data +'_'+model_name+'.png')
    
    os.chdir("..")

def plot_history(folder_,data, net):
    begin_ = 'hist_test_'+data+'_'+net
    beginr_ = 'hist_train_'+data+'_'+net

    filest = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(begin_)]))
    filesr = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(beginr_)]))
    

    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].set_title( net+', '+data+' validation set')
    axes[1].set_title( net+', '+data+' train set')

    for name in filest:
        hist = np.genfromtxt(name, delimiter=",") 
        axes[0].plot(np.arange(len(hist))+1, hist[:,0],label=name.split('.csv')[0][-5:] )
    

    for name in filesr:
        hist = np.genfromtxt(name, delimiter=",") 
        axes[1].plot(np.arange(len(hist))+1, hist[:,0], label=name.split('.csv')[0][-5:] )

    for i in range(2):
        axes[i].set_yscale('log')
        axes[i].set_xlabel('epochs')
        if i != 1: axes[i].set_ylabel('loss')
        axes[i].legend()
    
    fig.tight_layout()
    fig.savefig(f'history_{data}_{net}.png')


def plot_best_history(folder_,data,net):
    begin_ = 'hist_best_'+data+'_'+net
    filest = np.sort(np.asarray([join(folder_, f) for f in listdir(folder_) if isfile(join(folder_, f)) if f.startswith(begin_)]))
    
    HIST =[]
    for name in filest:
        hist = np.genfromtxt(name, delimiter=",")
        HIST.append([hist[0,0],float(name.split('.csv')[0][-3:])])

    HIST= np.array(HIST)


    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    
    axes.scatter(HIST[:,1],HIST[:,0], label='validation')

    HIST =[]
    for name in filest:
        hist = np.genfromtxt(name, delimiter=",")
        HIST.append([hist[1,0],float(name.split('.csv')[0][-3:])])
    
    HIST= np.array(HIST)

    axes.scatter(HIST[:,1],HIST[:,0], label='train')

    axes.set_yscale('log')
    axes.set_xlabel('p')
    
    axes.legend(loc='upper left')
    #fig.tight_layout()
    fig.savefig(f'history_onp_{data}_{net}.png')

#https://github.com/cpark321/uncertainty-deep-learning/blob/master/03.%20Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.ipynb
#https://discuss.pytorch.org/t/adversarial-training-with-pytorch/67980
#https://github.com/cpark321/uncertainty-deep-learning
#https://discuss.pytorch.org/t/adversarial-training-with-pytorch/67980

model_name = "FCRN_A"
type_ = 'test'

ps= [0.0,0.1,0.2,0.3,0.4, 0.5, 0.6]
#make_plots_with_errors(ps,'test',"FCRN_A","results")
#make_plots_with_errors(ps,'train',"FCRN_A","results")

#make_plots_with_errors(ps,'test',"UNet","results")
#make_plots_with_errors(ps,'train',"UNet","results")
ps= [0.0,0.1,0.2,0.3,0.4, 0.5, 0.6]
#make_plots_with_errors(ps,'test','ucsd',"UNet","results")
#make_plots_with_errors(ps,'train','ucsd',"UNet","results")

#plot_history('results','cell','FCRN_A_MC')
#plot_history('results','cell','UNet_MC')

#plot_history('results','ucsd','UNet_MC')


plot_best_history('results','cell','FCRN_A_MC')
plot_best_history('results','cell','UNet_MC')
plot_best_history('results','ucsd','UNet_MC')
