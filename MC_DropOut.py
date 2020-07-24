import numpy as np
import matplotlib
from matplotlib import pyplot


# --- Mak active dropout --- #
def make_active_dropout(model):
  model.eval()
  for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
      m.train()    


def MC_DropOut(looper_end, NN, network_architecture,dataset_name):

    looper_end.run()
    
    Predicted__values   = np.array(looper_end.predicted_values)
  
    Predicted__values2  = np.power(Predicted__values,2)

    for _ in range(NN):
        looper_end.run()
        temp   = np.array(looper_end.predicted_values)
        print("NN = ",NN)
        Predicted__values   += temp
        Predicted__values2  += np.power(temp,2)

    Predicted__values  = Predicted__values/(NN+1)
    Predicted__values2 = Predicted__values2/(NN+1)

    error = np.around(Predicted__values2 - np.power(Predicted__values,2),decimals=6)
    

    error = np.sqrt(error)
    #print(np.sqrt(Predicted__values2[0]))
    
    true_line = [[0, max(looper_end.true_values)]] * 2  # y = x
    figg, axes = pyplot.subplots()

    axes.set_xlabel('True value')
    axes.set_ylabel('Predicted value')
    axes.plot(*true_line, 'r-')
    axes.errorbar(looper_end.true_values,Predicted__values, error,fmt='.k',ecolor='red')

    matplotlib.pyplot.pause(0.01)
    matplotlib.pyplot.tight_layout()

    figg.savefig(f'error_{dataset_name}_{network_architecture}.png')

    result = np.stack((looper_end.true_values,Predicted__values, error),-1)
    np.savetxt(f'error_{dataset_name}_{network_architecture}.csv',result, delimiter=',')