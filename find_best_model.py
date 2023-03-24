import numpy as np
import pandas as pd
import os.path
from pathlib import Path


def read_k_fold_data(f):
    data = {}
    for i in range(3):
        data[i] = pd.read_csv(f.replace('XXXX',str(i)),delimiter=' ',header=None)
    return data

def estimate_validation_results(f):
    data = read_k_fold_data(f)
    avg_train_mse, avg_val_mse, avg_train_pc, avg_val_pc = [], [], [], []
    for i in range(3):
        avg_train_mse.append(data[i].iloc[0,0])    
        avg_val_mse.append(data[i].iloc[0,1])
        avg_train_pc.append(data[i].iloc[1].mean())
        avg_val_pc.append(data[i].iloc[2].mean())
    return avg_train_mse, avg_val_mse, avg_train_pc, avg_val_pc


def mean_validation_results(f):
    avg_train_mse, avg_val_mse,  avg_train_pc, avg_val_pc = estimate_validation_results(f)        
    return [np.mean(avg_train_mse), np.mean(avg_val_mse), np.mean(avg_train_pc), np.mean(avg_val_pc)]      

def mean_test_results(f):
    avg_train_mse, avg_val_mse,  avg_train_pc, avg_val_pc = estimate_validation_results(f)
    return [np.mean(avg_val_mse), np.mean(avg_val_pc)]



### total numer of hyperprmset space ---- 131220
def run_final(which, subject, NN):
    count=0
    start=1000
    col = ['index' ,'avg_train_mse', 'avg_val_mse', 'avg_test_mse', 'avg_train_pc', 'avg_val_pc', 'avg_test_pc']
    df = pd.DataFrame(columns = col)
    for i in range(9721):
        f1='text_out/stat_'+NN+'_'+which+'_'+ subject+'.hv_'+str(i)+'.CV_XXXX.txt'
        f2='text_out/stat_'+NN+'_'+which+'_'+ subject+'.hv_'+str(i)+'.fm.txt'
        try:
            df.loc[i] = 7*np.nan
            df.loc[i]['index']       = i 
            df.loc[i][['avg_train_mse', 'avg_val_mse', 'avg_train_pc', 'avg_val_pc']] = mean_validation_results(f1)
        except:
            None

        try:
            df.loc[i][[ 'avg_test_mse', 'avg_test_pc']] = mean_test_results(f2)
        except:
            None

    return df



class stat:
    def __init__(self, sub, NN):
        self.JA  = run_final("JA",  sub, NN)
        self.JM  = run_final("JM",  sub, NN)
        self.JRF = run_final("JRF", sub, NN)
        self.MA  = run_final("MA",  sub, NN)
        self.MF  = run_final("MF",  sub, NN)

class stat_CV:
    def __init__(self,NN):
        self.naive   = stat('naive',NN)
        self.exposed = stat('exposed',NN)


def compute_stat():
    df = stat_CV("NN")

    return df

df = compute_stat()

