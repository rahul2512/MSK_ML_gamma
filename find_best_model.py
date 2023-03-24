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
#            print("index not found--", i )
#            if tmp < start:
#                print("model_index = ",i,start,which)
#                start = tmp
#                return_model_stat(f)
#                count=count+1
#                print("XXXXXXXX-----",count,"-------XXXXXXXX")
#                if count>10:
#                    train_final_model(i,which)

    return df

df = run_final("JA",'exposed','NN')

