import numpy as np
import pandas as pd
import os.path
#from pathlib import Path
#from pytorch import create_final_model
hyper =  pd.read_csv('hyperparam.txt',delimiter='\s+')

def read_k_fold_data(f):
	data = {}
	for i in range(4):
		data[i] = pd.read_csv(f.replace('XXXX',str(i)),delimiter=' ',header=None)
	return data

def estimate_validation_results(f):
	data = read_k_fold_data(f)
	avg_train_mse, avg_val_mse, avg_train_pc, avg_val_pc = [], [], [], []
	for i in range(4):
		avg_train_mse.append(data[i].iloc[0,0])	
		avg_val_mse.append(data[i].iloc[0,1])
		avg_train_pc.append(data[i].iloc[1].mean())
		avg_val_pc.append(data[i].iloc[2].mean())
	return avg_train_mse, avg_val_mse, avg_train_pc, avg_val_pc




def return_model(f,criteria):
	avg_train_mse, avg_val_mse,  avg_train_pc, avg_val_pc = estimate_validation_results(f)
	if criteria == "avg_val_mse":
		if np.mean(avg_val_mse) == 0:
			return 1000
		else:
			return np.mean(avg_val_mse), np.mean(avg_val_pc)


def return_model_stat(f):
	avg_train_mse, avg_val_mse,  avg_train_pc, avg_val_pc = estimate_validation_results(f)
	print('avg_train_mse',np.mean(avg_train_mse))
	print('avg_val_mse',  np.mean(avg_val_mse))
	print('avg_train_pc', np.mean(avg_train_pc))
	print('avg_val_pc',   np.mean(avg_val_pc))
	print('---------XXXXXXXXX---------XXXX--------XXXXXXXXXX--------')

def run_final(which, what, subject, index):
    count=0
    start=10000
    for i in range(3889):
        fcv='text_out/stat_'+what+'_'+which+'_'+ subject+'.hv_'+str(i)+'.CV_3.txt'
        f='text_out/stat_'+what+'_'+which+'_'+ subject+'.hv_'+str(i)+'.fm.txt'

        try:
            tmp1,tmp2 = return_model(f,"avg_val_mse")
            if tmp1 < start:
                start=tmp1
                print('python',  '${path}/specific.py' , which,  subject, what, i, "&", "#", tmp1, tmp2, f, index, 'fcv = ',os.path.exists(fcv))
                index = index+1
        except:
            None
    return index

count=1
# SimpleRNN
for j in ["JRF", "JA", "MA", "MF", "JM"]:
    for k in ['exposed','naive']:
        for i in ["SimpleRNN", "BSimpleRNN", "LSTM", "BLSTM", "GRU", "BGRU"]:
            count = run_final(j, i, k, count)
        print('\n')

print("wait")
