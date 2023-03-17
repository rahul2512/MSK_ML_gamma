import numpy as np, time
import pandas as pd, keras
import os.path
from pathlib import Path

from pytorch import run_final_model, run_cross_valid, check_interpolation, combined_plot, save_outputs , plot_MSK_data
from pytorch import RNN_models,  feature_slist, feature_list, print_optimal_results, stat, specific_CV, specific, print_SI_table1, print_SI_table2, explore
from pytorch import print_SI_table3
from pytorch_utilities import hyper_param
from read_in_out import initiate_data, initiate_RNN_data, analysis_options, ML_analysis
import matplotlib.pyplot as plt
import sys
from barchart_err import barchart_error, barchart_params

path = './'
#path = '/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'

window = 10
fm = ML_analysis('final_model_list', path, window)

fm.LM.exposed.arg      = [11, 10, 8, 8,10]
fm.LM.naive.arg        = [0, 0, 0, 2, 0]
fm.LM.exposed.arch     = ['LM']*5
fm.LM.naive.arch       = ['LM']*5

fm.NN.exposed.arg        = [7560, 2286,  375, 34147, 2254]
fm.NN.naive.arg          = [7077, 6591,  377, 30380, 7646]
fm.NN.exposed.arch       = ['NN']*5
fm.NN.naive.arch         = ['NN']*5

# fm.RNN.exposed.arg       = [1252, 1836, 1537, 1489, 1416]
# fm.RNN.exposed.arch      = ['BLSTM','LSTM','BLSTM','LSTM','LSTM']

# fm.RNN.naive.arg         = [     52,     36,   694,  1037,  1934]  
# fm.RNN.naive.arch        = ['BLSTM', 'LSTM', 'GRU', 'GRU', 'LSTM']



def compute_stat(fm):
    for D in [fm.LM, fm.NN, fm.RNN]:
        for i in range(5):
            D.exposed = stat(D.exposed,i)
            D.naive   = stat(D.naive,i)
    return fm

def plot_final_results(fm):
    analysis_opt = analysis_options()        
    analysis_opt.save_name = 'final'
    analysis_opt.plot_subtitle   = [False, True]
    analysis_opt.legend_label   = ['RNN', 'NN']
    analysis_opt.window_size = [0,10]
    analysis_opt.data    = [fm.RNN.data,fm.NN.data]
    
    for i in range(5):    
        analysis_opt.feature   = fm.feature[i]

        analysis_opt.model_exposed_hyper_arg  = [fm.RNN.exposed.arg[i], fm.NN.exposed.arg[i]]
        analysis_opt.model_naive_hyper_arg    = [fm.RNN.naive.arg[i], fm.NN.naive.arg[i]]
        
        analysis_opt.model_exposed_arch  = [fm.RNN.exposed.arch[i],fm.NN.exposed.arch[i]]
        analysis_opt.model_naive_arch    = [fm.RNN.naive.arch[i],fm.NN.naive.arch[i]]

        combined_plot(analysis_opt)

    return None

def avg_stat(fm):
    for j in [fm.LM.exposed, fm.LM.naive, fm.NN.exposed,fm.NN.naive,fm.RNN.exposed,fm.RNN.naive]:
        a,b = [],[]
        for i in fm.feature:
            a = a + j.NRMSE[i]
            b = b + j.pc[i]
        print('%',np.around(np.mean(a),2),np.around(np.std(a),2), j.kind, j.subject, 'NRMSE')
        print('%',np.around(np.mean(b),2),np.around(np.std(b),2), j.kind, j.subject, 'pc')

explore(fm.NN, 1)