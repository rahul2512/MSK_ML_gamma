import numpy as np, time, keras, os.path, pandas as pd, sys, matplotlib.pyplot as plt
from barchart_err import barchart_error, barchart_params
from pytorch import run_final_model, run_cross_valid, check_interpolation, combined_plot, save_outputs, stat_new_data
from pytorch import RNN_models,  feature_slist, feature_list, stat, specific_CV, specific, explore, print_tables
from read_in_out import initiate_data, initiate_RNN_data, analysis_options, ML_analysis
#feat_order     = ['JA','JM','JRF','MA','MF']

window = 20
data_kind  =  ['LM', 'NN', 'RNN', 'CNN', 'CNNLSTM', 'convLSTM']
data_kind  =  ['NN','LM']
data_kind  =  ['NN']
fm = ML_analysis('final_model_list', data_kind, window)

should = 1

if should:
    # fm.LM.exposed.arg      = [12, 12, 12]
    # fm.LM.naive.arg        = [12, 12, 12]
    # fm.LM.exposed.arch     = ['LM']*3
    # fm.LM.naive.arch       = ['LM']*3

    fm.NN.exposed.arg        = [1080, 2809, 1118]
    fm.NN.naive.arg          = [1080, 6778, 2164]
    fm.NN.exposed.arch       = ['NN']*3
    fm.NN.naive.arch         = ['NN']*3

    # fm.VRNN = fm.RNN 
    # fm.LSTM = fm.RNN 
    # fm.GRU  = fm.RNN 
    
    # fm.VRNN.exposed.arg       = [4335, 2553, 4767]
    # fm.VRNN.naive.arg         = [2610, 1071, 5016] 
    # fm.VRNN.exposed.arch      = ['RNN']*3
    # fm.VRNN.naive.arch        = ['RNN']*3   ## (SimpleRNN, LSTM, GRU)

    # fm.LSTM.exposed.arg       = [4969, 4549, 871]
    # fm.LSTM.naive.arg         = [871, 4780, 2062]
    # fm.LSTM.exposed.arch      = ['RNN']*3
    # fm.LSTM.naive.arch        = ['RNN']*3   

    # fm.GRU.exposed.arg        = [983 , 5150, 983]
    # fm.GRU.naive.arg          = [2807, 4331, 4274]
    # fm.GRU.exposed.arch       = ['RNN']*3
    # fm.GRU.naive.arch         = ['RNN']*3   ## (SimpleRNN, LSTM, GRU)

def train_final_models(D):
    ## train final model with best-avg-validation accuracy
    for d  in D:
        for i in range(3):
            specific(d.exposed,i)
            specific(d.naive  ,i)
    return None

def compute_stat(fm):
    for D in [fm.LM, fm.NN]:
        for i in range(3):
            D.exposed = stat(D.exposed,i)
            D.naive   = stat(D.naive,i)
    return fm

def plot_final_results(fm):
    analysis_opt = analysis_options()        
    analysis_opt.save_name = 'final'
    analysis_opt.trial_ind = 2
    analysis_opt.plot_subtitle   = [False, True]
    analysis_opt.legend_label   = ['LM', 'NN']
    analysis_opt.window_size = [0,0]
    analysis_opt.data    = [fm.LM.data,fm.NN.data]
    
    for i in range(3):
        analysis_opt.feature   = fm.feature[i]

        analysis_opt.model_exposed_hyper_arg  = [fm.LM.exposed.arg[i], fm.NN.exposed.arg[i]]
        analysis_opt.model_naive_hyper_arg    = [fm.LM.naive.arg[i], fm.NN.naive.arg[i]]
        
        analysis_opt.model_exposed_arch  = [fm.LM.exposed.arch[i],fm.NN.exposed.arch[i]]
        analysis_opt.model_naive_arch    = [fm.LM.naive.arch[i],fm.NN.naive.arch[i]]

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

# hyper_index = int(sys.argv[1])
# explore(fm.CNN, hyper_index)
#train_final_models([fm.VRNN])

b = initiate_data('Braced_')
stat_new_data(fm.NN, b)

# fm = compute_stat(fm)
#plot_final_results(fm)
# print_tables(fm.NN)
# print_tables(fm.LM)
