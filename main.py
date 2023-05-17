import numpy as np, os.path, pandas as pd, sys, matplotlib.pyplot as plt
#from barchart_err import barchart_error, barchart_params
from pytorch import run_final_model, run_cross_valid, combined_plot, save_outputs, stat_new_data
from pytorch import feature_slist, feature_list, stat, specific, explore, print_tables, combined_plot_noise, learning_curve, plot_learning_curve
from read_in_out import initiate_data, initiate_RNN_data, analysis_options, ML_analysis
from joblib import Parallel, delayed
import copy
from pytorch_utilities import initiate_transformer

feat_order     = ['JA','JM','JRF']#,'MA','MF']

window = 10
window=20  ## when CNN
#data_kind  =  [ 'CNN', 'CNNLSTM']
#data_kind  =  ['NN','LM', 'RNN']

data_kind  =  ['CNN']
fm = ML_analysis('final_model_list', data_kind, window)
# hyper_arg = int(sys.argv[1])
# explore(fm.xgbr, hyper_arg)

should = 0
if should:
    None
    # fm.rf.exposed.arg      = [43, 43, 43]
    # fm.rf.naive.arg        = [43, 43, 43]
    # fm.rf.exposed.arch     = ['rf']*3
    # fm.rf.naive.arch       = ['rf']*3
    # fm.rf.exposed_unseen     = copy.deepcopy(fm.rf.exposed)

    fm.LM.exposed.arg      = [43, 43, 43]
    fm.LM.naive.arg        = [43, 43, 43]
    fm.LM.exposed.arch     = ['LM']*3
    fm.LM.naive.arch       = ['LM']*3
    fm.LM.exposed_unseen     = copy.deepcopy(fm.LM.exposed)

    # fm.NN.exposed.arg        = [2003, 2809, 2003]
    # fm.NN.naive.arg          = [4011, 8365, 3903]
    # fm.NN.exposed.arch       = ['NN']*3
    # fm.NN.naive.arch         = ['NN']*3
    # fm.NN.exposed_unseen     = copy.deepcopy(fm.NN.exposed)


#    fm.NN.exposed.arg        = [218, 2164, 2202]
#    fm.NN.naive.arg          = [3098, 6778, 4132]
#    fm.NN.exposed.arch       = ['NN']*3
#    fm.NN.naive.arch         = ['NN']*3
#    fm.NN.exposed_unseen     = copy.deepcopy(fm.NN.exposed)

#    fm.VRNN = copy.deepcopy(fm.RNN) 
#    fm.LSTM = copy.deepcopy(fm.RNN) 
#    fm.GRU  = copy.deepcopy(fm.RNN) 
    
 #   fm.VRNN.exposed.arg       = [3411, 3408, 3413]
 #   fm.VRNN.naive.arg         = [3169, 2136, 237  ] 
 #   fm.VRNN.exposed.arch      = ['RNN']*3
 #   fm.VRNN.naive.arch        = ['RNN']*3   ## (SimpleRNN, LSTM, GRU)

  #  fm.LSTM.exposed.arg       = [6869,5930,6869]
  #  fm.LSTM.naive.arg         = [7493,4108, 4396 ]
  #  fm.LSTM.exposed.arch      = ['RNN']*3
  #  fm.LSTM.naive.arch        = ['RNN']*3   

   # fm.GRU.exposed.arg        = [9441, 10346, 10301]
   # fm.GRU.naive.arg          = [10133,8596,10093 ]
   # fm.GRU.exposed.arch       = ['RNN']*3
   # fm.GRU.naive.arch         = ['RNN']*3   ## (SimpleRNN, LSTM, GRU)


#    fm.CNN.exposed.arg        = [105,555,253]
#    fm.CNN.naive.arg          = [105, 413,253]
#    fm.CNN.exposed.arg        = [248,410,106]
#    fm.CNN.naive.arg          = [52, 516,58]
#    fm.CNN.exposed.arch       = ['CNN']*3
#    fm.CNN.naive.arch         = ['CNN']*3
#    fm.CNN.exposed_unseen     = copy.deepcopy(fm.NN.exposed)


#    fm.CNNLSTM.exposed.arg        = [387, 1361, 1251]
#    fm.CNNLSTM.naive.arg          = [387, 907, 1251]

#    fm.CNNLSTM.exposed.arch       = ['CNNLSTM']*3
#    fm.CNNLSTM.naive.arch         = ['CNNLSTM']*3
#    fm.CNNLSTM.exposed_unseen     = copy.deepcopy(fm.NN.exposed)



def train_final_models(D):
    ## train final model with best-avg-validation accuracy
    for i in range(3):
        specific(D.exposed,i)
        specific(D.naive  ,i)
    return None

def compute_stat(f):
    for D in f:
        for i in range(3):
            D.exposed = stat(D.exposed,i)
            D.naive   = stat(D.naive,i)
            try:
                D.exposed_unseen.subject = 'exposed_unseen'
                D.exposed_unseen = stat(D.exposed_unseen, i)
            except:
                None
    return fm

def plot_final_results(fm):
    analysis_opt = analysis_options()        
    analysis_opt.save_name = 'final'
    analysis_opt.trial_ind = 2
    analysis_opt.plot_subtitle   = [False, True]
    analysis_opt.legend_label   = ['LM', 'NN']
    analysis_opt.window_size = [0,0]
    analysis_opt.data    = [fm.LM.data,fm.NN.data]
    analysis_opt.hyper    = [fm.LM.hyper,fm.NN.hyper]
    
    for i in range(3):
        analysis_opt.feature   = fm.feature[i]

        analysis_opt.model_exposed_hyper_arg  = [fm.LM.exposed.arg[i], fm.NN.exposed.arg[i]]
        analysis_opt.model_naive_hyper_arg    = [fm.LM.naive.arg[i], fm.NN.naive.arg[i]]
        
        analysis_opt.model_exposed_arch  = [fm.LM.exposed.arch[i],fm.NN.exposed.arch[i]]
        analysis_opt.model_naive_arch    = [fm.LM.naive.arch[i],fm.NN.naive.arch[i]]

        combined_plot(analysis_opt)
    return None

def plot_noise_results(fm):
    analysis_opt = analysis_options()        
    analysis_opt.save_name = 'final'
    analysis_opt.trial_ind = 2
    analysis_opt.plot_subtitle   = [True]
    analysis_opt.legend_label   = ['NN']
    analysis_opt.window_size = [0]
    analysis_opt.data    = [fm.NN.data]
    analysis_opt.hyper    = [fm.NN.hyper]
    
    for i in range(3):
        analysis_opt.feature   = fm.feature[i]

        analysis_opt.model_exposed_hyper_arg  = [ fm.NN.exposed.arg[i]]
        analysis_opt.model_naive_hyper_arg    = [ fm.NN.naive.arg[i]]
        
        analysis_opt.model_exposed_arch  = [fm.NN.exposed.arch[i]]
        analysis_opt.model_naive_arch    = [fm.NN.naive.arch[i]]

        combined_plot_noise(analysis_opt)
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
# explore(fm.LM, hyper_index)
# train_final_models(fm.rf)
# fm = compute_stat([fm.LM])
# print_tables(fm.LM)

#lc = learning_curve(fm.LM)
#lc = learning_curve(fm.NN)

#num_workers=3
#results = Parallel(n_jobs=num_workers)(delayed(learning_curve)(item) for item in [fm.LM, fm.NN, fm.VRNN, fm.LSTM, fm.GRU])
#results = Parallel(n_jobs=num_workers)(delayed(train_final_models)(item) for item in [fm.VRNN, fm.LSTM, fm.GRU])
#learning_curve(fm.CNNLSTM)
#train_final_models(fm.CNNLSTM)
# fm = compute_stat([fm.NN])
# plot_noise_results(fm)
# print_tables(fm.NN)

# b = initiate_data('Braced_')
# b = stat_new_data(fm.NN, b)
# print_tables(b)

###################
###################
###################
###################
u = fm.CNN.data.subject_exposed('JA',1)
train_in, train_out, val_in, val_out = u.train_in, u.train_out, u.test_in, u.test_out

import keras
input_shape = u.train_in.shape[1:]

model = initiate_transformer(
    input_shape,
    10,
    head_size=64,
    num_heads=8,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[32,32],
    mlp_dropout=0.2,
    dropout=0.1,
)

model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["mse"],
)
model.summary()

#callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(
    u.train_in,
    u.train_out,
    validation_data = (u.test_in, u.test_out),
    epochs=200,
    batch_size=128,
#    callbacks=callbacks,
)

model.evaluate(u.test_in, u.test_out, verbose=1)


