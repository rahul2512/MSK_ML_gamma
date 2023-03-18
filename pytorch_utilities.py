import torch as tr 
from torch import nn, sigmoid, tanh,relu
import keras
import numpy as np
from torch.nn import Linear 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorflow import keras
import sys
from keras.regularizers import l2
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional, Dropout
# from ann_visualizer.visualize import ann_viz;


### Initite NN model
def initiate_NN_model(inp_dim,out_dim,nbr_Hlayer,Neu_layer,activation,p_drop,lr,optim,loss,metric,kinit,final_act,regularizer_val):
    model = keras.Sequential()
    model.add(keras.layers.Dense(Neu_layer, input_shape=(inp_dim,), activation=activation))
    for i in range(nbr_Hlayer):
        model.add(Dense(Neu_layer, activation=activation,kernel_initializer=kinit))
        model.add(keras.layers.Flatten())
        model.add(Dropout(p_drop))
    model.add(keras.layers.Dense(out_dim, activation=final_act))
    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)
    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print("Initialised NN network")
    return model

def initiate_Linear_model(inp_dim,out_dim,nbr_Hlayer,Neu_layer,activation,p_drop,lr,optim,loss,metric,kinit,final_act,regularizer_val):
    #### rest of the parameters are redundnt but kept for generalisibilty of code
    # model = keras.Sequential()
    # model.add(keras.layers.Dense(out_dim, input_shape=(inp_dim,), activation='linear'))
    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)
    inputs = keras.layers.Input(shape=(inp_dim,))
    outputs = keras.layers.Dense(out_dim)(inputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print("Initialised Linear network")
    return model



## ENcoder-decoder architechutre of RNN
## consists of two recurrent neural networks (RNN) that act as an encoder and a decoder pair. 
## The encoder maps a variable-length source sequence to a fixed-length vector, and
## the decoder maps the vector representation back to a variable-length target sequence.
## https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/
## https://www.kaggle.com/code/kankanashukla/types-of-lstm
## https://machinelearningmastery.com/stacked-long-short-term-memory-networks/



def initiate_RNN_model(inp_dim, out_dim, t_dim, nbr_Hlayer, batch_size, units, loss, optim, act, p_drop, lr, kinit, final_act, metric, variant):
    dropout_layer = Dropout(rate=p_drop)
    model = keras.Sequential()
    #units: Positive integer, dimensionality of the output space.
    #inputs: A 3D tensor with shape [batch, timesteps, feature]
    if nbr_Hlayer == 0 :

        if variant == 'SimpleRNN':
            model.add(SimpleRNN(units=units,  input_shape = (None,inp_dim),activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False))
            print("Initialised SimpleRNN network")
    
        elif variant == 'BSimpleRNN':
            model.add(Bidirectional(SimpleRNN(units=units,  input_shape = (None,inp_dim),activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False)))
            print("Initialised Bidirectional SimpleRNN network")
    
        elif variant == 'LSTM':
            model.add(LSTM(units=units,  input_shape=(None, inp_dim), activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False))
            print("Initialised LSTM network")
    
        elif variant == 'BLSTM':
            model.add(Bidirectional(LSTM(units=units,  input_shape=(None, inp_dim), activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False)))
            print("Initialised Bidirectional LSTM network")
    
        elif variant == 'GRU':
            model.add(GRU(units=units, input_shape=(None, inp_dim), activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False)) 
            print("Initialised GRU network")
    
        elif variant == 'BGRU':
            model.add(Bidirectional(GRU(units=units, input_shape=(None, inp_dim), activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False))) 
            print("Initialised Bidirectional GRU network")

        else:
            print("incorrect choice of model")
            sys.exit()

    elif nbr_Hlayer > 0 :

        if variant == 'SimpleRNN':
            for i in range(nbr_Hlayer):
                model.add(SimpleRNN(units=units,  input_shape = (None,inp_dim),activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=True))
            model.add(SimpleRNN(units=units//2, activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False))
            print("Initialised SimpleRNN network")
    
        elif variant == 'BSimpleRNN':
            for i in range(nbr_Hlayer):
                model.add(Bidirectional(SimpleRNN(units=units,  input_shape = (None,inp_dim),activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=True)))
            model.add(Bidirectional(SimpleRNN(units=units//2, activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False)))
            print("Initialised Bidirectional SimpleRNN network")
    
        elif variant == 'LSTM':
            for i in range(nbr_Hlayer):
                model.add(LSTM(units=units,  input_shape = (None,inp_dim),activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=True))
            model.add(LSTM(units=units//2, activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False))
            print("Initialised LSTM network")
    
        elif variant == 'BLSTM':
            for i in range(nbr_Hlayer):
                model.add(Bidirectional(LSTM(units=units,  input_shape = (None,inp_dim),activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=True)))
            model.add(Bidirectional(LSTM(units=units//2, activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False)))
            print("Initialised Bidirectional LSTM network")
    
        elif variant == 'GRU':
            for i in range(nbr_Hlayer):
                model.add(GRU(units=units,  input_shape = (None,inp_dim),activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=True))
            model.add(    GRU(units=units//2, activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False))
            print("Initialised GRU network")
    
        elif variant == 'BGRU':
            for i in range(nbr_Hlayer):
                model.add(Bidirectional(GRU(units=units,  input_shape = (None,inp_dim),activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=True)))
            model.add(Bidirectional(GRU(units=units//2, activation=act, dropout = p_drop, kernel_initializer=kinit, return_sequences=False)))
            print("Initialised Bidirectional GRU network")

        else:
            print("incorrect choice of model")
            sys.exit()

    model.add(Dense(out_dim))
    
    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)
    model.compile(loss=loss, optimizer=opt, metrics=metric)
    return model


def initiate_LR_model(inp_dim,out_dim,nbr_Hlayer,Neu_layer,activation,p_drop,lr,optim,loss,metric,kinit,final_act,regularizer_val):
    #### rest of the parameters are redundnt but kept for generalisibilty of code
    model = keras.Sequential()
    model.add(keras.layers.Dense(out_dim, input_shape=(inp_dim,), activation='sigmoid'))
    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)
    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print("Initialised logistic regression network")
    return model



#### Code below generates a .txt file with row as the list of hypermeters 
#### for a given NN and then that NN hypermeters were cross-validated on cluster
def hyper_param_NN():
# batch-size {Automatic, 64, 1000, 2000}
# epoch = 40
# Adam, SGD, RMSprop
# mse
# relu, tanh, sigmoid
# random_uniform, random_normal, he_normal, xavier, glorot_uniform, glorot_normal (Xavier), 
    with open('hyperparam.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'p','regularizer_val','NN_variant', file=f)
        for optim in ['Adam', 'RMSprop', 'SGD']:
            for kinit in ['glorot_normal']:
                for batch_size in [64,128]:
                    for epoch in [50,100,200]:
                        for act in ['relu','tanh','sigmoid']:
                            for H_layer in [1,2,4,6,8]:
                                for metric in ['mse']:
                                    for loss in ['mse']:
                                        for lr in [0.001,0.005]:
                                            for p in [0,0.2]:
                                                for num_nodes in np.arange(200,1900,200):
                                                    for reg in [0]:
                                                        for NN_variant in ['NN']:
                                                            print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p,reg, NN_variant, file=f)
    return None


def hyper_param_LM():
# batch-size {Automatic, 64, 1000, 2000}
# epoch = 40
# Adam, SGD, RMSprop
# mse
# relu, tanh, sigmoid
# random_uniform, random_normal, he_normal, xavier, glorot_uniform, glorot_normal (Xavier), 
    with open('hyperparam_linear.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'p','regularizer_val', 'NN_variant', file=f)
        for optim in ['Adam', 'RMSprop', 'SGD']:
            for kinit in ['glorot_normal','random_normal', 'he_normal']:
                for batch_size in [64,256]:
                    for epoch in [50,100,200]:
                        for act in ['linear']:
                            for H_layer in [0]:
                                for metric in ['mse']:
                                    for loss in ['mse']:
                                        for lr in [0.001]:
                                            for p in [0]:
                                                for num_nodes in [200]:
                                                    for reg in [0]:
                                                        for NN_variant in ['LM']:
                                                            print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p,reg, NN_variant, file=f)
    return None



def hyper_param_RNN():
# batch-size {Automatic, 64, 1000, 2000}
# epoch = 40
# Adam, SGD, RMSprop
# mse
# relu, tanh, sigmoid
# random_uniform, random_normal, he_normal, xavier, glorot_uniform, glorot_normal (Xavier), 
    with open('hyperparam_RNN.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'p','regularizer_val','NN_variant', file=f)
        for optim in ['Adam', 'RMSprop']:
            for kinit in ['glorot_normal']:
                for batch_size in [64, 128]:
                    for epoch in [50,100, 200]:
                        for act in ['relu','tanh','sigmoid']:
                            for H_layer in [0, 1, 2, 3]:
                                for metric in ['mse']:
                                    for loss in ['mse']:
                                        for lr in [0.001]:
                                            for p in [0.1,0.2]:
                                                for num_nodes in [128, 256, 512]:
                                                    for reg in [0]:
                                                        for NN_variant in ['SimpleRNN','LSTM','GRU','BSimpleRNN','BLSTM','BGRU']:
                                                            print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p,reg, NN_variant, file=f)
    return None
