import torch as tr, keras, numpy as np, pandas as pd, sys
from torch import nn, sigmoid, tanh,relu
from torch.nn import Linear 
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorflow import keras
from keras.regularizers import l2
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional, Dropout, Flatten, ConvLSTM1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
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

def initiate_CNN_model(inp_dim, out_dim, t_dim, nbr_Hlayer, batch_size, units, loss, optim, act, pool_size, lr, kinit, final_act, metric, filt_size, variant, stride):
    # https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
    # small number of filters in the first layer, such as 32 or 64, and gradually increase the number of filters in deeper layers. 
    # as initial layers capture low-level features, while the deeper layers capture higher-level features that are more complex and abstract.
    model = keras.Sequential()

    #  (batch_size, timesteps, input_dim) -- input of conv1D layers
    model.add(Conv1D(filters = int(units), kernel_size=int(filt_size), strides=int(stride), activation=act, kernel_initializer=kinit, padding = 'same',input_shape=(t_dim, inp_dim)))
    model.add(MaxPooling1D(pool_size=int(pool_size), padding = 'same'))
    # (batch_size, new_timesteps, filters), where new_timesteps is the length of the output sequence after applying the convolutional operation

    for i in range(nbr_Hlayer-1): 
        model.add(Conv1D(filters=2*int(units), kernel_size=int(filt_size), activation=act, padding = 'same'))
        model.add(MaxPooling1D(pool_size=int(pool_size), padding = 'same'))

    model.add(Flatten())

    ## dropout can be used below but we are not using it here
    model.add(Dense(50, activation=act))
    model.add(Dense(out_dim))

    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)

    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print("Initialised CNN .....")
    # print(model.summary())
    return model



def initiate_CNNLSTM_model(inp_dim, out_dim, t_dim, nbr_Hlayer, batch_size, units, loss, optim, act, pool_size, lr, kinit, final_act, metric, filt_size, variant, stride, LSTM_units):
    model = keras.Sequential()


    if nbr_Hlayer == 1:
        model.add(Conv1D(filters = int(units), kernel_size=int(filt_size), strides=int(stride), activation=act, kernel_initializer=kinit, padding = 'same',input_shape=(t_dim, inp_dim)))
        model.add(MaxPooling1D(pool_size=int(pool_size), padding = 'same'))
    
        model.add(LSTM(units=LSTM_units, return_sequences=False))

    elif nbr_Hlayer > 1:
        model.add(Conv1D(filters = int(units), kernel_size=int(filt_size), strides=int(stride), activation=act, kernel_initializer=kinit, padding = 'same',input_shape=(t_dim, inp_dim)))
        model.add(MaxPooling1D(pool_size=int(pool_size), padding = 'same'))
        model.add(LSTM(units=LSTM_units, return_sequences=True))       
        for i in range(nbr_Hlayer-2):
            model.add(Conv1D(filters = int(units), kernel_size=int(filt_size), strides=int(stride), activation=act, kernel_initializer=kinit, padding = 'same'))
            model.add(MaxPooling1D(pool_size=int(pool_size), padding = 'same'))   
            model.add(LSTM(units=LSTM_units, return_sequences=True))
        model.add(Conv1D(filters = int(units), kernel_size=int(filt_size), strides=int(stride), activation=act, kernel_initializer=kinit, padding = 'same'))
        model.add(MaxPooling1D(pool_size=int(pool_size), padding = 'same'))   
        model.add(LSTM(units=LSTM_units, return_sequences=False))
            
    model.add(Dense(out_dim))

    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)

    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print("initiate_CNNLSTM_model .....")
    # print(model.summary())
    return model



def initiate_ConvLSTM_model(inp_dim, out_dim, t_dim, nbr_Hlayer, batch_size, units, loss, optim, act, pool_size, lr, kinit, final_act, metric, filt_size, variant, stride):
    #https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7
    model = keras.Sequential()

    if nbr_Hlayer == 1 :
        model.add(ConvLSTM1D(filters = int(units), kernel_size=int(filt_size), strides=int(stride), activation=act, kernel_initializer=kinit, padding='same', return_sequences=False, input_shape=(t_dim, inp_dim, 1)))
        model.add(MaxPooling1D(pool_size=int(pool_size), padding = 'same'))
    elif nbr_Hlayer > 1 :
        model.add(ConvLSTM1D(filters = int(units), kernel_size=int(filt_size), strides=int(stride), activation=act, kernel_initializer=kinit, padding='same', return_sequences=True, input_shape=(t_dim, inp_dim)))
        for i in range(nbr_Hlayer-2):
            model.add(ConvLSTM1D(filters=int(units), kernel_size=int(filt_size), activation=act, padding = 'same', return_sequences=True))
        model.add(ConvLSTM1D(filters=int(units), kernel_size=int(filt_size), activation=act, padding = 'same', return_sequences=False))
    else:
        sys.exit()

#    model.add(Dense(50, activation=act))
    model.add(Flatten())
    model.add(Dense(out_dim))

    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)

    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print("Initialised CNN .....")
    print(model.summary())
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



def hyper_param_CNN():
    with open('hyperparam_CNN.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'pool_size','regularizer_val','NN_variant','filt_size', 'stride', file=f)
        for optim in ['Adam', 'SGD', 'RMSprop']:
            for kinit in ['glorot_normal']:
                for batch_size in [64,128]:
                    for epoch in [50,100,200]:
                        for act in ['relu','tanh','sigmoid']:
                            for H_layer in [1,2]:
                                for metric in ['mse']:
                                    for loss in ['mse']:
                                        for lr in [0.001,0.005]:
                                            for pool_size in [2]:
                                                for num_nodes in [32,64]:
                                                    for reg in [0]:
                                                        for NN_variant in ['CNN']:
                                                            for filt_size in [3]:
                                                                for stride in [1,3]:
                                                                    print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, pool_size,reg, NN_variant, filt_size, stride, file=f)
    return None

def hyper_param_CNNLSTM():
    with open('hyperparam_CNNLSTM.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'pool_size','regularizer_val','NN_variant','filt_size', 'stride', 'LSTM_units',file=f)
        for optim in ['Adam', 'SGD', 'RMSprop']:
            for kinit in ['glorot_normal']:
                for batch_size in [64,128]:
                    for epoch in [50,100,200]:
                        for act in ['relu','tanh','sigmoid']:
                            for H_layer in [1,2]:
                                for metric in ['mse']:
                                    for loss in ['mse']:
                                        for lr in [0.001,0.005]:
                                            for pool_size in [2]:
                                                for num_nodes in [32,64]:
                                                    for reg in [0]:
                                                        for NN_variant in ['CNNLSTM']:
                                                            for filt_size in [3]:
                                                                for stride in [1,3]:
                                                                    for LSTM_units in [128, 256]:
                                                                        print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, pool_size,reg, NN_variant, filt_size, stride, LSTM_units, file=f)
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
