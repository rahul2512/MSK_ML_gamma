import keras, sys, numpy as np, xgboost as xgb
from keras.regularizers import l2
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional, Dropout, Flatten, ConvLSTM1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
# from ann_visualizer.visualize import ann_viz;
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


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




######################################
## Randomforest
######################################
def rf(X_Train, Y_Train, X_val, Y_val, n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, bootstrap, criterion):
    model = RandomForestRegressor(n_estimators=n_estimators, verbose=1, max_features=max_features, 
                                  max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                  bootstrap=bootstrap, criterion=criterion)
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, y_pred)
    print("Validation MSE --" , mse)
    return model

######################################
#GradientBoostingRegressor
###################################
def GBRT(X_Train, Y_Train, X_val, Y_val, n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, loss):
    model = GradientBoostingRegressor(n_estimators=n_estimators, verbose=1, max_features=max_features, 
                                      max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, loss=loss)
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, y_pred)
    print("Validation MSE --" , mse)
    return model

######################################
## xgboost
######################################
def xgbr(X_Train, Y_Train, X_val, Y_val, n_estimators, learning_rate,max_depth, objective, alpha,  lambda1):
    model = xgb.XGBRegressor(objective=objective, 
                             n_estimators=n_estimators, 
                             learning_rate=learning_rate, 
                             max_depth=max_depth, 
                             reg_alpha=alpha, 
                             reg_lambda=lambda1)
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, y_pred)
    print("Validation MSE --" , mse)
    return model

######################################
#SVR
###################################
from sklearn.svm import SVR
def SVR_model(X_Train, Y_Train, X_val, Y_val):
    model = SVR(kernel='rbf') 
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, y_pred)
    print("Validation MSE --" , mse)
    return model

###################
## Transformer code
###################
from tensorflow.keras import layers
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def initiate_transformer(
    input_shape,
    out_dim,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(out_dim, activation="relu")(x)
    return keras.Model(inputs, outputs)


#### Code below generates a .txt file with row as the list of hypermeters 
#### for a given NN and then that NN hypermeters were cross-validated on cluster
def hyper_param_rf():
    with open('./hyperparameters/hyperparam_rf.txt', 'w') as f:
        print('n_estimators', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'bootstrap','criterion', 'norm_out', file=f)
        for n_estimators in [200,400,600,800,1000]:
            for max_features in ['auto', 'sqrt']:
                for max_depth in [10, 30, 50, 70, 90, 110]:
                    for min_samples_split in [2,5,10]:
                        for min_samples_leaf in [1,2,4]:
                            for bootstrap in [True, False]:
                                for criterion in ['squared_error', 'absolute_error']:
                                    for norm_out in [0]:
                                        print(n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, bootstrap, criterion, norm_out, file=f)
    return None


def hyper_param_xgbr():
    with open('./hyperparameters/hyperparam_xgbr.txt', 'w') as f:
        print('n_estimators', 'learning_rate', 'max_depth', 'objective', 'alpha', 'lambda1', 'norm_out', file=f)
        for n_estimators in [200,400,600,800,1000]:
            for learning_rate in [0.01, 0.005]:
                for max_depth in [10, 30, 50, 70, 90, 110]:
                    for objective in ['reg:squarederror', 'reg:logistic']:
                        for alpha in [0.1, 0.2]:
                            for lambda1 in [0.1, 0.2]:
                                for norm_out in [0]:
                                    print(n_estimators, learning_rate, max_depth, objective, alpha, lambda1, norm_out, file=f)
    return None


def hyper_param_GBRT():
    with open('./hyperparameters/hyperparam_GBRT.txt', 'w') as f:
        print('n_estimators', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf','loss', 'norm_out', file=f)
        for n_estimators in [200,400,600,800,1000]:
            for max_features in ['auto', 'sqrt']:
                for max_depth in [10, 30, 50, 70, 90, 110]:
                    for min_samples_split in [2,5,10]:
                        for min_samples_leaf in [1,2,4]:
                            for loss in ['squared_error', 'absolute_error']:
                                for norm_out in [0]:
                                    print(n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, loss, norm_out, file=f)
    return None

#### Code below generates a .txt file with row as the list of hypermeters 
#### for a given NN and then that NN hypermeters were cross-validated on cluster
def hyper_param_NN():
# batch-size {Automatic, 64, 1000, 2000}
# epoch = 40
# Adam, SGD, RMSprop
# mse
# relu, tanh, sigmoid
# random_uniform, random_normal, he_normal, xavier, glorot_uniform, glorot_normal (Xavier), 
    with open('./hyperparameters/hyperparam_NN.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'p','regularizer_val','NN_variant','norm_out', file=f)
        for optim in ['Adam', 'SGD']:
            for kinit in ['glorot_normal']:
                for batch_size in [64,128]:
                    for epoch in [50,100,200]:
                        for act in ['relu','tanh','sigmoid']:
                            for H_layer in [2,4,6,8]:
                                for metric in ['rmse']:
                                    for loss in ['mse', 'rmse']:
                                        for lr in [0.001,0.005]:
                                            for p in [0.1, 0.2]:
                                                for num_nodes in np.arange(200,1100,200):
                                                    for reg in [0]:
                                                        for NN_variant in ['NN']:
                                                            for norm_out in [0, 1]:
                                                                print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p,reg, NN_variant, norm_out, file=f)
    return None



def hyper_param_CNN():
    with open('./hyperparameters/hyperparam_CNN.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'pool_size','regularizer_val','NN_variant','filt_size', 'stride', 'norm_out', file=f)
        for optim in ['Adam', 'SGD', 'RMSprop']:
            for kinit in ['glorot_normal']:
                for batch_size in [64,128]:
                    for epoch in [50,100,200]:
                        for act in ['relu','tanh','sigmoid']:
                            for H_layer in [1,2]:
                                for metric in ['rmse']:
                                    for loss in ['mse', 'rmse']:
                                        for lr in [0.001,0.005]:
                                            for pool_size in [2]:
                                                for num_nodes in [32,64]:
                                                    for reg in [0]:
                                                        for NN_variant in ['CNN']:
                                                            for filt_size in [3]:
                                                                for stride in [1,3]:
                                                                    for norm_out in [0, 1]:
                                                                        print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, pool_size,reg, NN_variant, filt_size, stride, norm_out, file=f)
    return None

def hyper_param_CNNLSTM():
    with open('./hyperparameters/hyperparam_CNNLSTM.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'pool_size','regularizer_val','NN_variant','filt_size', 'stride', 'LSTM_units', 'norm_out',file=f)
        for optim in ['Adam', 'SGD', 'RMSprop']:
            for kinit in ['glorot_normal']:
                for batch_size in [64,128]:
                    for epoch in [50,100,200]:
                        for act in ['relu','tanh','sigmoid']:
                            for H_layer in [1,2]:
                                for metric in ['rmse']:
                                    for loss in ['mse', 'rmse']:
                                        for lr in [0.001,0.005]:
                                            for pool_size in [2]:
                                                for num_nodes in [32,64]:
                                                    for reg in [0]:
                                                        for NN_variant in ['CNNLSTM']:
                                                            for filt_size in [3]:
                                                                for stride in [1,3]:
                                                                    for LSTM_units in [128, 256]:
                                                                        for norm_out in [0, 1]:
                                                                            print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, pool_size,reg, NN_variant, filt_size, stride, LSTM_units, norm_out, file=f)
    return None


def hyper_param_LM():
# batch-size {Automatic, 64, 1000, 2000}
# epoch = 40
# Adam, SGD, RMSprop
# mse
# relu, tanh, sigmoid
# random_uniform, random_normal, he_normal, xavier, glorot_uniform, glorot_normal (Xavier), 
    with open('./hyperparameters/hyperparam_LM.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'p','regularizer_val', 'NN_variant', 'norm_out', file=f)
        for optim in ['Adam', 'SGD']:
            for kinit in ['glorot_normal','random_normal', 'he_normal']:
                for batch_size in [64,256]:
                    for epoch in [50,100,200]:
                        for act in ['linear']:
                            for H_layer in [0]:
                                for metric in ['rmse']:
                                    for loss in ['mse', 'rmse']:
                                        for lr in [0.001]:
                                            for p in [0]:
                                                for num_nodes in [200]:
                                                    for reg in [0]:
                                                        for NN_variant in ['LM']:
                                                            for norm_out in [0, 1]:
                                                                print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p,reg, NN_variant, norm_out, file=f)
    return None



def hyper_param_RNN():
# batch-size {Automatic, 64, 1000, 2000}
# epoch = 40
# Adam, SGD, RMSprop
# mse
# relu, tanh, sigmoid
# random_uniform, random_normal, he_normal, xavier, glorot_uniform, glorot_normal (Xavier), 
    with open('./hyperparameters/hyperparam_RNN.txt', 'w') as f:
        print('NN_variant','optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'p','regularizer_val','norm_out',file=f)
#        for NN_variant in ['SimpleRNN','BSimpleRNN','LSTM','BLSTM','GRU','BGRU']:
        for NN_variant in ['BSimpleRNN','BLSTM','BGRU']:
            for optim in ['Adam', 'RMSprop']:
                for kinit in ['glorot_normal']:
                    for batch_size in [64, 128]:
                        for epoch in [50,100, 200]: 
                            for act in ['relu','tanh','sigmoid']:
                                for H_layer in [0, 1, 2, 3]:
                                    for metric in ['rmse']:
                                        for loss in ['mse', 'rmse']:
                                            for lr in [0.001]:
                                                for p in [0.1,0.2]:
                                                    for num_nodes in [128, 256, 512]:
                                                        for reg in [0]:
                                                            for norm_out in [0, 1]:
                                                                print(NN_variant, optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p,reg, norm_out, file=f)
    return None

hyper_param_xgbr()
# hyper_param_rf()
# hyper_param_GBRT()
# hyper_param_LM()
# hyper_param_NN()
# hyper_param_RNN()
# hyper_param_CNN()
# hyper_param_CNNLSTM()
