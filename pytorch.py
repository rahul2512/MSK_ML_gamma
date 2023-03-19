from pytorch_utilities import *
from read_in_out import analysis_options
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, DataLoader
import torch as tr, time
import numpy as np
import statsmodels.api as sm
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from fractions import Fraction
import sys, copy
import scipy
from scipy import signal
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib import gridspec
from scipy.interpolate import interp1d
from barchart_err import barchart_error, barchart_params


RNN_models = ['SimpleRNN','LSTM','GRU','BSimpleRNN','BLSTM','BGRU']
feature_list = ['Joint angles','Joint reaction forces','Joint moments',  'Muscle forces', 'Muscle activations']
feature_slist = ['JA','JRF','JM',  'MF', 'MA']


def plot_MSK_data(fm):
    for feature in fm.feature:
        col_number = fm.NN.data.o1.numer_of_features[feature]
        cols = fm.NN.data.o1.T1.columns[col_number]
        sparse = 10
        color = ['r','b','g','k','m']
        std = fm.NN.data.subject_exposed(feature).std
        if 'JRF' == feature:
            fig, ax = plt.subplots(4,3,figsize=(8,6),sharex=True)
            ax_list  = [ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2], ax[2,0], ax[2,1], ax[2,2], ax[3,0], ax[3,1], ax[3,2]]
            ss,b_xlabel = 8,9
            ylabel = [ 'Trunk \n Mediolateral', 'Trunk \n Proximodistal', 'Trunk \n Anteroposterior', 'Shoulder \n Mediolateral',
                      'Shoulder \n Proximodistal', 'Shoulder \n Anteroposterior', 'Elbow \n Mediolateral', 'Elbow \n Proximodistal',
                      'Elbow \n Anteroposterior', 'Wrist \n Mediolateral', 'Wrist \n Proximodistal', 'Wrist \n Anteroposterior']
            plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    
        elif 'MF'  == feature:
            fig, ax = plt.subplots(2,2,figsize=(4,3),sharex=True)
            ax_list = [ax[0,0],ax[0,1],ax[1,0] ,ax[1,1]]
            ss,b_xlabel = 7,1
            ylabel = ['Pectoralis major \n (Clavicle)','Biceps Brachii','Deltoid (Medial)','Brachioradialis']
            plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    
        elif 'MA' == feature:
            fig, ax = plt.subplots(2,2,figsize=(4,3),sharex=True)
            ax_list = [ax[0,0],ax[0,1],ax[1,0] ,ax[1,1]]
            ss,b_xlabel = 7,1
            ylabel = ['Pectoralis major \n (Clavicle)','Biceps Brachii','Deltoid (Medial)','Brachioradialis']
            plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    
        elif 'JM' == feature:
            fig, ax = plt.subplots(4,3,figsize=(8,6),sharex=True)
            ax_list  = [ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2], ax[2,0], ax[2,1], ax[2,2],  ax[3,1]]
            ax[3,0].remove()
            ax[3,2].remove()
            ss,b_xlabel = 8,7
            ylabel = [ 'Trunk Flexion / \n Extension', 'Trunk Internal / \n External Rotation', 'Trunk Right / \n Left Bending',
                      'Shoulder Flexion / \n Extension', 'Shoulder Abduction / \n Adduction', 'Shoulder Internal / \n External Rotation',
                      'Elbow Flexion / \n Extension', 'Elbow Pronation / \n Supination', 'Wrist Flexion / \n Extension', 'Wrist Radial / \n Ulnar Deviation']
            plot_list = ['(a)','(c)','(b)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    
        elif 'JA' == feature:
            new_order = [7,8,9,0,1,2,3,4,5,6]
            cols = cols[new_order]
            fig, ax = plt.subplots(4,3,figsize=(8,6),sharex=True)
            ax_list  = [ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2], ax[2,0], ax[2,1], ax[2,2], ax[3,1]]
            ax[3,0].remove()
            ax[3,2].remove()
    
            ss,b_xlabel = 8,7
            plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']    
            ylabel = ['Trunk Forward / \n Backward Bending', 'Trunk Right / \n Left Bending', 'Trunk Internal / \n External Rotation',
                      'Shoulder Flexion / \n Extension', 'Shoulder Abduction / \n Adduction', 'Shoulder Internal / \n External Rotation',
                      'Elbow Flexion / \n Extension', 'Elbow Pronation / \n Supination', 'Wrist Flexion / \n Extension', 'Wrist Radial / \n Ulnar Deviation']
        sub = ['Subject-1','Subject-2','Subject-3','Subject-4','Subject-5' ]
        no_lab = ['_no_legend_']*5
        lab = [sub,no_lab, no_lab]
        for enum, axu in enumerate(ax_list):
            for enum1, dsub in enumerate([fm.NN.data.o1,fm.NN.data.o2,fm.NN.data.o3,fm.NN.data.o4,fm.NN.data.o5]):
                for enum2, dtri in enumerate([dsub.T1, dsub.T2, dsub.T3]):
                    axu.plot(dtri[cols[enum]][::sparse].index,dtri[cols[enum]][::sparse], color=color[enum1],lw=0.2,label = lab[enum2][enum1])
            axu.set_ylabel(ylabel[enum],fontsize=ss-1)
            axu.tick_params(axis='x', labelsize=ss-1,   pad=2,length=3,width=0.5,direction= 'inout',which='major')
            axu.tick_params(axis='x', labelsize=ss-1, pad=2, length=3,width=0.5,direction= 'inout',which='minor')
            axu.tick_params(axis='y', labelsize=ss-1,   pad=2, length=3,width=0.5,direction= 'inout')

        if feature in  ['JRF','JA','JM']:
            ax[0,1].legend(fontsize=ss-1,loc = 'upper center',fancybox=True,ncol=5, frameon=True,framealpha=1, borderaxespad=-2.3)   
            plt.tight_layout(h_pad = 0.1,w_pad = -8)
        elif feature in  ['MA','MF']:
            ax[0,1].legend(fontsize=ss-1,loc = 'upper right',fancybox=True,ncol=5, frameon=True,framealpha=1, borderaxespad=-2.3)   
            plt.tight_layout(h_pad = 0.1,w_pad = -12)
            
        fig.savefig('./plots_out/MSK_data_'+feature+'.pdf',dpi=600)
        plt.show()
        plt.close()
    return None


def combined_plot(analysis_opt):
    # it plots the first trial and provides the statistics for each trial as well as average, std, iqr, min,max etc etc
    lll = len(analysis_opt.model_exposed_hyper_arg)
    save_name = analysis_opt.save_name

    color_list = ['r','b']
    ls_list = ['-','-']
    lsk = '--'
    for XX in range(lll):
                    
        feature = analysis_opt.feature 
        
        hyper_val_exp =  analysis_opt.model_exposed_hyper_arg[XX]
        hyper_val_naive = analysis_opt.model_naive_hyper_arg[XX]
        model_class_exp = analysis_opt.model_exposed_arch[XX]
        model_class_naive = analysis_opt.model_naive_arch[XX]

        model1 = load_model('exposed', feature, model_class_exp,   hyper_val_exp  )
        model2 = load_model('naive'  , feature, model_class_naive, hyper_val_naive)

        data = analysis_opt.data[XX]
        window = analysis_opt.window_size[XX]

        XE, YE = data.subject_exposed(feature).test_in, data.subject_exposed(feature).test_out
        XN, YN = data.subject_naive(feature).test_in,   data.subject_naive(feature).test_out
   
        SCE = data.subject_exposed(feature).std
        SCN = data.subject_naive(feature).std
    
        TE = data.subject_exposed(feature).time
        TN = data.subject_naive(feature).time

        plot_subtitle = analysis_opt.plot_subtitle[XX]

        NRMSE_list,  PC_list  = [],[]
        NRMSE2_list, PC2_list = [],[]
        RMSE_list, RMSE2_list = [], []
        YP1, YP2 = model1.predict(XE), model2.predict(XN)
        YT1, YT2 = np.array(YE),np.array(YN)
        a,b = np.shape(YT1)
        a2,b2 = np.shape(YT2)
        try:
            SCE,SCN = SCE.to_numpy(),SCN.to_numpy()
        except:
            SCE,SCN = SCE,SCN

        if 'MA' in feature:
            SCE,SCN = SCE*100,SCN*100

        YP1, YT1 = YP1*SCE, YT1*SCE
        YP2, YT2 = YP2*SCN, YT2*SCN
        #### the below loop is to set the time in terms of percentage of task

        zero_entries = np.where(TE==0)
        zero_entries = np.concatenate([zero_entries[0],np.array([a])])   #### adding last element
    
        zero_entries2 = np.where(TN==0)
        zero_entries2 = np.concatenate([zero_entries2[0],np.array([a2])])   #### adding last element
    
        count,aa = -1,[]
        for u in TE.to_numpy():
            if u == 0:
                count = count + 1
            aa.append(u+count)
    
        count,bb = -1,[]
        for v in TN.to_numpy():
            if v==0:
                count = count + 1
            bb.append(v+count)

        count=0

        if 'JRF' in feature:
            if XX == 0 :
                fig = plt.figure(figsize=(8,10.5))
                gs1 = gridspec.GridSpec(700, 560)
                gs1.update(left=0.065, right=0.98,top=0.945, bottom=0.06)
                d1, d2 =13, 10
                ax00 = plt.subplot(gs1[  0+d2:100  ,   0+d1:100 ])
                ax01 = plt.subplot(gs1[  0+d2:100  , 150+d1:250 ])
                ax10 = plt.subplot(gs1[120+d2:220  ,   0+d1:100 ])
                ax11 = plt.subplot(gs1[120+d2:220  , 150+d1:250 ])
                ax20 = plt.subplot(gs1[240+d2:340  ,   0+d1:100 ])
                ax21 = plt.subplot(gs1[240+d2:340  , 150+d1:250 ])
                ax30 = plt.subplot(gs1[360+d2:460  ,   0+d1:100 ])
                ax31 = plt.subplot(gs1[360+d2:460  , 150+d1:250 ])
                ax40 = plt.subplot(gs1[480+d2:580  ,   0+d1:100 ])
                ax41 = plt.subplot(gs1[480+d2:580  , 150+d1:250 ])
                ax50 = plt.subplot(gs1[600+d2:700  ,   0+d1:100 ])
                ax51 = plt.subplot(gs1[600+d2:700  , 150+d1:250 ])
        
                ax02 = plt.subplot(gs1[  0+d2:100  , 310+d1:410 ])
                ax03 = plt.subplot(gs1[  0+d2:100  , 460+d1:560 ])
                ax12 = plt.subplot(gs1[120+d2:220  , 310+d1:410 ])
                ax13 = plt.subplot(gs1[120+d2:220  , 460+d1:560 ])
                ax22 = plt.subplot(gs1[240+d2:340  , 310+d1:410 ])
                ax23 = plt.subplot(gs1[240+d2:340  , 460+d1:560 ])
                ax32 = plt.subplot(gs1[360+d2:460  , 310+d1:410 ])
                ax33 = plt.subplot(gs1[360+d2:460  , 460+d1:560 ])
                ax42 = plt.subplot(gs1[480+d2:580  , 310+d1:410 ])
                ax43 = plt.subplot(gs1[480+d2:580  , 460+d1:560 ])
                ax52 = plt.subplot(gs1[600+d2:700  , 310+d1:410 ])
                ax53 = plt.subplot(gs1[600+d2:700  , 460+d1:560 ])
        
                ax_list  = [ax00, ax01, ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41, ax50, ax51]
                ax_list2 = [ax02, ax03, ax12, ax13, ax22, ax23, ax32, ax33, ax42, ax43, ax52, ax53]
    
            ss,b_xlabel = 8,9
            ylabel = [ 'Trunk \n Mediolateral', 'Trunk \n Proximodistal', 'Trunk \n Anteroposterior', 'Shoulder \n Mediolateral',
                      'Shoulder \n Proximodistal', 'Shoulder \n Anteroposterior', 'Elbow \n Mediolateral', 'Elbow \n Proximodistal',
                      'Elbow \n Anteroposterior', 'Wrist \n Mediolateral', 'Wrist \n Proximodistal', 'Wrist \n Anteroposterior']
            plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    
        elif 'MF' in feature:
            if XX == 0:
                fig = plt.figure(figsize=(8,4))
                gs1 = gridspec.GridSpec(215, 560)
                gs1.update(left=0.07, right=0.98,top=0.84, bottom=0.15)
                d1, d2 =10, 10
                ax00 = plt.subplot(gs1[0:100 -d2    , 0+d1:100  ])
                ax01 = plt.subplot(gs1[0:100 -d2   , 150+d1:250 ])
                ax10 = plt.subplot(gs1[115+d2:215  , 0+d1:100 ])
                ax11 = plt.subplot(gs1[115+d2:215  , 150+d1:250 ])
        
                ax02 = plt.subplot(gs1[0:100 -d2   , 310+d1:410 ])
                ax03 = plt.subplot(gs1[0:100 -d2   , 460+d1:560 ])
                ax12 = plt.subplot(gs1[115+d2:215  , 310+d1:410 ])
                ax13 = plt.subplot(gs1[115+d2:215  , 460+d1:560 ])
        
                ax_list = [ax00,ax01,ax10 ,ax11]
                ax_list2= [ ax02,ax03, ax12 ,ax13 ]

            ss,b_xlabel = 8,1
            ylabel = ['Pectoralis major \n (Clavicle)','Biceps Brachii','Deltoid (Medial)','Brachioradialis']
            plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    
        elif 'MA' in feature:
            if XX == 0:
                fig = plt.figure(figsize=(8,4))
                gs1 = gridspec.GridSpec(215, 560)
                gs1.update(left=0.07, right=0.98,top=0.84, bottom=0.15)
                d1, d2 =10, 10
                ax00 = plt.subplot(gs1[0:100 -d2    , 0+d1:100  ])
                ax01 = plt.subplot(gs1[0:100 -d2   , 150+d1:250 ])
                ax10 = plt.subplot(gs1[115+d2:215  , 0+d1:100 ])
                ax11 = plt.subplot(gs1[115+d2:215  , 150+d1:250 ])
        
                ax02 = plt.subplot(gs1[0:100 -d2   , 310+d1:410 ])
                ax03 = plt.subplot(gs1[0:100 -d2   , 460+d1:560 ])
                ax12 = plt.subplot(gs1[115+d2:215  , 310+d1:410 ])
                ax13 = plt.subplot(gs1[115+d2:215  , 460+d1:560 ])
        
                ax_list = [ax00,ax01,ax10 ,ax11]
                ax_list2= [ ax02,ax03, ax12 ,ax13 ]
            ss,b_xlabel = 8,1
            ylabel = ['Pectoralis major \n (Clavicle)','Biceps Brachii','Deltoid (Medial)','Brachioradialis']
            plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    
    
    
        elif 'JM' in feature:
            if XX == 0:
                fig = plt.figure(figsize=(8,8.25))
                gs1 = gridspec.GridSpec(580, 560)
                gs1.update(left=0.075, right=0.98,top=0.945, bottom=0.08)
                d1, d2 =10, 10
                ax00 = plt.subplot(gs1[  0+d2:100  ,   0+d1:100 ])
                ax01 = plt.subplot(gs1[  0+d2:100  , 150+d1:250 ])
                ax10 = plt.subplot(gs1[120+d2:220  ,   0+d1:100 ])
                ax11 = plt.subplot(gs1[120+d2:220  , 150+d1:250 ])
                ax20 = plt.subplot(gs1[240+d2:340  ,   0+d1:100 ])
                ax21 = plt.subplot(gs1[240+d2:340  , 150+d1:250 ])
                ax30 = plt.subplot(gs1[360+d2:460  ,   0+d1:100 ])
                ax31 = plt.subplot(gs1[360+d2:460  , 150+d1:250 ])
                ax40 = plt.subplot(gs1[480+d2:580  ,   0+d1:100 ])
                ax41 = plt.subplot(gs1[480+d2:580  , 150+d1:250 ])
        
                ax02 = plt.subplot(gs1[  0+d2:100  , 310+d1:410 ])
                ax03 = plt.subplot(gs1[  0+d2:100  , 460+d1:560 ])
                ax12 = plt.subplot(gs1[120+d2:220  , 310+d1:410 ])
                ax13 = plt.subplot(gs1[120+d2:220  , 460+d1:560 ])
                ax22 = plt.subplot(gs1[240+d2:340  , 310+d1:410 ])
                ax23 = plt.subplot(gs1[240+d2:340  , 460+d1:560 ])
                ax32 = plt.subplot(gs1[360+d2:460  , 310+d1:410 ])
                ax33 = plt.subplot(gs1[360+d2:460  , 460+d1:560 ])
                ax42 = plt.subplot(gs1[480+d2:580  , 310+d1:410 ])
                ax43 = plt.subplot(gs1[480+d2:580  , 460+d1:560 ])
        
                ax_list  = [ax00, ax10, ax01, ax11, ax20, ax21, ax30, ax31, ax40, ax41]
                ax_list2 = [ax02, ax12, ax03, ax13, ax22, ax23, ax32, ax33, ax42, ax43]
        
            ss,b_xlabel = 8,7
    
    
            ylabel = [ 'Trunk Flexion / \n Extension', 'Trunk Internal / \n External Rotation', 'Trunk Right / \n Left Bending',
                      'Shoulder Flexion / \n Extension', 'Shoulder Abduction / \n Adduction', 'Shoulder Internal / \n External Rotation',
                      'Elbow Flexion / \n Extension', 'Elbow Pronation / \n Supination', 'Wrist Flexion / \n Extension', 'Wrist Radial / \n Ulnar Deviation']
            plot_list = ['(a)','(c)','(b)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    
        elif 'JA' in feature:
            new_order = [7,8,9,0,1,2,3,4,5,6]
            YP1 = YP1[:,new_order]
            YP2 = YP2[:,new_order]
            YT1 = YT1[:,new_order]
            YT2 = YT2[:,new_order]
            
            if XX == 0:
                fig = plt.figure(figsize=(8,8.25))
                gs1 = gridspec.GridSpec(580, 560)
                gs1.update(left=0.065, right=0.98,top=0.945, bottom=0.07)
                d1, d2 =13, 10
                ax00 = plt.subplot(gs1[  0+d2:100  ,   0+d1:100 ])
                ax01 = plt.subplot(gs1[  0+d2:100  , 150+d1:250 ])
                ax10 = plt.subplot(gs1[120+d2:220  ,   0+d1:100 ])
                ax11 = plt.subplot(gs1[120+d2:220  , 150+d1:250 ])
                ax20 = plt.subplot(gs1[240+d2:340  ,   0+d1:100 ])
                ax21 = plt.subplot(gs1[240+d2:340  , 150+d1:250 ])
                ax30 = plt.subplot(gs1[360+d2:460  ,   0+d1:100 ])
                ax31 = plt.subplot(gs1[360+d2:460  , 150+d1:250 ])
                ax40 = plt.subplot(gs1[480+d2:580  ,   0+d1:100 ])
                ax41 = plt.subplot(gs1[480+d2:580  , 150+d1:250 ])
        
                ax02 = plt.subplot(gs1[  0+d2:100  , 310+d1:410 ])
                ax03 = plt.subplot(gs1[  0+d2:100  , 460+d1:560 ])
                ax12 = plt.subplot(gs1[120+d2:220  , 310+d1:410 ])
                ax13 = plt.subplot(gs1[120+d2:220  , 460+d1:560 ])
                ax22 = plt.subplot(gs1[240+d2:340  , 310+d1:410 ])
                ax23 = plt.subplot(gs1[240+d2:340  , 460+d1:560 ])
                ax32 = plt.subplot(gs1[360+d2:460  , 310+d1:410 ])
                ax33 = plt.subplot(gs1[360+d2:460  , 460+d1:560 ])
                ax42 = plt.subplot(gs1[480+d2:580  , 310+d1:410 ])
                ax43 = plt.subplot(gs1[480+d2:580  , 460+d1:560 ])
        
                ax_list  = [ax00, ax01, ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41]
                ax_list2 = [ax02, ax03, ax12, ax13, ax22, ax23, ax32, ax33, ax42, ax43]
    
            ss,b_xlabel = 8,7   
            plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']    
            ylabel = ['Trunk Forward / \n Backward Bending', 'Trunk Right / \n Left Bending', 'Trunk Internal / \n External Rotation',
                      'Shoulder Flexion / \n Extension', 'Shoulder Abduction / \n Adduction', 'Shoulder Internal / \n External Rotation',
                      'Elbow Flexion / \n Extension', 'Elbow Pronation / \n Supination', 'Wrist Flexion / \n Extension', 'Wrist Radial / \n Ulnar Deviation']
    
         ######following loop computes stats
        for i in range(b):
            count = 1   ## computing for all trials
            for c in range(count+1):    ########## this loop is required to separate trials
                if c < 2:
                    
                    ttmmpp = np.arange(zero_entries[c],zero_entries[c+1])
            
                    PC =  scipy.stats.pearsonr(YP1[ttmmpp,i],YT1[ttmmpp,i])[0]
                    NRMSE =  mean_squared_error(YP1[ttmmpp,i], YT1[ttmmpp,i],squared=False)/SCE[i]
                    NRMSE_list.append(NRMSE)
                    RMSE =  mean_squared_error(YP1[ttmmpp,i], YT1[ttmmpp,i],squared=False)
                    RMSE_list.append(RMSE)
    
                    PC_list.append(PC)
    
                ttmmpp2 = np.arange(zero_entries2[c],zero_entries2[c+1])
                PC2  = scipy.stats.pearsonr(YP2[ttmmpp2,i],YT2[ttmmpp2,i])[0]
                NRMSE2  =  mean_squared_error(YP2[ttmmpp2,i], YT2[ttmmpp2,i],squared=False)/SCN[i]
                NRMSE2_list.append(NRMSE2)
                RMSE2  =  mean_squared_error(YP2[ttmmpp2,i], YT2[ttmmpp2,i],squared=False)
                RMSE2_list.append(RMSE2)
                PC2_list.append(PC2)
        NRMSE_list = np.around(NRMSE_list,2)
        NRMSE2_list = np.around(NRMSE2_list,2)
        RMSE_list = np.around(RMSE_list,2)
        RMSE2_list = np.around(RMSE2_list,2)
        PC_list  = np.around(PC_list,2)
        PC2_list = np.around(PC2_list,2)
        print("Printing statisics for ----", feature, "mean, std, max, min, iqr")
        print(np.around(np.mean(PC_list),2),' (' ,  np.around(np.std(PC_list),2),') &' , np.around(np.max(PC_list),2),' & ' , np.around(np.min(PC_list),2),' & ' ,  np.around(scipy.stats.iqr(PC_list),2),' & ' , 'PC Exposed')
        print(np.around(np.mean(NRMSE_list),2),' (' , np.around(np.std(NRMSE_list),2),') & ' , np.around(np.max(NRMSE_list),2),' & ' , np.around(np.min(NRMSE_list),2),' & ' ,  np.around(scipy.stats.iqr(NRMSE_list),2), 'NRMSE exposed')
    
        print(np.around(np.mean(PC2_list),2),'(' , np.around(np.std(PC2_list),2),') & ' , np.around(np.max(PC2_list),2),' & ' , np.around(np.min(PC2_list),2),' & ' ,  np.around(scipy.stats.iqr(PC2_list),2),' & ' , 'PC2 Naive')
        print(np.around(np.mean(NRMSE2_list),2),' ( ' , np.around(np.std(NRMSE2_list),2),') & ' , np.around(np.max(NRMSE2_list),2),' & ' , np.around(np.min(NRMSE2_list),2),' & ' ,  np.around(scipy.stats.iqr(NRMSE2_list),2), 'NRMSE2 Naive')
    
        print("below printing RMSE with units")
        print(np.around(np.mean(RMSE_list),2),'&' , np.around(np.std(RMSE_list),2),'& ' , np.around(np.max(RMSE_list),2),' & ' , np.around(np.min(RMSE_list),2),' & ' ,  np.around(scipy.stats.iqr(RMSE_list),2), 'RMSE_list exposed')
        print(np.around(np.mean(RMSE2_list),2),'&' , np.around(np.std(RMSE2_list),2),' & ' , np.around(np.max(RMSE2_list),2),' & ' , np.around(np.min(RMSE2_list),2),' & ' ,  np.around(scipy.stats.iqr(RMSE2_list),2), 'NRMSE2 Naive')
    
        sparse_plot=5
        for i in range(b):
            push_plot = 0
            count = 0   ## plotting first trial
            for c in range(count+1):    ########## this loop is required to separate trials
                ttmmpp = np.arange(zero_entries[c],zero_entries[c+1]) - window  ### to make the plot RNN compatible
                ttmmpp2 = np.arange(zero_entries2[c],zero_entries2[c+1]) - window
                if ax_list[i] == ax00:
                    label1  = analysis_opt.legend_label[XX] #+ ' prediction'
                    if label1 == 'NN':
                        label1 = 'FFNN'
                    if XX == 0:
                        label2 = 'MSK'
                else:
                    label1, label2 = '_no_legend_', '_no_legend_'

                if XX == 0:
                    ax_list[i].plot([aa[q] + push_plot for q in ttmmpp][window::sparse_plot] ,YT1[ttmmpp,i][window::sparse_plot],color='k',lw=0.9,ls = lsk, label=label2)
                ax_list[i].plot([aa[q] + push_plot for q in ttmmpp][window::sparse_plot] ,YP1[ttmmpp,i][window::sparse_plot],color=color_list[XX],ls = ls_list[XX], lw=0.7,label=label1)   ### np.arange(a)

                if XX == 0:
                    ax_list2[i].plot([bb[q] + push_plot for q in ttmmpp2][window::sparse_plot] ,YT2[ttmmpp2,i][window::sparse_plot],color='k',lw=0.9,ls = lsk, label ='_no_legend_')#,label=label2)
                ax_list2[i].plot([bb[q] + push_plot for q in ttmmpp2][window::sparse_plot] ,YP2[ttmmpp2,i][window::sparse_plot],color=color_list[XX], ls = ls_list[XX], lw=0.7,label ='_no_legend_')#,label=label1)   ### np.arange(a)

                Title =  scipy.stats.pearsonr(YP1[ttmmpp,i],YT1[ttmmpp,i])[0]
                NRMSE  = mean_squared_error(YP1[ttmmpp,i], YT1[ttmmpp,i],squared=False)
    
                Title2  = scipy.stats.pearsonr(YP2[ttmmpp2,i],YT2[ttmmpp2,i])[0]
                NRMSE2  = mean_squared_error(YP2[ttmmpp2,i], YT2[ttmmpp2,i],squared=False)
    
                push_plot = push_plot + 0.1
    
            NRMSE,NRMSE2 = NRMSE/SCE[i],NRMSE2/SCN[i]
    
            push2 = 0.05
            ax_list[i].set_xlim(0,count+1)
            ax_list2[i].set_xlim(0,count+1)
            ind = ['Trial '+str(i+1) for i in range(count+1)]
            Title = str(np.around(Title/(count+1),2))
            NRMSE = str(np.around(NRMSE/(count+1),2))
            Title2 = str(np.around(Title2/(count+1),2))
            NRMSE2 = str(np.around(NRMSE2/(count+1),2))
            if len(Title) == 3:
                Title = Title+'0'
            if len(Title2) == 3:
                Title2 = Title2+'0'
            if len(NRMSE) == 3:
                NRMSE = NRMSE+'0'
            if len(NRMSE2) == 3:
                NRMSE2 = NRMSE2+'0'
    
            Title = plot_list[i] + "  r = " + Title + ", NRMSE = " + NRMSE
            Title2 = plot_list[i] + "  r = " + Title2 + ", NRMSE = " + NRMSE2
            if plot_subtitle:
                ax_list[i].text(-0.25, 1.1, Title, transform=ax_list[i].transAxes, size=ss)#,fontweight='bold')
                ax_list2[i].text(-0.25, 1.1, Title2, transform=ax_list2[i].transAxes, size=ss)#,fontweight='bold')

            minor_ticks = []
            percent = ['0%','25%','50%','75%','100%']
            push3 = 0
            for sn in range(count+1):
                for sn1 in np.arange(sn,sn+1.25,0.25):
                    minor_ticks.append(sn1+push3)
                push3=push3+0.1
    
            ax_list[i].set_xticks(minor_ticks ,minor=True)
            ax_list[i].set_xticks(np.array(minor_ticks[2::5])+0.0005,minor=False)
    
            ax_list[i].set_ylabel(ylabel[i],fontsize=ss)
            # ax_list[i].yaxis.set_label_coords(-0.28,0.5)
    
            ax_list2[i].set_xticks(minor_ticks ,minor=True)
            ax_list2[i].set_xticks(np.array(minor_ticks[2::5])+0.0005,minor=False)
            ax_list2[i].set_ylabel(ylabel[i],fontsize=ss)
            # ax_list2[i].yaxis.set_label_coords(-0.28,0.5)
    
            for axx1,axx2 in zip(ax_list[0:len(ax_list)-2], ax_list2[0:len(ax_list2)-2]):
                axx1.set_xticklabels([],fontsize=ss,minor=False)
                axx2.set_xticklabels([],fontsize=ss,minor=False)
    
            for axx1,axx2 in zip(ax_list[-2:], ax_list2[-2:]):
                axx1.set_xticklabels([],fontsize=ss,minor=False)
                axx1.set_xticklabels(percent*(count+1),fontsize=ss,minor=True,rotation=45)
                axx1.set_xlabel("% of task completion",fontsize=ss)
    
                axx2.set_xticklabels([],fontsize=ss,minor=False)
                axx2.set_xticklabels(percent*(count+1),fontsize=ss,minor=True,rotation=45)
                axx2.set_xlabel("% of task completion",fontsize=ss)
    
            ax_list[i].tick_params(axis='x', labelsize=ss,   pad=14,length=3,width=0.5,direction= 'inout',which='major')
            ax_list[i].tick_params(axis='x', labelsize=ss-1, pad=2, length=3,width=0.5,direction= 'inout',which='minor')
            ax_list[i].tick_params(axis='y', labelsize=ss,   pad=3, length=3,width=0.5,direction= 'inout')
    
            ax_list2[i].tick_params(axis='x', labelsize=ss,   pad=14,length=3,width=0.5,direction= 'inout',which='major')
            ax_list2[i].tick_params(axis='x', labelsize=ss-1, pad=2, length=3,width=0.5,direction= 'inout',which='minor')
            ax_list2[i].tick_params(axis='y', labelsize=ss,   pad=3, length=3,width=0.5,direction= 'inout')
    
        if 'JM' in feature:
            ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(3.2, 1.5))
            ax00.text(0.9, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
            ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
    
        elif 'JA' in feature:
            ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(3.2, 1.5))
            ax00.text(0.9, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
            ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
    
        elif 'MF' in feature:
            ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(3.2, 1.5))
            ax00.text(0.94, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
            ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
    
        elif 'MA' in feature:
            ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(3.2, 1.5))
            ax00.text(0.94, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
            ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
    
        elif 'JRF' in feature:
            ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(3.2, 1.5))
            ax00.text(0.9, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
            ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
    
    # if scale_out == True:
    #     feature = feature + '_scaled_out'
    fig.savefig('./plots_out/Both_sub'+'_'+save_name+'_'+feature+'_combine'+'.pdf',dpi=600)
    plt.close()



def stat(fd, index):
    feature = fd.feature[index]
    
    hyper_val_exp =  fd.arg[index]
    model_class_exp = fd.arch[index]
    

    model1 = load_model(fd.subject, feature, model_class_exp,   hyper_val_exp  )
    param = model1.count_params()
    data = fd.data
    if fd.subject == 'exposed':
        XE, YE = data.subject_exposed(feature).test_in, data.subject_exposed(feature).test_out
        SCE = data.subject_exposed(feature).std
        TE = data.subject_exposed(feature).time
    elif fd.subject == 'naive':
        XE, YE = data.subject_naive(feature).test_in, data.subject_naive(feature).test_out
        SCE = data.subject_naive(feature).std
        TE = data.subject_naive(feature).time


    NRMSE_list,  PC_list, RMSE_list  = [],[], []
    YP1 = model1.predict(XE)
    YT1 = np.array(YE)
    a,b = np.shape(YT1)

    try:
        SCE = SCE.to_numpy()
    except:
        SCE = SCE

    if 'MA' in feature:
        SCE = SCE*100

    YP1, YT1 = YP1*SCE, YT1*SCE

    zero_entries = np.where(TE==0)
    zero_entries = np.concatenate([zero_entries[0],np.array([a])])   #### adding last element


    count,aa = -1,[]
    for u in TE.to_numpy():
        if u == 0:
            count = count + 1
        aa.append(u+count)

    count=0

    if 'JA' in feature:
        new_order = [7,8,9,0,1,2,3,4,5,6]
        YP1 = YP1[:,new_order]
        YT1 = YT1[:,new_order]
        

     ######following loop computes stats
    for i in range(b):
        count = 1   ## computing for all trials
        for c in range(count+1):    ########## this loop is required to separate trials
            if c < 2:
                
                ttmmpp = np.arange(zero_entries[c],zero_entries[c+1])
        
                PC =  scipy.stats.pearsonr(YP1[ttmmpp,i],YT1[ttmmpp,i])[0]
                NRMSE =  mean_squared_error(YP1[ttmmpp,i], YT1[ttmmpp,i],squared=False)/SCE[i]
                NRMSE_list.append(NRMSE)
                RMSE =  mean_squared_error(YP1[ttmmpp,i], YT1[ttmmpp,i],squared=False)
                RMSE_list.append(RMSE)

                PC_list.append(PC)

    fd.NRMSE[feature] = NRMSE_list
    fd.RMSE[feature]  = RMSE_list
    fd.pc[feature]    = PC_list
    fd.nparams.append(param)
    return fd

def create_PC_data(model,X1,Y2):
    Y1 = model.predict(X1)
    Y1,Y2 = np.array(Y1),np.array(Y2)
    Y1,Y2 = np.nan_to_num(Y1, nan=0),np.nan_to_num(Y2, nan=0)
    a,b = np.shape(Y1)
    PC = np.zeros(b)
    for i in range(b):
        PC[i] = np.around(scipy.stats.pearsonr(Y1[:,i],Y2[:,i])[0],3)
    return PC


def save_outputs(model, hyper_val, data, label, save_model, model_class):
    X_Train, Y_Train, X_val, Y_val = data.train_in, data.train_out, data.test_in, data.test_out
    feature = data.feature
    subject = data.subject
    train_error = create_PC_data(model,X_Train, Y_Train)
    val_error = create_PC_data(model,X_val, Y_val)   ## it is test error in case of final model
    mse = np.zeros(np.shape(train_error)[0])
    mse[0] = model.evaluate(X_Train, Y_Train,verbose=0)[0]
    mse[1] = model.evaluate(X_val, Y_val,verbose=0)[0]
    out = np.vstack([mse,train_error, val_error])
    out = np.nan_to_num(out, nan=0, posinf=2222)
    np.savetxt('./text_out/stat_'+ model_class + '_'  +feature + '_' + subject +'.'+ 'hv_'+ str(hyper_val) + label +'.txt',out,fmt='%1.6f')
    if save_model == True:
        model.save('./model_out/model_' + model_class + '_' + feature + '_' + subject +'.'+ 'hv_'+ str(hyper_val) + '.h5')
    return None

def run_NN(X_Train, Y_Train, X_val, Y_val,hyper_val,model_class, debug_mode=False):
    if model_class in 'RNN':
        dim = 2
    else:
        dim = 1 
    inp_dim = X_Train.shape[dim]
    out_dim = Y_Train.shape[1]
    t_dim = X_Train.shape[1]
    opt, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p , regularizer_val, NN_variant =   hyper_val

    if debug_mode == True:
        num_nodes=128
        H_layer=2
        epoch = 5
        print("Debug mode on ")
        print(inp_dim,out_dim,t_dim)

    if opt == 'Adam':
        optim = keras.optimizers.Adam
    elif opt == 'RMSprop':
        optim = keras.optimizers.RMSprop
    elif opt == 'SGD':
        optim = keras.optimizers.SGD
    #inp_dim, out_dim, nbr_Hlayer, Neu_layer, activation, p_drop, lr, optim,loss,metric,kinit
    final_act = None
    loss = keras.losses.mean_squared_error
    if model_class == 'NN':
        model = initiate_NN_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif model_class == 'LM':
        model = initiate_Linear_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif model_class == 'LR':
        model = initiate_LR_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif model_class == 'RNN':
        model = initiate_RNN_model(inp_dim, out_dim,  t_dim, H_layer, batch_size, num_nodes, loss, optim, act, p, lr, kinit, final_act, [metric], NN_variant)
    elif model_class == 'CNN':
        model = initiate_CNN_model(inp_dim, out_dim,  t_dim, H_layer, batch_size, num_nodes, loss, optim, act, p, lr, kinit, final_act, [metric], NN_variant)
        
    if debug_mode == True:
        history = model.fit(X_Train, Y_Train, validation_data = (X_val,Y_val),epochs=epoch, batch_size=batch_size, verbose=2,shuffle=True)
    else:
        history = model.fit(X_Train, Y_Train, validation_data = (X_val,Y_val),epochs=epoch, batch_size=batch_size, verbose=2,shuffle=True)
        
    return model


def run_final_model(data,hyper_arg,hyper_val,model_class, save_model=True):
    X_Train, Y_Train, X_Test, Y_Test = data.train_in, data.train_out, data.test_in, data.test_out
    if model_class == 'RNN':
        X_Train = X_Train
        Y_Train = Y_Train.to_numpy()
        X_Test = X_Test
        Y_Test = Y_Test.to_numpy()
    model = run_NN(X_Train, Y_Train, X_Test, Y_Test, hyper_val,  model_class)
    try:
        save_name_old = '.fm'
        save_outputs(model,hyper_arg, data, save_name_old, save_model, model_class)
    except:
        print("this index is creating problem in saving --- ",hyper_arg,hyper_val, data.feature)
    return model

def run_cross_valid(data,hyper_arg,hyper_val,model_class,save_model=False):

    Da = [data.cv1, data.cv2, data.cv3]
    try:
        for enum,d in enumerate(Da):
            X_Train, Y_Train, X_Test, Y_Test = d['train_in'], d['train_out'], d['val_in'], d['val_out']
            if model_class == 'RNN':
                X_Train = X_Train
                Y_Train = Y_Train.to_numpy()
                X_Test = X_Test
                Y_Test = Y_Test.to_numpy()
            model = run_NN(X_Train, Y_Train, X_Test, Y_Test, hyper_val,  model_class)

            try:
                save_name = '.CV_'+str(enum)
                save_outputs(model,hyper_arg, data, save_name, save_model ,model_class)
            except:
                print("this index is creating problem in saving --- ",hyper_arg,hyper_val, data.feature)

    except:
        None

    return None


def create_final_model(hyper_arg,hyper_val,which,pca,scale_out, model_class):
	model = run_final_model(which,hyper_arg,hyper_val,pca,scale_out, model_class)
	return model


def load_model(subject_condition,feature,model_type,hyper_arg):
    path = './model_out/model_'+model_type+'_'+feature+'_'+subject_condition+'.hv_'+str(hyper_arg)+'.h5'  
    model = keras.models.load_model(path)
    return model


def interpolate(xnew,x,y):
    f1 = interp1d(x, y, kind='cubic')
    ynew = f1(xnew)
    return ynew

def check_interpolation(data):
    # this plot the input IMU data before and after interpolation and help visualize the interpolation
    xnew = np.linspace(0, 1, num=data.o1.T1.shape[0], endpoint=True)
    x = data.i1.T1['time']
    columns = data.i1.T1.columns.to_list()
    for enum,fea in enumerate(columns):
        y = data.i1.T1[fea]
        ynew = interpolate(xnew, x, y)
        fig,ax = plt.subplots(2)
        lw = 0.4
        ax[1].plot(x,y,color='r',lw=lw,label = 'data')
        ax[0].plot(x,y,color='r',lw=lw,label = 'data')
        ax[0].plot(xnew,ynew,color='b',lw=lw,label = 'cubic')
        ax[0].set_title(str(enum)+ '  '+ fea)
        ax[0].legend()
        ax[1].legend()
        plt.show()
        plt.close()
        time.sleep(1)
    return None

def print_table_NN(hyper, h, extra):
    print(extra, '&', hyper.iloc[h]['kinit'], '&', hyper.iloc[h]['optim'], '&',hyper.iloc[h]['batch_size'], '&',hyper.iloc[h]['epoch'], '&',hyper.iloc[h]['act'], '&',hyper.iloc[h]['num_nodes'], 
          '&',hyper.iloc[h]['H_layer']+1, '&',hyper.iloc[h]['lr'], '&',hyper.iloc[h]['p'], '\\\\' )
    return None

def print_table_RNN(hyper, h, extra):
    print(extra, '&', hyper.iloc[h]['optim'], '&',hyper.iloc[h]['batch_size'], '&',hyper.iloc[h]['epoch'], '&',hyper.iloc[h]['act'], '&',hyper.iloc[h]['num_nodes'], 
          '&',hyper.iloc[h]['H_layer']+1, '&',hyper.iloc[h]['lr'], '&',hyper.iloc[h]['p'], '\\\\' )
    return None

def print_optimal_results(fm):
    print("\\multicolumn{10}{c}{} \\\\[-0.7em]")
    print("\\multicolumn{10}{c}{\\textbf{Optimal hyperparameters in \\textit{subject-exposed}  settings}}   \\\\")
    print("\\multicolumn{10}{c}{} \\\\[-0.7em]")
    for i in range(5):
        print_table_NN(fm.NN.hyper, fm.NN.exposed.arg[i],   fm.NN.feature_l[i] )
    print('\n')
    print("\\multicolumn{10}{c}{} \\\\[-0.7em]")
    print("\\multicolumn{10}{c}{\\textbf{Optimal hyperparameters in \\textit{subject-naive}  settings}}   \\\\")
    print("\\multicolumn{10}{c}{} \\\\[-0.7em]")
    for i in range(5):
        print_table_NN(fm.NN.hyper, fm.NN.naive.arg[i],   fm.NN.feature_l[i] )

    print('\n')
    print("\\multicolumn{10}{c}{} \\\\[-0.7em]")
    print("\\multicolumn{10}{c}{\\textbf{Optimal hyperparameters in \\textit{subject-exposed}  settings}}   \\\\")
    print("\\multicolumn{10}{c}{} \\\\[-0.7em]")
    for i in range(5):
        print_table_RNN(fm.RNN.hyper, fm.RNN.exposed.arg[i],   fm.RNN.feature_l[i]  + ' & ' + fm.RNN.exposed.arch[i] )
    print('\n')
    print("\\multicolumn{10}{c}{} \\\\[-0.7em]")
    print("\\multicolumn{10}{c}{\\textbf{Optimal hyperparameters in \\textit{subject-naive}  settings}}   \\\\")
    print("\\multicolumn{10}{c}{} \\\\[-0.7em]")
    for i in range(5):
        print_table_RNN(fm.RNN.hyper, fm.RNN.naive.arg[i],   fm.RNN.feature_l[i]  + ' & ' + fm.RNN.naive.arch[i] )
    
    


def specific_CV(data, feat, model_class, hyper_arg):
    subject = data.subject
    hyper   = data.hyper
    hyper_val =  hyper.iloc[hyper_arg]

    if subject == 'exposed':
        data = data.subject_exposed(feat)
    elif subject == 'naive':
        data = data.subject_naive(feat)
    model = run_cross_valid(data,hyper_arg,hyper_val,model_class,save_model=False)


def specific(fm, index):
    hyper_arg   = fm.arg[index]
    hyper_val   = fm.hyper.iloc[hyper_arg]
    model_class = fm.arch[index]
    feat        = fm.feature[index]
    if fm.subject == 'exposed':
        d = fm.data.subject_exposed(feat)
    elif fm.subject == 'naive':
        d = fm.data.subject_naive(feat)
    model = run_final_model(d,hyper_arg,hyper_val,model_class,save_model=True)
    return model


def tab1(fm, D):
    for i in range(5):
        print(fm.feature_l[i], ' & ' )
        for enum, d in enumerate(D):
            print(
                np.around(np.mean(         d.pc[fm.feature[i]]),2), 
                '(' + str(np.around(np.std(d.pc[fm.feature[i]]),2))+')', ' & ', 
                np.around(np.mean(         d.NRMSE[fm.feature[i]]),2), 
                '(' + str(np.around(np.std(d.NRMSE[fm.feature[i]]),2))+')'
                ) 
            if enum < 2:
                print(' & ')
            else:
                print('\\\\')
    return None

def print_SI_table1(fm):
    DE = [fm.LM.exposed, fm.NN.exposed, fm.RNN.exposed]
    DN = [fm.LM.naive,   fm.NN.naive,   fm.RNN.naive]
    tab1(fm,DE)
    print('\n \n')
    tab1(fm,DN)
    return None

def tab2(d):
    u = str(np.around(np.mean(d),2)) + ' & ' + str(np.around(np.std(d),2)) + ' & ' + str(np.around(np.max(d),2)) + ' & ' + str(np.around(np.min(d),2)) + ' & ' + str(np.around(scipy.stats.iqr(d),2))   
    return u

def tab22(fm, D):
    for i in range(5):
        print("\multirow{2}*{", fm.feature_l[i], '}')
        for k in D:
            print(' & ', k.arch[i], ' & ', tab2(k.pc[fm.feature[i]]),' & ', tab2(k.NRMSE[fm.feature[i]]),'\\\\')
        print("\\arrayrulecolor{lightgray} \\hline \\arrayrulecolor{black}")
    print("\n \n")
    return None

def print_SI_table2(fm):
    D = [fm.NN.exposed, fm.RNN.exposed]    
    tab22(fm, D)
    D = [fm.NN.naive, fm.RNN.naive]    
    tab22(fm, D)

def tab3(fm, D):
    for i in range(5):
        print("\multirow{2}*{", fm.feature_l2[i], '}')
        for k in D:
            print(' & ', k.arch[i], ' & ', tab2(k.RMSE[fm.feature[i]]), '\\\\')
        print("\\arrayrulecolor{lightgray} \\hline \\arrayrulecolor{black}")
    return None

def print_SI_table3(fm):
    D = [fm.NN.exposed, fm.RNN.exposed]    
    tab3(fm, D)
    print("\n \n")
    D = [fm.NN.naive, fm.RNN.naive]    
    tab3(fm, D)


def explore(data, hyper_arg):
    
    hyper_val =  data.hyper.iloc[hyper_arg]

    for label in data.feature:
        tmp_data1 = data.data.subject_exposed(label)
        tmp_data2 = data.data.subject_naive(label)
    
        for model_class in [data.what]:
            for Data in [tmp_data1,tmp_data2]:

                try:
                    model = run_cross_valid(Data,hyper_arg,hyper_val,model_class,save_model=False)        
                except:
                    None

                try:
                    model = run_final_model(Data,hyper_arg,hyper_val,model_class,save_model=False)        
                except:
                    None
