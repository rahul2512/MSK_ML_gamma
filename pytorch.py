from pytorch_utilities import *
from read_in_out import analysis_options, add_noise_to_trial
import pandas as pd
import numpy as np
import keras, random
import seaborn as sns
import matplotlib.pyplot as plt
import sys, copy
import scipy
from sklearn.preprocessing import StandardScaler
import random
from sklearn.metrics import mean_squared_error
from matplotlib import gridspec
from scipy.interpolate import interp1d
from barchart_err import barchart_error, barchart_params
from keras import backend as K


RNN_models = ['SimpleRNN','LSTM','GRU','BSimpleRNN','BLSTM','BGRU']
feature_list = ['Joint angles','Joint reaction forces','Joint moments',  'Muscle forces', 'Muscle activations']
feature_slist = ['JA','JRF','JM',  'MF', 'MA']
# Define custom RMSE loss function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

from keras.utils import get_custom_objects
get_custom_objects().update({'rmse': rmse})


def initiate_ax(feature):
    if 'JRF' == feature:
#        if XX == 0 :
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

    elif 'JM' == feature:
#        if XX == 0:
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

    elif 'JA' == feature:
#        if XX == 0:
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
    return fig, ax_list, ax_list2, ss, b_xlabel, ylabel, plot_list 

def end_ax(ax_list,ax_list2, feature, ss):

    if 'JM' in feature:
        ax_list[0].legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(3.2, 1.5))
        ax_list[0].text(0.9, 1.35, "(I) Subject-exposed", transform=ax_list[0].transAxes, size=ss+0.5,fontweight='bold')
        ax_list[0].text(4.35, 1.35, "(II) Subject-naive", transform=ax_list[0].transAxes, size=ss+0.5,fontweight='bold')

    elif 'JA' in feature:
        ax_list[0].legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(3.2, 1.5))
        ax_list[0].text(0.9, 1.35, "(I) Subject-exposed", transform=ax_list[0].transAxes, size=ss+0.5,fontweight='bold')
        ax_list[0].text(4.35, 1.35, "(II) Subject-naive", transform=ax_list[0].transAxes, size=ss+0.5,fontweight='bold')

    elif 'JRF' in feature:
        ax_list[0].legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(3.2, 1.5))
        ax_list[0].text(0.9, 1.35, "(I) Subject-exposed", transform=ax_list[0].transAxes, size=ss+0.5,fontweight='bold')
        ax_list[0].text(4.35, 1.35, "(II) Subject-naive", transform=ax_list[0].transAxes, size=ss+0.5,fontweight='bold')   

    for axl in [ax_list, ax_list2]:
        for ax in axl: 
            ax.tick_params(axis='x', labelsize=ss,   pad=14,length=3,width=0.5,direction= 'inout',which='major')
            ax.tick_params(axis='x', labelsize=ss-1, pad=2, length=3,width=0.5,direction= 'inout',which='minor')
            ax.tick_params(axis='y', labelsize=ss,   pad=3, length=3,width=0.5,direction= 'inout')

    percent = ['0%','25%','50%','75%','100%']
    for axx1,axx2 in zip(ax_list[0:len(ax_list)-2], ax_list2[0:len(ax_list2)-2]):
        axx1.set_xticklabels([],fontsize=ss,minor=False)
        axx2.set_xticklabels([],fontsize=ss,minor=False)

    for axl in [ax_list[-2:], ax_list2[-2:]]:
        for ax in axl:
            ax.set_xticklabels([],fontsize=ss,minor=False)
            ax.set_xticklabels(percent,fontsize=ss,minor=True,rotation=45)
            ax.set_xlabel("% of task completion",fontsize=ss)
   
    return ax_list, ax_list2

def combined_plot(analysis_opt):
    # it plots the first trial and provides the statistics for each trial as well as average, std, iqr, min,max etc etc
    lll = len(analysis_opt.model_exposed_hyper_arg)
    save_name = analysis_opt.save_name
    trial_ind = analysis_opt.trial_ind
    color_list = ['r','b']
    ls_list = ['-','-']
    lsk = '--'
    for XX in range(lll):
        feature = analysis_opt.feature 
        hyper_val_exp =  analysis_opt.model_exposed_hyper_arg[XX]
        hyper_val_naive = analysis_opt.model_naive_hyper_arg[XX]
        model_class_exp = analysis_opt.model_exposed_arch[XX]
        model_class_naive = analysis_opt.model_naive_arch[XX]
        hyper = analysis_opt.hyper[XX]
        data = analysis_opt.data[XX]
        window = analysis_opt.window_size[XX]
        norm_out = hyper.iloc[hyper_val_exp]['norm_out']

        model1 = load_model('exposed', feature, model_class_exp,   hyper_val_exp  )
        model2 = load_model('naive'  , feature, model_class_naive, hyper_val_naive)


        XE, YE  = data.subject_exposed(feature, norm_out).test_in_list[trial_ind], data.subject_exposed(feature, norm_out).test_out_list[trial_ind]
        XN, YN  = data.subject_naive(  feature, norm_out).test_in_list[trial_ind],   data.subject_naive(  feature, norm_out).test_out_list[trial_ind]
        sub_col = data.subject_exposed(feature, norm_out).sub_col
        
        SC = data.std_out[data.label[feature]]
        TE = np.linspace(0,1,YE.shape[0] + analysis_opt.window_size[XX])
        TN = np.linspace(0,1,YN.shape[0] + analysis_opt.window_size[XX])

        plot_subtitle = analysis_opt.plot_subtitle[XX]

        YP1, YP2 = model1.predict(XE), model2.predict(XN)
        YT1, YT2 = np.array(YE), np.array(YN)

        if XX == 0:
            fig, ax_list, ax_list2, ss, b_xlabel, ylabel, plot_list = initiate_ax(feature)

        if 'JA' == feature:
            new_order = [7,8,9,0,1,2,3,4,5,6]
            YP1 = YP1[:,new_order]
            YP2 = YP2[:,new_order]
            YT1 = YT1[:,new_order]
            YT2 = YT2[:,new_order]
            SC = SC[[sub_col[i] for i in new_order]]            
    
        sparse_plot=5
        for i, _  in enumerate(sub_col):
            push_plot = 0
            count = 0   ## plotting first trial
            if ax_list[i] == ax_list[0]:
                label1  = analysis_opt.legend_label[XX] #+ ' prediction'
                if label1 == 'NN':
                    label1 = 'FFNN'
                if XX == 0:
                    label2 = 'MSK'
            else:
                label1, label2 = '_no_legend_', '_no_legend_'

            if XX == 0:
                ax_list[i].plot( TE[window::sparse_plot], YT1[:,i][window::sparse_plot],color='k',lw=0.9,ls = lsk, label=label2)
                ax_list2[i].plot(TN[window::sparse_plot], YT2[:,i][window::sparse_plot],color='k',lw=0.9,ls = lsk, label ='_no_legend_')#,label=label2)
                
            ax_list[i].plot(TE[window::sparse_plot], YP1[:,i][window::sparse_plot],color=color_list[XX],ls = ls_list[XX], lw=0.7,label=label1)   ### np.arange(a)
            ax_list2[i].plot(TN[window::sparse_plot], YP2[:,i][window::sparse_plot],color=color_list[XX], ls = ls_list[XX], lw=0.7,label ='_no_legend_')#,label=label1)   ### np.arange(a)

            Title =  scipy.stats.pearsonr(YP1[:,i],YT1[:,i])[0]
            RMSE  = mean_squared_error(YP1[:,i], YT1[:,i],squared=False)

            Title2  = scipy.stats.pearsonr(YP2[:,i],YT2[:,i])[0]
            RMSE2  = mean_squared_error(YP2[:,i], YT2[:,i],squared=False)

            push_plot = push_plot + 0.1
    
            NRMSE, NRMSE2 = RMSE/SC[i], RMSE2/SC[i]
    
            ax_list[i].set_xlim(0,1)
            ax_list2[i].set_xlim(0,1)
            Title = str(np.around(Title,2))
            NRMSE = str(np.around(NRMSE,2))
            Title2 = str(np.around(Title2,2))
            NRMSE2 = str(np.around(NRMSE2,2))

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
    
    
        ax_list, ax_list2 = end_ax(ax_list,ax_list2, feature, ss)
            
    fig.savefig('./plots_out/Both_sub'+'_'+save_name+'_'+feature+'_combine'+'.pdf',dpi=600)
    plt.close()



def combined_plot_noise(analysis_opt):
    # it plots the first trial and provides the statistics for each trial as well as average, std, iqr, min,max etc etc
    lll = len(analysis_opt.model_exposed_hyper_arg)
    save_name = analysis_opt.save_name
    trial_ind = analysis_opt.trial_ind
    color_list = ['r','b']
    ls_list = ['-','-']
    lsk = '--'
    for XX in range(lll):
        feature = analysis_opt.feature 
        hyper_val_exp =  analysis_opt.model_exposed_hyper_arg[XX]
        hyper_val_naive = analysis_opt.model_naive_hyper_arg[XX]
        model_class_exp = analysis_opt.model_exposed_arch[XX]
        model_class_naive = analysis_opt.model_naive_arch[XX]
        hyper = analysis_opt.hyper[XX]
        data = analysis_opt.data[XX]
        window = analysis_opt.window_size[XX]
        norm_out = hyper.iloc[hyper_val_exp]['norm_out']

        model1 = load_model('exposed', feature, model_class_exp,   hyper_val_exp  )
        model2 = load_model('naive'  , feature, model_class_naive, hyper_val_naive)


        XE, YE  = data.subject_exposed(feature, norm_out).test_in_list[trial_ind], data.subject_exposed(feature, norm_out).test_out_list[trial_ind]
        XN, YN  = data.subject_naive(  feature, norm_out).test_in_list[trial_ind],   data.subject_naive(  feature, norm_out).test_out_list[trial_ind]
        sub_col = data.subject_exposed(feature, norm_out).sub_col
        
        SC = data.std_out[data.label[feature]]
        TE = np.linspace(0,1,YE.shape[0] + analysis_opt.window_size[XX])
        TN = np.linspace(0,1,YN.shape[0] + analysis_opt.window_size[XX])

        plot_subtitle = analysis_opt.plot_subtitle[XX]

        YP1, YP2 = model1.predict(XE), model2.predict(XN)
        YT1, YT2 = np.array(YE), np.array(YN)

        if XX == 0:
            fig, ax_list, ax_list2, ss, b_xlabel, ylabel, plot_list = initiate_ax(feature)

        if 'JA' == feature:
            new_order = [7,8,9,0,1,2,3,4,5,6]
            YP1 = YP1[:,new_order]
            YP2 = YP2[:,new_order]
            YT1 = YT1[:,new_order]
            YT2 = YT2[:,new_order]
            SC = SC[[sub_col[i] for i in new_order]]            
    
        sparse_plot=5
        for i, _  in enumerate(sub_col):
            push_plot = 0
            count = 0   ## plotting first trial
            if ax_list[i] == ax_list[0]:
                label1  = analysis_opt.legend_label[XX] #+ ' prediction'
                if label1 == 'NN':
                    label1 = 'FFNN'
                if XX == 0:
                    label2 = 'MSK'
            else:
                label1, label2 = '_no_legend_', '_no_legend_'

            if XX == 0:
                ax_list[i].plot( TE[window::sparse_plot], YT1[:,i][window::sparse_plot],color='k',lw=0.9,ls = lsk, label=label2)
                ax_list2[i].plot(TN[window::sparse_plot], YT2[:,i][window::sparse_plot],color='k',lw=0.9,ls = lsk, label ='_no_legend_')#,label=label2)
                
            ax_list[i].plot(TE[window::sparse_plot], YP1[:,i][window::sparse_plot],color=color_list[XX],ls = ls_list[XX], lw=0.7,label=label1)   ### np.arange(a)
            ax_list2[i].plot(TN[window::sparse_plot], YP2[:,i][window::sparse_plot],color=color_list[XX], ls = ls_list[XX], lw=0.7,label ='_no_legend_')#,label=label1)   ### np.arange(a)

            Title =  scipy.stats.pearsonr(YP1[:,i],YT1[:,i])[0]
            RMSE  = mean_squared_error(YP1[:,i], YT1[:,i],squared=False)

            Title2  = scipy.stats.pearsonr(YP2[:,i],YT2[:,i])[0]
            RMSE2  = mean_squared_error(YP2[:,i], YT2[:,i],squared=False)

            push_plot = push_plot + 0.1
    
            NRMSE, NRMSE2 = RMSE/SC[i], RMSE2/SC[i]
    
            ax_list[i].set_xlim(0,1)
            ax_list2[i].set_xlim(0,1)
            Title = str(np.around(Title,2))
            NRMSE = str(np.around(NRMSE,2))
            Title2 = str(np.around(Title2,2))
            NRMSE2 = str(np.around(NRMSE2,2))

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

#################################################################################
###########################above code is untouched and below it plots the noisy data
#################################################################################

        samples = 300
        M1 = np.zeros((samples,YP1.shape[0],YP1.shape[1]))
        M2 = np.zeros((samples,YP2.shape[0],YP2.shape[1]))
        for samp in range(samples):
            XEN, XNN = add_noise_to_trial(XE), add_noise_to_trial(XN)
            YPN1, YPN2 = model1.predict(XEN), model2.predict(XNN)
            if 'JA' == feature:
                new_order = [7,8,9,0,1,2,3,4,5,6]
                YPN1 = YPN1[:,new_order]
                YPN2 = YPN2[:,new_order]
            M1[samp] =  YPN1
            M2[samp] =  YPN2
        M1_mean, M1_std = np.mean(M1,axis=0), np.std(M1,axis=0)
        M2_mean, M2_std = np.mean(M2,axis=0), np.std(M2,axis=0)

        alpha=0.2
        for i, _  in enumerate(sub_col):
            mu1, sigma1 = YP1[:,i][window::sparse_plot], M1_std[:,i][window::sparse_plot]                
            mu2, sigma2 = YP2[:,i][window::sparse_plot], M2_std[:,i][window::sparse_plot]        
            ax_list[i].fill_between( TE[window::sparse_plot], mu1+sigma1, mu1-sigma1, facecolor=color_list[XX], alpha=alpha)
            ax_list2[i].fill_between(TN[window::sparse_plot], mu2+sigma2, mu2-sigma2, facecolor=color_list[XX], alpha=alpha)
            ax_list[i].plot( TE[window::sparse_plot], M1_mean[:,i][window::sparse_plot],color=color_list[XX], ls = ls_list[XX], lw=0.7,label='_no_legend_')   ### np.arange(a)
            ax_list2[i].plot(TN[window::sparse_plot], M2_mean[:,i][window::sparse_plot],color=color_list[XX], ls = ls_list[XX], lw=0.7,label ='_no_legend_')#,label=label1)   ### np.arange(a)

        ax_list, ax_list2 = end_ax(ax_list,ax_list2, feature, ss)            
    fig.savefig('./plots_out/Both_sub_noise'+'_'+save_name+'_'+feature+'_combine'+'.pdf',dpi=600)
    plt.close()

def stat(fd, index):
    feature = fd.feature[index]   
    hyper_val_exp =  fd.arg[index]
    norm_out = fd.hyper.loc[hyper_val_exp]['norm_out']
    model_class_exp = fd.arch[index]
   
    data = fd.data
    SCE = data.std_out[data.label[feature]]

    if fd.subject == 'exposed':
        tmp = data.subject_exposed(feature, norm_out)
        XE, YE = tmp.test_in_list, tmp.test_out_list
        model1 = load_model('exposed', feature, model_class_exp,   hyper_val_exp  )
    elif fd.subject == 'naive':
        tmp = data.subject_naive(feature, norm_out)
        XE, YE = tmp.test_in_list, tmp.test_out_list
        model1 = load_model('naive', feature, model_class_exp,   hyper_val_exp  )
    elif fd.subject == 'exposed_unseen':
        tmp = data.subject_exposed(feature, norm_out)
        XE, YE = tmp.super_test_in_list, tmp.super_test_out_list
        model1 = load_model('exposed', feature, model_class_exp,   hyper_val_exp  )

    param = model1.count_params()
    ntrials = len(XE)
    sub_col = tmp.sub_col
    df = pd.DataFrame(index = np.arange(ntrials),columns=sub_col)
    NRMSE, PC, RMSE  = copy.deepcopy(df), copy.deepcopy(df), copy.deepcopy(df)

    for n in range(ntrials):
        YP1 = model1.predict(XE[n])
        YT1 = np.array(YE[n])
    # if 'JA' in feature:
    #     new_order = [7,8,9,0,1,2,3,4,5,6]
    #     YP1 = YP1[:,new_order]
    #     YT1 = YT1[:,new_order]      
        for enum, col in enumerate(sub_col):
                PC[col].loc[n]    =  scipy.stats.pearsonr(YP1[:,enum],YT1[:,enum])[0]
                NRMSE[col].loc[n] =  mean_squared_error(  YP1[:,enum],YT1[:,enum],squared=False)/SCE[col]
                RMSE[col].loc[n]  =  mean_squared_error(  YP1[:,enum],YT1[:,enum],squared=False)
                
    fd.NRMSE[feature] = NRMSE
    fd.RMSE[feature]  = RMSE
    fd.pc[feature]    = PC
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

def run_NN(X_Train, Y_Train, X_val, Y_val, hyper_val, model_class, debug_mode=False):
    if model_class   == 'RNN':
        dim = 2
        NN_variant, opt, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p, regularizer_val, norm_out =   hyper_val
    elif model_class == 'CNN':
        dim = 2
        opt, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, pool_size, regularizer_val, NN_variant, filt_size, stride, norm_out = hyper_val
    elif model_class == 'convLSTM':
        dim = 2
        opt, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, pool_size, regularizer_val, NN_variant, filt_size, stride, norm_out = hyper_val
    elif model_class == 'CNNLSTM':
        dim = 2
        opt, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, pool_size, regularizer_val, NN_variant, filt_size, stride, LSTM_units, norm_out = hyper_val
    elif model_class   == 'NN':
        dim = 1 
        opt, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p , regularizer_val, NN_variant, norm_out =   hyper_val
    elif model_class   == 'LM':
        dim = 1 
        opt, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p , regularizer_val, NN_variant, norm_out =   hyper_val
    else:
        print('unrecog model description')
        sys.exit()
        
    inp_dim = X_Train.shape[dim]
    out_dim = Y_Train.shape[1]
    t_dim = X_Train.shape[1]

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

    if loss == 'mse':
        loss = keras.losses.mean_squared_error
    elif loss == 'rmse':
        loss = rmse

    if model_class == 'NN':
        model = initiate_NN_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif model_class == 'LM':
        model = initiate_Linear_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif model_class == 'LR':
        model = initiate_LR_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif model_class == 'RNN':
        model = initiate_RNN_model(inp_dim, out_dim,  t_dim, H_layer, batch_size, num_nodes, loss, optim, act, p, lr, kinit, final_act, [metric], NN_variant)
    elif model_class == 'CNN':
        model = initiate_CNN_model(inp_dim, out_dim,  t_dim, H_layer, batch_size, num_nodes, loss, optim, act, pool_size, lr, kinit, final_act, [metric], filt_size, NN_variant, stride)
    elif model_class == 'CNNLSTM':
        model = initiate_CNNLSTM_model(inp_dim, out_dim,  t_dim, H_layer, batch_size, num_nodes, loss, optim, act, pool_size, lr, kinit, final_act, [metric], filt_size, NN_variant, stride, LSTM_units)
    elif model_class == 'convLSTM':
        model = initiate_ConvLSTM_model(inp_dim, out_dim,  t_dim, H_layer, batch_size, num_nodes, loss, optim, act, pool_size, lr, kinit, final_act, [metric], filt_size, NN_variant, stride)
        X_Train = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1], X_Train.shape[2], 1))
        X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))

    if debug_mode == True:
        history = model.fit(X_Train, Y_Train, validation_data = (X_val,Y_val),epochs=epoch, batch_size=batch_size, verbose=2,shuffle=True)
    else:
        history = model.fit(X_Train, Y_Train, validation_data = (X_val,Y_val),epochs=epoch, batch_size=batch_size, verbose=0,shuffle=True)
        
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
        print("this index is creating problem in saving --- ",hyper_arg, hyper_val, data.feature)
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


def print_tables(d):
    title = ['Subject-exposed model','Subject-naive model', 'Subject-exposed model for unseen data']
    print(d.what, 'results \n')
    hyper = d.hyper
    
    for enumsub, sub in enumerate([d.exposed, d.naive, d.exposed_unseen]):
        print(title[enumsub], '\n')
        for enum, h in enumerate(sub.arg):
            if d.what == 'NN':
                print(d.feature_l[enum], '&', hyper.iloc[h]['kinit'], '&', hyper.iloc[h]['optim'], '&',hyper.iloc[h]['batch_size'], '&',hyper.iloc[h]['epoch'], '&',hyper.iloc[h]['act'], '&',hyper.iloc[h]['num_nodes'], 
                      '&',hyper.iloc[h]['H_layer']+1, '&',hyper.iloc[h]['lr'], '&',hyper.iloc[h]['p'], '\\\\' )
        print('\n \n')
        
        print('& \\multicolumn{15}{c}{\\textbf{Optimal hyperparameters for \\textit{', title[enumsub] ,'}}}   \\\\')
        for enum, h in enumerate(sub.feature):
            pc = pd.concat([sub.pc[h][col]    for col in sub.pc[h].columns],    axis=0, ignore_index=True)
            nr = pd.concat([sub.NRMSE[h][col] for col in sub.NRMSE[h].columns], axis=0, ignore_index=True)
            rm = pd.concat([sub.RMSE[h][col]  for col in sub.RMSE[h].columns],  axis=0, ignore_index=True)
            a, b = '&', '\\\\'
            print('\\arrayrulecolor{lightgray} \\hline \\arrayrulecolor{black}')
            print(d.feature_l[enum],  a, 
                  pc.mean().round(2), a, pc.std().round(2), a, pc.max().round(2), a, pc.min().round(2), a, (pc.quantile(0.75) - pc.quantile(0.25)).round(2), a, 
                  nr.mean().round(2), a, nr.std().round(2), a, nr.max().round(2), a, nr.min().round(2), a, (nr.quantile(0.75) - nr.quantile(0.25)).round(2), a, 
                  rm.mean().round(2), a, rm.std().round(2), a, rm.max().round(2), a, rm.min().round(2), a, (rm.quantile(0.75) - rm.quantile(0.25)).round(2), b                              
                  )
            if enumsub ==0:
                print('\\arrayrulecolor{lightgray} \\hline \\arrayrulecolor{black}')

        print('\n \n')
        
    return None

def print_table_RNN(hyper, h, extra):
    print(extra, '&', hyper.iloc[h]['optim'], '&',hyper.iloc[h]['batch_size'], '&',hyper.iloc[h]['epoch'], '&',hyper.iloc[h]['act'], '&',hyper.iloc[h]['num_nodes'], 
          '&',hyper.iloc[h]['H_layer']+1, '&',hyper.iloc[h]['lr'], '&',hyper.iloc[h]['p'], '\\\\' )
    return None

def specific(fm, index):
    hyper_arg   = fm.arg[index]
    hyper_val   = fm.hyper.iloc[hyper_arg]
    model_class = fm.arch[index]
    feat        = fm.feature[index]
    norm_out    = hyper_val['norm_out']
    if fm.subject == 'exposed':
        d = fm.data.subject_exposed(feat, norm_out)
    elif fm.subject == 'naive':
        d = fm.data.subject_naive(feat, norm_out)
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
    norm_out = hyper_val['norm_out']
    for label in data.feature:
        tmp_data1 = data.data.subject_exposed(label, norm_out)
        tmp_data2 = data.data.subject_naive(label, norm_out)
    
        for model_class in [data.what]:
            for Data in [tmp_data1,tmp_data2]:
                model = run_final_model(Data,hyper_arg,hyper_val,model_class,save_model=False)        
                try:
                    model = run_cross_valid(Data,hyper_arg,hyper_val,model_class,save_model=False)        
                except:
                    None

                try:
                    model = run_final_model(Data,hyper_arg,hyper_val,model_class,save_model=False)        
                except:
                    None

###############################################################################
###############################################################################
#####   Given a data set and a model, it computes the statistics
###############################################################################
###############################################################################

def stat_new_data(fd, data):
    data.exposed = fd.exposed
    data.naive   = fd.naive
    data.what  = fd.what
    data.hyper = fd.hyper
    data.feature_l = fd.feature_l
    for enumf, feat in enumerate(fd.feature):
        for sub in [fd.exposed, fd.naive]:
            hyper_val_exp =  sub.arg[enumf]
            model_class_exp = sub.arch[enumf]
            
            model1 = load_model(sub.subject, feat, model_class_exp,   hyper_val_exp  )
            col_labels = data.label[feat]
            SCE = fd.data.std_out[col_labels]
        
            XE, YE = [j for i in data.inp_all_list for j in i], [j[col_labels] for i in data.out_all_list for j in i]
            ntrials = len(XE)
            df = pd.DataFrame(index = np.arange(ntrials),columns=col_labels)
            NRMSE, PC, RMSE  = copy.deepcopy(df), copy.deepcopy(df), copy.deepcopy(df)
        
            for n in range(ntrials):
                
                YP = model1.predict(XE[n])
                YT = np.array(YE[n])
                for enum, col in enumerate(col_labels):
                        PC[col].loc[n]    =  scipy.stats.pearsonr(YP[:,enum],YT[:,enum])[0]
                        NRMSE[col].loc[n] =  mean_squared_error(  YP[:,enum],YT[:,enum],squared=False)/SCE[col]
                        RMSE[col].loc[n]  =  mean_squared_error(  YP[:,enum],YT[:,enum],squared=False)
                        
            if sub.subject == 'exposed':
                data.exposed.NRMSE[feat] = NRMSE
                data.exposed.RMSE[feat]  = RMSE
                data.exposed.pc[feat]    = PC
            elif sub.subject == 'naive':
                data.naive.NRMSE[feat]   = NRMSE
                data.naive.RMSE[feat]    = RMSE
                data.naive.pc[feat]      = PC
                
    return data


##############################################################################
##############################################################################
###### learning curve
##############################################################################
##############################################################################

def learning_curve(fm):
    ## learning curve are done using all the data i.e. validation accuracy is essentially test accuracy
    res = analysis_options("results for learning curve -- note only for naive models")
    res.model = fm.what
    res.subject = 'naive'
    nval = np.arange(5)  ### this allows picking random subjects to initialze or repeat the computation multiple times (with same subejcts) to check robustness
    res.RMSE_train = {}
    res.RMSE_test  = {}
    for enumf, feat in enumerate(fm.feature):
        hyper_arg = fm.naive.arg[enumf]
        model_class = fm.naive.arch[enumf]
        hyper_val = fm.hyper.loc[hyper_arg]
        norm_out  = hyper_val['norm_out']
        data = fm.data.subject_naive(feat,norm_out)
        nsub = len(data.train_in_list)
        res.RMSE_train[feat] = pd.DataFrame(index = np.arange(1, nsub), columns=nval)
        res.RMSE_test[feat]  = pd.DataFrame(index = np.arange(1, nsub), columns=nval)
        for r in nval:
            for n in np.arange(1,nsub):
                rand = random.choices(range(0, nsub), k=n)              
                tmp_train_in  = [ data.train_in_list[f] for f in rand]
                tmp_train_out = [ data.train_out_list[f] for f in rand]
                try:
                    X = pd.concat(tmp_train_in)
                except:
                    X = np.concatenate(tmp_train_in)  
                Y = pd.concat(tmp_train_out)
    
                model = run_NN(X, Y, data.test_in, data.test_out, hyper_val,  model_class, 0)
                res.RMSE_train[feat][r].loc[n] = model.evaluate(X, Y, verbose=2)[0]  ### test loss = 0 and test accuracy 1
                res.RMSE_test[feat][r].loc[n]  = model.evaluate(data.test_in, data.test_out, verbose=2)[0]  ### test loss = 0 and test accuracy 1
        res.RMSE_train[feat].to_csv('./lc_data/' + res.model + '.' + res.subject + '.' + feat + '.lc.train.txt' ,index=False, header=False)
        res.RMSE_test[feat].to_csv( './lc_data/' + res.model + '.' + res.subject + '.' + feat + '.lc.test.txt' ,index=False, header=False)
        ## columns are various nval trials and rows are number of subjects
    return fm



def plot_learning_curve(model_kind, subject_kind, feat):
    alpha = 0.2
    color = ['r','b']
    index = feature_slist.index(feat)
    yl = feature_list[index]
    s = 14
    train_err = pd.read_csv('./lc_data/' + model_kind + '.' + subject_kind+ '.' + feat + '.lc.train.txt', header=None)    
    val_err  = pd.read_csv('./lc_data/' + model_kind + '.' + subject_kind+ '.' + feat + '.lc.test.txt', header=None)       
    nsub = train_err.shape[0]
    fig, ax = plt.subplots()
    ind = np.arange(1,nsub+1)
    label = ['Training', 'Validation']
    for enum,d in enumerate([train_err,val_err]):
        ax.scatter(ind, d.mean(axis=1), color=color[enum], label=label[enum])
        ax.plot(ind, d.mean(axis=1), color=color[enum], label='_no_legend_')
        ax.fill_between(ind, d.mean(axis=1)+d.std(axis=1), d.mean(axis=1)-d.std(axis=1), facecolor=color[enum], alpha=alpha)
    ax.legend(fontsize=s)
    ax.tick_params(axis='both', labelsize=s,   pad=4,length=3,width=0.5,direction= 'inout',which='major')
    ax.set_xlabel("# of subjects", fontsize=s)
    ax.set_ylabel("RMSE loss (" + yl + ')', fontsize=s)
    plt.savefig('./plots_out/' + model_kind + '.' + subject_kind+ '.' + feat + '.lc.pdf', dpi=600)
    plt.show()
    
