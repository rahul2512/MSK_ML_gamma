#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:09:33 2023

@author: rsharma
"""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

def barchart_error(fm1):
    fm = deepcopy(fm1)   
    data = {0: [], 1: []}
    fig,ax = plt.subplots(2,2,sharex=True)
    for j in range(5):
        ML = {0:[], 1:[]}
        for D in [fm.LM, fm.NN, fm.RNN]:
            tmp = {0:[], 1:[]}
            for enum, R in enumerate([D.exposed, D.naive]):
                for Q in [R.pc, R.NRMSE]:
                    tmp[enum].append(np.mean(Q[D.feature[j]]))
                    tmp[enum].append(np.std( Q[D.feature[j]]))
            ML[0].append(tmp[0])
            ML[1].append(tmp[1])
        data[0].append(ML[0])
        data[1].append(ML[1])
    data[0] = np.array(data[0]) 
    data[1] = np.array(data[1]) 
    
    s = 8
    colors = ['red', 'blue', 'green', 'yellow', 'orange']
    hatch = ['/','//','///']
    hatch = [None]*3
    alpha = [1,1,1]
    x = np.array([0,4,8,12,16])
    label = ['LM','FFNN','RNN']
    no_label = ['_no_legend_']*3
    
    for i in range(3):
        means = data[0][:,i,0]
        stdev = data[0][:,i,1]
        ax[0,0].bar(x + i, means, yerr=stdev, alpha=alpha[i], width=1, align='center',color=colors[i], error_kw=dict(lw=0.5, capsize=1.5, capthick=.5, ecolor=colors[i]),label=no_label[i],hatch=hatch[i])
        ax[0,0].set_title('Subject-exposed',fontsize=s,pad=13,fontweight="bold")
    
    for i in range(3):
        means = data[1][:,i,0]
        stdev = data[1][:,i,1]
        ax[0,1].bar(x + i, means, yerr=stdev, alpha=alpha[i], width=1, align='center',color=colors[i], error_kw=dict(lw=0.5, capsize=1.5, capthick=.5, ecolor=colors[i]),label=no_label[i],hatch=hatch[i])
        ax[0,1].set_title('Subject-naive',fontsize=s,pad=13,fontweight="bold")
    
    for i in range(3):
        means = data[0][:,i,2]
        stdev = data[0][:,i,3]
        ax[1,0].bar(x + i, means, yerr=stdev, alpha=alpha[i], width=1, align='center',color=colors[i], error_kw=dict(lw=0.5, capsize=1.5, capthick=.5, ecolor=colors[i]),label=no_label[i],hatch=hatch[i])
    
    for i in range(3):
        means = data[1][:,i,2]
        stdev = data[1][:,i,3]
        ax[1,1].bar(x + i, means, yerr=stdev, alpha=alpha[i], width=1, align='center',color=colors[i], error_kw=dict(lw=0.5, capsize=1.5, capthick=.5, ecolor=colors[i]),label=label[i],hatch=hatch[i])
    
    ax[0,0].set_xticks(x+1)
    
    xlab = fm.feature_l
    ax[1,0].set_xticks(x+1)
    ax[1,1].set_xticks(x+1)
    ax[1,0].set_xticklabels(xlab,fontsize=s,rotation=30, ha= 'right')
    ax[1,1].set_xticklabels(xlab,fontsize=s,rotation=30, ha= 'right')
    plt.rcParams['hatch.linewidth'] = .3
    
    for i in range(2):
        for j in range(2):
            ax[i,j].tick_params(axis='both',labelsize=s,pad=3,length=3,width=0.5,direction= 'inout')
            ax[i,j].set_ylim(0)
            ax[i,j].grid(True,axis='y', lw=0.2, color = 'lightgray')                                      # Make grid lines visible
            ax[i,j].set_axisbelow(True)

    ax[0,0].set_ylim(0, 1.2)
    ax[0,1].set_ylim(0, 1.2)

    ax[1,1].legend(fontsize=s-1,loc='upper right',fancybox=True,ncol=1, frameon=True,framealpha=1)#, bbox_to_anchor=(1.25, 0))  
    
    ax[0,0].set_ylabel('r$_{avg}$',fontsize=s)
    ax[1,0].set_ylabel('NRMSE$_{avg}$',fontsize=s)
    plt.tight_layout()
    fig.savefig("./plots_out/Model_comparison_beta.pdf",dpi=600)
    plt.show()
    return data


def barchart_params(fm):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(6,3))
       
    d = 2
    s = 8
    colors = ['red', 'blue', 'green', 'yellow', 'orange']
    hatch = ['/','//','///']
    hatch = [None]*3
    alpha = [1,1,1]
    x = np.array([0,4,8,12,16])
    label = ['LM','FFNN','RNN']
    no_label = ['_no_legend_']*3
    where = 'center'
    for k, nparams in enumerate([fm.LM.exposed.nparams, fm.NN.exposed.nparams, fm.RNN.exposed.nparams]):
        ax[0].bar(x+k, nparams, alpha=alpha[0], width=1, align=where,color=colors[k], error_kw=dict(lw=0.5, capsize=1.5, capthick=.5, ecolor=colors[1]),label=label[k],hatch=hatch[0])
    for k, nparams in enumerate([fm.LM.naive.nparams,   fm.NN.naive.nparams,   fm.RNN.naive.nparams]):
        ax[1].bar(x+k, nparams, alpha=alpha[0], width=1, align=where,color=colors[k], error_kw=dict(lw=0.5, capsize=1.5, capthick=.5, ecolor=colors[1]),label=no_label[k],hatch=hatch[0])
   
   
    xlab = fm.feature_l
    plt.rcParams['hatch.linewidth'] = .3
    
    for i in range(2):
        ax[i].tick_params(axis='both',labelsize=s,pad=3,length=3,width=0.5,direction= 'inout')
        ax[i].set_ylim(0)
        ax[i].set_xticklabels(xlab,fontsize=s,rotation=30, ha= 'right')
        ax[i].set_xticks(x+1)
        ax[i].set_xticklabels(xlab,fontsize=s,rotation=30, ha= 'right')
        ax[i].set_ylabel('Number of parameters',fontsize=s)
        ax[i].grid(True,axis='y', lw=0.2, color = 'lightgray')                                       # Make grid lines visible
        ax[i].set_axisbelow(True)

    ax[0].legend(fontsize=s-1,loc='upper right',fancybox=True,ncol=1, frameon=True,framealpha=1)#, bbox_to_anchor=(1.25, 0))  
    
    ax[0].set_title('Subject-exposed',fontsize=s,fontweight="bold")
    ax[1].set_title('Subject-naive',fontsize=s,fontweight="bold")

    plt.tight_layout()
    fig.savefig("./plots_out/param_comparison_beta.pdf",dpi=600)
    plt.show()