### first section to read the marker data
import numpy as np, pandas as pd, copy
from scipy.interpolate import interp1d
import scipy.io as sio, sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

Weight = pd.read_csv('./Output/Weight', header=None)
Weight_moment = pd.read_csv('./Output/Weight_moment', header=None)
color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 
         'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k', 'teal',
         'deeppink','goldenrod','darkred','darkviolet']
ls = ['-','--',':']
feat_order     = ['JA','JM','JRF','MA','MF']
feat_order_l   = ['Joint angles','Joint moments', 'Joint reaction forces', 'Muscle forces', 'Muscle activations']
feat_order_l2  = ['Joint angles (degrees)','Joint moments (\\% Body Weight \\times Body Height )', 'Joint reaction forces (\\% Body Weight)', 'Muscle forces (\\% Body Weight)', 'Muscle activations (\\%)']
feat_order_tmp = ['JA']*10 + ['JM']*10 + ['JRF']*12 + ['MF']*4 + ['MA']*4 

############################################
## Some functions used later to handle data
############################################

def combine(d, how):
    if d.index in [1,2,3,4,8,12,13,14,15,16]:
        ul = [d.T1,d.T2,d.T3]
    elif d.index in [6,9,11]:
        ul = [d.T1]
    elif d.index in [5,10]:
        ul = [d.T1,d.T2]
    elif d.index in [7]:
        ul = [d.T1,d.T3]
    else:
        print("unrecognised index")
        sys.exit()            

    if how == 0:
        u = pd.concat(ul)   
    elif how ==1:
        u = np.concatenate(ul)
    return ul, u


def transform_trial_into_windows(i1,o1,window_size):
    s0,s1 = i1.shape
    tmp = np.zeros([s0-window_size+1,window_size,s1])
    st = i1.index[0]-1
    for enum, i in enumerate(i1.index.to_list()):
        if i >= st + window_size:
            tmp[enum-window_size+1] = i1.loc[i-window_size+1:i].to_numpy()  ###loc uses the 
    tmpo = o1.loc[st+window_size::]
    return tmp, tmpo

def transform_subject_into_windows(i1,o1,window_size):
    i1.T1, o1.T1 = transform_trial_into_windows(i1.T1, o1.T1, window_size)
    i1.T2, o1.T2 = transform_trial_into_windows(i1.T2, o1.T2, window_size)
    i1.T3, o1.T3 = transform_trial_into_windows(i1.T3, o1.T3, window_size)
    i1.all_list, i1.all = combine(i1,1)
    o1.all_list, o1.all = combine(o1,0)
    return i1, o1

def Muscle_process(Y,which):
#######  ListModify[list_] := {list[[1 ;; 5]] // Max, list[[6 ;; 7]] // Max,    list[[14 ;; 19]] // Max, list[[20 ;; 21]] // Max};
    Y.columns = np.arange(21)
    tmp = copy.deepcopy(Y.iloc[:,[0,1,2,3]])
    tmp.iloc[:,0] = Y.iloc[:,[0,1,2,3,4]].max(axis=1)
    tmp.iloc[:,1] = Y.iloc[:,[5,6]].max(axis=1)
    tmp.iloc[:,2] = Y.iloc[:,[13,14,15,16,17,18]].max(axis=1)
    tmp.iloc[:,3] = Y.iloc[:,[19,20]].max(axis=1)
    Y = tmp
    return Y


def filt(d):
    filters = pd.read_csv('./Output/'+d.add1+'frame_filters', header=None)
    d.filter = filters.iloc[d.index-1] 
    d.T1 = d.T1.iloc[d.filter[0]:d.filter[1]+1]
    d.T2 = d.T2.iloc[d.filter[2]:d.filter[3]+1]
    d.T3 = d.T3.iloc[d.filter[4]:d.filter[5]+1]
    return d

#################################################
# Classes to read data
#################################################
class subject_in:
    def __init__(self, index, add1=''):
        self.index = index
        self.add1 = add1
        self.path = './Input/' + self.add1
        self.T1 = pd.read_csv(self.path+'Marker_input_Subject'+str(self.index)+'_RGF_1.txt',engine='python',delimiter=',',header=None)
        self.T2 = pd.read_csv(self.path+'Marker_input_Subject'+str(self.index)+'_RGF_2.txt',engine='python',delimiter=',',header=None)
        self.T3 = pd.read_csv(self.path+'Marker_input_Subject'+str(self.index)+'_RGF_3.txt',engine='python',delimiter=',',header=None)

        self = filt(self)

        # ## add time columns
        self.T1[57] = np.linspace(0, 1, self.T1.shape[0])
        self.T2[57] = np.linspace(0, 1, self.T2.shape[0])
        self.T3[57] = np.linspace(0, 1, self.T3.shape[0])
        
        self.all_list, self.all = combine(self,0)

    def plot(self):
        for i in range(57):
            for enumc, T in enumerate([self.T1, self.T2, self.T3]):
                plt.plot(T[57],T[i],color=color[enumc])
            plt.ylabel(i)
            plt.xlabel('# Frames')
            plt.show()
            plt.close()
            input()

class subject_out:
    def __init__(self, index, add1=''):
        self.index = index
        self.add1 = add1
        self.path = './Output/' + self.add1
        self.order = feat_order[0:3]
        self.label = {}
        self.label['JA'] = ['SFE',	'SAA',	'SIR',	'EFE',	'EPS',	'WFE'	,'WAA',	'TFE',	'TAA',	'TIR']
        self.label['JM'] = ['SacrumPelvisFlexionExtensionMoment'	,'SacrumPelvisAxialMoment'	,'SacrumPelvisLateralMoment',	'GlenoHumeralFlexion'	,'GlenoHumeralAbduction',	
                         'GlenoHumeralExternalRotation'	,'ElbowFlexion',	'ElbowPronation',	'WristFlexion',	'WristAbduction']
        self.label['JRF'] = ['TML'	,'TPD'	,'TAP',	'GML',	'GPD',	'GAP',	'EML',	'EPD'	,'EAP',	'WML',	'WPD',	'WAP']
        # self.label['MA']  = ['MA1',	'MA2',	'MA3',	'MA4']
        # self.label['MF']  = ['MF1',	'MF2',	'MF3',	'MF4']
        self.col_labels = self.label[self.order[0]] + self.label[self.order[1]] + self.label[self.order[2]]  #+ self.label['MA'] + self.label['MF'] 
        
        self.T1 = pd.concat([pd.read_csv(self.path+self.order[0]+'_Subject' +str(self.index)+'_RGF_1.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[1]+'_Subject' +str(self.index)+'_RGF_1.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[2]+'_Subject' +str(self.index)+'_RGF_1.txt',engine='python',header=None)#,
#                             Muscle_process(pd.read_csv(self.path+self.order[3]+'_Subject' +str(self.index)+'_RGF_1.txt',engine='python',header=None),self.order[3]),
#                             Muscle_process(pd.read_csv(self.path+self.order[4]+'_Subject' +str(self.index)+'_RGF_1.txt',engine='python',header=None),self.order[4])
                             ],axis=1)

        self.T2 = pd.concat([pd.read_csv(self.path+self.order[0]+'_Subject' +str(self.index)+'_RGF_2.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[1]+'_Subject' +str(self.index)+'_RGF_2.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[2]+'_Subject' +str(self.index)+'_RGF_2.txt',engine='python',header=None)#,
#                             Muscle_process(pd.read_csv(self.path+self.order[3]+'_Subject' +str(self.index)+'_RGF_2.txt',engine='python',header=None),self.order[3]),
#                             Muscle_process(pd.read_csv(self.path+self.order[4]+'_Subject' +str(self.index)+'_RGF_2.txt',engine='python',header=None),self.order[4])
                             ],axis=1)

        self.T3 = pd.concat([pd.read_csv(self.path+self.order[0]+'_Subject' +str(self.index)+'_RGF_3.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[1]+'_Subject' +str(self.index)+'_RGF_3.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[2]+'_Subject' +str(self.index)+'_RGF_3.txt',engine='python',header=None)#,
#                             Muscle_process(pd.read_csv(self.path+self.order[3]+'_Subject' +str(self.index)+'_RGF_3.txt',engine='python',header=None),self.order[3]),
#                             Muscle_process(pd.read_csv(self.path+self.order[4]+'_Subject' +str(self.index)+'_RGF_3.txt',engine='python',header=None),self.order[4])
                             ],axis=1)

        self.T1.columns, self.T2.columns, self.T3.columns = self.col_labels, self.col_labels, self.col_labels
        
        self = filt(self)
        self.weight = Weight.iloc[index-1][0]
        self.weight_moment = Weight_moment.iloc[index-1][0]
        self.subject_scale = [1]*10 + [self.weight_moment]*10 + [self.weight]*12 #+ [1]*4 + [self.weight]*4
        self.T1, self.T2, self.T3 = self.T1/self.subject_scale, self.T2/self.subject_scale, self.T3/self.subject_scale
        self.all_list, self.all = combine(self,0)

    def plot(self):
        for enum, lab in enumerate(self.col_labels):
            for enumc, T in enumerate([self.T1, self.T2, self.T3]):
                plt.plot(np.linspace(0,1,len(T[lab])),T[lab],color=color[enumc])
            plt.ylabel(feat_order_tmp[enum] + ' -- ' + lab)
            plt.xlabel('# Frames')
            plt.show()
            plt.close()
            input()

#################################################
# Classes to read brace data and compare
#################################################
def compare_braced_input_data():
    for i in range(11):
        u = subject_in(i+1)
        b = subject_in(i+1, 'Braced_')
        for t in range(19):
        # for t in [16]:
            for enum, (d1, d2) in enumerate(zip([u.T1, u.T2, u.T3], [b.T1, b.T2, b.T3])):
                plt.plot(d1[57], d1[t*3+0],color=color[enum])
                plt.plot(d1[57], d1[t*3+1],color=color[enum])
                plt.plot(d1[57], d1[t*3+2],color=color[enum])
                plt.plot(d2[57], d2[t*3+0],color=color[enum], ls = '--')
                plt.plot(d2[57], d2[t*3+1],color=color[enum], ls = '--')
                plt.plot(d2[57], d2[t*3+2],color=color[enum], ls = '--')
                # plt.plot(d2[t*3+2].index, d2[t*3+2],color=color[enum], ls = '--')
            plt.title("Subject " + str(i+1) + "   Marker" + str(t+1))
            plt.show()
            plt.close()
            input()
    return None

def compare_braced_output_data():
    for i in range(11):
        u = subject_out(i+1)
        b = subject_out(i+1, 'Braced_')
        print(u.col_labels)
        for enum, t in enumerate(u.col_labels[0:]):
            for enum, (d1, d2) in enumerate(zip([u.T1, u.T2, u.T3], [b.T1, b.T2, b.T3])):
                plt.plot(np.linspace(0,1,len(d1[t])), d1[t],color=color[enum])
                plt.plot(np.linspace(0,1,len(d2[t])), d2[t],color=color[enum], ls = '--')
            plt.title("Subject " + str(i+1) + "   " + t)
            plt.show()
            plt.close()
            input()
    return None

#################################################
# Initialising data class
#################################################

class cv_data:
    def __init__(self):
        self.cv1 = {}
        self.cv2 = {}
        self.cv3 = {}
        self.train_in = None
        self.train_out = None
        self.test_in = None
        self.test_out = None

class initiate_data:

    def __init__(self, add1=''):

        if add1[0:4] == 'Miss':
            add2 = ''
        else:
            add2 = add1

        self.i1,  self.o1  = subject_in(1,  add1), subject_out(1,  add2)
        self.i2,  self.o2  = subject_in(2,  add1), subject_out(2,  add2)
        self.i3,  self.o3  = subject_in(3,  add1), subject_out(3,  add2)
        self.i4,  self.o4  = subject_in(4,  add1), subject_out(4,  add2)
        self.i5,  self.o5  = subject_in(5,  add1), subject_out(5,  add2)   ## T2 and T3 are same
        self.i6,  self.o6  = subject_in(6,  add1), subject_out(6,  add2)   ## All three are same
        self.i7,  self.o7  = subject_in(7,  add1), subject_out(7,  add2)   ## T1 and T2 are same 
        self.i8,  self.o8  = subject_in(8,  add1), subject_out(8,  add2)
        self.i9,  self.o9  = subject_in(9,  add1), subject_out(9,  add2)   ## all are same
        self.i10, self.o10 = subject_in(10, add1), subject_out(10, add2)  ## T2 and T3 same
        self.i11, self.o11 = subject_in(11, add1), subject_out(11, add2)  ## all three are same

        self.inp     = [self.i1, self.i2, self.i3, self.i4, self.i5, self.i6, self.i7, self.i8, self.i9, self.i10, self.i11] 
        self.out     = [self.o1, self.o2, self.o3, self.o4, self.o5, self.o6, self.o7, self.o8, self.o9, self.o10, self.o11] 
        self.inp_all = [self.i1.all, self.i2.all, self.i3.all, self.i4.all, self.i5.all, self.i6.all, self.i7.all, self.i8.all, 
                        self.i9.all, self.i10.all, self.i11.all]  
        self.out_all = [self.o1.all, self.o2.all, self.o3.all, self.o4.all, self.o5.all, self.o6.all, self.o7.all, self.o8.all, 
                        self.o9.all, self.o10.all, self.o11.all]  
        self.inp_all_list = [self.i1.all_list, self.i2.all_list, self.i3.all_list, self.i4.all_list, self.i5.all_list, self.i6.all_list, self.i7.all_list, self.i8.all_list, 
                        self.i9.all_list, self.i10.all_list, self.i11.all_list] 
        self.out_all_list = [self.o1.all_list, self.o2.all_list, self.o3.all_list, self.o4.all_list, self.o5.all_list, self.o6.all_list, self.o7.all_list, self.o8.all_list, 
                        self.o9.all_list, self.o10.all_list, self.o11.all_list] 

        if add1 != 'Braced_':
            self.i12, self.o12 = subject_in(12, add1), subject_out(12, add2) 
            self.i13, self.o13 = subject_in(13, add1), subject_out(13, add2)
            self.i14, self.o14 = subject_in(14, add1), subject_out(14, add2)
            self.i15, self.o15 = subject_in(15, add1), subject_out(15, add2)
            self.i16, self.o16 = subject_in(16, add1), subject_out(16, add2)

            self.inp     += [self.i12, self.i13, self.i14, self.i15, self.i16]
            self.out     += [self.o12, self.o13, self.o14, self.o15, self.o16]
            self.inp_all += [self.i12.all, self.i13.all, self.i14.all, self.i15.all, self.i16.all]
            self.out_all += [self.o12.all, self.o13.all, self.o14.all, self.o15.all, self.o16.all]
            self.inp_all_list += [self.i12.all_list, self.i13.all_list, self.i14.all_list, self.i15.all_list, self.i16.all_list]
            self.out_all_list += [self.o12.all_list, self.o13.all_list, self.o14.all_list, self.o15.all_list, self.o16.all_list]


        self.col_labels = self.o1.col_labels
        self.label = self.o1.label
        self.data_class = 'normal'
        self.std_out = pd.concat(self.out_all).std()
        self.std_dummy = copy.deepcopy(self.std_out)
        self.std_dummy[self.col_labels] = np.ones(32)

    def plot(self):
        for e, label  in enumerate(self.col_labels):
            fig,ax = plt.subplots()
            for enum,data in enumerate(self.out):
                for enumls, T in enumerate([data.T1, data.T2, data.T3]):
                    plt.plot(np.linspace(0,1,len(T[label])),T[label],lw =1,color=color[enum],ls=ls[enumls])
            plt.xlabel('# Frames')
            plt.ylabel(feat_order_tmp[e] + ' -- ' + label)
            plt.show()
            plt.close()
            input()

    def subject_naive(self,feature, norm_out):
        cv = cv_data()
        cv.feature = feature
        cv.subject = 'naive'
        cv.data_class = self.data_class
        sub_col = self.label[feature]
        cv.sub_col = sub_col
        ## normalize the outputs features
        if int(norm_out):
            scale = self.std_out[sub_col]
            print('Normalizing outputs')
        else:
            scale = self.std_dummy[sub_col]
            print('Crude outputs')
        ## held-out test data 2, 5, 9, 15
        ## remaining data list 1,3,4, 6,7,8, 10,11,12, 13,14, 16
        HO = [1, 4, 8, 14]  #python indexing
        shuffled = [12,  0, 10,  7, 9, 11,  2,  6,  5, 15, 13,  3]
                    
        V1, T1 = shuffled[0:4],   shuffled[4:12]
        V2, T2 = shuffled[4:8],   shuffled[0:4] + shuffled[8:12]
        V3, T3 = shuffled[8:12],  shuffled[0:8] 

        if self.data_class in ['LM','normal']:
            cv.cv1['train_in']  = pd.concat([self.inp_all[i] for i in T1])
            cv.cv1['val_in']    = pd.concat([self.inp_all[i] for i in V1])
            cv.cv2['train_in']  = pd.concat([self.inp_all[i] for i in T2])
            cv.cv2['val_in']    = pd.concat([self.inp_all[i] for i in V2])
            cv.cv3['train_in']  = pd.concat([self.inp_all[i] for i in T3])
            cv.cv3['val_in']    = pd.concat([self.inp_all[i] for i in V3])
            cv.train_in_list    = [self.inp_all[i] for i in shuffled]
            cv.train_in         = pd.concat(cv.train_in_list)
            cv.test_in          = pd.concat([self.inp_all[i] for i in HO])
            cv.test_in_list     = [self.inp_all_list[i] for i in HO]
            cv.test_in_list     = [j for i in cv.test_in_list for j in i]  ## seperating each trials

        elif self.data_class in ['RNN','CNN','CNNLSTM']:
            cv.cv1['train_in']  = np.concatenate([self.inp_all[i] for i in T1])
            cv.cv1['val_in']    = np.concatenate([self.inp_all[i] for i in V1])
            cv.cv2['train_in']  = np.concatenate([self.inp_all[i] for i in T2])
            cv.cv2['val_in']    = np.concatenate([self.inp_all[i] for i in V2])
            cv.cv3['train_in']  = np.concatenate([self.inp_all[i] for i in T3])
            cv.cv3['val_in']    = np.concatenate([self.inp_all[i] for i in V3])
            cv.train_in_list    = [self.inp_all[i] for i in shuffled]
            cv.train_in         = np.concatenate(cv.train_in_list)
            cv.test_in          = np.concatenate([self.inp_all[i] for i in HO])
            cv.test_in_list     = [self.inp_all_list[i] for i in HO]
            cv.test_in_list     = [j for i in cv.test_in_list for j in i]  ## seperating each trials
        else:
            print("incorrect data class")
            sys.exit()

        cv.cv1['train_out'] = pd.concat([self.out_all[i] for i in T1])[sub_col]/scale 
        cv.cv1['val_out']   = pd.concat([self.out_all[i] for i in V1])[sub_col]/scale
        cv.cv2['train_out'] = pd.concat([self.out_all[i] for i in T2])[sub_col]/scale 
        cv.cv2['val_out']   = pd.concat([self.out_all[i] for i in V2])[sub_col]/scale
        cv.cv3['train_out'] = pd.concat([self.out_all[i] for i in T3])[sub_col]/scale 
        cv.cv3['val_out']   = pd.concat([self.out_all[i] for i in V3])[sub_col]/scale

        cv.train_out_list = [self.out_all[i][sub_col] for i in shuffled]
        cv.train_out = pd.concat(cv.train_out_list)[sub_col]/scale

        cv.test_out  = pd.concat([self.out_all[i] for i in HO])[sub_col]/scale

        cv.test_out_list  = [self.out_all_list[i] for i in HO] #
        cv.test_out_list  = [j[sub_col]/scale for i in cv.test_out_list for j in i]  ## seperating each trials

        cv.time = cv.test_in[57]

        return cv 

    def subject_exposed(self, feature, norm_out):
        cv = cv_data()
        cv.feature = feature
        cv.subject = 'exposed'
        cv.data_class = self.data_class
        sub_col = self.label[feature]

        if int(norm_out):
            scale = self.std_out[sub_col]
            print('Normalizing outputs')
        else:
            scale = self.std_dummy[sub_col]
            print('Crude outputs')

        cv.sub_col = sub_col

        ## python indexing
        ## [0,1,2,3,7,11,12,13,14,15] -- T1, T2, T3
        ## [5,8,10]  --- T1
        ## [4,9]    --- T1,T2
        ## [6]      --- T1,T3
        ## super held-out test data 5,8,10

        ## held-out test data 
        ## remaining data list 1,3,4, 6,7,8, 10,11,12, 13,14, 16
        HO = [0,1,2,3,7,11,12,13,14,15]  #Trial1, python indexing
        rem1 = [5,8,10]  #super held-out test data
        rem2 = [4,9]
        rem3 = [6]

        cv.train_in_list  = [self.inp[i].T2 for i in HO] + [self.inp[i].T3 for i in HO] + [self.inp[i].T1 for i in rem2] + [self.inp[i].T2 for i in rem2] + [self.inp[i].T1 for i in rem3] + [self.inp[i].T3 for i in rem3] 
        cv.train_out_list = [self.out[i].T2[sub_col] for i in HO] + [self.out[i].T3[sub_col] for i in HO]   + [self.out[i].T1[sub_col] for i in rem2]  + [self.out[i].T2[sub_col] for i in rem2] + [self.out[i].T1[sub_col] for i in rem3] + [self.out[i].T3[sub_col] for i in rem3] 

        T1_in  = [self.inp[i].T3 for i in HO] + [self.inp[i].T1 for i in rem2] + [self.inp[i].T2 for i in rem2] + [self.inp[i].T1 for i in rem3] + [self.inp[i].T3 for i in rem3] 
        T1_out = [self.out[i].T3 for i in HO] + [self.out[i].T1 for i in rem2] + [self.out[i].T2 for i in rem2] + [self.out[i].T1 for i in rem3] + [self.out[i].T3 for i in rem3] 
        V1_in  = [ self.inp[i].T2 for i in HO] 
        V1_out = [self.out[i].T2 for i in HO] 

        T2_in  = [self.inp[i].T2 for i in HO] + [self.inp[i].T1 for i in rem2] + [self.inp[i].T2 for i in rem2] + [self.inp[i].T1 for i in rem3] + [self.inp[i].T3 for i in rem3] 
        T2_out = [self.out[i].T2 for i in HO] + [self.out[i].T1 for i in rem2] + [self.out[i].T2 for i in rem2] + [self.out[i].T1 for i in rem3] + [self.out[i].T3 for i in rem3] 
        V2_in  = [self.inp[i].T3 for i in HO] 
        V2_out = [self.out[i].T3 for i in HO] 
                    
        T3_in  = [self.inp[i].T2 for i in HO]   + [self.inp[i].T3 for i in HO] + [self.inp[i].T2 for i in rem2] + [self.inp[i].T3 for i in rem3] 
        T3_out = [self.out[i].T2 for i in HO]   + [self.out[i].T3 for i in HO] + [self.out[i].T2 for i in rem2] + [self.out[i].T3 for i in rem3] 
        V3_in  = [self.inp[i].T1 for i in rem2] + [self.inp[i].T1 for i in rem3]
        V3_out = [self.out[i].T1 for i in rem2] + [self.out[i].T1 for i in rem3]

        if self.data_class in ['LM','normal']:
            cv.cv1['train_in']  = pd.concat(T1_in)
            cv.cv1['val_in']    = pd.concat(V1_in)
            cv.cv2['train_in']  = pd.concat(T2_in)
            cv.cv2['val_in']    = pd.concat(V2_in)
            cv.cv3['train_in']  = pd.concat(T3_in)
            cv.cv3['val_in']    = pd.concat(V3_in)
            cv.train_in         = pd.concat(cv.train_in_list)
            cv.test_in          = pd.concat([self.inp[i].T1  for i in HO])
            cv.test_in_list          = [self.inp[i].T1  for i in HO]
            cv.super_test_in_list    = [self.inp[i].T1  for i in rem1]

        elif self.data_class in ['RNN','CNN','CNNLSTM']:
            cv.cv1['train_in']  = np.concatenate(T1_in)
            cv.cv1['val_in']    = np.concatenate(V1_in)
            cv.cv2['train_in']  = np.concatenate(T2_in)
            cv.cv2['val_in']    = np.concatenate(V2_in)
            cv.cv3['train_in']  = np.concatenate(T3_in)
            cv.cv3['val_in']    = np.concatenate(V3_in)
            cv.train_in         = np.concatenate(cv.train_in_list)
            cv.test_in          = np.concatenate([self.inp[i].T1  for i in HO])
            cv.test_in_list          = [self.inp[i].T1  for i in HO]
        else:
            print("incorrect data class")
            sys.exit()

        cv.cv1['train_out'] = pd.concat(T1_out)[sub_col]/scale 
        cv.cv1['val_out']   = pd.concat(V1_out)[sub_col]/scale
        cv.cv2['train_out'] = pd.concat(T2_out)[sub_col]/scale 
        cv.cv2['val_out']   = pd.concat(V2_out)[sub_col]/scale
        cv.cv3['train_out'] = pd.concat(T3_out)[sub_col]/scale 
        cv.cv3['val_out']   = pd.concat(V3_out)[sub_col]/scale

        cv.train_out           = pd.concat(cv.train_out_list)[sub_col]/scale
        cv.test_out            = pd.concat([self.out[i].T1 for i in HO])[sub_col]/scale         
        cv.test_out_list       = [self.out[i].T1[sub_col]/scale for i in HO]         
        cv.super_test_out_list = [self.out[i].T1[sub_col]/scale for i in rem1]
        
        cv.time = cv.test_in[57]
        return cv 


class initiate_RNN_data(initiate_data):
    def __init__(self, window_size, add1=''):
        initiate_data.__init__(self, add1)
        self.data_class = 'RNN'
        self.window = window_size
        self.i1, self.o1 = transform_subject_into_windows(self.i1, self.o1, window_size)
        self.i2, self.o2 = transform_subject_into_windows(self.i2, self.o2, window_size)
        self.i3, self.o3 = transform_subject_into_windows(self.i3, self.o3, window_size)
        self.i4, self.o4 = transform_subject_into_windows(self.i4, self.o4, window_size)
        self.i5, self.o5 = transform_subject_into_windows(self.i5, self.o5, window_size)
        self.i6, self.o6 = transform_subject_into_windows(self.i6, self.o6, window_size)
        self.i7, self.o7 = transform_subject_into_windows(self.i7, self.o7, window_size)
        self.i8, self.o8 = transform_subject_into_windows(self.i8, self.o8, window_size)
        self.i9, self.o9 = transform_subject_into_windows(self.i9, self.o9, window_size)
        self.i10, self.o10 = transform_subject_into_windows(self.i10, self.o10, window_size)
        self.i11, self.o11 = transform_subject_into_windows(self.i11, self.o11, window_size)

        self.inp     = [self.i1, self.i2, self.i3, self.i4, self.i5, self.i6, self.i7, self.i8, self.i9, self.i10, self.i11] 
        self.out     = [self.o1, self.o2, self.o3, self.o4, self.o5, self.o6, self.o7, self.o8, self.o9, self.o10, self.o11] 
        self.inp_all = [self.i1.all, self.i2.all, self.i3.all, self.i4.all, self.i5.all, self.i6.all, self.i7.all, self.i8.all, 
                        self.i9.all, self.i10.all, self.i11.all]  
        self.out_all = [self.o1.all, self.o2.all, self.o3.all, self.o4.all, self.o5.all, self.o6.all, self.o7.all, self.o8.all, 
                        self.o9.all, self.o10.all, self.o11.all]  
        self.inp_all_list = [self.i1.all_list, self.i2.all_list, self.i3.all_list, self.i4.all_list, self.i5.all_list, self.i6.all_list, self.i7.all_list, self.i8.all_list, 
                        self.i9.all_list, self.i10.all_list, self.i11.all_list] 
        self.out_all_list = [self.o1.all_list, self.o2.all_list, self.o3.all_list, self.o4.all_list, self.o5.all_list, self.o6.all_list, self.o7.all_list, self.o8.all_list, 
                        self.o9.all_list, self.o10.all_list, self.o11.all_list] 

        if add1 != 'Braced_':
            self.i12, self.o12 = transform_subject_into_windows(self.i12, self.o12, window_size)
            self.i13, self.o13 = transform_subject_into_windows(self.i13, self.o13, window_size)
            self.i14, self.o14 = transform_subject_into_windows(self.i14, self.o14, window_size)
            self.i15, self.o15 = transform_subject_into_windows(self.i15, self.o15, window_size)
            self.i16, self.o16 = transform_subject_into_windows(self.i16, self.o16, window_size)

            self.inp     += [self.i12, self.i13, self.i14, self.i15, self.i16]
            self.out     += [self.o12, self.o13, self.o14, self.o15, self.o16]
            self.inp_all += [self.i12.all, self.i13.all, self.i14.all, self.i15.all, self.i16.all]
            self.out_all += [self.o12.all, self.o13.all, self.o14.all, self.o15.all, self.o16.all]
            self.inp_all_list += [self.i12.all_list, self.i13.all_list, self.i14.all_list, self.i15.all_list, self.i16.all_list]
            self.out_all_list += [self.o12.all_list, self.o13.all_list, self.o14.all_list, self.o15.all_list, self.o16.all_list]


#################################################
# Classes used for analysis
#################################################

class analysis_options:
    def __init__(self, what=None):
        self.what = what


class subject:
    def __init__(self, what, data, hyper, kind):
        self.kind    = kind
        self.subject =  what
        self.arg     =  None
        self.arch    =  None
        self.nparams =  []
        self.feature = feat_order
        self.feature = feat_order[0:3]
        self.feature_l   = feat_order_l
        self.feature_l2  = feat_order_l2
        self.NRMSE   =  {key: None for key in self.feature}
        self.RMSE    =  {key: None for key in self.feature}
        self.pc      =  {key: None for key in self.feature}
        self.data    =  data
        self.hyper   = hyper        
        
class ML:
    def __init__(self, what, window):

        if what == 'NN':
            self.data  = initiate_data()
            self.hyper = pd.read_csv('./hyperparameters/hyperparam_NN.txt',delimiter='\s+')
        elif what == 'LM':
            self.data  = initiate_data()
            self.hyper = pd.read_csv('./hyperparameters/hyperparam_LM.txt',delimiter='\s+')
        elif what == 'rf':
            self.data  = initiate_data()
            self.hyper = pd.read_csv('./hyperparameters/hyperparam_rf.txt',delimiter='\s+')
        elif what == 'GBRT':
            self.data  = initiate_data()
            self.hyper = pd.read_csv('./hyperparameters/hyperparam_GBRT.txt',delimiter='\s+')
        elif what == 'RNN':
            self.data  = initiate_RNN_data(window_size=window)
            self.hyper = pd.read_csv('./hyperparameters/hyperparam_RNN.txt',delimiter='\s+')
        elif what == 'CNN':
            self.data  = initiate_RNN_data(window_size=window)
            self.hyper = pd.read_csv('./hyperparameters/hyperparam_CNN.txt',delimiter='\s+')
        elif what == 'CNNLSTM':
            self.data  = initiate_RNN_data(window_size=window)
            self.hyper = pd.read_csv('./hyperparameters/hyperparam_CNNLSTM.txt',delimiter='\s+')
        elif what == 'convLSTM':
            self.data  = initiate_RNN_data(window_size=window)
            self.hyper = pd.read_csv('./hyperparameters/hyperparam_CNN.txt',delimiter='\s+')

        self.what = what
        self.exposed        =  subject('exposed', self.data, self.hyper, self.what)
        self.naive          =  subject('naive'  , self.data, self.hyper, self.what)
        self.feature = feat_order
        self.feature = feat_order[0:3]
        self.feature_l   = feat_order_l
        self.feature_l2  = feat_order_l2

class ML_analysis:
    def __init__(self, what, data_kind, window):
        self.what = what
        
        if 'LM' in data_kind:
            self.LM  = ML('LM', window)
        if 'rf' in data_kind:
            self.rf  = ML('rf', window)
        if 'GBRT' in data_kind:
            self.GBRT  = ML('GBRT', window)
        if 'NN' in data_kind:
            self.NN  = ML('NN', window)
        if 'RNN' in data_kind:
            self.RNN = ML('RNN', window)
        if 'CNN' in data_kind:
            self.CNN = ML('CNN', window)
        if 'CNNLSTM' in data_kind:
            self.CNNLSTM = ML('CNNLSTM', window)
        if 'convLSTM' in data_kind:
            self.convLSTM = ML('convLSTM', window)

        self.feature = feat_order
        self.feature = feat_order[0:3]
        self.feature_l   = feat_order_l
        self.feature_l2  = feat_order_l2
        

#################################################
#################################################
# Functions to create missing marker data or adding noise into the data
#################################################
#################################################

### During the experiments, some of the marker data is occulued and leads to missing input data
### Here, we will first artifically and randomly create such data sets ..

def intro_nan(data, prob_of_missing, number_of_miss):
    ## 19 markers in the input data
    markers = data.columns.shape[0]//3
    # prob_of_missing i.e. how many frames should have missing markers
    for i in data.index:
        if np.random.binomial(1, prob_of_missing):
            ## how many marker to miss
            marker_index = np.random.choice(markers, number_of_miss, replace=False)
            for m in marker_index:
                marker_ind = np.arange(3*m,3*m+3)
                data.loc[i][marker_ind] = [np.nan]*3
    return data

def create_artifical_input_data_with_missing_markers():
    path = './Input/'
    prob_of_missing = 0.1
    number_of_miss = 1
    for index in np.arange(1, 17, 1):
        for trial in np.arange(1, 4, 1):
            data = pd.read_csv(path+'Marker_input_Subject'+str(index)+'_RGF_'+str(trial)+'.txt',engine='python',delimiter=',',header=None)
            data = intro_nan(data, prob_of_missing, number_of_miss)
            data.to_csv(path+'Miss_1_Marker_input_Subject'+str(index)+'_RGF_'+str(trial) + '.txt',  sep=',', index=False, header=None)
    return data

#data = create_artifical_input_data_with_missing_markers()
        
def continuous_noise(t):
    l = np.shape(t)[0]
    ### need to pick a maximum values for A, w, Phi  
    A = 10
    w = 2*np.pi*6   ## 6Hz 
    phi = 2*np.pi
    tmpA = A*np.random.rand(l)
    tmpw = w*np.random.rand(l)
    tmpp = phi*np.random.rand(l)
    # np.sin(np.pi/2) = 1 ---> input is radian
    f = tmpA*np.sin(tmpw*t+tmpp)
    return f 

def add_noise_to_trial(T_in):
    T = copy.deepcopy(T_in)
    time_col = T.columns[-1]
    samples = T.shape[0]
    for col in T.columns[:-1]:
        cn = continuous_noise(T[time_col])
        offset = np.random.normal(0, 2, 1) #https://www.mdpi.com/2075-1729/12/6/819#B15-life-12-00819
        OMC_noise = np.random.normal(0, 2.89, samples) #https://www.sciencedirect.com/science/article/pii/S0966636204000682  ## 1-5 mm
        T[col] = T[col] + cn + np.full(samples, offset)
        # plt.plot(cn)
        # plt.show()
        # plt.close()
    return T

       