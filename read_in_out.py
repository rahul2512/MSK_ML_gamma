### first section to read the marker data
import numpy as np, pandas as pd, copy
from scipy.interpolate import interp1d
import scipy.io as sio, sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


filters = pd.read_csv('./Output/frame_filters', header=None)
Weight = pd.read_csv('./Output/Weight', header=None)
Weight_moment = pd.read_csv('./Output/Weight_moment', header=None)
color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 
         'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k', 'teal',
         'deeppink','goldenrod','darkred','darkviolet']
ls = ['-','--',':']

def transform_data_into_windows(d1,o1,window_size):
    s0,s1 = d1.shape
    tmp = np.zeros([s0-window_size,window_size,s1])
    for i in d1.index.to_list():
        if i > window_size-1:
            tmp[i-window_size] = d1.iloc[i-window_size:i].to_numpy()

    tmpo = o1.iloc[window_size::]
    return tmp, tmpo

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

def combine(d):
    if d.index in [1,2,3,4,8,12,13,14,15,16]:
        u = pd.concat([d.T1,d.T2,d.T3])
    elif d.index in [6,9,11]:
        u = d.T1
    elif d.index in [5,10]:
        u = pd.concat([d.T1,d.T2])
    elif d.index in [7]:
        u = pd.concat([d.T1,d.T3])
    else:
        print("unrecognised index")
        sys.exit()            
    return u

def filt(d):
    d.filter = filters.iloc[d.index-1] 
    d.T1 = d.T1.iloc[d.filter[0]:d.filter[1]+1]
    d.T2 = d.T2.iloc[d.filter[2]:d.filter[3]+1]
    d.T3 = d.T3.iloc[d.filter[4]:d.filter[5]+1]
    return d

class subject_in:
    def __init__(self,index):
        self.index = index
        self.path = './Input/'
        self.T1 = pd.read_csv(self.path+'Marker_input_Subject'+str(self.index)+'_RGF_1.txt',engine='python',delimiter=',',header=None)
        self.T2 = pd.read_csv(self.path+'Marker_input_Subject'+str(self.index)+'_RGF_2.txt',engine='python',delimiter=',',header=None)
        self.T3 = pd.read_csv(self.path+'Marker_input_Subject'+str(self.index)+'_RGF_3.txt',engine='python',delimiter=',',header=None)

        self = filt(self)

        # ## add time columns
        self.T1[57] = np.linspace(0, 1, self.T1.shape[0])
        self.T2[57] = np.linspace(0, 1, self.T2.shape[0])
        self.T3[57] = np.linspace(0, 1, self.T3.shape[0])
        
        self.all = combine(self)

        # ## add dummy variables for 
        # self.RNN_T1 = None
        # self.RNN_T2 = None
        # self.RNN_T3 = None

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
    def __init__(self,index):
        self.index = index
        self.path = './Output/'
        self.order = ['JA','JM','JRF','MA','MF']
        self.label = {}
        self.label['JA'] = ['SFE',	'SAA',	'SIR',	'EFE',	'EPS',	'WFE'	,'WAA',	'TFE',	'TAA',	'TIR']
        self.label['JM'] = ['SacrumPelvisFlexionExtensionMoment'	,'SacrumPelvisAxialMoment'	,'SacrumPelvisLateralMoment',	'GlenoHumeralFlexion'	,'GlenoHumeralAbduction',	
                         'GlenoHumeralExternalRotation'	,'ElbowFlexion',	'ElbowPronation',	'WristFlexion',	'WristAbduction']
        self.label['JRF'] = ['TML'	,'TPD'	,'TAP',	'GML',	'GPD',	'GAP',	'EML',	'EPD'	,'EAP',	'WML',	'WPD',	'WAP']
        self.label['MA']  = ['MA1',	'MA2',	'MA3',	'MA4']
        self.label['MF']  = ['MF1',	'MF2',	'MF3',	'MF4']
        self.col_labels = self.label['JA'] + self.label['JM'] + self.label['JRF'] + self.label['MA'] + self.label['MF'] 
        
        self.T1 = pd.concat([pd.read_csv(self.path+self.order[0]+'_Subject' +str(self.index)+'_RGF_1.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[1]+'_Subject' +str(self.index)+'_RGF_1.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[2]+'_Subject' +str(self.index)+'_RGF_1.txt',engine='python',header=None),
                             Muscle_process(pd.read_csv(self.path+self.order[3]+'_Subject' +str(self.index)+'_RGF_1.txt',engine='python',header=None),self.order[3]),
                             Muscle_process(pd.read_csv(self.path+self.order[4]+'_Subject' +str(self.index)+'_RGF_1.txt',engine='python',header=None),self.order[4])
                             ],axis=1)

        self.T2 = pd.concat([pd.read_csv(self.path+self.order[0]+'_Subject' +str(self.index)+'_RGF_2.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[1]+'_Subject' +str(self.index)+'_RGF_2.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[2]+'_Subject' +str(self.index)+'_RGF_2.txt',engine='python',header=None),
                             Muscle_process(pd.read_csv(self.path+self.order[3]+'_Subject' +str(self.index)+'_RGF_2.txt',engine='python',header=None),self.order[3]),
                             Muscle_process(pd.read_csv(self.path+self.order[4]+'_Subject' +str(self.index)+'_RGF_2.txt',engine='python',header=None),self.order[4])
                             ],axis=1)

        self.T3 = pd.concat([pd.read_csv(self.path+self.order[0]+'_Subject' +str(self.index)+'_RGF_3.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[1]+'_Subject' +str(self.index)+'_RGF_3.txt',engine='python',header=None),
                             pd.read_csv(self.path+self.order[2]+'_Subject' +str(self.index)+'_RGF_3.txt',engine='python',header=None),
                             Muscle_process(pd.read_csv(self.path+self.order[3]+'_Subject' +str(self.index)+'_RGF_3.txt',engine='python',header=None),self.order[3]),
                             Muscle_process(pd.read_csv(self.path+self.order[4]+'_Subject' +str(self.index)+'_RGF_3.txt',engine='python',header=None),self.order[4])
                             ],axis=1)

        self.T1.columns, self.T2.columns, self.T3.columns = self.col_labels, self.col_labels, self.col_labels
        
        self = filt(self)
        self.weight = Weight.iloc[index-1][0]
        self.weight_moment = Weight_moment.iloc[index-1][0]
        self.subject_scale = [1]*10 + [self.weight_moment]*10 + [self.weight]*12 + [1]*4 + [self.weight]*4
        self.T1, self.T2, self.T3 = self.T1/self.subject_scale, self.T2/self.subject_scale, self.T3/self.subject_scale
        self.all = combine(self)

        # self.RNN_T1 = None
        # self.RNN_T2 = None
        # self.RNN_T3 = None

    def plot(self):
        for enum, lab in enumerate(self.col_labels):
            for enumc, T in enumerate([self.T1, self.T2, self.T3]):
                plt.plot(np.linspace(0,1,len(T[lab])),T[lab],color=color[enumc])
            plt.ylabel(lab)
            plt.xlabel('# Frames')
            plt.show()
            plt.close()
            input()


class cv_data:
    def __init__(self):
        self.cv1 = None
        self.cv2 = None
        self.cv3 = None
        self.train_in = None
        self.train_out = None
        self.test_in = None
        self.test_out = None

class initiate_data:

    def __init__(self):
        self.i1,  self.o1  = subject_in(1),  subject_out(1)
        self.i2,  self.o2  = subject_in(2),  subject_out(2)
        self.i3,  self.o3  = subject_in(3),  subject_out(3)
        self.i4,  self.o4  = subject_in(4),  subject_out(4)
        self.i5,  self.o5  = subject_in(5),  subject_out(5)   ## T2 and T3 are same
        self.i6,  self.o6  = subject_in(6),  subject_out(6)   ## All three are same
        self.i7,  self.o7  = subject_in(7),  subject_out(7)   ## T1 and T2 are same 
        self.i8,  self.o8  = subject_in(8),  subject_out(8)
        self.i9,  self.o9  = subject_in(9),  subject_out(9)   ## all are same
        self.i10, self.o10 = subject_in(10), subject_out(10)  ## T2 and T3 same
        self.i11, self.o11 = subject_in(11), subject_out(11)  ## all three are same
        self.i12, self.o12 = subject_in(12), subject_out(12) 
        self.i13, self.o13 = subject_in(13), subject_out(13)
        self.i14, self.o14 = subject_in(14), subject_out(14)
        self.i15, self.o15 = subject_in(15), subject_out(15)
        self.i16, self.o16 = subject_in(16), subject_out(16)
                
        self.inp = [self.i1, self.i2, self.i3, self.i4, self.i5, self.i6, self.i7, self.i8, self.i9, self.i10, self.i11, self.i12, self.i13, self.i14, self.i15, self.i16]
        self.out = [self.o1, self.o2, self.o3, self.o4, self.o5, self.o6, self.o7, self.o8, self.o9, self.o10, self.o11, self.o12, self.o13, self.o14, self.o15, self.o16]

        self.inp_all = [self.i1.all, self.i2.all, self.i3.all, self.i4.all, self.i5.all, self.i6.all, self.i7.all, self.i8.all, 
                        self.i9.all, self.i10.all, self.i11.all, self.i12.all, self.i13.all, self.i14.all, self.i15.all, self.i16.all]
        self.out_all = [self.o1.all, self.o2.all, self.o3.all, self.o4.all, self.o5.all, self.o6.all, self.o7.all, self.o8.all, 
                        self.o9.all, self.o10.all, self.o11.all, self.o12.all, self.o13.all, self.o14.all, self.o15.all, self.o16.all]

        self.col_labels = self.o1.col_labels
        self.label = self.o1.label
        self.data_class = 'normal'
        self.std_out = pd.concat(self.out_all).std()
        self.std_dummy = copy.deepcopy(self.std_out)
        self.std_dummy[self.col_labels] = np.ones(40)

    def plot(self):
        for label  in self.col_labels:
            fig,ax = plt.subplots()
            for enum,data in enumerate(self.out):
                for enumls, T in enumerate([data.T1, data.T2, data.T3]):
                    plt.plot(np.linspace(0,1,len(T[label])),T[label],lw =1,color=color[enum],ls=ls[enumls])
            plt.xlabel('# Frames')
            plt.ylabel(label)
            plt.show()
            plt.close()
            input()

    def subject_naive(self,feature):
        cv = cv_data()
        cv.feature = feature
        cv.subject = 'naive'
        cv.data_class = 'normal'
        sub_col = self.label[feature]
        scale = self.std_out[sub_col]
        scale = self.std_dummy[sub_col]
        ## held-out test data 2, 5, 9, 15
        ## remaining data list 1,3,4, 6,7,8, 10,11,12, 13,14, 16
        HO = [1, 4, 8, 14]  #python indexing
        shuffled = [12,  0, 10,  7, 9, 11,  2,  6,  5, 15, 13,  3]
                    
        V1, T1 = shuffled[0:4],   shuffled[4:12]
        V2, T2 = shuffled[4:8],   shuffled[0:4] + shuffled[8:12]
        V3, T3 = shuffled[8:12],  shuffled[0:8] 
        
        cv.cv1 = {'train_in': pd.concat([self.inp_all[i] for i in T1]), 
                  'train_out':pd.concat([self.out_all[i] for i in T1])[sub_col]/scale, 
                  'val_in':   pd.concat([self.inp_all[i] for i in V1]),
                  'val_out':  pd.concat([self.out_all[i] for i in V1])[sub_col]/scale
                  }
        cv.cv2 = {'train_in': pd.concat([self.inp_all[i] for i in T2]), 
                  'train_out':pd.concat([self.out_all[i] for i in T2])[sub_col]/scale, 
                  'val_in':   pd.concat([self.inp_all[i] for i in V2]),
                  'val_out':  pd.concat([self.out_all[i] for i in V2])[sub_col]/scale
                  }
        cv.cv3 = {'train_in': pd.concat([self.inp_all[i] for i in T3]), 
                  'train_out':pd.concat([self.out_all[i] for i in T3])[sub_col]/scale, 
                  'val_in':   pd.concat([self.inp_all[i] for i in V3]),
                  'val_out':  pd.concat([self.out_all[i] for i in V3])[sub_col]/scale
                  }

        cv.train_in  = pd.concat([self.inp_all[i] for i in shuffled])
        cv.train_out = pd.concat([self.out_all[i] for i in shuffled])[sub_col]/scale
        cv.test_in   = pd.concat([self.inp_all[i] for i in HO])
        cv.test_out  = pd.concat([self.out_all[i] for i in HO])[sub_col]/scale

        return cv 

    def subject_exposed(self, feature):
        cv = cv_data()
        cv.feature = feature
        cv.subject = 'exposed'
        cv.data_class = 'normal'
        sub_col = self.label[feature]
        scale = self.std_out[sub_col]
        scale = self.std_dummy[sub_col]

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
        cv.test_in   = pd.concat([self.inp[i].T1 for i in HO])
        cv.test_out  = pd.concat([self.out[i].T1 for i in HO])[sub_col]/scale

        train_in_list  = [self.inp[i].T2 for i in HO] + [self.inp[i].T3 for i in HO] + [self.inp[i].T1 for i in rem2] + [self.inp[i].T2 for i in rem2] + [self.inp[i].T1 for i in rem3] + [self.inp[i].T3 for i in rem3] 
        train_out_list = [self.out[i].T2 for i in HO] + [self.out[i].T3 for i in HO] + [self.out[i].T1 for i in rem2] + [self.out[i].T2 for i in rem2] + [self.out[i].T1 for i in rem3] + [self.out[i].T3 for i in rem3] 
                                 
        cv.train_in  = pd.concat(train_in_list)
        cv.train_out  = pd.concat(train_out_list)[sub_col]/scale
        
        T1_in  = [self.inp[i].T3 for i in HO] + [self.inp[i].T1 for i in rem2] + [self.inp[i].T2 for i in rem2] + [self.inp[i].T1 for i in rem3] + [self.inp[i].T3 for i in rem3] 
        T1_out = [self.out[i].T3 for i in HO] + [self.out[i].T1 for i in rem2] + [self.out[i].T2 for i in rem2] + [self.out[i].T1 for i in rem3] + [self.out[i].T3 for i in rem3] 
        V1_in  = [ self.inp[i].T2 for i in HO] 
        V1_out = [self.out[i].T2 for i in HO] 

        T2_in  = [self.inp[i].T2 for i in HO] + [self.inp[i].T1 for i in rem2] + [self.inp[i].T2 for i in rem2] + [self.inp[i].T1 for i in rem3] + [self.inp[i].T3 for i in rem3] 
        T2_out = [self.out[i].T2 for i in HO] + [self.out[i].T1 for i in rem2] + [self.out[i].T2 for i in rem2] + [self.out[i].T1 for i in rem3] + [self.out[i].T3 for i in rem3] 
        V2_in  = [self.inp[i].T3 for i in HO] 
        V2_out = [self.out[i].T3 for i in HO] 
                    
        T3_in  = [self.inp[i].T2 for i in HO] + [self.inp[i].T3 for i in HO] + [self.inp[i].T2 for i in rem2]  + [self.inp[i].T3 for i in rem3] 
        T3_out = [self.out[i].T2 for i in HO] + [self.out[i].T3 for i in HO] + [self.out[i].T2 for i in rem2] + [self.out[i].T3 for i in rem3] 
        V3_in  = [self.inp[i].T1 for i in rem2] + [self.inp[i].T1 for i in rem3]
        V3_out = [self.out[i].T1 for i in rem2] + [self.out[i].T1 for i in rem3]

        cv.cv1 = {'train_in': pd.concat(T1_in), 
                  'train_out':pd.concat(T1_out)[sub_col]/scale, 
                  'val_in':   pd.concat(V1_in),
                  'val_out':  pd.concat(V1_out)[sub_col]/scale
                  }
        cv.cv2 = {'train_in': pd.concat(T2_in), 
                  'train_out':pd.concat(T2_out)[sub_col]/scale, 
                  'val_in':   pd.concat(V2_in),
                  'val_out':  pd.concat(V2_out)[sub_col]/scale
                  }
        cv.cv3 = {'train_in': pd.concat(T3_in), 
                  'train_out':pd.concat(T3_out)[sub_col]/scale, 
                  'val_in':   pd.concat(V3_in),
                  'val_out':  pd.concat(V3_out)[sub_col]/scale
                  }
        # cv.time = cv.test_in['time']
        return cv 

class initiate_RNN_data:

    def __init__(self,window_size):
        self.o1 = subject_out(1)
        self.o2 = subject_out(2)
        self.o3 = subject_out(3)
        self.o4 = subject_out(4)
        self.o5 = subject_out(5)
        
        self.i1 = subject_in(1)
        self.i2 = subject_in(2)
        self.i3 = subject_in(3)
        self.i4 = subject_in(4)
        self.i5 = subject_in(5)
        
        self.i1.RNN_T1, self.o1.RNN_T1 = transform_data_into_windows(self.i1.T1, self.o1.T1, window_size)
        self.i1.RNN_T2, self.o1.RNN_T2 = transform_data_into_windows(self.i1.T2, self.o1.T2, window_size)
        self.i1.RNN_T3, self.o1.RNN_T3 = transform_data_into_windows(self.i1.T3, self.o1.T3, window_size)

        self.i2.RNN_T1, self.o2.RNN_T1 = transform_data_into_windows(self.i2.T1, self.o2.T1, window_size)
        self.i2.RNN_T2, self.o2.RNN_T2 = transform_data_into_windows(self.i2.T2, self.o2.T2, window_size)
        self.i2.RNN_T3, self.o2.RNN_T3 = transform_data_into_windows(self.i2.T3, self.o2.T3, window_size)

        self.i3.RNN_T1, self.o3.RNN_T1 = transform_data_into_windows(self.i3.T1, self.o3.T1, window_size)
        self.i3.RNN_T2, self.o3.RNN_T2 = transform_data_into_windows(self.i3.T2, self.o3.T2, window_size)
        self.i3.RNN_T3, self.o3.RNN_T3 = transform_data_into_windows(self.i3.T3, self.o3.T3, window_size)

        self.i4.RNN_T1, self.o4.RNN_T1 = transform_data_into_windows(self.i4.T1, self.o4.T1, window_size)
        self.i4.RNN_T2, self.o4.RNN_T2 = transform_data_into_windows(self.i4.T2, self.o4.T2, window_size)
        self.i4.RNN_T3, self.o4.RNN_T3 = transform_data_into_windows(self.i4.T3, self.o4.T3, window_size)

        self.i5.RNN_T1, self.o5.RNN_T1 = transform_data_into_windows(self.i5.T1, self.o5.T1, window_size)
        self.i5.RNN_T2, self.o5.RNN_T2 = transform_data_into_windows(self.i5.T2, self.o5.T2, window_size)
        self.i5.RNN_T3, self.o5.RNN_T3 = transform_data_into_windows(self.i5.T3, self.o5.T3, window_size)
        self.data_class = 'RNN'

    def subject_naive(self,feature):
        cv = cv_data()
        cv.feature = feature
        cv.subject = 'naive'
        cv.data_class = 'RNN'
        columns = self.o1.RNN_T1.columns.to_list()
        sc = self.o1.numer_of_features[feature]
        sub_col = columns[sc[0]:sc[-1]+1]

        val_in_1  = np.concatenate([self.i1.RNN_T1, self.i1.RNN_T2, self.i1.RNN_T3])
        val_in_2  = np.concatenate([self.i2.RNN_T1, self.i2.RNN_T2, self.i2.RNN_T3])
        val_in_3  = np.concatenate([self.i3.RNN_T1, self.i3.RNN_T2, self.i3.RNN_T3])
        val_in_4  = np.concatenate([self.i4.RNN_T1, self.i4.RNN_T2, self.i4.RNN_T3])

        val_out_1 = pd.concat([self.o1.RNN_T1, self.o1.RNN_T2, self.o1.RNN_T3])[sub_col]
        val_out_2 = pd.concat([self.o2.RNN_T1, self.o2.RNN_T2, self.o2.RNN_T3])[sub_col]
        val_out_3 = pd.concat([self.o3.RNN_T1, self.o3.RNN_T2, self.o3.RNN_T3])[sub_col]
        val_out_4 = pd.concat([self.o4.RNN_T1, self.o4.RNN_T2, self.o4.RNN_T3])[sub_col]

        train_in_1 = np.concatenate([self.i2.RNN_T1, self.i2.RNN_T2, self.i2.RNN_T3, self.i3.RNN_T1, self.i3.RNN_T2, self.i3.RNN_T3, self.i4.RNN_T1, self.i4.RNN_T2, self.i4.RNN_T3])
        train_in_2 = np.concatenate([self.i1.RNN_T1, self.i1.RNN_T2, self.i1.RNN_T3, self.i3.RNN_T1, self.i3.RNN_T2, self.i3.RNN_T3, self.i4.RNN_T1, self.i4.RNN_T2, self.i4.RNN_T3])
        train_in_3 = np.concatenate([self.i1.RNN_T1, self.i1.RNN_T2, self.i1.RNN_T3, self.i2.RNN_T1, self.i2.RNN_T2, self.i2.RNN_T3, self.i4.RNN_T1, self.i4.RNN_T2, self.i4.RNN_T3])
        train_in_4 = np.concatenate([self.i1.RNN_T1, self.i1.RNN_T2, self.i1.RNN_T3, self.i2.RNN_T1, self.i2.RNN_T2, self.i2.RNN_T3, self.i3.RNN_T1, self.i3.RNN_T2, self.i3.RNN_T3])

        train_out_1 = pd.concat([self.o2.RNN_T1, self.o2.RNN_T2, self.o2.RNN_T3, self.o3.RNN_T1, self.o3.RNN_T2, self.o3.RNN_T3, self.o4.RNN_T1, self.o4.RNN_T2, self.o4.RNN_T3])[sub_col]
        train_out_2 = pd.concat([self.o1.RNN_T1, self.o1.RNN_T2, self.o1.RNN_T3, self.o3.RNN_T1, self.o3.RNN_T2, self.o3.RNN_T3, self.o4.RNN_T1, self.o4.RNN_T2, self.o4.RNN_T3])[sub_col]
        train_out_3 = pd.concat([self.o1.RNN_T1, self.o1.RNN_T2, self.o1.RNN_T3, self.o2.RNN_T1, self.o2.RNN_T2, self.o2.RNN_T3, self.o4.RNN_T1, self.o4.RNN_T2, self.o4.RNN_T3])[sub_col]
        train_out_4 = pd.concat([self.o1.RNN_T1, self.o1.RNN_T2, self.o1.RNN_T3, self.o2.RNN_T1, self.o2.RNN_T2, self.o2.RNN_T3, self.o3.RNN_T1, self.o3.RNN_T2, self.o3.RNN_T3])[sub_col]


        cv.train_in  = np.concatenate([val_in_1,train_in_1])
        cv.train_out = pd.concat([val_out_1,train_out_1])  
        std = cv.train_out.std()
        cv.std = std

        cv.test_in  = np.concatenate([self.i5.RNN_T1, self.i5.RNN_T2, self.i5.RNN_T3])
        cv.time  = pd.concat([pd.DataFrame(np.linspace(0,1,self.i5.RNN_T1.shape[0]),columns=['time']), 
                              pd.DataFrame(np.linspace(0,1,self.i5.RNN_T2.shape[0]),columns=['time']),
                              pd.DataFrame(np.linspace(0,1,self.i5.RNN_T3.shape[0]),columns=['time'])])

        cv.test_out = pd.concat([self.o5.RNN_T1, self.o5.RNN_T2, self.o5.RNN_T3])[sub_col]  #randomly chosen

        cv.train_out, cv.test_out = cv.train_out/std, cv.test_out/std 
        cv.cv1 = {'train_in':train_in_1, 'train_out':train_out_1, 'val_in':val_in_1,'val_out':val_out_1}
        cv.cv2 = {'train_in':train_in_2, 'train_out':train_out_2, 'val_in':val_in_2,'val_out':val_out_2}
        cv.cv3 = {'train_in':train_in_3, 'train_out':train_out_3, 'val_in':val_in_3,'val_out':val_out_3}
        cv.cv4 = {'train_in':train_in_4, 'train_out':train_out_4, 'val_in':val_in_4,'val_out':val_out_4}

        return cv 

    def subject_exposed(self, feature):
        cv = cv_data()
        cv.feature = feature
        cv.subject = 'exposed'
        cv.data_class = 'RNN'
        columns = self.o1.RNN_T1.columns.to_list()
        sc = self.o1.numer_of_features[feature]
        sub_col = columns[sc[0]:sc[-1]+1]
        
        val_in_1 = np.concatenate([self.i4.RNN_T1, self.i5.RNN_T1, self.i1.RNN_T2])
        val_in_2 = np.concatenate([self.i2.RNN_T2, self.i3.RNN_T2, self.i4.RNN_T2])
        val_in_3 = np.concatenate([self.i5.RNN_T2, self.i1.RNN_T3, self.i2.RNN_T3])
        val_in_4 = np.concatenate([self.i3.RNN_T3, self.i4.RNN_T3, self.i5.RNN_T3])

        val_out_1 = pd.concat([self.o4.RNN_T1, self.o5.RNN_T1, self.o1.RNN_T2])[sub_col]
        val_out_2 = pd.concat([self.o2.RNN_T2, self.o3.RNN_T2, self.o4.RNN_T2])[sub_col]
        val_out_3 = pd.concat([self.o5.RNN_T2, self.o1.RNN_T3, self.o2.RNN_T3])[sub_col]
        val_out_4 = pd.concat([self.o3.RNN_T3, self.o4.RNN_T3, self.o5.RNN_T3])[sub_col]

        train_in_1  = np.concatenate([self.i2.RNN_T2, self.i3.RNN_T2, self.i4.RNN_T2, self.i5.RNN_T2, self.i1.RNN_T3, self.i2.RNN_T3, self.i3.RNN_T3, self.i4.RNN_T3, self.i5.RNN_T3])
        train_in_2  = np.concatenate([self.i5.RNN_T2, self.i1.RNN_T3, self.i2.RNN_T3, self.i3.RNN_T3, self.i4.RNN_T3, self.i5.RNN_T3, self.i4.RNN_T1, self.i5.RNN_T1, self.i1.RNN_T2])
        train_in_3  = np.concatenate([self.i3.RNN_T3, self.i4.RNN_T3, self.i5.RNN_T3, self.i4.RNN_T1, self.i5.RNN_T1, self.i1.RNN_T2, self.i2.RNN_T2, self.i3.RNN_T2, self.i4.RNN_T2])
        train_in_4  = np.concatenate([self.i4.RNN_T1, self.i5.RNN_T1, self.i1.RNN_T2, self.i2.RNN_T2, self.i3.RNN_T2, self.i4.RNN_T2, self.i5.RNN_T2, self.i1.RNN_T3, self.i2.RNN_T3])

        train_out_1 = pd.concat([self.o2.RNN_T2, self.o3.RNN_T2, self.o4.RNN_T2, self.o5.RNN_T2, self.o1.RNN_T3, self.o2.RNN_T3, self.o3.RNN_T3, self.o4.RNN_T3, self.o5.RNN_T3])[sub_col]
        train_out_2 = pd.concat([self.o5.RNN_T2, self.o1.RNN_T3, self.o2.RNN_T3, self.o3.RNN_T3, self.o4.RNN_T3, self.o5.RNN_T3, self.o4.RNN_T1, self.o5.RNN_T1, self.o1.RNN_T2])[sub_col]
        train_out_3 = pd.concat([self.o3.RNN_T3, self.o4.RNN_T3, self.o5.RNN_T3, self.o4.RNN_T1, self.o5.RNN_T1, self.o1.RNN_T2, self.o2.RNN_T2, self.o3.RNN_T2, self.o4.RNN_T2])[sub_col]
        train_out_4 = pd.concat([self.o4.RNN_T1, self.o5.RNN_T1, self.o1.RNN_T2, self.o2.RNN_T2, self.o3.RNN_T2, self.o4.RNN_T2, self.o5.RNN_T2, self.o1.RNN_T3, self.o2.RNN_T3])[sub_col]

        cv.train_in  = np.concatenate([val_in_1,train_in_1])
        cv.train_out = pd.concat([val_out_1,train_out_1])  
        std = cv.train_out.std()
        cv.std = std

        cv.test_in  = np.concatenate([self.i1.RNN_T1, self.i2.RNN_T1, self.i3.RNN_T1])
        cv.time  = pd.concat([pd.DataFrame(np.linspace(0,1,self.i1.RNN_T1.shape[0]),columns=['time']), 
                              pd.DataFrame(np.linspace(0,1,self.i2.RNN_T1.shape[0]),columns=['time']),
                              pd.DataFrame(np.linspace(0,1,self.i3.RNN_T1.shape[0]),columns=['time'])])


        cv.test_out = pd.concat([self.o1.RNN_T1, self.o2.RNN_T1, self.o3.RNN_T1])[sub_col]   #randomly chosen
        cv.train_out, cv.test_out = cv.train_out/std, cv.test_out/std 

        cv.cv1 = {'train_in':train_in_1, 'train_out':train_out_1, 'val_in':val_in_1,'val_out':val_out_1}
        cv.cv2 = {'train_in':train_in_2, 'train_out':train_out_2, 'val_in':val_in_2,'val_out':val_out_2}
        cv.cv3 = {'train_in':train_in_3, 'train_out':train_out_3, 'val_in':val_in_3,'val_out':val_out_3}
        cv.cv4 = {'train_in':train_in_4, 'train_out':train_out_4, 'val_in':val_in_4,'val_out':val_out_4}

        return cv 
    
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
        self.feature = ['JA','JRF','JM','MF','MA']
        self.NRMSE   =  {key: None for key in self.feature}
        self.RMSE    =  {key: None for key in self.feature}
        self.pc      =  {key: None for key in self.feature}
        self.data    =  data
        self.hyper   = hyper        
        self.feature_l   = ['Joint angles','Joint reaction forces','Joint moments',  'Muscle forces', 'Muscle activations']
        self.feature_l2  = ['Joint angles (degrees)','Joint reaction forces (\\% Body Weight)','Joint moments (\\% Body Weight \\times Body Height )',  'Muscle forces (\\% Body Weight)', 'Muscle activations (\\%)']

class ML:
    def __init__(self, what, path, window):

        if what == 'NN':
            self.data  = initiate_data()
            self.hyper = pd.read_csv(path+'hyperparam.txt',delimiter='\s+')
        elif what == 'LM':
            self.data  = initiate_data()
            self.hyper = pd.read_csv(path+'hyperparam_linear.txt',delimiter='\s+')
        elif what == 'RNN':
            self.data  = initiate_RNN_data(window_size=window)
            self.hyper = pd.read_csv(path+'hyperparam_RNN.txt',delimiter='\s+')

        self.what = what
        self.exposed =  subject('exposed', self.data, self.hyper, self.what)
        self.naive   =  subject('naive'  , self.data, self.hyper, self.what)
        self.feature_l  = ['Joint angles','Joint reaction forces','Joint moments',  'Muscle forces', 'Muscle activations']
        self.feature_l2  = ['Joint angles (degrees)','Joint reaction forces (\\% Body Weight)','Joint moments (\\% Body Weight \\times Body Height )',  'Muscle forces (\\% Body Weight)', 'Muscle activations (\\%)']
        self.feature = ['JA','JRF','JM','MF','MA']

class ML_analysis:
    def __init__(self, what, path, window):
        self.what = what
        self.LM  = ML('LM',  path, window)
        self.NN  = ML('NN',  path, window)
        # self.RNN = ML('RNN', path, window)
        self.path = path        

        self.feature_l  = ['Joint angles','Joint reaction forces','Joint moments',  'Muscle forces', 'Muscle activations']
        self.feature_l2  = ['Joint angles (degrees)','Joint reaction forces (\\% Body Weight)','Joint moments (\\% Body Weight \\times Body Height )',  'Muscle forces (\\% Body Weight)', 'Muscle activations (\\%)']
        self.feature = ['JA','JRF','JM','MF','MA']