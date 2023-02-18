### first section to read the marker data
import numpy as np, pandas as pd, copy
from scipy.interpolate import interp1d



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
    Y.columns = [which + str(i) for i in range(4)]
    return Y

class subject_in:
    def __init__(self,index,path):
        self.index = index
        self.path = path + 'IMU_input/'
        ## [start1,end1,start2,end2,start3,end3]
        if self.index   == 1:        
            self.filter = np.array([72,401, 57 ,384, 49,355])
        elif self.index == 2:        
            self.filter = np.array([56,366, 46 ,319, 49,331])
        elif self.index == 3:        
            self.filter = np.array([62,455, 63 ,452, 77,448])
        elif self.index == 4:        
            self.filter = np.array([67,465, 103,427, 71,516])
        elif self.index == 5:        
            self.filter = np.array([64,590, 60 ,581, 51,535])
        else:
            print("Data for specified subject is not available")

        self.T1 = pd.read_csv(self.path+'IMU_Subject_'+str(self.index)+'_RGF-1-noe.csv',engine='python',delimiter=',')
        columns = self.T1.columns.to_list()
        columns = columns[1::]   ## remove first column, unnecessary index
        self.T1 = self.T1[columns]

        self.T2 = pd.read_csv(self.path+'IMU_Subject_'+str(self.index)+'_RGF-2-noe.csv',engine='python',delimiter=',')
        self.T2 = self.T2[columns]

        self.T3 = pd.read_csv(self.path+'IMU_Subject_'+str(self.index)+'_RGF-3-noe.csv',engine='python',delimiter=',')
        self.T3 = self.T3[columns]
 
        ## remove terminal frames
        self.T1 = self.T1.iloc[self.filter[0]:self.filter[1]+1]
        self.T2 = self.T2.iloc[self.filter[2]:self.filter[3]+1]
        self.T3 = self.T3.iloc[self.filter[4]:self.filter[5]+1]

        ## reset time columns
        self.T1['time'] = (self.T1['time'] - self.T1['time'].loc[self.filter[0]])/(self.T1['time'].loc[self.filter[1]]- self.T1['time'].loc[self.filter[0]])
        self.T2['time'] = (self.T2['time'] - self.T2['time'].loc[self.filter[2]])/(self.T2['time'].loc[self.filter[3]]- self.T2['time'].loc[self.filter[2]])
        self.T3['time'] = (self.T3['time'] - self.T3['time'].loc[self.filter[4]])/(self.T3['time'].loc[self.filter[5]]- self.T3['time'].loc[self.filter[4]])

        ## add dummy variables for 
        self.RNN_T1 = None
        self.RNN_T2 = None
        self.RNN_T3 = None

class subject_out:
    def __init__(self,index,path):
        self.index = index
        self.path = path + 'Marker_output/'
        self.order = ['JA','JM','JRF','MA','MF']
        self.numer_of_features = {'JA':np.arange(0,10), 'JM':np.arange(10,20), 'JRF':np.arange(20,32), 'MA':np.arange(32,36), 'MF':np.arange(36,40)}
        ### Note that the below filter remove the terminal frames but this has already been done for the files... So this is just for info
        ## [start1,end1,start2,end2,start3,end3]
        # JM scale --> 13.4103681 , 12.16228104, 11.6657577, 8.5365639, 11.0091744
        #JRF or MF --> 7.32807 , 7.196616 , 6.44517, 4.93443,   6.59232
        if self.index   == 1:        
            self.filter = np.array([120, 668, 95 , 640, 82 , 592])
            self.subject_scale = [1]*10 + [13.4103681]*10 + [7.32807]*12 + [1]*4 + [7.32807]*4
        elif self.index == 2:        
            self.filter = np.array([93 , 610, 77 , 532, 82 , 552])
            self.subject_scale = [1]*10 + [12.16228104]*10 + [7.196616]*12 + [1]*4 + [7.196616]*4
        elif self.index == 3:        
            self.filter = np.array([103, 758, 105, 753, 128, 747])
            self.subject_scale = [1]*10 + [11.6657577]*10 + [6.44517]*12 + [1]*4 + [6.44517]*4
        elif self.index == 4:        
            self.filter = np.array([112, 775, 172, 712, 118, 860])
            self.subject_scale = [1]*10 + [8.5365639]*10 + [4.93443]*12 + [1]*4 + [4.93443]*4
        elif self.index == 5:        
            self.filter = np.array([107, 983, 100, 968, 85 , 892])
            self.subject_scale = [1]*10 + [11.0091744]*10 + [6.59232]*12 + [1]*4 + [6.59232]*4
        else:
            print("Data for specified subject is not available")

        self.T1 = pd.concat([pd.read_csv(self.path+'angles_vicon_Subject'         + str(self.index)+'_RGF_1.txt',engine='python',delimiter='\s+'),
                             pd.read_csv(self.path+'JM_vicon_Subject'             + str(self.index)+'_RGF_1.txt',engine='python',delimiter='\s+'),
                             pd.read_csv(self.path+'JRF_vicon_Subject'            + str(self.index)+'_RGF_1.txt',engine='python',delimiter='\s+'),
                             Muscle_process(pd.read_csv(self.path+'Muscle_activity_vicon_Subject'+ str(self.index)+'_RGF_1.txt',engine='python',delimiter='\s+'),'MA'),
                             Muscle_process(pd.read_csv(self.path+'Muscle_force_vicon_Subject'   + str(self.index)+'_RGF_1.txt',engine='python',delimiter='\s+'),'MF')
                             ],axis=1)
        
        self.T2 = pd.concat([pd.read_csv(self.path+'angles_vicon_Subject'         + str(self.index)+'_RGF_2.txt',engine='python',delimiter='\s+'),
                             pd.read_csv(self.path+'JM_vicon_Subject'             + str(self.index)+'_RGF_2.txt',engine='python',delimiter='\s+'),
                             pd.read_csv(self.path+'JRF_vicon_Subject'            + str(self.index)+'_RGF_2.txt',engine='python',delimiter='\s+'),
                             Muscle_process(pd.read_csv(self.path+'Muscle_activity_vicon_Subject'+ str(self.index)+'_RGF_2.txt',engine='python',delimiter='\s+'),'MA'),
                             Muscle_process(pd.read_csv(self.path+'Muscle_force_vicon_Subject'   + str(self.index)+'_RGF_2.txt',engine='python',delimiter='\s+'),'MF')
                             ],axis=1)
        
        self.T3 = pd.concat([pd.read_csv(self.path+'angles_vicon_Subject'         + str(self.index)+'_RGF_3.txt',engine='python',delimiter='\s+'),
                             pd.read_csv(self.path+'JM_vicon_Subject'             + str(self.index)+'_RGF_3.txt',engine='python',delimiter='\s+'),
                             pd.read_csv(self.path+'JRF_vicon_Subject'            + str(self.index)+'_RGF_3.txt',engine='python',delimiter='\s+'),
                             Muscle_process(pd.read_csv(self.path+'Muscle_activity_vicon_Subject'+ str(self.index)+'_RGF_3.txt',engine='python',delimiter='\s+'),'MA'),
                             Muscle_process(pd.read_csv(self.path+'Muscle_force_vicon_Subject'   + str(self.index)+'_RGF_3.txt',engine='python',delimiter='\s+'),'MF')
                             ],axis=1)
        self.T1 = self.T1/self.subject_scale
        self.T2 = self.T2/self.subject_scale
        self.T3 = self.T3/self.subject_scale
        self.RNN_T1 = None
        self.RNN_T2 = None
        self.RNN_T3 = None

class cv_data:
    def __init__(self):
        self.cv1 = None
        self.cv2 = None
        self.cv3 = None
        self.cv4 = None
        self.train_in = None
        self.train_out = None
        self.test_in = None
        self.test_out = None

def interpolate(xnew,x,y):
    f1 = interp1d(x, y, kind='cubic')
    ynew = f1(xnew)
    return ynew

def interpolate_input(inp,out):
    xnew = np.linspace(0, 1, num=out.shape[0], endpoint=True)
    x = inp['time']
    columns = inp.columns.to_list()
    tmp = pd.DataFrame(np.zeros((out.shape[0],inp.shape[1])))
    tmp.columns = columns
    for enum,fea in enumerate(columns):   #ignoreing the time column
        y = inp[fea]
        tmp[fea] = interpolate(xnew,x,y)
    return tmp

def interpolate_all_input(inp,out):
    inp.T1 = interpolate_input(inp.T1,out.T1)
    inp.T2 = interpolate_input(inp.T2,out.T2)
    inp.T3 = interpolate_input(inp.T3,out.T3)
    return inp


class initiate_data:

    def __init__(self,path):
        self.o1 = subject_out(1,path)
        self.o2 = subject_out(2,path)
        self.o3 = subject_out(3,path)
        self.o4 = subject_out(4,path)
        self.o5 = subject_out(5,path)
        
        self.i1 = subject_in(1,path)
        self.i2 = subject_in(2,path)
        self.i3 = subject_in(3,path)
        self.i4 = subject_in(4,path)
        self.i5 = subject_in(5,path)
        
        self.i1 = interpolate_all_input(self.i1,self.o1)
        self.i2 = interpolate_all_input(self.i2,self.o2)
        self.i3 = interpolate_all_input(self.i3,self.o3)
        self.i4 = interpolate_all_input(self.i4,self.o4)
        self.i5 = interpolate_all_input(self.i5,self.o5)
        self.data_class = 'normal'

    def subject_naive(self,feature):
        cv = cv_data()
        cv.feature = feature
        cv.subject = 'naive'
        cv.data_class = 'normal'
        columns = self.o1.T1.columns.to_list()
        sc = self.o1.numer_of_features[feature]
        sub_col = columns[sc[0]:sc[-1]+1]

        val_in_1 = pd.concat([self.i1.T1, self.i1.T2, self.i1.T3])
        val_in_2 = pd.concat([self.i2.T1, self.i2.T2, self.i2.T3])
        val_in_3 = pd.concat([self.i3.T1, self.i3.T2, self.i3.T3])
        val_in_4 = pd.concat([self.i4.T1, self.i4.T2, self.i4.T3])

        val_out_1 = pd.concat([self.o1.T1, self.o1.T2, self.o1.T3])[sub_col]
        val_out_2 = pd.concat([self.o2.T1, self.o2.T2, self.o2.T3])[sub_col]
        val_out_3 = pd.concat([self.o3.T1, self.o3.T2, self.o3.T3])[sub_col]
        val_out_4 = pd.concat([self.o4.T1, self.o4.T2, self.o4.T3])[sub_col]

        train_in_1 = pd.concat([self.i2.T1, self.i2.T2, self.i2.T3, self.i3.T1, self.i3.T2, self.i3.T3, self.i4.T1, self.i4.T2, self.i4.T3])
        train_in_2 = pd.concat([self.i1.T1, self.i1.T2, self.i1.T3, self.i3.T1, self.i3.T2, self.i3.T3, self.i4.T1, self.i4.T2, self.i4.T3])
        train_in_3 = pd.concat([self.i1.T1, self.i1.T2, self.i1.T3, self.i2.T1, self.i2.T2, self.i2.T3, self.i4.T1, self.i4.T2, self.i4.T3])
        train_in_4 = pd.concat([self.i1.T1, self.i1.T2, self.i1.T3, self.i2.T1, self.i2.T2, self.i2.T3, self.i3.T1, self.i3.T2, self.i3.T3])

        train_out_1 = pd.concat([self.o2.T1, self.o2.T2, self.o2.T3, self.o3.T1, self.o3.T2, self.o3.T3, self.o4.T1, self.o4.T2, self.o4.T3])[sub_col]
        train_out_2 = pd.concat([self.o1.T1, self.o1.T2, self.o1.T3, self.o3.T1, self.o3.T2, self.o3.T3, self.o4.T1, self.o4.T2, self.o4.T3])[sub_col]
        train_out_3 = pd.concat([self.o1.T1, self.o1.T2, self.o1.T3, self.o2.T1, self.o2.T2, self.o2.T3, self.o4.T1, self.o4.T2, self.o4.T3])[sub_col]
        train_out_4 = pd.concat([self.o1.T1, self.o1.T2, self.o1.T3, self.o2.T1, self.o2.T2, self.o2.T3, self.o3.T1, self.o3.T2, self.o3.T3])[sub_col]


        cv.train_in  = pd.concat([val_in_1,train_in_1])
        cv.train_out = pd.concat([val_out_1,train_out_1])  
        std = cv.train_out.std()
        cv.std = std
        cv.test_in = pd.concat([self.i5.T1, self.i5.T2, self.i5.T3])
        cv.test_out = pd.concat([self.o5.T1, self.o5.T2, self.o5.T3])[sub_col]  #randomly chosen
        cv.train_out, cv.test_out = cv.train_out/std, cv.test_out/std 
        cv.cv1 = {'train_in':train_in_1, 'train_out':train_out_1, 'val_in':val_in_1,'val_out':val_out_1}
        cv.cv2 = {'train_in':train_in_2, 'train_out':train_out_2, 'val_in':val_in_2,'val_out':val_out_2}
        cv.cv3 = {'train_in':train_in_3, 'train_out':train_out_3, 'val_in':val_in_3,'val_out':val_out_3}
        cv.cv4 = {'train_in':train_in_4, 'train_out':train_out_4, 'val_in':val_in_4,'val_out':val_out_4}
        cv.time = cv.test_in['time']

        return cv 

    def subject_exposed(self, feature):
        cv = cv_data()
        cv.feature = feature
        cv.subject = 'exposed'
        cv.data_class = 'normal'
        columns = self.o1.T1.columns.to_list()
        sc = self.o1.numer_of_features[feature]
        sub_col = columns[sc[0]:sc[-1]+1]
        
        val_in_1 = pd.concat([self.i4.T1, self.i5.T1, self.i1.T2])
        val_in_2 = pd.concat([self.i2.T2, self.i3.T2, self.i4.T2])
        val_in_3 = pd.concat([self.i5.T2, self.i1.T3, self.i2.T3])
        val_in_4 = pd.concat([self.i3.T3, self.i4.T3, self.i5.T3])

        val_out_1 = pd.concat([self.o4.T1, self.o5.T1, self.o1.T2])[sub_col]
        val_out_2 = pd.concat([self.o2.T2, self.o3.T2, self.o4.T2])[sub_col]
        val_out_3 = pd.concat([self.o5.T2, self.o1.T3, self.o2.T3])[sub_col]
        val_out_4 = pd.concat([self.o3.T3, self.o4.T3, self.o5.T3])[sub_col]

        train_in_1  = pd.concat([self.i2.T2, self.i3.T2, self.i4.T2, self.i5.T2, self.i1.T3, self.i2.T3, self.i3.T3, self.i4.T3, self.i5.T3])
        train_in_2  = pd.concat([self.i5.T2, self.i1.T3, self.i2.T3, self.i3.T3, self.i4.T3, self.i5.T3, self.i4.T1, self.i5.T1, self.i1.T2])
        train_in_3  = pd.concat([self.i3.T3, self.i4.T3, self.i5.T3, self.i4.T1, self.i5.T1, self.i1.T2, self.i2.T2, self.i3.T2, self.i4.T2])
        train_in_4  = pd.concat([self.i4.T1, self.i5.T1, self.i1.T2, self.i2.T2, self.i3.T2, self.i4.T2, self.i5.T2, self.i1.T3, self.i2.T3])

        train_out_1 = pd.concat([self.o2.T2, self.o3.T2, self.o4.T2, self.o5.T2, self.o1.T3, self.o2.T3, self.o3.T3, self.o4.T3, self.o5.T3])[sub_col]
        train_out_2 = pd.concat([self.o5.T2, self.o1.T3, self.o2.T3, self.o3.T3, self.o4.T3, self.o5.T3, self.o4.T1, self.o5.T1, self.o1.T2])[sub_col]
        train_out_3 = pd.concat([self.o3.T3, self.o4.T3, self.o5.T3, self.o4.T1, self.o5.T1, self.o1.T2, self.o2.T2, self.o3.T2, self.o4.T2])[sub_col]
        train_out_4 = pd.concat([self.o4.T1, self.o5.T1, self.o1.T2, self.o2.T2, self.o3.T2, self.o4.T2, self.o5.T2, self.o1.T3, self.o2.T3])[sub_col]

        cv.train_in  = pd.concat([val_in_1,train_in_1])
        cv.train_out = pd.concat([val_out_1,train_out_1])  

        std = cv.train_out.std()
        cv.std = std
        cv.test_in  = pd.concat([self.i1.T1, self.i2.T1, self.i3.T1])
        cv.test_out = pd.concat([self.o1.T1, self.o2.T1, self.o3.T1])[sub_col]   #randomly chosen
        cv.train_out, cv.test_out = cv.train_out/std, cv.test_out/std 

        cv.cv1 = {'train_in':train_in_1, 'train_out':train_out_1, 'val_in':val_in_1,'val_out':val_out_1}
        cv.cv2 = {'train_in':train_in_2, 'train_out':train_out_2, 'val_in':val_in_2,'val_out':val_out_2}
        cv.cv3 = {'train_in':train_in_3, 'train_out':train_out_3, 'val_in':val_in_3,'val_out':val_out_3}
        cv.cv4 = {'train_in':train_in_4, 'train_out':train_out_4, 'val_in':val_in_4,'val_out':val_out_4}

        cv.time = cv.test_in['time']
        return cv 

class initiate_RNN_data:

    def __init__(self,path,window_size):
        self.o1 = subject_out(1,path)
        self.o2 = subject_out(2,path)
        self.o3 = subject_out(3,path)
        self.o4 = subject_out(4,path)
        self.o5 = subject_out(5,path)
        
        self.i1 = subject_in(1,path)
        self.i2 = subject_in(2,path)
        self.i3 = subject_in(3,path)
        self.i4 = subject_in(4,path)
        self.i5 = subject_in(5,path)
        
        self.i1 = interpolate_all_input(self.i1,self.o1)
        self.i2 = interpolate_all_input(self.i2,self.o2)
        self.i3 = interpolate_all_input(self.i3,self.o3)
        self.i4 = interpolate_all_input(self.i4,self.o4)
        self.i5 = interpolate_all_input(self.i5,self.o5)

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
            self.data  = initiate_data(path)
            self.hyper = pd.read_csv(path+'hyperparam.txt',delimiter='\s+')
        elif what == 'LM':
            self.data  = initiate_data(path)
            self.hyper = pd.read_csv(path+'hyperparam_linear.txt',delimiter='\s+')
        elif what == 'RNN':
            self.data  = initiate_RNN_data(path, window_size=window)
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
        self.RNN = ML('RNN', path, window)
        self.path = path        

        self.feature_l  = ['Joint angles','Joint reaction forces','Joint moments',  'Muscle forces', 'Muscle activations']
        self.feature_l2  = ['Joint angles (degrees)','Joint reaction forces (\\% Body Weight)','Joint moments (\\% Body Weight \\times Body Height )',  'Muscle forces (\\% Body Weight)', 'Muscle activations (\\%)']
        self.feature = ['JA','JRF','JM','MF','MA']