import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from src.data_grabbing.data_grabber import DataGrabber
from sklearn.model_selection import StratifiedKFold
from src.algos.utils import SkeletonUtils
from src.preprocessing.Data import Data
from src.preprocessing.UAVHFileNames import UAVHFileNames

class UAVHData(Data):
    '''
    Data container for UAVHuman dataset.
    The dataset has separate 'Train' and 'Test' subsets (defined by the dataset creators).
    'Train' subset contains 16724 videos of either 1 or 2 subjects performing one of 155 actions
    ...in skeletal form (extracted using AlphaPose) obtained using UAV. 'Test' subset contains 6307 videos.
    Raw data and basic preprocessing script are available at https://github.com/SUTDCV/UAV-Human/tree/master/uavhumanactiontools
    Methods:
    load_full_landmarks: Loads either train or test samples and labels.
    load_landmarks: Loads train samples with validation split
    load_test_train: Mixes train and test set and loads with validation split
    load_pairwise_landmarks: Loads all samples from two chosen action classes (for AUC ROC optimization)
    '''

    # SINGLE_LABELS=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
    # 23,24,25,26,27,28,29,30,31,32,33,34,35,
    # 36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,
    # 72,73,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,
    # 117,118,119,120,121,122,123,124,125,126,127,128,129,136,137,138,139,140,141,
    # 142,143,144,145,147,149,150,151,152,153,154]
    # DOUBLE_LABELS=[74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,
    # 93,94,95,96,97,98,130,131,132,133,134,135,146,148]
    # # JOINT_LABELS=[75,76,78,88,89,93,95,97,98,130,132,133]

    DEFAULT_DATA_PATH="/scratch/gale/UAVHuman/newData/"

    def __init__(self, data_path=DEFAULT_DATA_PATH):
        super(UAVHData, self).__init__(data_path)
        '''
        Args:
        data_path: data location
        '''
        self.file_names=UAVHFileNames(self.path)
        self.N_TIMESTEPS = 305
        self.N_AXES =2
        self.N_JOINTS = SkeletonUtils.UAVHuman_Joints()
        self.N_BONES = SkeletonUtils.UAVHuman_Bones()
        self.joints_to_bones_matrix=SkeletonUtils.joints_to_bones_matrix(SkeletonUtils.UAVHuman_Joints_graph_edges(),self.N_JOINTS)
        self.N_PERSONS = 2
        self.N_CLASSES=155
        self.datagrabber=DataGrabber()


    def get_timesteps(self):
        return self.N_TIMESTEPS

    def get_axes(self):
        return self.N_AXES

    def get_joints(self):
        return self.N_JOINTS

    def get_subjects(self):
        return self.N_PERSONS

    def get_classes(self):
        return self.N_CLASSES

    def get_name(self):
        return 'UAVHuman'

    def get_joints_graph(self):
        return SkeletonUtils.UAVHuman_Joints_graph()
    def get_bones_graph(self):
        return SkeletonUtils.UAVHuman_Bones_graph()

    def load_labels(self,flag):
        return np.load(self.file_names.label_file_name(flag))

    def _cross_validation_fold(self,one_hot_label,cross_val_fold_num=5,verbose=0):
        fold_num=cross_val_fold_num

        label=np.argmax(one_hot_label,axis=-1)

        def verboseprint(*args,**kwargs):
            if verbose==0:
                return
            else:
                print(*args,**kwargs)

        k_fold = StratifiedKFold(cross_val_fold_num)
        k_fold.get_n_splits(np.zeros(len(label)),label)
        k_fold_idx_dict = dict()

        verboseprint('Create {}-fold for cross validation...'.format(cross_val_fold_num))
        for k, (train_idx, val_idx) in enumerate(k_fold.split(np.zeros(len(label)),label)):
            k_fold_idx_dict.update({str(k):{'train':train_idx, 'val':val_idx}})
            verboseprint(k+1,'- fold:','Trainset size: ',len(train_idx),' Valset size: ',len(val_idx))
        return k_fold_idx_dict

    def _get_iter(self,data,label):
        """
        Returns iterators over the requested
        subset of the data.
        """
        return iter(list(zip(data, label)))


    def load_landmarks(self,fold_idx=0):
        '''
        Loads train-test data split according to fold_idx
        Args:
        fold_idx: fold index
        '''
        print('load_landmarks(',fold_idx,')')
        data, label = self.load_full_landmarks('train')

        fold_idx_dict=self._cross_validation_fold(label)

        train_index = fold_idx_dict[str(fold_idx)]['train']
        val_index = fold_idx_dict[str(fold_idx)]['val']

        train_data = data[train_index]
        train_label = label[train_index]

        val_data = data[val_index]
        val_label = label[val_index]

        return train_data,train_label,val_data,val_label

    def load_sampled_landmarks(self,fold_idx=0):
        '''
        Loads landmarks with a random sample of 100 frames.
        Args:
        fold_idx: fold index
        '''
        train_data, train_label, val_data, val_label = self.load_landmarks(fold_idx)
        mask = np.random.choice(self.N_TIMESTEPS,100,replace=False)
        return train_data[:,mask],train_label[:,mask], val_data[:,mask], val_label[:,mask]


    def load_test_train(self,fold_idx=0):
        '''
        Loads and mixes train and test datasets
        Args:
        fold_idx: fold index
        '''
        path = self.path
        print('Load test and train data')

        X_train, y_train=self.load_full_landmarks(flag='train')
        X_test, y_test=self.load_full_landmarks(flag='test')
        data=np.concatenate((X_train,X_test),axis=0)
        label=np.concatenate((y_train,y_test),axis=0)
        print("combined data shape", data.shape, label.shape)

        def shuffle_in_unison(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        shuffle_in_unison(data,label)

        fold_idx_dict=self._cross_validation_fold(label)
        train_index = fold_idx_dict[str(fold_idx)]['train']
        val_index = fold_idx_dict[str(fold_idx)]['val']

        train_data = data[train_index]
        train_label = label[train_index]

        val_data = data[val_index]
        val_label = label[val_index]

        print('Test-train combined split data shapes', train_data.shape, val_data.shape)

        return train_data,train_label,val_data,val_label

    def load_pairwise_landmarks(self, classes):
        '''
        Loads all samples from two chosen action classes (for AUC ROC optimization)
        classes: a tuple (class1,class2)
        '''
        X_train, y_train=self.load_full_landmarks(flag='train')
        X_test, y_test=self.load_full_landmarks(flag='test')

        ind_classes_train=np.nonzero(y_train[:,classes])
        ind_classes_test=np.nonzero(y_test[:,classes])

        X_train_cut=X_train[ind_classes_train[0]]
        X_test_cut=X_test[ind_classes_test[0]]
        y_train_cut=y_train[ind_classes_train[0]]
        y_train_cut=y_train_cut[:,classes]
        y_train_cut=np.argmax(y_train_cut,axis=1)

        y_test_cut=y_test[ind_classes_test[0]]
        y_test_cut=y_test_cut[:,classes]
        y_test_cut=np.argmax(y_test_cut,axis=1)

        y_train_cut=y_train_cut.reshape(-1,1)
        y_test_cut=y_test_cut.reshape(-1,1)

        return X_train_cut,y_train_cut,X_test_cut,y_test_cut

    def load_full_landmarks(self, flag='train',mode='regular',raw=False,old_data=False):
        '''
        Loads either train or test samples and labels.
        Args:
        flag: 'train' for train dataset, 'test' for test
        mode: mode of splitting the dataset, currently 'regular' only
        raw: whether to load raw or preprocessed data
        old_data: data after historical pre_processing
        '''
        if raw:
            full_flag='raw_'+flag
        else:
            full_flag='prenorm_'+flag

        if old_data:
            full_flag='old_'+flag
            flag=full_flag
        print('Load full',full_flag,'data')
        data=np.load(self.file_names.data_file_name(full_flag))
        label=np.load(self.file_names.label_file_name(flag))

        print('data shape:',data.shape)
        print('label shape',label.shape)
        # currently returning data without confidence hence [:,:,:2,:] slicing
        if raw:
            return data[:,:,[0,2],:], label
        else:
            return data[:,:,[0,1],:], label
