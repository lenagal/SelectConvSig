import numpy as np
import tensorflow as tf
from src.algos.utils.misc import to_categorical
from src.Autoencoder import SigAutoencoder
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from src.data_grabbing.NTUDataGen import NTUDataGen
from sklearn.model_selection import StratifiedKFold
import src.algos.utils.SkeletonUtils as SkeletonUtils
from src.preprocessing.Data import Data
from src.preprocessing.NTUFileNames import NTUFileNames

class NTUData(Data):
    '''
    Data container for NTU RGB+D dataset.
    By setting dataset='partial' or 'all' one can work with NTU RGB+D 60 or 120 respectively
    The available evaluation modes are
    Methods:
    load_full_landmarks: Loads either 'train' or 'test' samples and labels.
    ...The dataset is split into 'train' and 'test' by setting "mode"='cross-view','cross-subject','cross-setup'
    '''

    TRAIN_SUBJECTS_120=[1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
                    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
                    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
    TRAIN_SUBJECTS_60=[1, 2, 4, 5, 8, 9, 13, 14, 15,
                        16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

    TRAIN_CAMERAS=[2,3]

    SETUP_IDX = 0
    CAMERA_IDX = 1
    PERFORMER_IDX = 2
    REPLICATION_IDX = 3

    DEFAULT_DATA_PATH="/scratch/gale/UAVHuman/NTURGBData/"

    def __init__(self, data_path=DEFAULT_DATA_PATH, dataset = 'partial'):
        '''
            Args:
            dataset = 'full' (120) or 'partial' (60)
        '''
        super(NTUData, self).__init__(data_path)

        self.N_TIMESTEPS = 300
        self.N_AXES = 3
        self.N_JOINTS = 25
        self.N_BONES = 24
        self.N_PERSONS = 2
        self.joints_to_bones_matrix=SkeletonUtils.joints_to_bones_matrix(SkeletonUtils.NTU_Joints_graph_edges(),self.N_JOINTS)

        self.dataset = dataset
        self.file_names=NTUFileNames(self.path,dataset)
        if self.dataset=='all':
            self.train_subjects=self.TRAIN_SUBJECTS_120
            self.N_CLASSES=120
        elif self.dataset=='partial':
            self.train_subjects=self.TRAIN_SUBJECTS_60
            self.N_CLASSES=60
        self.supplemental_info=np.load(self.file_names.supplemental_info_file_name())

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
        return 'NTU'

    def get_joints_graph(self):
        return SkeletonUtils.NTU_Joints_graph()
    def get_bones_graph(self):
        return SkeletonUtils.NTU_Bones_graph()

    def load_full_landmarks(self,flag='',mode='cross-subject',raw=False):

        print('Loading',flag,mode,'data from',self.dataset,'dataset')
        self.data = np.load(self.file_names.data_file_name())
        self.data = self.data.transpose([0,1,4,3,2]).reshape((-1,self.N_TIMESTEPS,self.N_AXES,self.N_JOINTS*self.N_PERSONS))
        self.label = to_categorical(np.load(self.file_names.label_file_name())-1,self.N_CLASSES)

        if mode=='cross-view':
            return self._load_cross_view_data(flag)
        elif mode=='cross-subject':
            return self._load_cross_subject_data(flag)
        elif mode=='cross-setup':
            return self._load_cross_setup_data(flag)
        else:
            raise ValueError('mode can only be cross-view, cross-subject or cross-setup. recieved '+mode)


    def _load_cross_subject_data(self,flag):
        train_idx=[]
        test_idx=[]

        for i in range(len(self.data)):
            if self.supplemental_info[i][self.PERFORMER_IDX] in self.train_subjects:
                train_idx.append(i)
            else:
                test_idx.append(i)
        if flag=='train':
            train_data=self.data[train_idx]
            train_label=self.label[train_idx]
            # print('data shape:',train_data.shape)
            return train_data,train_label
        elif flag=='test':
            test_data=self.data[test_idx]
            test_label=self.label[test_idx]
            # print('data shape:',test_data.shape)
            return test_data,test_label
        else:
            raise ValueError('flag must be test or train. recieved '+flag)

    def _load_cross_view_data(self,flag):
        train_idx=[]
        test_idx=[]

        for i in range(len(self.data)):
            if self.supplemental_info[i][self.CAMERA_IDX] in self.TRAIN_CAMERAS:
                train_idx.append(i)
            else:
                test_idx.append(i)

        if flag=='train':
            train_data=self.data[train_idx]
            train_label=self.label[train_idx]
            # print('data shape:',train_data.shape)
            return train_data,train_label
        elif flag=='test':
            test_data=self.data[test_idx]
            test_label=self.label[test_idx]
            # print('data shape:',test_data.shape)
            return test_data,test_label
        else:
            raise ValueError('flag must be test or train. recieved '+flag)

    def _load_cross_setup_data(self,flag):
        train_idx=[]
        test_idx=[]

        for i in range(len(self.data)):
            if self.supplemental_info[i][self.SETUP_IDX] % 2 == 0:
                train_idx.append(i)
            else:
                test_idx.append(i)

        if flag=='train':
            train_data=self.data[train_idx]
            train_label=self.label[train_idx]
            # print('data shape:',train_data.shape)
            return train_data,train_label
        elif flag=='test':
            test_data=self.data[test_idx]
            test_label=self.label[test_idx]
            # print('data shape:',test_data.shape)
            return test_data,test_label
        else:
            raise ValueError('flag must be test or train. recieved '+flag)
