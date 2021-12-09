from psfdataset import transforms, PSFDataset, PSFZippedDataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from src.Autoencoder import SigAutoencoder
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from src.data_grabbing.NTUDataGen import NTUDataGen
from sklearn.model_selection import StratifiedKFold
import src.algos.utils.SkeletonUtils as SkeletonUtils

class LinConvNTUData():

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

    def __init__(self, load_data=True, dataset = 'partial', debug=False):
        '''
            Args:
            dataset = 'all' (120) or 'partial' (60)
        '''
        self.N_TIMESTEPS = 300
        self.N_AXES = 3
        self.N_JOINTS = 25
        self.N_BONES = 24
        self.N_PERSONS = 2
        self.joints_to_bones_matrix=SkeletonUtils.joints_to_bones_matrix(SkeletonUtils.NTU_Joints_graph_edges(),self.N_JOINTS)
        self.N_CLASSES=120 if dataset=='all' else 60

        self.classes=self.N_CLASSES
        self._debug = debug
        self.dataset = dataset
        self.datagrabber = NTUDataGen(load_data=load_data,dataset=dataset)
        self.fold_idx_dict = self._cross_validation_fold()
        self.bodies='both'


    def _cross_validation_fold(self,data_input=None,cross_val_fold_num=5,verbose=0):
        if data_input is None:
            data, label = self.datagrabber.train_data, self.datagrabber.train_label
        else:
            data, label = data_input
        fold_num=cross_val_fold_num


        def verboseprint(*args,**kwargs):
            if verbose==0:
                return
            else:
                print(*args,**kwargs)

        # class_num=self.classes
        # verboseprint('Check data imbalance...')
        # class_cnt = np.zeros((class_num,))
        # for l in label:
        #     class_cnt[l] += 1
        # verboseprint(class_cnt)
        # verboseprint('Avg sample num: ',class_cnt.mean())
        # verboseprint('Max sample num: ',class_cnt.max())
        # verboseprint('Min sample num: ',class_cnt.min())

        k_fold = StratifiedKFold(cross_val_fold_num)
        k_fold.get_n_splits(data,label)
        k_fold_idx_dict = dict()

        verboseprint('Create {}-fold for cross validation...'.format(cross_val_fold_num))
        for k, (train_idx, val_idx) in enumerate(k_fold.split(data,label)):
            k_fold_idx_dict.update({str(k):{'train':train_idx, 'val':val_idx}})
            verboseprint(k+1,'- fold:','Trainset size: ',len(train_idx),' Valset size: ',len(val_idx))
        return k_fold_idx_dict

    def _get_iter(self,data,label):
        """
        Returns iterators over the requested
        subset of the data.
        """
        return iter(list(zip(data, label)))

    def load_trial_landmarks(self,fold_idx=0):
        print('load_landmarks(',fold_idx,')')
        data, labels = self.load_full_landmarks()

        train_index = self.fold_idx_dict[str(fold_idx)]['train']
        val_index = self.fold_idx_dict[str(fold_idx)]['val']

        train_data = data[train_index]
        train_label = labels[train_index]

        val_data = data[val_index]
        val_label = labels[val_index]

        return train_data,train_label,val_data,val_label

    def load_full_landmarks(self):
        data=self.datagrabber.train_data.copy()
        data=data.reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])
        label=to_categorical(self.datagrabber.train_label-1,self.classes)
        return data, label

    def load_evaluation_data(self,flag,bones=False):
        if flag=='cross-view':
            return self._load_cross_view_data(bones=bones)
        elif flag=='cross-subject':
            return self._load_cross_subject_data(bones=bones)
        elif flag=='cross-setup':
            return self._load_cross_setup_data(bones=bones)


    def _load_cross_subject_data(self,bones=False):
        if self.dataset=='all':
            train_subjects=self.TRAIN_SUBJECTS_120
        elif self.dataset=='partial':
            train_subjects=self.TRAIN_SUBJECTS_60

        train_idx=[]
        test_idx=[]

        for i in range(len(self.datagrabber.train_data)):
            if self.datagrabber.supplemental_info[i][self.PERFORMER_IDX] in train_subjects:
                train_idx.append(i)
            else:
                test_idx.append(i)
        if not bones:
            train_data = self.datagrabber.train_data[train_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])
            val_data = self.datagrabber.train_data[test_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])
        else:
            train_data = self.datagrabber.bones_data[train_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])
            val_data = self.datagrabber.bones_data[test_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])

        train_label = to_categorical(self.datagrabber.train_label[train_idx]-1,self.classes)
        val_label = to_categorical(self.datagrabber.train_label[test_idx]-1,self.classes)

        return train_data, train_label, val_data, val_label


    def _load_cross_view_data(self,bones=False):
        train_idx=[]
        test_idx=[]

        for i in range(len(self.datagrabber.train_data)):
            if self.datagrabber.supplemental_info[i][self.CAMERA_IDX] in self.TRAIN_CAMERAS:
                train_idx.append(i)
            else:
                test_idx.append(i)

        if not bones:
            train_data = self.datagrabber.train_data[train_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])
            val_data = self.datagrabber.train_data[test_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])
        else:
            train_data = self.datagrabber.bones_data[train_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])
            val_data = self.datagrabber.bones_data[test_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])

        train_label = to_categorical(self.datagrabber.train_label[train_idx]-1,self.classes)
        val_label = to_categorical(self.datagrabber.train_label[test_idx]-1,self.classes)

        return train_data, train_label, val_data, val_label

    def _load_cross_setup_data(self,bones=False):
        train_idx=[]
        test_idx=[]

        for i in range(len(self.datagrabber.train_data)):
            if self.datagrabber.supplemental_info[i][self.SETUP_IDX] % 2 == 0:
                train_idx.append(i)
            else:
                test_idx.append(i)

        if not bones:
            train_data = self.datagrabber.train_data[train_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])
            val_data = self.datagrabber.train_data[test_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])
        else:
            train_data = self.datagrabber.bones_data[train_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])
            val_data = self.datagrabber.bones_data[test_idx].reshape((-1,self.N_TIMESTEPS,self.N_PERSONS*self.N_JOINTS,self.N_AXES)).transpose([0,1,3,2])

        train_label = to_categorical(self.datagrabber.train_label[train_idx]-1,self.classes)
        val_label = to_categorical(self.datagrabber.train_label[test_idx]-1,self.classes)

        return train_data, train_label, val_data, val_label
