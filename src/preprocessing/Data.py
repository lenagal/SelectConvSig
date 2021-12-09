import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from src.data_grabbing.data_grabber import DataGrabber
from sklearn.model_selection import StratifiedKFold
from src.algos.utils import SkeletonUtils

class Data:

    def __init__(self, data_path):
        '''
        Interface for data container class
        Args:
        data_path: data location
        '''
        self.path=data_path

    def get_timesteps(self):
        raise NotImplementedError("Method get_timesteps was not implemented")
    def get_axes(self):
        raise NotImplementedError("Method get_axes was not implemented")
    def get_joints(self):
        raise NotImplementedError("Method get_joints was not implemented")
    def get_subjects(self):
        raise NotImplementedError("Method get_persons was not implemented")
    def get_classes(self):
        raise NotImplementedError("Method get_classes was not implemented")
    def get_joints_graph(self):
        raise NotImplementedError("Method get_joints_graph was not implemented")
    def get_bones_graph(self):
        raise NotImplementedError("Method get_joints_graph was not implemented")


    def get_name(self):
        raise NotImplementedError("Method get_name was not implemented")

    def load_landmarks(self,fold_idx=0):
        '''
        returns train_data, train_label,val_data, val_label
        '''
        raise NotImplementedError("Method load_landmarks was not implemented")


    def load_full_landmarks(self,flag,mode,raw=False):
        '''
        returns data, label
        '''
        raise NotImplementedError("Method load_full_landmarks was not implemented")
