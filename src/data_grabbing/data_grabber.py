import os
import numpy as np
import pickle



class DataGrabber:

    '''
        Extracts the raw data from the path into the attributes train_data, train_label as np.arrays of shapes (N,C,T,V,M)
        ... alias (sample,coordinate, frame alias time, joint alias vertex, person ID) and (N,).
        The attribute dataset_folder_path specifies the raw data path.
        The attribute k_fold_idx_dict is a nested dictionary specifying the indices of cross_val_fold_num=5 stratified
        ... folds of the (input) data, split into a training and a validation set. It has the structure [N_fold][z] where
        ... N_fold ranges over '0',...,str(cross_val_fold_num-1), z ranges over 'train', 'val'.
    '''

    #TODO should receive data_path as argument
    def __init__(self):
        #self.dataset_folder_path = "".join(__file__.split("/")[:-1]) + "/raw_data/" --- OLD VERSION, had a bug
        #self.dataset_folder_path = os.path.dirname(__file__) + "/raw_data/"
        self.dataset_folder_path = "/scratch/gale/UAVHuman/newData"

        self.test_data, self.test_label = self._load_uav_data('test')

        self.train_data, self.train_label = self._load_uav_data('train')

    def _load_uav_data(self, flag = 'train'):
        data_uav = np.load(os.path.join(self.dataset_folder_path,'{}_data.npy'.format(flag)))#mmap_mode = None, 'r‘
        with open(os.path.join(self.dataset_folder_path,'{}_label.pkl'.format(flag)), 'rb') as f:
                print('loading labels')
                sample_name, label_uav = pickle.load(f)
                label_uav = np.array(label_uav)
        print(label_uav.shape)
        print(data_uav.shape)
        #N,C,T,V,M = data_uav.shape
        #print(N,C,T,V,M)
        #
        # if flag=='train':
        #     with open(os.path.join(self.dataset_folder_path,'{}_label.pkl'.format(flag)), 'rb') as f:
        #         print('loading labels')
        #         sample_name, label_uav = pickle.load(f)
        #         label_uav = np.array(label_uav)
        # #print(label_uav.shape)
        return data_uav,label_uav

    '''
    def _get_data_batch(self):
        raise NotImplementedError

    def _get_all_data(self):
        raise NotImplementedError

    ER ???
    '''

if __name__ == "__main__":
    obj = DataGrabber()
