from src.preprocessing.FileNames import FileNames
import os

class NTUFileNames(FileNames):
    def __init__(self,data_path='',dataset='partial'):
        super(NTUFileNames, self).__init__(data_path)

        self.data_name='nturgb_data'
        self.label_name='nturgb_label'
        self.supplemental_info_name='nturgb_supplemental'
        if dataset!='all' and dataset!='partial':
            raise ValueError('dataset option can only be partial or all. recieved '+dataset)
        self.dataset=dataset

    def data_file_name(self):
        return os.path.join(self.data_path,self.data_name+'_{}'.format(self.dataset)+'.npy')

    def label_file_name(self):
        return os.path.join(self.data_path,self.label_name+'_{}'.format(self.dataset)+'.npy')

    def supplemental_info_file_name(self):
        return os.path.join(self.data_path,self.supplemental_info_name+'_{}'.format(self.dataset)+'.npy')
