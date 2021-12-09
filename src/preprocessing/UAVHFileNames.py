from src.preprocessing.FileNames import FileNames
import os

class UAVHFileNames(FileNames):
    '''
    returns file names of labels and data for saving/loading
    all methods recieve a flag ('prenorm_test','prenorm_train','raw_train','raw_test',...) and supply a path to the file.
    debug option adds a 'debug' to the name to prevent overwriting original data.
    '''
    def __init__(self,data_path):
        super(UAVHFileNames, self).__init__(data_path)

    def data_file_name(self,flag,debug=False):
        if debug:
            return os.path.join(self.data_path,flag+'_debug_data.npy')
        else:
            return os.path.join(self.data_path,flag+'_data.npy')

    def label_file_name(self,flag,debug=False):
        if debug:
            return os.path.join(self.data_path,flag+'_debug_label.npy')
        else:
            return os.path.join(self.data_path,flag+'_label.npy')
