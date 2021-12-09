class FileNames():
    def __init__(self,data_path=''):
        self.data_path=data_path

    def data_file_name(self,flag):
        raise NotImplementedError("Method data_save_name was not implemented")

    def label_file_name(self,flag):
        raise NotImplementedError("Method label_save_name was not implemented")
