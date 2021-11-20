from src.preprocessing.NTUData import NTUData
from src.algos.SelectConvSig import SelectConvSig

class SelectConvSigNTU(SelectConvSig):

    def __init__(self,NTU_dataset='partial',*args, **kwargs):
        super(SelectConvSigNTU,self).__init__(*args, **kwargs)
        self.NTU_dataset=NTU_dataset
        print('Training with {} NTU-RGB'.format(NTU_dataset))

    def getData(self):
        return NTUData(dataset=self.NTU_dataset)
