from src.preprocessing.NTUData import NTUData
from src.algos.Trainer import Trainer

class TrainerNTU(Trainer):

    def __init__(self,NTU_dataset='partial',*args, **kwargs):
        super(TrainerNTU,self).__init__(*args, **kwargs)
        self.NTU_dataset=NTU_dataset
        print('Training with {} NTU-RGB'.format(NTU_dataset))

    def getData(self):
        return NTUData(dataset=self.NTU_dataset)
