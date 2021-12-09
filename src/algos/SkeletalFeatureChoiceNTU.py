from src.preprocessing.NTUData import NTUData
from src.algos.SkeletalFeatureChoice import SkeletalFeatureChoice

class SkeletalFeatureChoiceNTU(SkeletalFeatureChoice):

    def __init__(self,NTU_dataset='partial',*args, **kwargs):
        super(SkeletalFeatureChoiceNTU,self).__init__(*args, **kwargs)
        self.NTU_dataset=NTU_dataset
        print('Training with {} NTU-RGB'.format(NTU_dataset))

    def getData():
        return NTUData(dataset=self.NTU_dataset)
