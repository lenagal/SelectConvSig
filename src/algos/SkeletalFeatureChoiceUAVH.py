from src.preprocessing.UAVHData import UAVHData
from src.algos.SkeletalFeatureChoice import SkeletalFeatureChoice

class SkeletalFeatureChoiceUAVH(SkeletalFeatureChoice):

    def __init__(self,*args,**kwargs):
        super(SkeletalFeatureChoiceUAVH,self).__init__(*args, **kwargs)
        print("Training with UAVHuman database")

    def getData(self):
        return UAVHData()
