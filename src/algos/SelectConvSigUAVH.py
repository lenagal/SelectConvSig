from src.preprocessing.UAVHData import UAVHData
from src.algos.SelectConvSig import SelectConvSig

class SelectConvSigUAVH(SelectConvSig):

    def __init__(self,*args,**kwargs):
        super(SelectConvSigUAVH,self).__init__(*args, **kwargs)
        print("Training with UAVHuman database")

    def getData(self):
        return UAVHData()
