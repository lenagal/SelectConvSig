from src.preprocessing.UAVHData import UAVHData
from src.algos.Trainer import Trainer

class TrainerUAVH(Trainer):

    def __init__(self,*args,**kwargs):
        super(TrainerUAVH,self).__init__(*args, **kwargs)
        print("Training with UAVHuman database")

    def getData(self):
        return UAVHData()
