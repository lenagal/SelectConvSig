import os
import tensorflow as tf
from src.preprocessing.pre_normaliser import preNormaliser
from src.algos.TrainerUAVH import TrainerUAVH
from src.algos.TrainerNTU import TrainerNTU
from src.algos.RNNJB import RNNJB
from src.algos.RNNJBSoft import RNNJBSoft
from src.algos.RNNStream import RNNStream
from src.algos.SkeletalFeatureChoiceUAVH import SkeletalFeatureChoiceUAVH
from src.algos.SkeletalFeatureChoiceNTU import SkeletalFeatureChoiceNTU
from src.algos.RNNTriples import RNNTriples
from src.algos.RNNBase import RNNBase
from src.algos.siglinconvRNNTriplesbyAction import SkeletalFeatureChoice as SkeletalFeatureChoice
from src.algos.TeacherStudent import TeacherStudent
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # block Tensorflow notifications
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# path to trained model weights
weights_path= "/scratch/gale/UAVHuman/rnnmodels/"
# path to save predicted labels in .csv format
predictions_path= "/home/gale/UAVHuman/predictions/"
# data location
data_path="/scratch/gale/UAVHuman/newData/"

class SelectConvSigSwitch:
    '''
    Class for pre-processing datasets
    ...and training and evaluating classifiers for Skeletal Action recognition using Graph Convolution and path signature method.
    Currently works with UAVHuman dataset (https://github.com/SUTDCV/UAV-Human/tree/master/uavhumanactiontools)
    (set 'dataset'='UAVHuman') and NTU RGB+D dataset (set 'dataset'='NTURGB')
    UAVHuman dataset can be pre-processed by setting preprocess=True, NTU RGB+D is used as supplied by creators.
    Available classification methods (algo) are
    'RNNJB' - using Graph convolution spatially on Joints and Bones streams + signatures and RNN temporally
    'RNNStream'-adds certain Triplets of Human joints as a stream to the above
    'SkeletalFeatureChoice'-chooses Triplets to use in Triplets stream using L1 regularization with signature/RNN based classifier
    'ROCFeatureChoice'-chooses Triplets to use in Triplets stream using maximization of AUC ROC (in development)
    'RNNBase'-Uses 2 Depthwise Convolutional layers spatially + signature and RNN temporally
    '''
    def __init__(self, algo, dataset, NTU_dataset=None, preprocess=False, epochs=50, batch_size=16, accumulate_steps=0,debug=False):

        '''
        Loads specified dataset (NTU RGB+D 120 or UAV-Human) and and runs pre-processesing if needed.
        Args:
        algo: classification method to be used
        dataset: either "NTURGB" or "UAVHuman"
        NTU_dataset: 'partial' for NTU-RGB+D 60, 'all' for NTU-RGB+D 120
        preprocess: whether to run preprocessing on raw data or load ready data (for UAVHuman dataset only)
        ...should be run once on raw data, saves preprocessed data and it can be loaded afterwards
        batch_size: batch size
        epochs: epochs
        accumulate_steps: increases batch size times the value (to work with batch sizes that don't fit in memory)
        NTU_dataset: type of NTU dataset used: 'partial' for NTU-RGB 60 and 'full' for  NTU-RGB 120.
        ...if not working with NTU RGB+D set to None
        '''
        self.algo = algo
        self.dataset=dataset
        self.NTU_dataset=NTU_dataset
        self.batch_size=batch_size
        self.epochs=epochs
        self.accumulate_steps=accumulate_steps

        if preprocess:
            if dataset=='UAVHuman':
                print('Running pre-normalizer on UAVHuman dataset')
                '''
                Data prenormalization settings:
                data_path
                pad: whether to perform the padding procedure (replacing zero frames with valid frames)
                centre: whether to centre the skeleta around the first person's torso or not, 0 = not, 1 = frame-wise,
                ... 2 = sample-wise
                add_center_joint: adds joint at the center of the torso of the skelton
                switchBody: whether to switch the bodies in order to reduce the total energy
                eliminateSpikes: whether to try to find outlying frames and replace them with valid frames
                scale: whether to scale the data to fit into the unit square (preserves proportions), 0 = not, 1 = frame-wise
                smoothen: whether to apply a savgol filter to smoothen the data
                '''
                pad=True
                centre=1
                add_center_joint=True
                switchBody =True
                eliminateSpikes = True
                scale = 2
                smoothen = False
                self.pre_normaliser = preNormaliser(data_path=data_path, pad=pad,centre=centre,add_center_joint=add_center_joint, switchBody=switchBody,
                    eliminateSpikes=eliminateSpikes,scale=scale,parallel=False,smoothen=smoothen,debug=debug)
                print('pre-normaliser finished')
            else:
                print('NTU RGB+D dataset has no custom pre-normalisation procedure currently. Will load standard data instead')
        else:
            print('Loading data')
        print("Classification algorithm:", self.algo)

        # With classifier using Joints+Bones streams
        if self.algo=='RNNJB':
            if self.dataset=='UAVHuman':
                self.trainer = TrainerUAVH(classifier=RNNJB,
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps)
            elif self.dataset=='NTURGB':
                self.trainer = TrainerNTU(classifier=RNNJB,
                    NTU_dataset=self.NTU_dataset,
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps)
            else:
                raise NotImplementedError

        # Teacher-student training on UAVHuman dataset
        # (where designated test set has signinficant dissimilarities to train set)
        elif self.algo=='TeacherStudent':
            if self.dataset=='UAVHuman':
                self.trainer = TeacherStudent(teacher=RNNJB,
                    student=RNNJBSoft,
                    temperature=8,
                    signature=2,
                    weights=weights_path+'finalJB64wRotlogits_20211112-1632.h5',
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps)
            else:
                raise NotImplementedError

        # With classifier using Joints+Bones+selected Triples streams
        elif self.algo=='RNNStream' :
            if self.dataset=='UAVHuman':
                self.trainer_triples=SkeletalFeatureChoiceUAVH(classifier=RNNTriples,
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps,
                    mask=mask)
                self.trainer = TrainerUAVH(classifier=RNNStream,
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps,
                    mask=mask)

            elif self.dataset=='NTURGB':
                self.trainer_triples=SkeletalFeatureChoiceNTU(classifier=RNNTriples,
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps,
                    mask=mask)

                self.trainer = TrainerNTURGB(classifier=RNNStream,
                    NTU_dataset=self.NTU_dataset,
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps,
                    mask=mask)
            else:
                raise NotImplementedError

        # Selecting data for triples stream using l1 regularization
        elif self.algo=='SkeletalFeatureChoice' :
            if self.dataset=='UAVHuman':
                self.trainer= SkeletalFeatureChoiceUAVH(classifier=RNNTriples,
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps)
            elif self.dataset=='NTURGB':
                self.trainer= SkeletalFeatureChoiceNTU(classifier=RNNTriples,
                    NTU_dataset=self.NTU_dataset,
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps)
            else:
                raise NotImplementedError

        # With basic classifier using depthwise convolution spatially
        elif self.algo=='RNNBase':
            if self.dataset=='UAVHuman':
                self.trainer = TrainerUAVH(classifier=RNNBase,
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps)
            elif self.dataset=='NTURGB':
                self.trainer = TrainerNTU(classifier=RNNBase,
                    NTU_dataset=self.NTU_dataset,
                    batch_size=self.batch_size,
                    accumulate_steps=self.accumulate_steps)
            else:
                raise NotImplementedError

        # selecting data for triples stream using AUC ROC
        elif self.algo=='ROCFeatureChoice' :
            self.classifierTriples = SigLinConv(classifier=SigLinConvRNNTriples,
                pre_normaliser=self.pre_normaliser,
                classes=N_CLASSES,transform='landmarks',batch_size=self.batch_size,
                load_dejanked_data=False,
                accumulate_steps=self.accumulate_steps,checkpointed_save=True)

            self.classifier = SigLinConv(classifier=SkeletalFeatureChoice,
                pre_normaliser=self.pre_normaliser,
                classes=N_CLASSES,transform='landmarks',batch_size=self.batch_size,
                load_dejanked_data=False,
                accumulate_steps=self.accumulate_steps,checkpointed_save=True)
        else:
            raise NotImplementedError

    def run(self):
        '''
            Performs training in specified mode.
        '''
        if self.algo=='RNNJB':

            self.trainer.finalize(signature=2,modes=['cross-subject'], epochs=self.epochs,
                    saved_paths=[weights_path+"JBNTU60_"+datetime.now().strftime("%Y%m%d-%H%M")+".h5"],
                    weights=None)

        elif self.algo=='TeacherStudent':
            self.trainer.student_train(
                epochs=self.epochs,
                saved_path=weights_path+"teacherstudent_iter2_ep90"+datetime.now().strftime("%m%d-%H%M")+".h5",
                iter=2,
                render_plot=False)

        elif self.algo=='RNNStreams' :
            self.trainer_triples.triples_choice(signature=2,
                            weights=weights_path+"triples00120211110-1715.h5",percentile=90)
            # self.trainer.finalize(signature=2,
            #         saved_paths=[weights_path+'streams'+datetime.now().strftime("%Y%m%d-%H%M")+".h5"],
            #         epochs=self.epochs,
            #         batch_size=self.batch_size,
            #         mask=weights_path+"l001+dominant_weights_percentile_"+str(90)+".pkl")

        elif self.algo=='SkeletalFeatureChoice' :
            self.trainer.finalize_triples(mode='regular',signature=2,
                            saved_path=weights_path+"triples001"+datetime.now().strftime("%Y%m%d-%H%M")+".h5",
                            weights=None,
                            epochs=self.epochs)
            # percentile=99.9
            # self.trainer.triples_choice(signature=2,
            #                 weights=weights_path+"triples00120211110-1715.h5",saved_path=weights_path+"l001+dominant_weights_percentile_"+str(percentile)+".pkl",percentile=percentile,show_plot=True)

        elif self.algo=='RNNBase':
            self.trainer.finalize(signature=2,modes=['cross-subject','cross-view'], epochs=self.epochs,
                    saved_paths=[weights_path+"finalBase"+datetime.now().strftime("%Y%m%d-%H%M")+".h5"])

        # elif self.algo=='ROCFeatureChoice':
        #     classes=[1,32]
        #     #
        #     mask=self.classifierTriples.triples_choice(signature=2,
        #                     weights=weights_path+"TriplesChoice",percentile=90)
        #     # mask=None
        #     self.classifier.pairwise_triples_AUC(signature=2, classes=classes,
        #         saved_path=weights_path+"BCE_AUCtriples",
        #     # # # weights=weights_path+"finalwith00001tripleswavg.h5",
        #         epochs=self.epochs,batch_size=self.batch_size,mask=mask)
        #
        #     self.classifier.triples_choice(signature=2,
        #                     weights=weights_path+'BCE_AUCtriples_'+'_'.join(str(c) for c in classes)+'.h5',
        #                     percentile=99,show_plot=True,mask=mask)

        else:
            raise NotImplementedError

if __name__ == "__main__":
    a = SelectConvSigSwitch(algo='TeacherStudent', dataset='UAVHuman', NTU_dataset=None, preprocess=False,
        batch_size=32,accumulate_steps=8, epochs=1)
    a.run()
