import pickle
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.algos.utils.AUCloss import AUCLoss

class SkeletalFeatureChoice:
    '''
    This class contains methods for selecting most relevant skeletal features (i.e. combinations of joints)
    for further training.
    To work with different datasets one needs to realize a getData() method in a child class.
    Child classes currently realised are SkeletalFeatureChoiceNTU and SkeletalFeautureChoiceUAVH for
    ...working with NTU RGB+D and UAVHuman datasets respectively.
    Methods:
    getData(): abstract method that returns a dataset container class of type Data
    finalize_triples(): trains the SkeletalFeatureChoiceRNN classifier on the data of all joint triples
    ...with the objective to use the resulting weights for selecting the most relevant ones.

    '''
    def __init__(self, classifier, batch_size, accumulate_steps=0):
        '''
        classifier: classifier object to train/predict
        batch_size: batch_size
        accumulate steps: increases batch size times the value
        ...(to work with batch sizes that don't fit in memory)
        '''
        self.classifier=classifier
        self.batch_size=batch_size
        self.accumulate_steps=accumulate_steps
    def getData(self):
        '''
        Abstract method that creates instance of either NTU RGB+D or UAVH dataset class
        ...returned object should be of type Data (should realize Data interface)
        '''
        raise NotImplementedError("Method getData was not implemented")

    def triples_choice(self, signature, weights, saved_path, mask=None, percentile=None,show_plot=False):
        '''
        Picks triples with weights above set treshhold (dominant triples),
        ...from weights of the classifier TriplesChoiceRNN with l1-regularization.
        signature:signature degree of the trained classifier.
        weights: weights of the trained classifier.
        saved_path: location to save the dominant triples list.
        mask: path to previously saved triples list.
        percentile: percentile cut-off (picks up triples with coeffiecient above that)
        show_plot: shows plot of weights coeffiecients distribution
        '''
        data_wrapper = self.getData()
        classifier= self.classifier(signature_degree=signature, data_wrapper=data_wrapper, weights=weights, accumulate_steps=self.accumulate_steps, mask=mask)

        weights=np.squeeze(np.array(classifier.model.get_layer(name="spatial_config_layer_t1").get_weights()[0]))
        print("full triples weights data shape:", np.shape(weights))
        num_channels=weights.shape[1]

        if percentile is not None:
            dom_triples=np.nonzero((weights>np.percentile(weights,q=percentile,axis=0,keepdims=True))|(weights<np.percentile(weights,q=100-percentile,axis=0,keepdims=True)))
        #needs debugging - not currently working as should
        else:
            dom_triples=np.nonzero((weights>0.01)|(weights<-0.01))

        triples_list=np.unique(dom_triples[0])

        with open(saved_path,"wb") as fp:
            pickle.dump(triples_list,fp)

        if show_plot:
            print("percentile shape",np.percentile(weights,q=percentile,axis=0))
            print("dominant triples list", triples_list, "dominant triples amount",len(triples_list))
            #
            fig1, axs = plt.subplots(num_channels,2,sharex=True)

            fig1.suptitle("Triples and dominant triples by weight")

            for i in range(num_channels):
                axs[i,0].scatter(weights[:,i],np.zeros_like(weights[:,i])+i)

            x=dom_triples[0]
            y=dom_triples[1]
            j=0

            for i in y:
                axs[i,1].scatter(weights[x[j],i],i)
                j+=1

            plt.show()

        return triples_list

    def analyze(self, signature, weights, mask=None):
        data_wrapper = LinConvData(pre_normaliser=self.pre_normaliser,
         transform=self.transform,
         bodies='both',load_dejanked_data=self.load_dejanked_data,no_head=self.no_head,with_center_joint=self.with_center_joint)

        classifier= self.classifier(signature_degree=signature, data_wrapper=data_wrapper, weights=weights,lr=0.001,accumulate_steps=self.accumulate_steps,no_head=self.no_head,with_center_joint=self.with_center_joint,mask=mask)
        weights=np.squeeze(np.array(classifier.model.get_layer(name="spatial_config_layer_t1").get_weights()[0]))
        print("full weights data shape:", np.shape(weights))
        num_channels=weights.shape[1]

        # fig1, axs = plt.subplots(num_channels,2,sharex=True)
        #
        # fig1.suptitle("Triples and dominant triples by weight")
        #
        # for i in range(num_channels):
        #     axs[i,0].scatter(weights[:,i],np.zeros_like(weights[:,i])+i)
        #
        # plt.show()
        print(np.squeeze(np.array(classifier.model.get_layer(name='scores'+'_sig_2').get_weights()[0])))


    def finalize_triples(self, mode, signature, epochs, saved_path, weights=None, triples_list=None,render_plot=True):
        '''Trains the classifier on triples of skeletal joints from train data and evaluates on separate test data.
        mode: defines test-train evaluation split of the dataset. Currently implemented
        ...'regular' for UAVHuman;
        'cross-view', 'cross-subject', 'cross-setup' for NTU-RGB+D;
        signature: signature degree
        epochs: number of training epochs
        weights: pre-trained weights file
        saved_path: paths to save trained models
        render_plot: whether to show training metrics plot
        triples_list: list of select triples; uses all if triples_list=None
        '''
        print("Finalize triples training....")
        data_wrapper = self.getData()

        X_train, y_train=data_wrapper.load_full_landmarks(flag='train',mode=mode)
        X_test, y_test=data_wrapper.load_full_landmarks(flag='test',mode=mode)

        print('data shape:train', X_train.shape, y_train.shape,'test',X_test.shape, y_test.shape)
        classifier= self.classifier(signature_degree=signature, data_wrapper=data_wrapper, weights=weights, accumulate_steps=self.accumulate_steps,mask=triples_list)
        history=classifier.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size)
        print("Evaluate on test data")
        results = classifier.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print("test loss, test acc:", results)

        if saved_path is not None:
            classifier.model.save_weights(saved_path)
            print('model saved at',saved_path)
        # Plot training accuracy
        if render_plot:
            fig,axs=plt.subplots(1,2)

            axs[0].plot(history.history['accuracy'])
            axs[0].set_title('accuracy')
            axs[0].set_ylabel('accuracy')
            axs[0].set_xlabel('epoch')
            axs[0].legend(['train'])

            axs[1].plot(history.history['loss'])
            axs[1].set_title('loss')
            axs[1].set_ylabel('loss')
            axs[1].set_xlabel('epoch')
            axs[1].legend(['train'])

            plt.show()

    def pairwise_triples_AUC(self,signature, classes, saved_path, weights=None, epochs=15, lr=0.001,mask=None):
        data_wrapper = LinConvData(pre_normaliser=self.pre_normaliser,
         transform=self.transform,
         bodies='both',load_dejanked_data=self.load_dejanked_data,no_head=self.no_head,with_center_joint=self.with_center_joint)

        X_train, y_train, X_test, y_test=data_wrapper.load_pairwise_landmarks(classes)

        print('data shape:train', X_train.shape, y_train.shape,'test',X_test.shape, y_test.shape)
        # get weights over qth percentile
        classifier= self.classifier(signature_degree=signature, data_wrapper=data_wrapper, weights=weights,lr=lr,accumulate_steps=self.accumulate_steps,no_head=self.no_head,with_center_joint=self.with_center_joint,mask=mask)
        # # classifier.model.get_layer(name="spatial_config_layer_t1")
        print('classifier initialized')
        history=classifier.model.fit(X_train[:,:,:2,:], y_train, epochs=epochs, batch_size=self.batch_size)
        #
        print("Evaluate on test data")
        results = classifier.model.evaluate(X_test[:,:,:2,:], y_test, batch_size=self.batch_size)
        print("loss, acc, AUC", results)

        if saved_path is not None:
            saved_path=saved_path+'_'+'_'.join(str(c) for c in classes)+'.h5'
            classifier.model.save_weights(saved_path)
            print('model saved at',saved_path)
