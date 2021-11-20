import tensorflow as tf
import iisignature
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling1D, concatenate, Dense, Dropout, LSTM, Input, InputLayer, Embedding, Flatten, Conv1D, Conv2D, Conv3D, MaxPooling2D, Reshape, Lambda, Permute ,BatchNormalization, GaussianNoise
from tensorflow.keras.regularizers import l1
from src.algos.utils.sigutils import * #SP,CLF
from src.preprocessing.linconv_data import LinConvData
from src.preprocessing.linconv_NTU_data import LinConvNTUData
from tensorflow.keras.optimizers import Adam
from src.algos.utils.RotateAxisMovie import RotateAxisMovie
from src.algos.utils.ZoomMovie import ZoomMovie
from src.algos.utils.GraphLinConv import GraphLinConv
from src.algos.utils.GradientAccumulation import CustomTrainStep
from src.algos.DepthwiseConv3D import DepthwiseConv3D
from src.algos.utils import SkeletonUtils
from src.algos.utils.TripleSig import TripleSig
from src.algos.utils.AUCloss import AUCLoss


import math


class SkeletalFeatureChoice:
    #model parameters
    N_SEGMENTS = 32
    DROP_RATE_1 = 0.5
    DROP_RATE_2 = 0.8

    '''
    A model for choosing best discriminative features in the form of paths of skeletal joints
    for Skeletal Human Action Recognition task.
    We use the signature method for parametrization of paths and optimize the AUC metric
    '''
    def __init__(self, signature_degree, data_wrapper, weights=None,
            lr=0.001,accumulate_steps=0,no_head=False,with_center_joint=False,mask=None):
        '''
        Args:
        signature_degree: degree of signature requested
        data_wrapper=an instance of a feature extractor container LinConvData
        weights=path to weigths file or None
        ...
        '''
        self.data_wrapper=data_wrapper
        self.accumulate_steps=accumulate_steps
        self.no_head=no_head
        self.with_center_joint=with_center_joint
        self.mask=mask
        classes=2

        if isinstance(data_wrapper,LinConvData):
            print('UAV Human data wrapper')
            self.subjects=2
            self.N_HIDDEN_NEURONS = 256

        elif isinstance(data_wrapper,LinConvNTUData):
            print('NTU data wrapper')
            self.subjects=2
            self.axes=3
            self.N_HIDDEN_NEURONS = 256

        self.input_shape=(data_wrapper.N_TIMESTEPS, data_wrapper.N_AXES, data_wrapper.N_JOINTS*self.subjects)

        self.TRIPLES_FILTER_SIZE_1=5

        self.model=self.build_model(self.input_shape,signature_degree,classes)

        if weights is not None:
        #TO DO:exctract classes from weights file and throw exeption
        #if it doesn't agree with passed classes, extract signature and throw
        #if it doesn't agree with weights

            print('load trained weights')
            initial_weights = [layer.get_weights() for layer in self.model.layers]
            print('load weights to layers')
            self.model.load_weights(weights)

            for layer, initial in zip(self.model.layers, initial_weights):
                weights = layer.get_weights()
                if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                    print(f'Checkpoint contained no weights for layer {layer.name}!')
            print('-------------------')

        adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # loss=AUCLoss()
        loss=tf.keras.losses.BinaryCrossentropy()
        self.model.compile(loss=loss, optimizer=adam, metrics = ['AUC','accuracy'])


    def build_model(self,input_shape,signature_deg,classes):
        DIM_TRIPLES=self.TRIPLES_FILTER_SIZE_1*6

        # logsiglen_joints = iisignature.logsiglength(DIM_JOINTS, signature_deg)
        # logsiglen_bones = iisignature.logsiglength(DIM_BONES, signature_deg)
        logsiglen_triples = iisignature.logsiglength(DIM_TRIPLES, signature_deg)

        joints_input = Input(input_shape)

        print('input_shape', input_shape)
        rotated_joints=RotateAxisMovie(axis=1,name='rotation',min_angle=-math.pi/3,max_angle=math.pi/3)(joints_input)
        # zoomed_joints=ZoomMovie(rescaled=True,name='zoom')(rotated_joints)
        noisy_joints=GaussianNoise(0.01,name='noise1')(rotated_joints)

        # convolution on triples
        triples_input=TripleSig(random=False, name='triples_input',mask=self.mask)(noisy_joints)
        print('triples layer:',triples_input.shape)
        #
        transpose_layer=Permute((1,3,2))(triples_input)
        sp_config_layer=Conv1D(self.TRIPLES_FILTER_SIZE_1,1,activation='relu',padding='same',kernel_regularizer=l1(0.01), name="spatial_config_layer_t1")(transpose_layer)
        # kernel_regularizer=l1(0.001)
        print('SigLinConvRNN Debug triples:',sp_config_layer.shape)
        reshape_layer_triple=Reshape((self.data_wrapper.N_TIMESTEPS,self.TRIPLES_FILTER_SIZE_1*6))(sp_config_layer)

        # triple scores
        mid_output = Lambda(lambda x: SP(x, self.N_SEGMENTS), output_shape=(self.N_SEGMENTS, DIM_TRIPLES), name='start_position_triple')(reshape_layer_triple)
        #
        hidden_layer = Lambda(lambda x: CLF(x, self.N_SEGMENTS, signature_deg, logsiglen_triples), output_shape=(self.N_SEGMENTS, logsiglen_triples), name='logsig_triple')(reshape_layer_triple)
        #
        hidden_layer = Reshape((self.N_SEGMENTS, logsiglen_triples),name='Reshape_hidden_layer_triple')(hidden_layer)
        #
        BN_layer = BatchNormalization(name='batch_normalization_layer_triple'+'_sig_'+str(signature_deg))(hidden_layer)
        #
        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer], axis=-1)
        #
        # LSTM
        lstm_layer = LSTM(units=self.N_HIDDEN_NEURONS, return_sequences=True,name='lstm_layer_triple'+'_sig_'+str(signature_deg))(mid_input)
        #
        # Dropout
        drop_layer = Dropout(self.DROP_RATE_2)(lstm_layer)
        output_layer = Flatten()(drop_layer)
        output_layer = Dense(classes-1, activation='sigmoid',name='output_layer_triple'+'_sig_'+str(signature_deg))(output_layer)

        if self.accumulate_steps==0:
            model = Model(inputs=joints_input, outputs=output_layer)
        else:
            model = CustomTrainStep(n_gradients=self.accumulate_steps,inputs=joints_input, outputs=output_layer)

        model.summary()

        return model
