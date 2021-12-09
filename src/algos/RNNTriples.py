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

import math


class RNNTriples:
    #model parameters
    N_SEGMENTS = 32
    DROP_RATE_1 = 0.5
    DROP_RATE_2 = 0.8
    '''
    NN for Human Action Recognition from Skeletal Data that uses
    --convolution layer to analyze spatial data with L1 regularization
    --log-signature to encode temporal data
    --LSTM to analyze temporal data
    the input data is in the form of all possible triples of joints of
    ...human skeleton.  L1 regularization pushes coefficients of most
    ...triples to 0 thus allowing to choose the most relevant triples for
    ...further analysis
    '''
    #model parameters
    # LSTM segments
    N_SEGMENTS = 32
    DROP_RATE_1 = 0.5
    DROP_RATE_2 = 0.8

    def __init__(self, signature_degree, data_wrapper, weights=None,
            lr=0.001,accumulate_steps=0,mask=None):
        '''
        Args:
        signature_degree: degree of signature
        data_wrapper: Data container class
        weights:path to trained weigths file or None
        lr: learning rate (using Adam optimizer)
        accumulate steps: number of steps before performing gradient adjustment
        ...(to achieve desired batch_size in case it doesn't fit in memory)
        mask:path to file containing list of triples to train on. If None trains on all triples.
        '''
        self.data_wrapper=data_wrapper
        self.accumulate_steps=accumulate_steps
        self.mask=mask

        self.subjects=data_wrapper.get_subjects()
        self.classes=data_wrapper.get_classes()
        self.axes=data_wrapper.get_axes()
        self.input_shape=(data_wrapper.get_timesteps(), data_wrapper.get_axes(), data_wrapper.get_joints()*data_wrapper.get_subjects())

        self.N_HIDDEN_NEURONS = 256
        #number of output channels in convolution layer
        self.TRIPLES_FILTER_SIZE_1=2
        # dimension of triples signatures (degree 2)
        self.TRIPLE_SIG_DIM=6

        self.model=self.build_model(self.input_shape,signature_degree,self.classes)

        if weights is not None:
            print('load trained weights')
            initial_weights = [layer.get_weights() for layer in self.model.layers]
            print('load weights to layers')
            self.model.load_weights(weights,by_name=True)

            for layer, initial in zip(self.model.layers, initial_weights):
                weights = layer.get_weights()
                if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                    print(f'Checkpoint contained no weights for layer {layer.name}!')
            print('-------------------')

        adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])

    def build_model(self,input_shape,signature_deg,classes):
        # calculating spatial dimension
        DIM_TRIPLES=self.TRIPLES_FILTER_SIZE_1*self.TRIPLE_SIG_DIM
        logsiglen_triples = iisignature.logsiglength(DIM_TRIPLES, signature_deg)

        joints_input = Input(input_shape)
        print('input_shape', input_shape)

        # applies random rotation and noise to data
        rotated_joints=RotateAxisMovie(axis=1,name='rotation',min_angle=-math.pi/3,max_angle=math.pi/3)(joints_input)
        noisy_joints=GaussianNoise(0.01,name='noise1')(rotated_joints)

        # obtaining triples list
        triples_input=TripleSig(random=False, name='triples_input',mask=self.mask)(noisy_joints)
        print('triples layer shape:',triples_input.shape)
        transpose_layer=Permute((1,3,2))(triples_input)
        # convolutional layer
        sp_config_layer=Conv1D(self.TRIPLES_FILTER_SIZE_1,1,activation='relu',padding='same',kernel_regularizer=l1(0.01),name="spatial_config_layer_t1")(transpose_layer)
        print('SigLinConvRNN Debug triples:',sp_config_layer.shape)
        reshape_layer_triple=Reshape((self.data_wrapper.N_TIMESTEPS,DIM_TRIPLES))(sp_config_layer)
        # calculating each segment start position
        mid_output = Lambda(lambda x: SP(x, self.N_SEGMENTS), output_shape=(self.N_SEGMENTS, DIM_TRIPLES), name='start_position_triple')(reshape_layer_triple)
        #alculating segments log signatures
        hidden_layer = Lambda(lambda x: CLF(x, self.N_SEGMENTS, signature_deg, logsiglen_triples), output_shape=(self.N_SEGMENTS, logsiglen_triples), name='logsig_triple')(reshape_layer_triple)
        hidden_layer = Reshape((self.N_SEGMENTS, logsiglen_triples),name='Reshape_hidden_layer_triple')(hidden_layer)
        BN_layer = BatchNormalization(name='batch_normalization_layer_triple'+'_sig_'+str(signature_deg))(hidden_layer)
        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer], axis=-1)
        # LSTM
        lstm_layer = LSTM(units=self.N_HIDDEN_NEURONS, return_sequences=True,name='lstm_layer_triple'+'_sig_'+str(signature_deg))(mid_input)
        # Dropout
        drop_layer = Dropout(self.DROP_RATE_2)(lstm_layer)
        output_layer = Flatten()(drop_layer)
        output_layer = Dense(classes, activation='softmax',name='output_layer_triple'+'_sig_'+str(signature_deg))(output_layer)

        if self.accumulate_steps==0:
            model = Model(inputs=joints_input, outputs=output_layer)
        else:
            model = CustomTrainStep(n_gradients=self.accumulate_steps,inputs=joints_input, outputs=output_layer)

        model.summary()

        return model
