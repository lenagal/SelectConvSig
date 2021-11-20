import tensorflow as tf
import iisignature
from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling1D, concatenate, Dense, Dropout, LSTM, Input, InputLayer, Embedding, Flatten, Conv1D, Conv2D, MaxPooling2D, Reshape, Lambda, Permute ,BatchNormalization, GaussianNoise
from src.algos.utils.sigutils import * #SP,CLF
from src.algos.utils.linconv import LinConv
from tensorflow.keras.optimizers import Adam
from src.algos.utils.RotateAxisMovie import RotateAxisMovie
from src.algos.utils.GradientAccumulation import CustomTrainStep
import math


class SelectConvSigRNNBase:
    #model parameters
    N_SEGMENTS = 32
    DROP_RATE_2 = 0.8
    '''
    NN for Human Action Recognition from Skeletal Data that uses
    --2 Depthwise Convolutional Layers to analyze spatial data
    --log-signature to encode temporal data
    --LSTM to analyze temporal data
    '''
    def __init__(self, signature_degree, data_wrapper, weights=None,
            lr=0.001,accumulate_steps=0):
        '''
        Args:
        signature_degree: degree of signature
        data_wrapper: Data container class
        weights:path to trained weigths file or None
        lr: learning rate (using Adam optimizer)
        accumulate steps: number of steps before performing gradient adjustment
        ...(to achieve desired batch_size in case it doesn't fit in memory)
        '''
        self.data_wrapper=data_wrapper
        self.accumulate_steps=accumulate_steps

        self.subjects=data_wrapper.get_subjects()
        self.classes=data_wrapper.get_classes()
        self.input_shape=(data_wrapper.get_timesteps(), data_wrapper.get_axes(), data_wrapper.get_joints()*data_wrapper.get_subjects())

        self.N_HIDDEN_NEURONS = 256
        self.FILTER_SIZE_1 = self.input_shape[2]
        self.FILTER_SIZE_2 = self.FILTER_SIZE_1

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
        dimV=input_shape[1]
        logsiglen = iisignature.logsiglength(self.FILTER_SIZE_2, signature_deg)
        joints_input = Input(input_shape)
        print('input_shape', joints_input)
        # applies random rotation, zoom and noise to data
        rotated_joints=RotateAxisMovie(axis=1,name='rotation',min_angle=-math.pi/3,max_angle=math.pi/3)(joints_input)
        noise_layer=GaussianNoise(0.01,name='noise1')(rotated_joints)
        #Convolutional layer that earns optimal linear configurations of the joint vectors
        joint_config_layer=LinConv(self.FILTER_SIZE_1,1,dim=dimV,activation='relu',use_bias=False,name="spatial_config_layer1")(noise_layer)
        reshape_layer = Reshape((-1, self.FILTER_SIZE_2*input_shape[1]),name='reshape_temp_config')(joint_config_layer)
        # temporal convolution
        temp_config_layer = Conv1D(self.FILTER_SIZE_2, 5, padding='same',name='temp_config_layer')(reshape_layer)
        # calculating each segment start position
        mid_output = Lambda(lambda x: SP(x, self.N_SEGMENTS), output_shape=(self.N_SEGMENTS, self.FILTER_SIZE_2), name='start_position')(temp_config_layer)
        # calculating segments log signatures
        hidden_layer = Lambda(lambda x: CLF(x, self.N_SEGMENTS, signature_deg, logsiglen), output_shape=(self.N_SEGMENTS, logsiglen), name='logsig')(temp_config_layer)
        hidden_layer = Reshape((self.N_SEGMENTS, logsiglen),name='Reshape_hidden_layer')(hidden_layer)
        BN_layer = BatchNormalization(name='batch_normalization_layer'+'_sig_'+str(signature_deg))(hidden_layer)
        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer], axis=-1)
        # LSTM
        lstm_layer = LSTM(units=self.N_HIDDEN_NEURONS, return_sequences=True,name='lstm_layer'+'_sig_'+str(signature_deg))(mid_input)
        # Dropout
        drop_layer = Dropout(self.DROP_RATE_2)(lstm_layer)
        output_layer = Flatten()(drop_layer)
        output_layer = Dense(classes, activation='softmax',name='output_layer'+'_sig_'+str(signature_deg))(output_layer)

        if self.accumulate_steps==0:
            model = Model(inputs=joints_input, outputs=output_layer)
        else:
            model = CustomTrainStep(n_gradients=self.accumulate_steps,inputs=joints_input, outputs=output_layer)

        model.summary()

        return model
