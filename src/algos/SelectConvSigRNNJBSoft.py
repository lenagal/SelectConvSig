import tensorflow as tf
import iisignature
from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling1D, concatenate, Dense, Dropout, LSTM, Input, InputLayer, Embedding, Flatten, Conv1D, Conv2D, Conv3D, MaxPooling2D, Reshape, Lambda, Permute, BatchNormalization, GaussianNoise, Softmax
from src.algos.utils.sigutils import * #SP,CLF
from tensorflow.keras.optimizers import Adam
from src.algos.utils.RotateAxisMovie import RotateAxisMovie
from src.algos.utils.ZoomMovie import ZoomMovie
from src.algos.utils.GraphLinConv import GraphLinConv
from src.algos.utils.GradientAccumulation import CustomTrainStep
from src.algos.utils import SkeletonUtils
from src.algos.utils.scale_layer import scale

import math


class SelectConvSigRNNJBSoft:
    '''
    NN for Human Action Recognition from Skeletal Data that uses
    --graph convolution to analyze spatial data
    --log-signature to encode temporal data
    --LSTM to analyze temporal data
    Uses Joints and Bones data streams
    '''
    #model parameters
    # LSTM segments
    N_SEGMENTS = 32
    DROP_RATE_1 = 0.5
    DROP_RATE_2 = 0.8

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
        self.signature=signature_degree
        self.subjects=data_wrapper.get_subjects()
        self.classes=data_wrapper.get_classes()
        self.input_shape=(data_wrapper.get_timesteps(), data_wrapper.get_axes(), data_wrapper.get_joints()*data_wrapper.get_subjects())

        self.N_HIDDEN_NEURONS = 256
        self.FILTER_SIZE_1 = 32
        self.FILTER_SIZE_4 = 32
        self.FILTER_SIZE_2=64
        self.T=8

        self.model=self.build_model(self.input_shape,signature_degree,self.classes)
        # temperature

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
        self.model.compile(loss=tf.keras.losses.KLDivergence(), optimizer=adam, metrics = ['accuracy'])

    def build_model(self,input_shape,signature_deg,classes):
        bones_graph=SkeletonUtils.UAVHuman_Bones_graph()
        joints_graph=SkeletonUtils.UAVHuman_Joints_graph(Extended_neighbours=False)
        #filter size for concatenated joints and bones
        FILTER_SIZE_3=self.FILTER_SIZE_4*2
        # log signature dimension
        logsiglen = iisignature.logsiglength(FILTER_SIZE_3, signature_deg)
        joints_input = Input(input_shape)
        print('input_shape to NN model', input_shape)

        # applies random rotation, zoom and noise to data
        rotated_joints=RotateAxisMovie(axis=1,name='rotation',min_angle=-math.pi/3,max_angle=math.pi/3)(joints_input)
        zoomed_joints=ZoomMovie(rescaled=True,name='zoom')(rotated_joints)
        noisy_joints=GaussianNoise(0.01,name='noise1')(zoomed_joints)

        #graph convolution on joints - 2x32+2x64 layers
        # input and output of GraphLinConv TimexAxesxJointsxFilters
        reshape_input_layer=Reshape(input_shape+(1,))(noisy_joints)
        joint_config_layer=GraphLinConv(self.FILTER_SIZE_1,kernel_size=1,graph=joints_graph, activation='relu',padding='same',use_bias=False,name="spatial_config_layer_j1")(reshape_input_layer)
        bn_layer= BatchNormalization(name='bn_layer_js1')(joint_config_layer)
        # drop_layer = Dropout(self.DROP_RATE_1)(bn_layer)
        drop_layer=bn_layer
        temp_config_layer=GraphLinConv(self.FILTER_SIZE_1,kernel_size=3,graph=joints_graph,activation='relu',padding='same',use_bias=False,name="temp_config_layer_j1")(drop_layer)
        bn_layer= BatchNormalization(name='bn_layer_jt1')(temp_config_layer)

        joint_config_layer=GraphLinConv(self.FILTER_SIZE_1,kernel_size=1,activation='relu',graph=joints_graph, padding='same',use_bias=False,name="spatial_config_layer_j2")(bn_layer)
        bn_layer= BatchNormalization(name='bn_layer_js2')(joint_config_layer)
        drop_layer = Dropout(self.DROP_RATE_1)(bn_layer)
        # drop_layer=bn_layer
        temp_config_layer=GraphLinConv(self.FILTER_SIZE_1,kernel_size=3,activation='relu',graph=joints_graph, padding='same',use_bias=False,name="temp_config_layer_j2")(drop_layer)
        bn_layer= BatchNormalization(name='bn_layer_jt2')(temp_config_layer)

        joint_config_layer=GraphLinConv(self.FILTER_SIZE_2,kernel_size=1,graph=joints_graph,activation='relu',padding='same',use_bias=False,name="spatial_config_layer_j3")(bn_layer)
        bn_layer= BatchNormalization(name='bn_layer_js3')(joint_config_layer)
        # drop_layer = Dropout(self.DROP_RATE_1)(bn_layer)
        drop_layer=bn_layer
        temp_config_layer=GraphLinConv(self.FILTER_SIZE_2,kernel_size=3,graph=joints_graph,activation='relu',padding='same',use_bias=False,name="temp_config_layer_j3")(drop_layer)
        bn_layer= BatchNormalization(name='bn_layer_jt3')(temp_config_layer)

        joint_config_layer=GraphLinConv(self.FILTER_SIZE_2,kernel_size=1,graph=joints_graph, activation='relu',padding='same',use_bias=False,name="spatial_config_layer_j4")(bn_layer)
        bn_layer= BatchNormalization(name='bn_layer_js4')(joint_config_layer)
        drop_layer = Dropout(self.DROP_RATE_1)(bn_layer)
        temp_config_layer=GraphLinConv(self.FILTER_SIZE_2,kernel_size=3,graph=joints_graph,activation='relu',padding='same',use_bias=False, separate_axes=True, name="temp_config_layer_j4")(drop_layer)
        bn_layer= BatchNormalization(name='bn_layer_jt4')(temp_config_layer)

        # to TxJxFxA
        permute_layer=Permute((1,3,4,2))(bn_layer)
        # to Tx1xFxA
        print('SigLinConvRNN Debug 0 joints:',permute_layer.shape)
        contraction_layer=Conv3D(self.FILTER_SIZE_4,(1,self.data_wrapper.N_JOINTS*self.subjects,self.FILTER_SIZE_2),padding='same', strides=(1,self.data_wrapper.N_JOINTS*self.subjects,self.FILTER_SIZE_2))(permute_layer)
        print('SigLinConvRNN Debug 1 joints:',contraction_layer.shape)
        # from TxJx1xA to TxFilters
        reshape_layer_joint=Reshape((self.data_wrapper.N_TIMESTEPS,self.FILTER_SIZE_4))(contraction_layer)
        concat_layer1=reshape_layer_joint

        # calculating bones data
        bones_input = Reshape((input_shape[0],input_shape[1],self.data_wrapper.N_JOINTS,self.data_wrapper.N_PERSONS),name='bones_reshape_1')(noisy_joints)
        joints_to_bones_matrix=tf.constant(self.data_wrapper.joints_to_bones_matrix)
        bones_input = Lambda(lambda x:tf.tensordot(x[0],x[1],axes=[[3],[1]]),name='joints_to_bones_matrix')([bones_input,joints_to_bones_matrix])
        bones_input = Permute([1,2,4,3])(bones_input)
        bones_input = Reshape((input_shape[0],input_shape[1],self.data_wrapper.N_BONES*self.data_wrapper.N_PERSONS),name='bones_reshape_2')(bones_input)

         #graph convolution on bones - 2x32+2x64 layers
        reshape_input_layer=Reshape((input_shape[0],input_shape[1],self.data_wrapper.N_BONES*self.data_wrapper.N_PERSONS) +(1,))(bones_input)
        joint_config_layer=GraphLinConv(self.FILTER_SIZE_1,kernel_size=1,activation='relu',graph=bones_graph,padding='same',use_bias=False,name="spatial_config_layer_b1")(reshape_input_layer)
        bn_layer= BatchNormalization(name='bn_layer_bs1')(joint_config_layer)
        # drop_layer = Dropout(self.DROP_RATE_1)(bn_layer)
        drop_layer=bn_layer
        temp_config_layer=GraphLinConv(self.FILTER_SIZE_1,kernel_size=3,activation='relu',graph=bones_graph,padding='same',use_bias=False,name="temp_config_layer_b1")(drop_layer)
        bn_layer= BatchNormalization(name='bn_layer_bt1')(temp_config_layer)
        #
        joint_config_layer=GraphLinConv(self.FILTER_SIZE_1,kernel_size=1,activation='relu',graph=bones_graph,padding='same',use_bias=False,name="spatial_config_layer_b2")(bn_layer)
        bn_layer= BatchNormalization(name='bn_layer_bs2')(joint_config_layer)
        # drop_layer = Dropout(self.DROP_RATE_1)(bn_layer)
        drop_layer=bn_layer
        temp_config_layer=GraphLinConv(self.FILTER_SIZE_1,kernel_size=3,activation='relu',graph=bones_graph,padding='same',use_bias=False,name="temp_config_layer_b2")(drop_layer)
        bn_layer= BatchNormalization(name='bn_layer_bt2')(temp_config_layer)

        joint_config_layer=GraphLinConv(self.FILTER_SIZE_2,kernel_size=1,activation='relu',graph=bones_graph,padding='same',use_bias=False,name="spatial_config_layer_b3")(bn_layer)
        bn_layer= BatchNormalization(name='bn_layer_bs3')(joint_config_layer)
        drop_layer = Dropout(self.DROP_RATE_1)(bn_layer)
        temp_config_layer=GraphLinConv(self.FILTER_SIZE_2,kernel_size=3,activation='relu',graph=bones_graph,padding='same',use_bias=False,name="temp_config_layer_b3")(drop_layer)
        bn_layer= BatchNormalization(name='bn_layer_bt3')(temp_config_layer)

        joint_config_layer=GraphLinConv(self.FILTER_SIZE_2,kernel_size=1,activation='relu',graph=bones_graph,padding='same',use_bias=False,name="spatial_config_layer_b4")(bn_layer)
        bn_layer= BatchNormalization(name='bn_layer_bs4')(joint_config_layer)
        drop_layer = Dropout(self.DROP_RATE_1)(bn_layer)
        temp_config_layer=GraphLinConv(self.FILTER_SIZE_2,kernel_size=3,activation='relu',graph=bones_graph,padding='same',use_bias=False,separate_axes=True,name="temp_config_layer_b4")(drop_layer)
        print('SigLinConvRNN Debug 0:',temp_config_layer.shape)
        bn_layer= BatchNormalization(name='bn_layer_bt4')(temp_config_layer)


        permute_layer=Permute((1,3,4,2))(bn_layer)
        print('SigLinConvRNN Debug 0 bones:',permute_layer.shape,self.data_wrapper.N_BONES*self.subjects)
        contraction_layer=Conv3D(self.FILTER_SIZE_4,(5,self.data_wrapper.N_BONES*self.subjects,self.FILTER_SIZE_2),padding='same',strides=(1,self.data_wrapper.N_JOINTS*self.subjects,self.FILTER_SIZE_2))(permute_layer)
        print('SigLinConvRNN Debug 1 bones:',contraction_layer.shape)
        # from TxJx1xA to TxFilters
        reshape_layer_bone=Reshape((self.data_wrapper.N_TIMESTEPS,self.FILTER_SIZE_4))(contraction_layer)
        concat_layer1 = concatenate([reshape_layer_joint,reshape_layer_bone], axis=-1)

        # calculating each segment start position
        mid_output = Lambda(lambda x: SP(x, self.N_SEGMENTS), output_shape=(self.N_SEGMENTS, FILTER_SIZE_3), name='start_position')(concat_layer1)
        # calculating segments log signatures
        hidden_layer = Lambda(lambda x: CLF(x, self.N_SEGMENTS, signature_deg, logsiglen), output_shape=(self.N_SEGMENTS, logsiglen), name='logsig')(concat_layer1)

        hidden_layer = Reshape((self.N_SEGMENTS, logsiglen),name='Reshape_hidden_layer')(hidden_layer)

        BN_layer = BatchNormalization(name='batch_normalization_layer'+'_sig_'+str(signature_deg))(hidden_layer)

        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer], axis=-1)

        # LSTM
        lstm_layer = LSTM(units=self.N_HIDDEN_NEURONS, return_sequences=True,name='lstm_layer'+'_sig_'+str(signature_deg))(mid_input)

        # Dropout
        drop_layer = Dropout(self.DROP_RATE_2)(lstm_layer)
        output_layer = Flatten()(drop_layer)
        logit_layer = Dense(classes, activation=None,name='output_layer'+'_sig_'+str(signature_deg))(output_layer)
        # temperature regularization
        scaled_layer=scale(temp=self.T)(logit_layer)
        output_layer=Softmax()(scaled_layer)

        if self.accumulate_steps==0:
            model = Model(inputs=joints_input, outputs=output_layer)
        else:
            model = CustomTrainStep(n_gradients=self.accumulate_steps,inputs=joints_input, outputs=output_layer)

        model.summary()

        return model
