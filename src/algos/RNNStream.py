import tensorflow as tf
import iisignature
from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling1D, concatenate, Dense, Dropout, LSTM, Input, InputLayer, Embedding, Flatten, Conv1D, Conv2D, Conv3D, MaxPooling2D, Reshape, Lambda, Permute ,BatchNormalization, GaussianNoise
from src.algos.utils.sigutils import * #SP,CLF
from tensorflow.keras.optimizers import Adam
from src.algos.utils.RotateAxisMovie import RotateAxisMovie
from src.algos.utils.ZoomMovie import ZoomMovie
from src.algos.utils.GraphLinConv import GraphLinConv
from src.algos.utils.GradientAccumulation import CustomTrainStep
from src.algos.utils import SkeletonUtils
import math
from src.algos.utils.TripleSig import TripleSig
from src.algos.utils.WeightedAvg import WeightedAvg


class RNNStream:
    '''
    NN for Human Action Recognition from Skeletal Data that uses
    --graph convolution to analyze spatial data
    --log-signature to encode temporal data
    --LSTM to analyze temporal data
    Uses Joints, Bones and Triples data streams
    '''
    #model parameters
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
        mask: a list of triples to train with in triples stream
        '''

        self.data_wrapper=data_wrapper
        self.accumulate_steps=accumulate_steps
        self.mask=mask

        self.subjects=data_wrapper.get_subjects()
        self.classes=data_wrapper.get_classes()
        self.input_shape=(data_wrapper.get_timesteps(), data_wrapper.get_axes(), data_wrapper.get_joints()*data_wrapper.get_subjects())

        self.N_HIDDEN_NEURONS = 256
        self.FILTER_SIZE_1 = 32
        self.FILTER_SIZE_4 = 32
        self.FILTER_SIZE_2=64
        self.TRIPLES_FILTER_SIZE_1=6

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
        bones_graph=SkeletonUtils.UAVHuman_Bones_graph(no_head=self.no_head)
        joints_graph=SkeletonUtils.UAVHuman_Joints_graph(Extended_neighbours=False,no_head=self.no_head,with_center=self.with_center_joint)

        DIM_JOINTS=self.FILTER_SIZE_4
        DIM_BONES=self.FILTER_SIZE_4
        DIM_TRIPLES=self.TRIPLES_FILTER_SIZE_1*6
        # log signature dimensions
        logsiglen_joints = iisignature.logsiglength(DIM_JOINTS, signature_deg)
        logsiglen_bones = iisignature.logsiglength(DIM_BONES, signature_deg)
        logsiglen_triples = iisignature.logsiglength(DIM_TRIPLES, signature_deg)

        joints_input = Input(input_shape)

        print('input_shape', input_shape)

        # applies random rotation, zoom and noise to data
        rotated_joints=RotateAxisMovie(axis=1,name='rotation',min_angle=-math.pi/3,max_angle=math.pi/3)(joints_input)
        # zoomed_joints=ZoomMovie(rescaled=True,name='zoom')(rotated_joints)
        noisy_joints=GaussianNoise(0.01,name='noise1')(rotated_joints)

        #convolution on joints
        # input and output of GraphLinConv TxAxJxF
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
        temp_config_layer=GraphLinConv(self.FILTER_SIZE_2,kernel_size=5,graph=joints_graph,activation='relu',padding='same',use_bias=False, separate_axes=True, name="temp_config_layer_j4")(drop_layer)
        bn_layer= BatchNormalization(name='bn_layer_jt4')(temp_config_layer)

        # to TxJxFxA
        permute_layer=Permute((1,3,4,2))(bn_layer)
        # to Tx1xFxA
        print('SigLinConvRNN Debug 0 joints:',permute_layer.shape)
        # contraction_layer=DepthwiseConv3D((1,self.data_wrapper.N_JOINTS*self.subjects,self.FILTER_SIZE_1),padding='same',depth_multiplier=self.FILTER_SIZE_4, strides=(1,self.data_wrapper.N_JOINTS*self.subjects,self.FILTER_SIZE_1))(permute_layer)
        contraction_layer=Conv3D(self.FILTER_SIZE_4,(1,self.data_wrapper.N_JOINTS*self.subjects,self.FILTER_SIZE_2),padding='same', strides=(1,self.data_wrapper.N_JOINTS*self.subjects,self.FILTER_SIZE_2))(permute_layer)
        print('SigLinConvRNN Debug 1 joints:',contraction_layer.shape)
        # from TxJx1xA to TxJ*A
        # reshape_layer_joint=Reshape((self.data_wrapper.N_TIMESTEPS,self.data_wrapper.N_AXES*self.FILTER_SIZE_4))(contraction_layer)
        reshape_layer_joint=Reshape((self.data_wrapper.N_TIMESTEPS,self.FILTER_SIZE_4))(contraction_layer)


        # calculating bones data
        bones_input = Reshape((input_shape[0],input_shape[1],self.data_wrapper.N_JOINTS,self.data_wrapper.N_PERSONS),name='bones_reshape_1')(noisy_joints)
        joints_to_bones_matrix=tf.constant(self.data_wrapper.joints_to_bones_matrix)
        bones_input = Lambda(lambda x:tf.tensordot(x[0],x[1],axes=[[3],[1]]),name='joints_to_bones_matrix')([bones_input,joints_to_bones_matrix])
        bones_input = Permute([1,2,4,3])(bones_input)
        bones_input = Reshape((input_shape[0],input_shape[1],self.data_wrapper.N_BONES*self.data_wrapper.N_PERSONS),name='bones_reshape_2')(bones_input)

        # convolution on bones.
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
        # from TxJx1xA to TxFilters
        print('SigLinConvRNN Debug 1 bones:',contraction_layer.shape)
        reshape_layer_bone=Reshape((self.data_wrapper.N_TIMESTEPS,self.FILTER_SIZE_4))(contraction_layer)

        # convolution on triples
        triples_input=TripleSig(random=False, name='triples_input',mask=self.mask)(noisy_joints)
        print('triples layer:',triples_input.shape)
        transpose_layer=Permute((1,3,2))(triples_input)
        sp_config_layer=Conv1D(self.TRIPLES_FILTER_SIZE_1,1,activation='relu',padding='same',name="spatial_config_layer_t1")(transpose_layer)

        print('SigLinConvRNN triples layer shape:',sp_config_layer.shape)
        reshape_layer_triple=Reshape((self.data_wrapper.N_TIMESTEPS,self.TRIPLES_FILTER_SIZE_1*6))(sp_config_layer)

        # joint scores
        # calculating each segment start position
        mid_output = Lambda(lambda x: SP(x, self.N_SEGMENTS), output_shape=(self.N_SEGMENTS, DIM_JOINTS), name='start_position_joint')(reshape_layer_joint)
        # calculating segments log signatures
        hidden_layer = Lambda(lambda x: CLF(x, self.N_SEGMENTS, signature_deg, logsiglen_joints), output_shape=(self.N_SEGMENTS, logsiglen_joints), name='logsig_joint')(reshape_layer_joint)

        hidden_layer = Reshape((self.N_SEGMENTS, logsiglen_joints),name='Reshape_hidden_layer_joint')(hidden_layer)

        BN_layer = BatchNormalization(name='batch_normalization_layer_joint'+'_sig_'+str(signature_deg))(hidden_layer)

        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer], axis=-1)

        # LSTM
        lstm_layer_joint = LSTM(units=self.N_HIDDEN_NEURONS, return_sequences=True,name='lstm_layer_j'+'_sig_'+str(signature_deg))(mid_input)

        # Dropout
        drop_layer = Dropout(self.DROP_RATE_2)(lstm_layer_joint)
        output_layer_joint = Flatten()(drop_layer)
        output_layer_joint = Dense(classes, activation='softmax',name='output_layer_j'+'_sig_'+str(signature_deg))(output_layer_joint)

        # bone scores
        # calculating each segment start position
        mid_output = Lambda(lambda x: SP(x, self.N_SEGMENTS), output_shape=(self.N_SEGMENTS, DIM_BONES), name='start_position_bone')(reshape_layer_bone)
        # calculating segments log signatures
        hidden_layer = Lambda(lambda x: CLF(x, self.N_SEGMENTS, signature_deg, logsiglen_bones), output_shape=(self.N_SEGMENTS, logsiglen_joints), name='logsig_bone')(reshape_layer_bone)

        hidden_layer = Reshape((self.N_SEGMENTS, logsiglen_bones),name='Reshape_hidden_layer_bone')(hidden_layer)

        BN_layer = BatchNormalization(name='batch_normalization_layer_bone'+'_sig_'+str(signature_deg))(hidden_layer)

        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer], axis=-1)

        # LSTM
        lstm_layer_bone = LSTM(units=self.N_HIDDEN_NEURONS, return_sequences=True,name='lstm_layer_b'+'_sig_'+str(signature_deg))(mid_input)

        # Dropout
        drop_layer = Dropout(self.DROP_RATE_2)(lstm_layer_bone)
        output_layer_bone = Flatten()(drop_layer)
        output_layer_bone = Dense(classes, activation='softmax',name='output_layer_b'+'_sig_'+str(signature_deg))(output_layer_bone)

        # triple scores
        # calculating each segment start position
        mid_output = Lambda(lambda x: SP(x, self.N_SEGMENTS), output_shape=(self.N_SEGMENTS, DIM_TRIPLES), name='start_position_triple')(reshape_layer_triple)
        # calculating segments log signatures
        hidden_layer = Lambda(lambda x: CLF(x, self.N_SEGMENTS, signature_deg, logsiglen_triples), output_shape=(self.N_SEGMENTS, logsiglen_triples), name='logsig_triple')(reshape_layer_triple)
        #
        hidden_layer = Reshape((self.N_SEGMENTS, logsiglen_triples),name='Reshape_hidden_layer_triple')(hidden_layer)
        #
        BN_layer = BatchNormalization(name='batch_normalization_layer_t'+'_sig_'+str(signature_deg))(hidden_layer)
        #
        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer], axis=-1)
        #
        # LSTM
        lstm_layer_triple = LSTM(units=self.N_HIDDEN_NEURONS, return_sequences=True,name='lstm_layer_t'+'_sig_'+str(signature_deg))(mid_input)

        # Dropout
        drop_layer = Dropout(self.DROP_RATE_2)(lstm_layer_triple)
        output_layer_triple = Flatten()(drop_layer)
        output_layer_triple = Dense(classes, activation='softmax',name='output_layer_t'+'_sig_'+str(signature_deg))(output_layer_triple)

        # lstm_layer_concatenate=concatenate([lstm_layer_joint,lstm_layer_bone,lstm_layer_triple],axis=-2)
        # Dropout
        # drop_layer = Dropout(self.DROP_RATE_2)(lstm_layer_concatenate)
        # output_layer = Flatten()(drop_layer)

        # taking linear combination (with learned coefficients) of scores from the three data streams
        output_layer_stack=tf.stack([output_layer_joint,output_layer_bone,output_layer_triple],axis=2)
        output_layer = WeightedAvg(1, activation=None, name='scores'+'_sig_'+str(signature_deg))(output_layer_stack)
        output_layer=Flatten()(output_layer)

        if self.accumulate_steps==0:
            model = Model(inputs=joints_input, outputs=output_layer)
        else:
            model = CustomTrainStep(n_gradients=self.accumulate_steps,inputs=joints_input, outputs=output_layer)

        model.summary()

        return model
