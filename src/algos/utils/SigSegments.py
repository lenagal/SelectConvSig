import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
import random
from src.algos.utils.deg2_sig import deg2_sig

def sigLength(dim,signature_deg):
    return int((dim**(signature_deg+1)-1)/(dim-1)-1)

class SigSegments(Layer):
    '''
    input in shape Batch X Time X Landmarks
    output shape Batch x N_SEGMENTS x Signatures
    '''
    def __init__(self,N_SEGMENTS,signature_deg=2,**kwargs):
        super(SigSegments, self).__init__(**kwargs)
        self.N_SEGMENTS=N_SEGMENTS
        self.signature_deg=signature_deg

    def build(self,input_shape):
        if self.signature_deg!=2:
            raise ValueError('SigSegments: signature degrees other than 2 currently not supported')
        self.N_TIMESTEPS=input_shape[1]
        self.N_LANDMARKS=input_shape[2]
        self.sig_length=sigLength(self.N_LANDMARKS,self.signature_deg)
        self.batch_time_transposition=[1,0,2]


        self.start_vec = np.linspace(1, self.N_TIMESTEPS, self.N_SEGMENTS + 1)
        self.start_vec = [int(round(x)) - 1 for x in self.start_vec]

    def call(self,input):
        sig_list=[]
        for i in range(self.N_SEGMENTS):
            path=input[:,self.start_vec[i]:self.start_vec[i+1]]
            path=tf.transpose(path,perm=self.batch_time_transposition)
            path=tf.reshape(path,(path.shape[0],-1,1,path.shape[2]))
            sig=deg2_sig(path)
            sig_list.append(tf.reshape(deg2_sig(path),(-1,self.sig_length)))

        sig = tf.stack(sig_list,axis=1)

        return sig
