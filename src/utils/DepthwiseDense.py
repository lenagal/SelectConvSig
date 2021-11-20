import tensorflow as tf
from tensorflow.keras.layers import Layer

class DepthwiseDense(Layer):
    '''
        input shape:
        dimV x rank
        output shape:
        dimV x units
    '''
    def __init__(self,units,activation=None,use_bias=False,**kwargs):
        super(DepthwiseDense, self).__init__(**kwargs)
        self.units=units
        if activation is not None:
            self.activation=getattr(tf.nn,activation)
        else:
            self.activation=(lambda x:x)
        self.use_bias=use_bias

    def build(self,input_shape):
        self.dimV=input_shape[1]
        self.rank=input_shape[2]

        self.operator=self.add_weight(
            shape=(self.units,self.rank),
            initializer="glorot_uniform",
            trainable=True,constraint=tf.keras.constraints.UnitNorm(axis=1)
        )
        if self.use_bias:
            self.bias=self.add_weight(
                shape=(self.units,),
                initializer="glorot_uniform",
                trainable=True
            )
        else:
            self.bias=tf.zeros((self.units,))

    def call(self,inputs):
        return self.activation(
            tf.transpose(
                tf.tensordot(self.operator,inputs,axes=[[1],[2]]),
            perm=[1,2,0])+self.bias
        )
