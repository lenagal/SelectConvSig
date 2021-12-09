import tensorflow as tf
from tensorflow.keras.layers import Layer

class scale(Layer):

    def __init__(self,temp,**kwargs):
        super(scale,self).__init__(**kwargs)
        self.T=temp

    def __call__(self,x):
        return tf.truediv(x,self.T)
