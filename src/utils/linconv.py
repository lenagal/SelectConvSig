import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

class LinConv(Layer):
    """Convolutional layer for positional data (i.e. vectors in a linear space).
    Finds features in the form of linear combinations of positional data (Landmarks).
    Convolves with the same kernel in the depth direction (axes) in contrast to the convention for the traditional convolution.
    Accepts input of shape (Time, Axes, Landmarks).
    """
    # Need to change input shape to (T,L,A)
    def __init__(self,filters,kernel_size,dim,strides=1,activation=None,padding='same',use_bias=False,**kwargs):
        """Args
        filters=number of filters
        kernel_size=temporal size of kernel (int)
        dim=dimesion of coordinate space
        activation=activation function (ReLu or None) (string)
        use_bias=not implemented
        """
        self.filters = filters
        #kernels of shape(n,1)
        self.kernel_size = (kernel_size,1)
        self.strides=(strides,1)
        self.activation=activation
        self.use_bias=use_bias
        self.dim=dim
        self.padding=padding
        super(LinConv, self).__init__(**kwargs)
        # self.initializer=keras.initializers.get('glorot_uniform')

    def build(self, input_shape):
        shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer='glorot_uniform')
        dim=input_shape[1]
        super(LinConv, self).build(input_shape)

    def call(self, x):

        conv_coord=[]
        for i in range(self.dim):
            conv_coord.append(K.conv2d(tf.expand_dims(x[:,:,i],axis=2),tf.expand_dims(self.kernel[:,0],axis=1),strides=self.strides,padding=self.padding)[:,:,0,:])

        z = K.stack(conv_coord,axis=2)

        if self.activation=='relu':
            return K.relu(z)
        elif self.activation is None:
            return z
        else:
            raise ValueError('Unsupported activation' + str(self.activation))


    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config=super(LinConv,self).get_config()
        config.update({"filters":self.filters,"kernel_size":self.kernel_size[0],"dim":self.dim,"strides":self.strides[0],"activation":self.activation,"use_bias":self.use_bias})
        return config
