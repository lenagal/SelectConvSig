import tensorflow as tf
from keras.layers.pooling import _Pooling2D

class LinAvgPool(Pooling2D):
    """Average Pooling Layer for positional data (i.e. vectors in a linear space).

    Averages over the coordinates for each vector landmark.

    Accepts input of shape (Time,Axes,Landmarks).
    """
    def __init__(self, pool_size=2, strides=None, padding='valid', data_format='channels_last', **kwargs):
        def avg(input, pool_size, strides, padding):

        super(MaxPooling2D, self).__init__(
            tf.nn.avg_pool,
            pool_size=pool_size,
            strides=(strides,1),
            padding=padding,
            data_format=data_format,
            **kwargs)

    def build(self, input_shape):
        shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer='glorot_uniform')
        dim=input_shape[1]
        super(LinConv, self).build(input_shape)

    def call(self, x):

        conv_coord=[]
        for i in range(self.dim):
            conv_coord.append(K.conv2d(tf.expand_dims(x[:,:,i],axis=2),tf.expand_dims(self.kernel[:,0],axis=1),strides=self.strides)[:,:,0,:])

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
