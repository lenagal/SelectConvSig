import tensorflow as tf
from tensorflow.keras.layers import Layer
from src.algos.utils import SkeletonUtils

from skimage.util.shape import view_as_windows #for mem_striding

class GraphLinConv(Layer):
    '''
        Does a simple graph convolution on a skeleton + time convolution
        args:
        filters = number of filters
        kernel_size = time convolution kernel size
        padding...
        graph - instance of ConvGraph
        input shape: Time x axes x joints*persons x channels
        if separate_axes:
            output shape: Time x axes x joints*persons x filters
        else:
            output shape: Time x joints*persons x filters
    '''

    def __init__(self, filters, kernel_size, graph,activation=None,
                padding='same', separate_axes=True, N_PERSONS=2, use_bias=False, **kwargs):
        super(GraphLinConv, self).__init__(**kwargs)
        self.filters=filters
        self.kernel_size=kernel_size
        self.graph=graph
        self.separate_axes=separate_axes
        self.N_PERSONS=N_PERSONS
        self.N_GRAPH_NEIGHBORS=self.graph.N_NEIGHBOURS

        if activation is not None:
            self.activation=getattr(tf.nn,activation)
        else:
            self.activation=(lambda x:x)
        self.padding=padding

    def build(self,input_shape):
        self.N_TIMESTEPS=input_shape[1]
        self.N_AXES=input_shape[2]
        self.channels=input_shape[4]
        self.time_paddings=tf.constant([[0,0],[0,self.kernel_size-1],[0,0],[0,0],[0,0]])
        self.shift_paddings=[]
        for i in range(self.kernel_size):
            self.shift_paddings.append(
                tf.constant([[0,0],[self.kernel_size-i-1,i],[0,0],[0,0],[0,0]])
            )
        self.shift_start=tf.constant([0,self.kernel_size-1,0,0,0])
        self.tilted_data_batch_reshape=(-1,self.kernel_size,self.N_TIMESTEPS-self.kernel_size+1,self.N_AXES,self.graph.N_VERTICES,self.N_PERSONS,self.channels)
        self.truncated_result_shape=(-1,self.N_TIMESTEPS-self.kernel_size+1,self.N_AXES,self.graph.N_VERTICES*self.N_PERSONS,self.filters)
        self.kernel=self.add_weight(
            shape=(self.N_GRAPH_NEIGHBORS,self.kernel_size,self.filters,self.channels),
            initializer="glorot_uniform",
            trainable=True,
        )
        if not self.separate_axes:
            self.axes_functional=self.add_weight(
                shape=(self.filters,self.N_AXES),
                initializer="glorot_uniform",
                trainable=True,
            )

    def call(self, data_batch):
        matrix =self.graph.neighbor_matrix(self.kernel)
        shift_list=[]
        for i in range(self.kernel_size):
            shift_list.append(
                tf.slice(
                    tf.pad(data_batch,self.shift_paddings[i]),
                    self.shift_start, tf.shape(data_batch)
                )
            )

        tilted_data_batch=tf.stack(shift_list,axis=1)
        tilted_data_batch=tilted_data_batch[:,:,:self.N_TIMESTEPS-self.kernel_size+1]
        tilted_data_batch=tf.reshape(
                tilted_data_batch,
                self.tilted_data_batch_reshape
        )
        #tilted_data_batch shape: batch x shift x truncated time x axes x vertices x persons x channels
        #matrix shape: time_window x filters x channels x vertices x vertices
        result = tf.tensordot(tilted_data_batch,matrix,axes=[[1,6,4],[0,2,4]])
        #result shape: batch x truncated time x axes x persons x filters x vertices
        result = tf.transpose(result,perm=[0,1,2,5,3,4])
        result = tf.reshape(result,self.truncated_result_shape)

        if not self.separate_axes:
            result=tf.transpose(result,perm=[4,0,1,2,3])
            result=tf.vectorized_map(
                                    lambda xy: tf.tensordot(xy[0], xy[1], axes=[[3], [0]]),
                                    elems=(result, self.axes_functional)
                                    )
            result=tf.transpose(result,perm=[1,2,3,4,0])
        if self.padding=='valid':
            return self.activation(result)
        elif self.padding=='same':
            return self.activation(tf.pad(result,self.time_paddings))
