import tensorflow as tf
from tensorflow.keras.layers import Layer
import math


class RotateAxis(Layer):
    '''
        Keras layer that applies a random rotation on a coordinate axis.
        the rotation is applied to either the first two or first three coords
        according to value of 'rotate_dim'.

        axis - the axis to rotate.

        min_angle,max_angle - range of possible rotations

        note: only applies rotation in training. Outside of training Layer is disabled.
    '''

    def __init__(self,axis,rotate_dim=2,min_angle=0,max_angle=2*math.pi,debug=False,**kwargs):
        super(RotateAxis, self).__init__(**kwargs)
        self.axis=axis
        self.min_angle=min_angle
        self.max_angle=max_angle
        self.rotate_dim=rotate_dim

        self.debug=debug

    def build(self,input_shape):
        self.transposition = tuple(range(self.axis))+(len(input_shape)-1,)+tuple(range(self.axis+1,len(input_shape)-1))+(self.axis,)
        self.axis_dim = input_shape[self.axis]
        # print("ROTATE LAYER AXIS DIM:",self.axis_dim)

    def call(self, inputs, training=None):
        if not training and self.debug==False:
            return inputs

        angle=tf.random.uniform((),minval=self.min_angle,maxval=self.max_angle)
        # tf.print('RotateAxis debug 1:',angle)
        rotation_eye=tf.eye(self.axis_dim)

        rotation = tf.tensor_scatter_nd_update(rotation_eye,
                                        [[0,0],[0,1],[1,0],[1,1]],
                                        [tf.math.cos(angle),-tf.math.sin(angle),
                                        tf.math.sin(angle),tf.math.cos(angle)])

        if self.rotate_dim==3:
            angle2=tf.random.uniform((),minval=self.min_angle,maxval=self.max_angle)
            rotation_eye2=tf.eye(self.axis_dim)

            rotation2 = tf.tensor_scatter_nd_update(rotation_eye2,
                                            [[1,1],[1,2],[2,1],[2,2]],
                                            [tf.math.cos(angle2),-tf.math.sin(angle2),
                                            tf.math.sin(angle2),tf.math.cos(angle2)])
            rotation=tf.matmul(rotation,rotation2)
        expanded_rotation = tf.expand_dims(rotation,axis=0)

        transposed_input = tf.transpose(inputs,self.transposition)
        transposed_shape=tf.shape(transposed_input)
        batched_transposed_input = tf.reshape(transposed_input,(-1,1,tf.shape(transposed_input)[-1]))

        rotated_batched_transposed_input = tf.matmul(batched_transposed_input, expanded_rotation)
        rotated_transposed_input = tf.reshape(rotated_batched_transposed_input,transposed_shape)
        rotated_input = tf.transpose(rotated_transposed_input,self.transposition)

        return rotated_input
