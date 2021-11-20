import tensorflow as tf
from tensorflow.keras.layers import Layer
import math


class RotateAxisMovie(Layer):
    '''
        Keras layer that applies a continuous random rotation on a coordinate axis to a movie.
        the rotation is applied to either the first two or first three coords
        according to value of 'rotate_dim'. if rotate_dim=2 then a 3D rotation is applied
        to the (0,x,y) followed by projection back to the 2D plane.

        input must be of form Time x other axes

        axis - the axis to rotate. axis>0

        min_angle,max_angle - range of possible rotations

        note: only applies rotation in training. Outside of training Layer is disabled.
    '''

    def __init__(self,axis,rotate_dim=2,min_angle=0,max_angle=2*math.pi,debug=False,**kwargs):
        super(RotateAxisMovie, self).__init__(**kwargs)
        self.axis=axis+1
        self.min_angle=min_angle
        self.max_angle=max_angle
        self.rotate_dim=rotate_dim

        self.debug=debug

    def build(self,input_shape):
        self.axis_dim = input_shape[self.axis]
        self.rank=len(input_shape)
        self.N_TIMESTEPS=input_shape[1]
        def proj_start(i):
            if i==self.axis:
                return 1
            else:
                return 0
        self.projection_slice_start=[proj_start(i) for i in range(self.rank)]
        self.padded_shape=[input_shape[i]+proj_start(i) for i in range(2,self.rank)]
        self.batch_time_transposition=[1,0]+list(range(2,self.rank))
        self.axis_last_transposition=list(range(self.axis))+[self.rank-1]+list(range(self.axis+1,self.rank-1))+[self.axis]

        @tf.function
        def rotate_and_project(input_batch,rotation_movie):
            N_BATCH=tf.shape(input_batch)[0]
            self.tiling_indices=tf.tensor_scatter_nd_update(tf.ones((3,),dtype=tf.int32),[[0]],[N_BATCH])

            # transform input from (x,y) to (0,x,y)
            paddings=[[0,0] for i in range(self.axis)]+[[1,0]]+[[0,0] for i in range(self.rank-self.axis-1)]
            padded_input=tf.pad(input_batch,paddings=paddings)

            #transpose to Time x Batch x ...
            padded_input=tf.transpose(padded_input,self.batch_time_transposition)
            rotated_input=tf.transpose(
                            tf.vectorized_map(
                            lambda xy: tf.tensordot(xy[0], xy[1], axes=[[self.axis-self.rank], [-1]]),
                            elems=(padded_input, rotation_movie)
                            ),
                            self.axis_last_transposition
                            )
            #transpose back to Batch x Time X ...
            rotated_input=tf.transpose(rotated_input,self.batch_time_transposition)
            #project back to plane
            return tf.slice(rotated_input,self.projection_slice_start,tf.shape(input_batch))

        @tf.function
        def rotate_only(input_batch,rotation_movie):
            #transpose to Time x Batch x ...
            transposed_input=tf.transpose(input_batch,self.batch_time_transposition)
            rotated_input=tf.transpose(
                            tf.vectorized_map(
                            lambda xy: tf.tensordot(xy[0], xy[1], axes=[[self.axis-self.rank], [-1]]),
                            elems=(transposed_input, rotation_movie)
                            ),
                            self.axis_last_transposition
                            )
            #transpose back to Batch x Time X ...
            rotated_input=tf.transpose(rotated_input,self.batch_time_transposition)
            return rotated_input

        if self.rotate_dim==2:
            self.rotate_fn=rotate_and_project
        elif self.rotate_dim==3:
            self.rotate_fn=rotate_only

    def call(self, input, training=None):
        if not training and self.debug==False:
            return input

        angle1=tf.random.uniform((),minval=self.min_angle,maxval=self.max_angle)
        angle1_movie=tf.slice(tf.range(angle1,delta=angle1/self.N_TIMESTEPS),[0],[self.N_TIMESTEPS])

        angle2=tf.random.uniform((),minval=self.min_angle,maxval=self.max_angle)
        angle2_movie=tf.slice(tf.range(angle2,delta=angle2/self.N_TIMESTEPS),[0],[self.N_TIMESTEPS])

        if self.rotate_dim==3:
            eye_dim=self.axis_dim

        elif self.rotate_dim==2:
            eye_dim=self.axis_dim+1
        rotation_eye=tf.eye(eye_dim)
        rotation_eye_movie=tf.reshape(
                        tf.tile(rotation_eye,[self.N_TIMESTEPS,1]),
                        (self.N_TIMESTEPS,eye_dim,eye_dim)
                        )
        rotation_eye_movie_transposed=tf.transpose(rotation_eye_movie,[1,2,0])
        rotation1_movie = tf.transpose(
                            tf.tensor_scatter_nd_update(rotation_eye_movie_transposed,
                                        [[0,0],[0,1],[1,0],[1,1]],
                                        [tf.math.cos(angle1_movie),-tf.math.sin(angle1_movie),
                                        tf.math.sin(angle1_movie),tf.math.cos(angle1_movie)]),
                            [2,0,1]
                            )

        rotation2_movie = tf.transpose(
                        tf.tensor_scatter_nd_update(rotation_eye_movie_transposed,
                                        [[1,1],[1,2],[2,1],[2,2]],
                                        [tf.math.cos(angle2_movie),-tf.math.sin(angle2_movie),
                                        tf.math.sin(angle2_movie),tf.math.cos(angle2_movie)]),
                        [2,0,1]
                        )

        rotation_movie=tf.vectorized_map(lambda xy: tf.tensordot(xy[0], xy[1], axes=[[-1], [-2]]),
               elems=(rotation1_movie, rotation2_movie))

        return self.rotate_fn(input,rotation_movie)
