import tensorflow as tf
from tensorflow.keras.layers import Layer
import math


class ZoomMovie(Layer):
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

    def __init__(self,min_zoom=0,max_zoom=1,rescaled=True,debug=False,**kwargs):
        super(ZoomMovie, self).__init__(**kwargs)
        # self.axis=axis+1
        self.min_zoom=min_zoom
        self.max_zoom=max_zoom
        self.rescaled=rescaled

        self.debug=debug

    def build(self,input_shape):
        # self.axis_dim = input_shape[self.axis]
        self.rank=len(input_shape)
        self.batch_time_transposition=[1,0]+list(range(2,self.rank))
        self.N_TIMESTEPS=input_shape[1]

        @tf.function
        def zoom_movie(input_batch,zoom_ratio_movie):
            #transpose to Time x Batch x ...
            transposed_input=tf.transpose(input_batch,self.batch_time_transposition)
            zoomed_input=tf.vectorized_map(
                            lambda xy: tf.multiply(xy[0], xy[1]),
                            elems=(transposed_input, zoom_ratio_movie)
                            )
            #transpose back to Batch x Time X ...
            zoomed_input=tf.transpose(zoomed_input,self.batch_time_transposition)
            return zoomed_input

        self.zoom_fn=zoom_movie

    def call(self, input, training=None):
        if not training and self.debug==False:
            return input

        zoom_start_ratio=tf.random.uniform((),minval=self.min_zoom,maxval=self.max_zoom)
        zoom_end_ratio=tf.random.uniform((),minval=self.min_zoom,maxval=self.max_zoom)

        if self.rescaled:
            if zoom_start_ratio>zoom_end_ratio:
                zoom_end_ratio=zoom_end_ratio/zoom_start_ratio
                zoom_start_ratio=tf.constant(1.,dtype=tf.float32)
            else:
                zoom_start_ratio=zoom_start_ratio/zoom_end_ratio
                zoom_end_ratio=tf.constant(1.,dtype=tf.float32)

        zoom_ratio_movie=tf.slice(tf.range(zoom_start_ratio,zoom_end_ratio,delta=(zoom_end_ratio-zoom_start_ratio)/self.N_TIMESTEPS),[0],[self.N_TIMESTEPS])
        return self.zoom_fn(input,zoom_ratio_movie)
