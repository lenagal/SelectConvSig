import tensorflow as tf

@tf.function
def deg2_sig(path):
    '''
    computes signature of path of vectors to degree 2
    path - array (or tensor) of shape T x Batch x TimeBatch  x V

    output shape Batch x TimeBatch x Sig
    '''

    length=path.shape[0]
    space_dim=path.shape[3]
    time_batch_size=path.shape[2]
    #compute degree 1 components
    shifted_left=tf.roll(path,shift=-1,axis=0)
    deg1=(shifted_left-path)[:-1]
    #compute degree 2 components - need to adjust for batch processing

    deg2_list=[]
    for i in range(length-1):
        deg2_list.append(
                tf.vectorized_map(lambda z: tf.vectorized_map(lambda y: tf.vectorized_map(lambda x: tf.tensordot(x[0], x[1], axes=[[], []]),
               elems=(y[0], y[1])),
               elems=(z[0], z[1])),
               elems=(deg1, tf.roll(deg1,shift=-i,axis=0)))
               )
        #shape - length-1 x Batch x TimeBatch x space_dim x space_dim
    deg2=tf.concat(deg2_list,axis=0)
    #put them together
    deg1=tf.reduce_sum(deg1,axis=0)
    deg2=tf.reduce_sum(deg2,axis=0)/2
    deg2=tf.reshape(deg2,[-1,time_batch_size,space_dim*space_dim])

    return tf.concat([deg1,deg2],axis=-1)

@tf.function
def deg3_sig(path):
    '''
    computes signature of path of vectors to degree 3
    path - array (or tensor) of shape T x Batch x TimeBatch  x V

    output shape Batch x TimeBatch x Sig
    '''

    length=path.shape[0]
    space_dim=path.shape[3]
    time_batch_size=path.shape[2]
    #compute degree 1 components
    shifted_left=tf.roll(path,shift=-1,axis=0)
    deg1=(shifted_left-path)[:-1]
    #compute degree 2 components - need to adjust for batch processing

    deg2_list=[]
    for i in range(length-1):
        deg2_list.append(
                tf.vectorized_map(lambda z: tf.vectorized_map(lambda y: tf.vectorized_map(lambda x: tf.tensordot(x[0], x[1], axes=[[], []]),
               elems=(y[0], y[1])),
               elems=(z[0], z[1])),
               elems=(deg1, tf.roll(deg1,shift=-i,axis=0)))
               )
        #shape - length-1 x Batch x TimeBatch x space_dim x space_dim
    deg2=tf.concat(deg2_list,axis=0)

    deg3_list=[]
    for i in range(length-1):
        for j in range(length-1):
            deg3_list.append(
                    tf.vectorized_map(lambda z: tf.vectorized_map(lambda y: tf.vectorized_map(lambda x: tf.tensordot(x[0], x[1], axes=[[], []]),
                   elems=(y[0], y[1])),
                   elems=(z[0], z[1])),
                   elems=(deg2_list[i], tf.roll(deg1,shift=-j,axis=0)))
                   )
            #shape - length-1 x Batch x TimeBatch x space_dim x space_dim x space_dim

    deg3=tf.concat(deg3_list,axis=0)
    #put them together
    deg1=tf.reduce_sum(deg1,axis=0)
    deg2=tf.reduce_sum(deg2,axis=0)/2
    deg2=tf.reshape(deg2,[-1,time_batch_size,space_dim**2])
    deg3=tf.reduce_sum(deg3,axis=0)/6
    deg3=tf.reshape(deg2,[-1,time_batch_size,space_dim**3])

    return tf.concat([deg1,deg2,deg3],axis=-1)
