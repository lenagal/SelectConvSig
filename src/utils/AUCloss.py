import tensorflow as tf

class AUCLoss(tf.keras.losses.Loss):

    '''
    loss for optimizing the area under ROC curve
    '''

    def __init__(self, name="AUCLoss",**kwargs):
        super(AUCLoss,self).__init__(name=name,**kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):

        def difference_matrix(v,batch,threshold=0):
                # print('AUCLoss DEBUG batch:',batch,threshold)
                t=tf.ones([batch,1])*threshold
                return v @ tf.transpose(tf.ones([batch,1])) - tf.ones([batch,1]) @ tf.transpose(v) - t@tf.transpose(tf.ones([batch,1]))

        batch=tf.shape(y_true)[0]
        threshold=tf.constant(0.4,dtype=tf.float32)
        cost = tf.reduce_sum((tf.nn.relu(-difference_matrix(y_pred,batch,threshold)) * tf.nn.relu(difference_matrix(tf.cast(y_true,tf.float32),batch)))**2)
        return cost

# if __name__=='__main__':
#     loss=AUCLoss()
#     y=[[1.],[0.],[0.],[1.]]
#     x=[[0.5],[0.6],[0.8],[1.]]
#     print(loss(tf.constant(y),tf.constant(x)))
