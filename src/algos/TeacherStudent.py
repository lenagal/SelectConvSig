import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.algos.utils.scale_layer import scale
from src.preprocessing.UAVHData import UAVHData


class TeacherStudent:
    '''
    This class performs training following simple teacher-student scheme:
    ...itiratively using trained classifier-created soft distilled labels on
    ...(unlabelled) test set to train new classifer. Dataset used is UAVHuman
    Methods:
    getData: returns the UAVHuman data container
    student_train: performs the iterative training
    mix_data: collates train set and test set with classifier predicted labels.
    '''

    def __init__(self, teacher, student, temperature, signature, weights, batch_size, accumulate_steps=0):
        '''
        teacher: trained classifier object to predict initial labels
        student: student classifier that learns from mixed train and test set with teacher labels
        temperature: distillation coefficient for soft labels
        signature: signature degree
        weights: teachers weights
        batch_size: batch_size
        accumulate steps: increases batch size times the value
        ...(to work with batch sizes that don't fit in memory)
        '''
        self.student=student
        self.teacher=teacher
        self.T=temperature
        self.batch_size=batch_size
        self.accumulate_steps=accumulate_steps
        self.signature=signature
        self.weights=weights

    def getData(self):
        return UAVHData()

    def student_train(self, epochs, saved_path, iter, render_plot=False):
        '''
        Performs training following teacher-student scheme.
        Trained model's soft predicted labels on the test set are used to
        ...train the next model from scratch on the mixed train test set.
        epochs: number of training epochs in each iteration
        saved_path: path to save trained model
        iter: number of training cycles
        initial_weights: pre-trained weights file
        render_plot: whether to show training metrics plot
        '''
        print("Teacher-student training")
        data_wrapper = self.getData()
        # initialing teacher classifier
        teacher=self.teacher(signature_degree=self.signature, data_wrapper=data_wrapper, weights=self.weights, accumulate_steps=self.accumulate_steps)
        X_test, y_test=data_wrapper.load_full_landmarks(flag='test')
        X_train, y_train=data_wrapper.load_full_landmarks(flag='train')
        y_test_pred=tf.nn.softmax(scale(temp=self.T)(teacher.model.predict(X_test)))
        y_train_pred=tf.nn.softmax(scale(temp=self.T)(teacher.model.predict(X_train)))
        # creates a new dataset from test and train datasets and model predicted labels
        X_train, y_train=self.mix_data(y_train_pred,y_test_pred)
        print('Initial train data shape', X_train.shape, y_train.shape)
        history=[]

        for i in range(iter):
             student= self.student(signature_degree=self.signature, data_wrapper=data_wrapper, weights=None, accumulate_steps=self.accumulate_steps)
             print('classifier {} initialized'.format(i+1))
             h=student.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size)
             history.append(h)
             # Evaluating interim model on full test data
             print("Evaluate model {} on test data".format(i+1))
             results = student.model.evaluate(X_test, y_test, batch_size=self.batch_size)
             print("test loss, test acc:", results)
             y_test_pred=student.model.predict(X_test)
             X_train, y_train=self.mix_data(y_train_pred,y_test_pred)

        student.model.save_weights(saved_path)
        print('model saved at',saved_path)

        if render_plot:
            for h in history:
                fig,axs=plt.subplots(1,2)

                axs[0].plot(h.history['accuracy'])
                axs[0].set_title('accuracy')
                axs[0].set_ylabel('accuracy')
                axs[0].set_xlabel('epoch')
                axs[0].legend(['train'])

                axs[1].plot(h.history['loss'])
                axs[1].set_title('loss')
                axs[1].set_ylabel('loss')
                axs[1].set_xlabel('epoch')
                axs[1].legend(['train'])

            plt.show()

    def mix_data(self,y_train,y_test):
        '''
        Collates train set and test set with classifier predicted labels.
        y_test: test set labels
        '''
        data_wrapper=self.getData()
        X_train,train_labels=data_wrapper.load_full_landmarks(flag='train')
        X_test,test_labels=data_wrapper.load_full_landmarks(flag='test')
        print('debug:X_train,y_old,y_new,X_test,y_old,y_new',
            X_train.shape,train_labels.shape,y_train.shape,X_test.shape,test_labels.shape,y_test.shape)
        data=np.concatenate((X_train,X_test))
        labels=np.concatenate((y_train,y_test))

        def shuffle_in_unison(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)
            return a,b

        return shuffle_in_unison(data,labels)
