import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class Trainer:
    '''
    This class realises different training modes for Human Actions classifiers.
    To work with different datasets one needs to realize a getData() method in a child class.
    Child classes currently realised are SelectConvSigNTU and SelectConvSigUAVH
    ...working with NTU RGB+D and UAVHuman datasets respectively.
    Methods:
    getData: abstract method that returns a dataset container class of type Data
    trial_train: trains the classifier on train data with fixed test and validation split
    finalize:Trains the classifier on full train data and evaluates on separate test data
    fine_tune_sig: Loads weights of a model trained with signature computed with lower degree
    ...of accuracy and trains it with signature computed up to higher term to improve the model accuracy.
    validate:divides the training set into 5 folds and performs cross-validation
    predict:predicts labels on test set using trained model and writes them to .csv file
    '''

    def __init__(self, classifier, batch_size, accumulate_steps=0):
        '''
        classifier: classifier object to train/predict
        batch_size: batch_size
        accumulate steps: increases batch size times the value
        ...(to work with batch sizes that don't fit in memory)
        '''
        self.classifier=classifier
        self.batch_size=batch_size
        self.accumulate_steps=accumulate_steps

    def getData(self):
        '''
        Abstract method that creates instance of either NTU RGB+D or UAVH dataset class
        ...returned object should be of type Data (should realize Data interface)
        '''
        raise NotImplementedError("Method getData was not implemented")


    def trial_train(self, signature, epochs, weights=None, saved_path=None, render_plot=True):
        '''Trains the classifier on train data with fixed test and validation split.
        ...For training models when testing is on unseen test data.
        signature: signature degree
        epochs: number of training epochs
        weights: pre-trained weights file
        saved_path: path to save trained model
        render_plot: whether to show training metrics plot
        '''
        print("Trial training....")
        data_wrapper = self.getData()
        classifier= self.classifier(signature_degree=signature, data_wrapper=data_wrapper,weights=weights, accumulate_steps=self.accumulate_steps)
        X_train, y_train, X_test, y_test = data_wrapper.load_landmarks(0)
        print('Trial train data shape', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        history=classifier.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size, validation_split=0.1)

        print("test loss, test acc:", results)
        if saved_path is not None:
            classifier.model.save_weights(saved_path)
            print('model saved at',saved_path)

        # Plot training accuracy
        if render_plot:

            fig,axs=plt.subplots(1,2)

            axs[0].plot(history.history['accuracy'])
            axs[0].plot(history.history['val_accuracy'])
            axs[0].set_title('accuracy')
            axs[0].set_ylabel('accuracy')
            axs[0].set_xlabel('epoch')
            axs[0].legend(['train', 'test'])

            axs[1].plot(history.history['loss'])
            axs[1].plot(history.history['val_loss'])
            axs[1].set_title('loss')
            axs[1].set_ylabel('loss')
            axs[1].set_xlabel('epoch')
            axs[1].legend(['train', 'test'])

            plt.show()


    def finalize(self, modes, signature, epochs, saved_paths, weights=None, render_plot=True):
        '''Trains the classifier on full train data and evaluates on separate test data.
        modes: defines test-train evaluation split of the dataset. Currently implemented
        ...'regular' for UAVHuman;
        'cross-view', 'cross-subject', 'cross-setup' for NTU-RGB+D;
        signature: signature degree
        epochs: number of training epochs
        weights: pre-trained weights file
        saved_paths: paths to save trained models
        render_plot: whether to show training metrics plot
        '''
        print("Final model training....")
        paths=saved_paths.copy()
        data_wrapper = self.getData()
        history=[]
        results=[]

        for mode in modes:
            X_train, y_train=data_wrapper.load_full_landmarks(flag='train',mode=mode)
            X_test, y_test=data_wrapper.load_full_landmarks(flag='test',mode=mode)

            print('data shape:train', X_train.shape, y_train.shape,'test',X_test.shape, y_test.shape)
            classifier= self.classifier(signature_degree=signature, data_wrapper=data_wrapper, weights=weights, accumulate_steps=self.accumulate_steps)
            print('classifier initialized')
            h=classifier.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size)
            history.append(h)
            print("Evaluate on test data")
            r = classifier.model.evaluate(X_test, y_test, batch_size=self.batch_size)
            results.append(r)

            if saved_paths is not None:
                classifier.model.save_weights(saved_paths.pop(0))

        if paths is not None:
            print('model saved at',paths)

        print("test loss, test acc:", results)

        # Plot training accuracy
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
        return paths

    def fine_tune_sig(self, old_sig, new_sig, weights, saved_path, epochs, lr, render_plot=True):
        '''
        Loads weights of a model trained with signature computed with lower degree (up to term old_sig)
        ...and trains it with signature computed up to term new_sig to improve the model accuracy.
        old_sig: number of loaded model signature terms (old signature degree)
        new_sig: number of new signature terms (new signature degree)
        weights: pre-trained weights file
        saved_path: path to save trained model
        epochs: number of training epochs
        lr: learning rate
        render_plot: whether to show training metrics plot
        '''
        if old_sig<new_sig:
            print("Continue training with signature",new_sig)
        else:
            raise ValueError("old_sig value should be less than new_sig")

        data_wrapper = self.getData()
        trained_classifier=self.classifier(signature_degree=old_sig, data_wrapper=data_wrapper,weights=weights,accumulate_steps=self.accumulate_steps)

        W = trained_classifier.model.get_layer('lstm_layer_sig_'+str(old_sig)).get_weights()[0]
        U = trained_classifier.model.get_layer('lstm_layer_sig_'+str(old_sig)).get_weights()[1]
        b = trained_classifier.model.get_layer('lstm_layer_sig_'+str(old_sig)).get_weights()[2]
        print('lstm_layer shape', old_sig, W.shape, U.shape, b.shape)

        new_classifier=self.classifier(signature_degree=new_sig, data_wrapper=data_wrapper, weights=None,accumulate_steps=self.accumulate_steps)
        initial_weights = [layer.get_weights() for layer in new_classifier.model.layers]
        print('load weights to layers')
        new_classifier.model.load_weights(weights, by_name=True)

        print('load weights to output layer')
        new_classifier.model.get_layer(index=-1).set_weights(trained_classifier.model.get_layer(index=-1).get_weights())

        for layer, initial in zip(new_classifier.model.layers, initial_weights):
            weights = layer.get_weights()
            if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                print(f'Checkpoint contained no weights for layer {layer.name}!')

        print('-------------------')
        print('analyze LSTM layer')
        W_n = new_classifier.model.get_layer('lstm_layer_sig_'+str(new_sig)).get_weights()[0]
        U_n = new_classifier.model.get_layer('lstm_layer_sig_'+str(new_sig)).get_weights()[1]
        b_n = new_classifier.model.get_layer('lstm_layer_sig_'+str(new_sig)).get_weights()[2]
        print('lstm_layer shape', new_sig, W_n.shape, U_n.shape, b_n.shape)
        W=np.pad(W,((0,W_n.shape[0]-W.shape[0]),(0,0)),'constant')
        print(W)
        new_classifier.model.get_layer('lstm_layer_sig_'+str(new_sig)).set_weights((W,U,b))

        if np.array_equal(new_classifier.model.get_layer('lstm_layer_sig_'+str(new_sig)).get_weights()[0],W):
            print('weigths were set succesfully')
        else:
            print('weigths were not set')
        X_train, y_train = data_wrapper.load_full_landmarks(flag='train')
        X_test, y_test=data_wrapper.load_full_landmarks(flag='test')

        adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        print("Evaluate trained model on on test data")
        trained_classifier.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])
        results = trained_classifier.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print("test loss, test acc:", results)

        print("Evaluate new model on on test data")
        adam1 = Adam(learning_rate=1e-05, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        new_classifier.model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics = ['accuracy'])
        results = new_classifier.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print("test loss, test acc:", results)

        print('run classifier with signature level'+str(new_sig))
        history=new_classifier.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size)
        print("Evaluate on test data")
        results = new_classifier.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print("test loss, test acc:", results)

        if saved_path is not None:
            new_classifier.model.save_weights(saved_path)
            print('model saved at',saved_path)

    def validate(self,signature,epochs):
        '''
        Divides the training set into 5 folds and performs cross-validation
        ...For training models when testing is on unseen test data.
        signature: signature degree
        epochs: number of training epochs
        '''
        print("Model Cross-validation")
        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []

        for idx in range(5):
            data_wrapper = self.getData()
            X_train, y_train, X_test, y_test=data_wrapper.load_landmarks(idx)

            classifier= self.classifier(signature_degree=signature, data_wrapper=data_wrapper,weights=None, accumulate_steps=self.accumulate_steps)
            print('classifier initialized')
            history=classifier.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size)
            scores = classifier.model.evaluate(X_test, y_test, verbose=0)

            print(f'Score for fold {idx+1}: {classifier.model.metrics_names[0]} of {scores[0]}; {classifier.model.metrics_names[1]} of {scores[1]*100}%')

            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0] * 100)

        #Provide average scores
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
            print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')

    def predict(self,signature, weights, predictions=None):
        '''
        Predicts labels on test set using trained model and writes them to .csv file
        signature: signature: signature degree
        weights: path to pre-trained weights file
        predictions: path to predictions file
        '''
        print("Predict test labels")
        data_wrapper = self.getData()
        X_test, y_test = data_wrapper.load_full_landmarks(flag='test')
        print('Test data loaded. Shape:', X_test.shape)
        classifier= self.classifier(signature_degree=signature, data_wrapper=data_wrapper,weights=weights,accumulate_steps=self.accumulate_steps)
        print('classifier initialized')
        results= classifier.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print("test loss, test acc:", results)
        pred_hot=classifier.model.predict(X_test)
        pred=np.argmax(pred_hot,axis=-1)

        if predictions is not None:
            predictions_list=[]
            for i,result in np.ndenumerate(pred):
                predictions_list.append({'index':i[0],'category':result})

            predictions_dataframe=pd.DataFrame(predictions_list)
            predictions_dataframe.to_csv(predictions,index=False)
            print("Predictions saved at",predictions)

        # print('Sanity check: evaluate on train data')
        # X_train, y_train=data_wrapper.load_full_landmarks(flag='train')
        # results = classifier.model.evaluate(X_train, y_train, batch_size=self.batch_size)
        # print("test loss, test acc:", results)
        return pred_hot


    def student_train(self, signature, epochs, saved_path, iter, initial_weights=None,render_plot=False):
        '''
        Performs training following teacher-student scheme.
        Trained model's "hard" (one-hot) predicted labels on the test set are used to
        ...train the next model from scratch on the mixed train test set.
        signature: signature degree
        epochs: number of training epochs
        saved_path: path to save trained model
        iter: number of training cycles
        initial_weights: pre-trained weights file
        render_plot: whether to show training metrics plot
        '''
        print("Teacher-student training")
        data_wrapper = self.getData()
        X_test, y_test=data_wrapper.load_full_landmarks(flag='test')

        if initial_weights is not None:
            X_train, y_train=self.mix_data(self.predict(signature,initial_weights))
        else:
            X_train, y_train=data_wrapper.load_full_landmarks(flag='train')

        print('Initial train data shape', X_train.shape, y_train.shape)

        for i in range(iter):
             classifier= self.classifier(signature_degree=signature, data_wrapper=data_wrapper, weights=None, accumulate_steps=self.accumulate_steps)
             print('classifier {} initialized'.format(i+1))
             history=classifier.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size)
             # Evaluating interim model on full test data
             print("Evaluate model {} on test data".format(i+1))
             results = classifier.model.evaluate(X_test, y_test, batch_size=self.batch_size)
             print("test loss, test acc:", results)
             X_train,y_train=self.mix_data(classifier.model.predict(X_test))

        classifier.model.save_weights(saved_path)
        print('model saved at',saved_path)

        if render_plot:

            fig,axs=plt.subplots(1,2)

            axs[0].plot(history.history['accuracy'])
            axs[0].plot(history.history['val_accuracy'])
            axs[0].set_title('accuracy')
            axs[0].set_ylabel('accuracy')
            axs[0].set_xlabel('epoch')
            axs[0].legend(['train', 'test'])

            axs[1].plot(history.history['loss'])
            axs[1].plot(history.history['val_loss'])
            axs[1].set_title('loss')
            axs[1].set_ylabel('loss')
            axs[1].set_xlabel('epoch')
            axs[1].legend(['train', 'test'])

            plt.show()

    def mix_data(self,y_test):
        '''
        Collates train set and test set with classifier predicted labels.
        y_test: test set labels
        '''
        data_wrapper=self.getData()
        X_train, y_train=data_wrapper.load_full_landmarks(flag='train')
        X_test,y=data_wrapper.load_full_landmarks(flag='test')
        print('debug:Xtrain,ytrain,Xtest,ytest,yold',X_train.shape,y_train.shape,X_test.shape,y_test.shape,y.shape)
        data=np.concatenate((X_train,X_test))
        labels=np.concatenate((y_train,y_test))

        def shuffle_in_unison(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)
            return a,b

        return shuffle_in_unison(data,labels)
