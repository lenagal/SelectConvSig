from psfdataset import transforms, PSFDataset, PSFZippedDataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from src.Autoencoder import SigAutoencoder
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from src.data_grabbing.data_grabber import DataGrabber
from sklearn.model_selection import StratifiedKFold
from src.algos.utils import SkeletonUtils

class LinConvData:
    SINGLE_LABELS=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
    23,24,25,26,27,28,29,30,31,32,33,34,35,
    36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,
    72,73,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,
    117,118,119,120,121,122,123,124,125,126,127,128,129,136,137,138,139,140,141,
    142,143,144,145,147,149,150,151,152,153,154]
    DOUBLE_LABELS=[74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,
    93,94,95,96,97,98,130,131,132,133,134,135,146,148]
    # JOINT_LABELS=[75,76,78,88,89,93,95,97,98,130,132,133]

    DEFAULT_DATA_PATH="/scratch/gale/UAVHuman/newData/"

    def __init__(self,pre_normaliser=None,transform='landmarks',classes=155, debug=False, bodies='both',
                autoencoded_latent_dim=50,load_autoencoded_data=False,add_center_joint=False,with_center_joint=True,
                load_dejanked_data=False, no_head=False,
                data_path=DEFAULT_DATA_PATH):
        '''
        Args:
        input_shape: (N_TIMESTEPS, N_AXES, N_JOINTS, N_PERSONS)
        pre_normaliser: object is defined if data is not to be loaded. The path signature features are
            ... directly created using 'self.sig_transform' and saved in 'self.sig_data' (a pair (trainingset, valset)
            ... of class PSFDataset or PSFZippedDataset.
        transform: shape of extracted data
        '''
        self.no_head=no_head
        self.N_TIMESTEPS = 305
        self.N_AXES =2
        self.N_JOINTS = SkeletonUtils.UAVHuman_Joints(no_head=self.no_head,with_center=with_center_joint)
        self.N_PERSONS = 2
        self.N_CLASSES=classes
        self.transform = transform
        self._debug = debug
        self.pre_normaliser = pre_normaliser
        self.path=data_path
        self.bodies=bodies
        self.load_autoencoded_data=load_autoencoded_data
        self.datagrabber=DataGrabber()
        self.fold_idx_dict=self._cross_validation_fold()
        self.autoencoded_latent_dim=autoencoded_latent_dim
        self.add_center_joint=add_center_joint
        self.load_with_center_joint=with_center_joint
        self.load_dejanked_data=load_dejanked_data

        if self.add_center_joint or self.load_with_center_joint:
            self.N_JOINTS = SkeletonUtils.UAVHuman_Joints(with_center=True,no_head=self.no_head)
            self.joints_to_bones_matrix=SkeletonUtils.joints_to_bones_matrix(SkeletonUtils.UAVHuman_Joints_graph_edges(no_head=self.no_head),self.N_JOINTS)
            self.N_BONES = len(SkeletonUtils.UAVHuman_Joints_graph_edges(no_head=self.no_head))
        if pre_normaliser is None:
            self.load_transform=True
        else:
            self.load_transform=False

        # if self.transform=='landmarks':
        #      self.train_data, self.train_label, self.val_data, self.val_label=self.load_landmarks()
    def _allowed_labels(self):
        if self.bodies=='both':
            return range(self.N_CLASSES)
        elif self.bodies=='single':
            return LinConvData.SINGLE_LABELS
        elif self.bodies=='double':
            return LinConvData.DOUBLE_LABELS

    def _label_translator(self,label):
        ''' takes original label and gives new label, i.e. the position of the
        original label in the current allowed label list. throws ValueError if
        not present'''
        return self._allowed_labels().index(label)

    def _inverse_label_translator(self,new_label):
        return self._allowed_labels()[new_label]

    def _cross_validation_fold(self,data_input=None,cross_val_fold_num=5,verbose=0,process_labels=True):
        if data_input is None:
            data_uav, label_uav = self.datagrabber.train_data, self.datagrabber.train_label
        elif data_input=='load':
            data_uav, label_uav = self.datagrabber._load_uav_data()
        else:
            data_uav, label_uav = data_input
        # length_uav = lengths   ER: ???
        fold_num=cross_val_fold_num

        if self.pre_normaliser is not None:
            if self.pre_normaliser.debug:
                data_uav=data_uav[:self.pre_normaliser.debug]
                label_uav=label_uav[:self.pre_normaliser.debug]
                fold_num=2
        if process_labels:
            label_uav=self.get_body_type_labels(label_uav)
        data_uav=data_uav[:label_uav.shape[0]]


        def verboseprint(*args,**kwargs):
            if verbose==0:
                return
            else:
                print(*args,**kwargs)

        # LABELS=self._allowed_labels()
        # class_num=len(LABELS)
        # verboseprint('Check data imbalance...')
        # class_cnt = np.zeros((class_num,)))
        # for l in label_uav:
        #     class_cnt[l] += 1
        # verboseprint(class_cnt)
        # verboseprint('Avg sample num: ',class_cnt.mean())
        # verboseprint('Max sample num: ',class_cnt.max())
        # verboseprint('Min sample num: ',class_cnt.min())

        k_fold = StratifiedKFold(cross_val_fold_num)
        k_fold.get_n_splits(data_uav,label_uav)
        k_fold_idx_dict = dict()

        verboseprint('Create {}-fold for cross validation...'.format(cross_val_fold_num))
        for k, (train_idx, val_idx) in enumerate(k_fold.split(data_uav,label_uav)):
            k_fold_idx_dict.update({str(k):{'train':train_idx, 'val':val_idx}})
            verboseprint(k+1,'- fold:','Trainset size: ',len(train_idx),' Valset size: ',len(val_idx))
        return k_fold_idx_dict

    def get_body_type(self,data,labels,by_label=False,translate_labels=True):
        '''get data and labels for single,double or both bodies from preNormaliser
            shape of input data = Batch x Coords x Time x Joints x People
            shape of output data = Batch x Time x Coords x (Joints x People)
        '''
        if labels is not None and len(labels.shape)>1:
            labels=np.argmax(labels, axis=1)
        B, C, T, J, P = data.shape

        if by_label:
            if self.bodies=='both':
                return data.reshape((B,C,T,J*P)).transpose([0,2,1,3]), labels
            elif self.bodies=='single':
                single_indices=[]
                for i in range(labels.shape[0]):
                    if labels[body] in self._allowed_labels():
                        single_indices.append(i)
                if translate_labels:
                    print('Translating single body labels')
                    for i in single_indices:
                        new_label=self._label_translator(labels[body])
                        np.put(labels,i,new_label)
                return data[single_indices,:,:,:,0].transpose([0,2,1,3]),labels[single_indices]
            elif self.bodies=='double':
                double_indices=[]
                for i in range(labels.shape[0]):
                    if labels[body] in self._allowed_labels():
                        double_indices.append(i)
                if translate_labels:
                    print('Translating double body labels')
                    for i in double_indices:
                        new_label=self._label_translator(labels[body])
                        np.put(labels,i,new_label)
                return data[double_indices].reshape((-1,C,T,J*P)).transpose([0,2,1,3]),labels[double_indices]
        else:
            if self.bodies=='both':
                if labels is None:
                    return data.reshape((B,C,T,J*P)).transpose([0,2,1,3])
                else:
                    return data.reshape((B,C,T,J*P)).transpose([0,2,1,3]), labels
            elif self.bodies=='single':
                single_indices=[]
                for i in range(data.shape[0]):
                    if np.array_equal(data[i,:,:,:,1],np.zeros_like(data[i,:,:,:,1])):
                        single_indices.append(i)
                if translate_labels and labels is not None:
                    for i in single_indices:
                        new_label=self._label_translator(labels[body])
                        np.put(labels,i,new_label)
                if labels is None:
                    return data[single_indices,:,:,:,0].transpose([0,2,1,3])
                else:
                    return data[single_indices,:,:,:,0].transpose([0,2,1,3]),labels[single_indices]
            elif self.bodies=='double':
                double_indices=[]
                for i in range(data.shape[0]):
                    if not np.array_equal(data[i,:,:,:,1],np.zeros_like(data[i,:,:,:,1])):
                        double_indices.append(i)
                if translate_labels and labels is not None:
                    for i in double_indices:
                        new_label=self._label_translator(labels[body])
                        np.put(labels,i,new_label)
                if labels is None:
                    return data[double_indices].reshape((-1,C,T,J*P)).transpose([0,2,1,3])
                else:
                    return data[double_indices].reshape((-1,C,T,J*P)).transpose([0,2,1,3]),labels[double_indices]

    def get_body_type_labels(self,labels,translate_labels=True):

        LABELS=self._allowed_labels()

        label_list=[]
        for i in range(labels.shape[0]):
            if labels[i] in LABELS:
                label_list.append(i)
        if translate_labels:
            for i in label_list:
                new_label=self._label_translator(labels[i])
                np.put(labels,i,new_label)
        return labels[label_list]

    def _load_data(self,fold_idx=0,full_data=False):
        '''
            Will only be applied if data is to be loaded. Loads the data and the fold indices from the prenormaliser.

        '''
        isFastTestSubset = self._debug # --- for preliminary testing purposes only

        data_uav,label_uav = self.load_full_landmarks('train')

        print('Prenormalized data shape:',data_uav.shape)
        print('Prenormalized label shape',label_uav.shape)

        train_index = self.fold_idx_dict[str(fold_idx)]['train'] if not isFastTestSubset else fold_idx_dict[str(fold_idx)]['train'][::100]
        val_index = self.fold_idx_dict[str(fold_idx)]['val'] if not isFastTestSubset else fold_idx_dict[str(fold_idx)]['val'][::100]

        train_data = data_uav[train_index]
        train_label = label_uav[train_index]
        # train_length = length_uav[train_index]   # see Weixin's notebook: can be used to add the temporal length of a movie as a feature; then self._get_iter(...) has to be adapted

        val_data = data_uav[val_index]
        val_label = label_uav[val_index]
        # val_length = length_uav[val_index]   # see above

        print('Test split data shape', train_data.shape, val_data.shape)

        return train_data, train_label, val_data, val_label

    def _get_iter(self,data,label):
        """
        Returns iterators over the requested
        subset of the data.
        """
        return iter(list(zip(data, label)))

    def load_autoencoded_signature(self,fold_idx=0):
        path=self.path
        train_data = np.load(os.path.join(path,'{}_body_{}_sig_autoencoded.npy'.format(self.bodies,'train')))
        val_data = np.load(os.path.join(path,'{}_body_{}_sig_autoencoded.npy'.format(self.bodies,'val')))
        train_label=np.load(path+self.bodies+'_body_uav_train_landmarks_labels.npy')
        val_label=np.load(path+self.bodies+'_body_uav_val_landmarks_labels.npy')

        return train_data,train_label,val_data,val_label

    def load_full_autoencoded_signature(self,flag):
        path=self.path

        label=np.load(os.path.join(path,'{}_body_full_{}_landmarks_labels.npy'.format(self.bodies,flag)))
        data = np.load(os.path.join(path,'{}_body_full_{}_sig_autoencoded.npy'.format(self.bodies,flag)))
        return data, label

    def load_autoencoded(self,fold_idx=0):
        path = self.path
        latent_dim=self.autoencoded_latent_dim

        if self.load_autoencoded==True:
            print('Load autoencoded trial train data')
            train_data=np.load(os.path.join(path,self.bodies+'_body_trial_{}_autoencoded_dim_{}.npy'.format('train',latent_dim)))
            val_data=np.load(os.path.join(path,self.bodies+'_body_trial_{}_autoencoded_dim_{}.npy'.format('val',latent_dim)))

            train_label=np.load(path+self.bodies+'_body_uav_train_landmarks_labels.npy')
            val_label=np.load(path+self.bodies+'_body_uav_val_landmarks_labels.npy')

            return train_data, train_label, val_data, val_label
        else:
            dimV=2
            train_data, train_label, val_data, val_label = self.load_landmarks(fold_idx)
            N_JOINTS=train_data.shape[3]

            if self.bodies=='single':
                autoencoder=SigAutoencoder(latent_dim,N_JOINTS=N_JOINTS,load_weights=True,save_file_name_suffix='_single_body')
            else:
                autoencoder=SigAutoencoder(latent_dim,N_JOINTS=N_JOINTS,load_weights=True,save_file_name_suffix='_double_body')

            print('Encoding data using autoencoder for',self.bodies,'body type')
            def encode_data(data):
                '''input shape B x T X C X (JxP)
                    output shape B x T x LD'''
                B,T,C,JP=data.shape
                LD=latent_dim
                data=data.transpose([0,1,3,2])
                print('encode_data debug 1:',data.shape)
                data=data.reshape((-1, JP,C))

                data=autoencoder.encode(data[:,:,:dimV])
                return data.reshape((B,T,LD))

            train_data=encode_data(train_data)
            val_data=encode_data(val_data)

            print('Saving autoencoded trial train data')
            np.save(os.path.join(path,self.bodies+'_body_trial_{}_autoencoded_dim_{}.npy'.format('train',latent_dim)),train_data)
            np.save(os.path.join(path,self.bodies+'_body_trial_{}_autoencoded_dim_{}.npy'.format('val',latent_dim)),val_data)

        print('autoencoded data shape',train_data.shape, val_data.shape)

        return train_data, train_label, val_data, val_label

    def load_full_autoencoded(self,flag='train'):
        path = self.path
        latent_dim=self.autoencoded_latent_dim

        if self.load_autoencoded==True:
            print('Load autoencoded trial train data')
            data=np.load(os.path.join(path,self.bodies+'_body_full_{}_autoencoded_dim_{}.npy'.format(flag,latent_dim)))

            label=np.load(os.path.join(path,self.bodies+'_body_full_{}_landmarks_labels.npy'.format(flag)))

            return data, label
        else:
            dimV=2
            data, label = self.load_full_landmarks(flag)
            N_JOINTS=data.shape[3]

            if self.bodies=='single':
                autoencoder=SigAutoencoder(latent_dim,N_JOINTS=N_JOINTS,load_weights=True,save_file_name_suffix='_single_body')
            else:
                autoencoder=SigAutoencoder(latent_dim,N_JOINTS=N_JOINTS,load_weights=True,save_file_name_suffix='_double_body')

            print('Encoding data using autoencoder for',self.bodies,'body type')
            def encode_data(data):
                '''input shape B x T x C x (JxP)
                    output shape B x T x LD'''
                B,T,C,JP=data.shape
                LD=latent_dim
                data=data.transpose([0,1,3,2])
                print('encode_data debug 2:',data.shape)
                data=data.reshape((-1, JP,C))

                data=autoencoder.encode(data[:,:,:dimV])
                return data.reshape((B,T,LD))

            data=encode_data(data)

            print('Saving autoencoded full',flag,'data')
            np.save(os.path.join(path,self.bodies+'_body_trial_{}_autoencoded_dim_{}.npy'.format(flag,latent_dim)),data)

        print('autoencoded data shape',data.shape)

        return data,label

    def load_landmarks(self,fold_idx=0):
        print('load_landmarks(',fold_idx,')')
        data, labels = self.load_full_landmarks('train')

        train_index = self.fold_idx_dict[str(fold_idx)]['train']
        val_index = self.fold_idx_dict[str(fold_idx)]['val']

        train_data = data[train_index]
        train_label = labels[train_index]

        val_data = data[val_index]
        val_label = labels[val_index]

        return train_data,train_label,val_data,val_label

    def load_sampled_landmarks(self,fold_idx=0):
        '''
            Either loads the landmark data from path, or creates from pre_normaliser and saves to file

        '''

        path = self.path
        if self.load_transform==True:
            print('Load data')
            train_data=np.load(path+self.bodies+'_body_sampled_train_landmarks.npy')
            train_label_trans=np.load(path+self.bodies+'_body_sampled_train_landmarks_labels.npy')

            val_data=np.load(path+self.bodies+'_body_sampled_val_landmarks.npy')
            val_label_trans=np.load(path+self.bodies+'_body_sampled_val_landmarks_labels.npy')

            print('reshaped data',train_data.shape, val_data.shape)
            return train_data,train_label_trans, val_data, val_label_trans

        else:
            print('Create data')

            train_data, train_label, val_data, val_label = self._load_data(fold_idx)
            mask = np.random.choice(self.N_TIMESTEPS,100,replace=False)
            # transforming data [N,C,T,V,M] to [N,T,V,M,C] to [N,T,C,V*M] -- changed AG
            # train_data = np.transpose(train_data,(0,2,1,3))
            # train_data=train_data.reshape(train_data.shape[:3] + (self.N_JOINTS*self.N_PERSONS,))
            # val_data = np.transpose(val_data,(0,2,1,3))
            # val_data=val_data.reshape(val_data.shape[:3]+(self.N_JOINTS*self.N_PERSONS,))
            train_label_trans = to_categorical(train_label,len(self._allowed_labels()))
            val_label_trans =to_categorical(val_label,len(self._allowed_labels()))

            print('reshaped data',train_data.shape, val_data.shape)

            print('Save landmark data')
            np.save(path+self.bodies+'_body_sampled_train_landmarks.npy', train_data[:,mask])
            np.save(path+self.bodies+'_body_sampled_train_landmarks_labels.npy', train_label_trans)

            np.save(path+self.bodies+'_body_sampled_val_landmarks.npy', val_data[:,mask])
            np.save(path+self.bodies+'_body_sampled_val_landmarks_labels.npy', val_label_trans)


            return train_data[:,mask],train_label_trans[:,mask], val_data[:,mask], val_label_trans[:,mask]


    def load_test_train(self,fold_idx=0):
        '''
            Either loads the landmark data from path, or creates from pre_normaliser and saves to file
            The path is ~/scr/preprocessing/LinConvData.
        '''
        path = self.path
        print('Load test and train data')

        data=np.concatenate((np.load(os.path.join(path,self.bodies+'_body_full_train_landmarks.npy')),
            np.load(os.path.join(path,self.bodies+'_body_full_test_landmarks.npy'))),
            axis=0)
        label=np.concatenate((np.load(os.path.join(path,self.bodies+'_body_full_train_landmarks_labels.npy')),
            np.load(os.path.join(path,self.bodies+'_body_full_test_landmarks_labels.npy'))),
            axis=0)
        print("full data shape", data.shape, label.shape)

        def shuffle_in_unison(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        shuffle_in_unison(data,label)

        fold_idx_dict=self._cross_validation_fold(data_input=(data,np.argmax(label,axis=1)),process_labels=False)
        train_index = fold_idx_dict[str(fold_idx)]['train']
        val_index = fold_idx_dict[str(fold_idx)]['val']

        train_data = data[train_index]
        train_label = label[train_index]

        val_data = data[val_index]
        val_label = label[val_index]

        print('Test split data shape', train_data.shape, val_data.shape)

        return train_data,train_label, val_data, val_label

    def load_pairwise_landmarks(self, classes):
        X_train, y_train=self.load_full_landmarks(flag='train')
        X_test, y_test=self.load_full_landmarks(flag='test')

        ind_classes_tr=np.nonzero(y_train[:,classes])
        ind_classes_test=np.nonzero(y_test[:,classes])

        X_train_cut=X_train[ind_classes_tr[0]]
        X_test_cut=X_test[ind_classes_test[0]]
        y_train_cut=y_train[ind_classes_tr[0]]
        y_train_cut=y_train_cut[:,classes]
        y_train_cut=np.argmax(y_train_cut,1)

        y_test_cut=y_test[ind_classes_test[0]]
        y_test_cut=y_test_cut[:,classes]
        y_test_cut=np.argmax(y_test_cut,1)

        y_train_cut=y_train_cut.reshape(-1,1)
        y_test_cut=y_test_cut.reshape(-1,1)

        return X_train_cut, y_train_cut, X_test_cut, y_test_cut

    def load_full_landmarks(self, flag='train',raw=False):

        path = self.path
        if self.load_transform==True:
            print('Load full',flag,'data')
            if self.load_with_center_joint:
                if self.load_dejanked_data:
                    data=np.load(os.path.join(path,self.bodies+'_body_full_{}_landmarks_dejanked.npy'.format(flag)))
                    label_trans=np.load(os.path.join(path,self.bodies+'_body_full_{}_label_dejanked.npy'.format(flag)))

                    print('loading dejanked',flag,'data')
                else:
                    if raw:
                        label_trans=np.load(os.path.join(path,self.bodies+'_body_full_{}_landmarks_labels.npy'.format(flag)))
                        data=np.load(os.path.join(path,self.bodies+'_body_full_{}_landmarks_with_center_raw.npy'.format(flag)))
                    else:
                        label_trans=np.load(os.path.join(path,self.bodies+'_body_full_{}_landmarks_labels.npy'.format(flag)))
                        data=np.load(os.path.join(path,self.bodies+'_body_full_{}_landmarks_with_center.npy'.format(flag)))
            else:
                if self.load_dejanked_data:
                    data=np.load(os.path.join(path,self.bodies+'_body_full_{}_landmarks_dejanked.npy'.format(flag)))
                    label_trans=np.load(os.path.join(path,self.bodies+'_body_full_{}_label_dejanked.npy'.format(flag)))
                    data=data.reshape((-1,305,2,18,2))[:,:,:,:17,:].reshape((-1,305,2,34))
                    print('loading dejanked',flag,'data')
                else:
                    if raw:
                        data=np.load(os.path.join(path,self.bodies+'_body_full_raw_{}_landmarks.npy'.format(flag)))
                        label_trans=np.load(os.path.join(path,self.bodies+'_body_full_raw_{}_landmarks_labels.npy'.format(flag)))
                        print('raw')
                    else:
                        data=np.load(os.path.join(path,self.bodies+'_body_full_{}_landmarks.npy'.format(flag)))
                        label_trans=np.load(os.path.join(path,self.bodies+'_body_full_{}_landmarks_labels.npy'.format(flag)))
                        print('scaled')
            # data=np.load(os.path.join(path,'{}_data.npy'.format(flag)))
            # with open(os.path.join(path,'{}_label.pkl'.format(flag)), 'rb') as f:
            #         print('loading labels')
            #         sample_name, label_uav = pickle.load(f)
            #         label = np.array(label_uav)
            # data = np.transpose(data,(0,2,1,3,4))
            # data=data.reshape(data.shape[:3] + (self.N_JOINTS*self.N_PERSONS,))
            # label_trans = to_categorical(label,self.N_CLASSES)

        else:
            print('Create full data')
            if flag=='train':
                data = self.pre_normaliser.train_prenorm_data.copy()
                label = self.pre_normaliser.train_prenorm_label.copy()
            elif flag=='test':
                data = self.pre_normaliser.test_prenorm_data.copy()
                label = self.pre_normaliser.test_prenorm_label.copy()

            print('Prenormalized data shape:',data.shape)
            print('Prenormalized label shape',label.shape)
            # transforming data [N,C,T,V,M] to [N,T,V,M,C] to [N,T,C,V*M]
            # data = np.transpose(data,(0,2,1,3,4))
            # print('transposed data shape', data.shape)
            # data=data.reshape(data.shape[:3] + (self.N_JOINTS*self.N_PERSONS,))
            # print('reshapen data shape', data.shape)
            # done in get_body_type AG

            print('Get',self.bodies,'body data')
            data,label = self.get_body_type(data,label,by_label=True)
            print(data.shape,label.shape)
            label_trans = to_categorical(label,len(self._allowed_labels()))
            print('Save landmark data')

            np.save(os.path.join(path,self.bodies+'_body_full_{}_landmarks.npy'.format(flag)), data)
            np.save(os.path.join(path,self.bodies+'_body_full_{}_landmarks_labels.npy'.format(flag)), label_trans)

        if self.add_center_joint:
            print('Adding center joint as joint 17')
            data=self._add_center_joint(data)
            np.save(os.path.join(path,self.bodies+'_body_full_{}_landmarks_with_center.npy'.format(flag)), data)
        if self.no_head:
            shape=data.shape
            data=data.reshape(shape[:3]+(int(shape[3]/2),2))
            data=data[:,:,:,5:]
            data=data.reshape(shape[:3]+(self.N_JOINTS*2,))
        return data, label_trans

    def _add_center_joint(self,data):
        if self.bodies=='single':
            bodies=1
        else:
            bodies=2
        data=data.reshape((-1,305,3,17,bodies))
        new_data=np.zeros((data.shape[0],305,3,18,bodies))
        new_data[:,:,:,:17,:]=data

        for i in range(len(data)):
            print(i,'added',end='\r')
            for frame in range(305):
                for body in range(bodies):
                    x_center=(data[i,frame,0,5,body]+data[i,frame,0,6,body]+data[i,frame,0,11,body]+data[i,frame,0,12,body])/4
                    y_center=(data[i,frame,1,5,body]+data[i,frame,1,6,body]+data[i,frame,1,11,body]+data[i,frame,1,12,body])/4
                    c_center=(data[i,frame,2,5,body]+data[i,frame,2,6,body]+data[i,frame,2,11,body]+data[i,frame,2,12,body])/4
                    new_data[i,frame,:,17,body]=np.array([x_center,y_center,c_center])

        return new_data.reshape((-1,305,3,18*bodies))

    def load_test_landmarks(self,load=False):

        path = self.path
        if load:
            print('Load test data from file')
            test_data=np.load(path+self.bodies+'_body_full_test_landmarks.npy')
        else:
            print('Create full test data')
            test_data = self.pre_normaliser.test_prenorm_data.copy()
            print('Prenormalized test data shape:',test_data.shape)
            # transforming data [N,C,T,V,M] to [N,T,V,M,C] to [N,T,C,V*M]
            test_data = np.transpose(test_data,(0,2,1,3,4))
            test_data=test_data.reshape(test_data.shape[:3] + (self.N_JOINTS*self.N_PERSONS,))
            print('Get',self.bodies,'body data')
            test_data = get_body_type(data,labels=None,by_label=False)
            print('Save test landmark data')
            np.save(path+self.bodies+'_body_full_test_landmarks.npy', test_data)

        return test_data


    def sig_transform(self):
        '''
            Either loads the data from path, or creates the desired PSFDataset object (trainingset, validation set). These
            ... objects contain the desired PSF transform. Then trainingset and validation set are filled with the data so that
            ... they are ready to be used by torch for training and validation. In the second case, data will be saved
            ... in the path. The path is ~/scr/preprocessing/PSF.
        '''

        path = os.path.dirname(__file__) + "/LinConvData/"
        if self.load_transform==True:
            print('Load data')
            trainingset = PSFDataset()
            valset = PSFDataset()
            trainingset.load(path+"uav_train_"+self.transform)
            valset.load(path+"uav_val_"+self.transform)
        else:
            print('Create data')
            if self.transform=='pairs':
                tr = transforms.Compose([
                        transforms.spatial.Tuples(2)

                ])
                trainingset = PSFDataset(transform=tr)
                valset = PSFDataset(transform=tr)

                train_data, train_label, val_data, val_label = self._load_data()
                # transforming data [N,C,T,V,M] to [N,T,V,M,C] to [N,T,V*M,C]
                train_data = np.transpose(train_data,(0,2,3,4,1))
                train_data = train_data.reshape(*train_data.shape[:2],-1,*train_data.shape[4:])
                val_data = np.transpose(val_data,(0,2,3,4,1))
                val_data = val_data.reshape(*val_data.shape[:2],-1,*val_data.shape[4:])
                if self._debug:
                # consider only the 5 first joints of the first person
                    train_data = train_data[:,:,:5,:]
                    val_data = val_data[:,:,:5,:]
                print('Training data shape:',train_data.shape)
                print('Validation data shape:',val_data.shape)
                trainingset.fill_from_iterator(self._get_iter(train_data, train_label))
                valset.fill_from_iterator(self._get_iter(val_data, val_label))
            else:
                raise NotImplementedError

            print('Save PSF.')
            trainingset.save(path+"uav_train_"+self.transform)
            valset.save(path+"uav_val_"+self.transform)

        print("Number of trainingset elements:", len(trainingset))
        print("Number of validation set elements", len(valset))
        print("Dimension of feature vector:", trainingset.get_data_dimension())

        return trainingset, valset
