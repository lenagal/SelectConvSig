import pickle
import tensorflow as tf
from tensorflow.keras.layers import Layer
import random
import src.algos.utils.tf_sig as tf_sig
import math
import itertools

class TripleSig(Layer):
    '''
    if random is True elects a random selection of triples of joints using seed and
    ...returns their signatures (as a path of length 3) for each time point.
    If random is False returns list of signatues of all possible triples of joints
    ...in lexicographic order (picks out a subset of these if mask is not None)
    input shape : Batch X Time X Axes X Joints
    output shape: Batch x Time x Triple_indices x signatures

    Args:
    triple_amount: amount of triples to choose
    random: whether to pick random triples
    signature_deg: degree of signature to apply to triples
    seed: seed for random selection
    mask: path to file with list of triples to select out of all possible triples
    (contains indices for the lexicographically ordered list of triples)
    '''
    def __init__(self,triple_amount=0,random=False,signature_deg=2,seed=None,mask=None,**kwargs):
        super(TripleSig, self).__init__(**kwargs)
        self.triple_amount=triple_amount
        self.seed=seed
        self.signature_deg=signature_deg
        self.mask=mask

    def build(self,input_shape):
        self.space_dim=input_shape[2]
        self.N_JOINTS=input_shape[3]

        if self.seed is not None:
            random.seed(self.seed)

        if self.signature_deg==2:
            self.sig_transform=tf_sig.deg2_sig
        elif self.signature_deg==3:
            self.sig_transform=tf_sig.deg3_sig
        else:
            raise ValueError('TripleSig: signatures above 3 not supported')

        if random is True:
            self.triple_list=[]
            for i in range(self.triple_amount):
                self.triple_list.append(random.sample(range(self.N_JOINTS),3))
        else:
            self.triple_list=list(itertools.combinations(range(self.N_JOINTS), 3))

            if self.mask is not None:
                with open(mask,"rb") as fp:
                    list_mask=pickle.load(fp)
                self.triple_list=list(self.triple_list[i] for i in list_mask)

    def call(self,input):
        joint_triple_list=[]

        for triple in self.triple_list:
            joint_triple_list.append([
                input[:,:,:,triple[0]],
                input[:,:,:,triple[1]],
                input[:,:,:,triple[2]]
            ])
        signatures_list=[]
        for joint_triple in joint_triple_list:
            signatures_list.append(self.sig_transform(tf.stack(joint_triple)))
            # print('sig shape',signatures_list[-1].shape)


        return tf.stack(signatures_list,axis=2)
