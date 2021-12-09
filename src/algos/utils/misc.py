import numpy as np

def to_categorical(int_array,num_classes=None):
    '''
    takes an integer array and returns it in one-hot encoding
    '''
    if num_classes is None:
        num_classes=np.max(int_array)

    return np.eye(num_classes)[int_array]
