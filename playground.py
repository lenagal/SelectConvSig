import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # block Tensorflow crap
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # block GPU

import psutil
process = psutil.Process(os.getpid())

import tensorflow as tf

import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Reshape,Lambda
from src.algos.utils import SkeletonUtils
import matplotlib.pyplot as plt
from matplotlib import animation
import timeit
from src.preprocessing.pre_normaliser import preNormaliser

# from src.algos.siglinconvRNN import SigLinConvRNN
# from src.algos.siglinconvRNNBase import SigLinConvRNNBase
# from src.algos.siglinconvRNNAuto import SigLinConvRNNAuto
from src.preprocessing.linconv_data import LinConvData
# from src.preprocessing.linconv_NTU_data import LinConvNTUData
# from src.algos.utils.GraphLinConv import GraphLinConv
# from src.algos.utils.DepthwiseDense import DepthwiseDense
from src.preprocessing.dejankTestData import dejank
# from src.algos.utils.ZoomMovie import ZoomMovie
from src.algos.utils.SigSegments import SigSegments

N_PERSONS=2
N_TIMESTEPS=305

def visualize(data,title='',ignore_head=False):
    if ignore_head:
        start=5
    else:
        start=0
    AXES=[0,1]
    x0, x1 = np.min(data[:, AXES[0], start:, :]), np.max(data[:, AXES[0], start:, :])
    y0, y1 = np.min(data[:, AXES[1], start:, :]), np.max(data[:, AXES[1], start:, :])

    ratio = (y1 - y0) / (x1 - x0)

    xh = 5
    yh = ratio * 5

    fig, ax = plt.subplots(figsize=(xh, yh))

    plt.xlim((x0, x1))
    plt.ylim((y0, y1))
    plt.title(title)

    bones = SkeletonUtils.UAVHuman_Joints_graph_edges()

    p_type = ['b-', 'g-', 'g-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    pose = []

    for m in range(N_PERSONS):
        a = []
        for i in range(len(bones)):
            a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
        pose.append(a)

    def animate(t):

        for m in range(N_PERSONS):

            for i, bone in enumerate(bones):
                x1 = data[t, AXES, bone[0], m]
                x2 = data[t, AXES, bone[1], m]
                if (x1.sum() != 0 and x2.sum() != 0):
                    pose[m][i].set_xdata(data[t, AXES[0], [bone[0], bone[1]], m])
                    pose[m][i].set_ydata(data[t, AXES[1], [bone[0], bone[1]], m])

        return np.array(pose).flatten()

    anim = animation.FuncAnimation(fig, animate, frames=N_TIMESTEPS, interval=10, blit=True)
    plt.show()

# pre_normaliser = preNormaliser(pad=True, centre=1, rotate=0, switchBody =True, eliminateSpikes = True, scale = 2, parallel = False, smoothen = False,
#              setPerson0 = 1, confidence = 0,debug=0)
# print('Init pre_normaliser finished')
# data_wrapper=LinConvData(pre_normaliser=None, add_center_joint=False)
# # raw_data,raw_labels=data_wrapper.load_full_landmarks('train',raw=True)
# data,labels=data_wrapper.load_full_landmarks('test')
#
# LABEL= 79
#
# labels=np.argmax(labels,axis=-1)
#
# label_indices = np.where(labels == LABEL)[0]
#
# # playgroundRawData=raw_data[label_indices[:100],:,:2,:]
#
# playgroundData=data[label_indices[:100],:,:2,:]
#
# # np.save('/scratch/gale/playgroundData.npy',playgroundData)
# #
playgroundData=np.load('/scratch/gale/playgroundData.npy')

input_layer=Input((305,2,36))
reshape=Reshape((305,72))(input_layer)
output_layer=SigSegments(32,signature_deg=2)(reshape)

test_model=Model(inputs=[input_layer],outputs=[output_layer])

print(test_model(playgroundData).shape)
# for i in range(0,5):
#     # raw_sample=playgroundRawData[i]
#     sample=playgroundData[i]
#     rotated_sample=test_model(sample.reshape(1,305,2,36),training=True).numpy()
#     # visualize(raw_sample.reshape((305,2,18,2)))
#     visualize(sample.reshape((305,2,18,2)))
#     visualize(rotated_sample.reshape((305,2,18,2)))



# data_wrapper=LinConvData()
# data,labels=data_wrapper.load_full_landmarks('train')
#
# LABEL= 74
#
# labels=np.argmax(labels,axis=-1)
#
# label_indices = np.where(labels == LABEL)[0]
#
# playgroundData=data[label_indices[:100],:,:2,:]
#
# np.save('/scratch/gale/playgroundData.npy',playgroundData)

# playgroundData=np.load('/scratch/gale/playgroundData.npy')
#
# input_layer=Input((305,2,36))
# output_layer=RotateAxisMovie(axis=1,min_angle=0,max_angle=1)(input_layer)
#
# test_model=Model(inputs=[input_layer],outputs=[output_layer])
#
# sample=playgroundData[1]
# rotated_sample=test_model(sample.reshape(1,305,2,36),training=True).numpy()
# visualize(sample.reshape((305,2,18,2)))
# visualize(rotated_sample.reshape((305,2,18,2)))





#####################DeJank data###########################
# data_wrapper=LinConvData(with_center_joint=True)
# path = data_wrapper.path




# for flag in ['train']:
#
#     data,labels=data_wrapper.load_full_landmarks(flag)
#     data=data[:,:,:2].reshape((-1,305,2,18,2))
#     new_data_list=[]
#     new_label_list=[]
#     changes=0
#     # changed_list=[]
#     for i,datum in enumerate(data):
#         new_datum=dejank(datum)
#         if not np.all(new_datum==datum):
#             changes+=1
#             # changed_list.append(i)
#         else:
#             new_data_list.append(new_datum.reshape((305,2,36)))
#             new_label_list.append(labels[i].copy())
#         print(i,'done',changes,'changed',end='\r')
#
#
#     new_data=np.array(new_data_list)
#     new_label=np.array(new_label_list)
#     np.save(os.path.join(path,'both_body_full_{}_landmarks_dejanked.npy'.format(flag)), new_data)
#     np.save(os.path.join(path,'both_body_full_{}_label_dejanked.npy'.format(flag)), new_label)


# LABEL= 74
# ##########develop dejank#################
# data_wrapper=LinConvData()
# data,labels=data_wrapper.load_full_landmarks('train')
#
#
#
# labels=np.argmax(labels,axis=-1)
#
# label_indices = np.where(labels == LABEL)[0]
#
# playgroundData=data[label_indices[:100],:,:2,:]
#
# np.save('/scratch/gale/playgroundData.npy',playgroundData)
#
# # playgroundData=np.load('/scratch/gale/playgroundData.npy')
#
# AMOUNT=5
#
# for i in range(AMOUNT):
#     sample_film=playgroundData[i].reshape((305,2,18,2))
#     visualize(sample_film,title='before: Label - '+str(LABEL)+' Sample - '+str(i))
#     processed_sample_film = dejank(sample_film,return_metrics=False)
#     # print(change_list)
#     visualize(processed_sample_film,title='after: Label - '+str(LABEL)+' Sample - '+str(i)+' '+metrics)
