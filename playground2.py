import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # block Tensorflow crap
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # block GPU

import psutil
process = psutil.Process(os.getpid())

import tensorflow as tf

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Reshape,Lambda
from src.algos.utils import SkeletonUtils
import matplotlib.pyplot as plt
from matplotlib import animation
import timeit

# from src.algos.siglinconvRNN import SigLinConvRNN
# from src.algos.siglinconvRNNBase import SigLinConvRNNBase
# from src.algos.siglinconvRNNAuto import SigLinConvRNNAuto
from src.preprocessing.linconv_data import LinConvData
# from src.preprocessing.linconv_NTU_data import LinConvNTUData
# from src.algos.utils.GraphLinConv import GraphLinConv
# from src.algos.utils.DepthwiseDense import DepthwiseDense
from src.preprocessing.dejankTestData import dejank

N_PERSONS=2
N_TIMESTEPS=305

def visualize(data,title=''):
    AXES=[0,1]
    x0, x1 = np.min(data[:, AXES[0], :, :]), np.max(data[:, AXES[0], :, :])
    y0, y1 = np.min(data[:, AXES[1], :, :]), np.max(data[:, AXES[1], :, :])

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

    anim = animation.FuncAnimation(fig, animate, frames=N_TIMESTEPS, interval=20, blit=True)
    plt.show()

flag='test'

data_wrapper=LinConvData(load_with_center_joint=True)
path = data_wrapper.path

data,labels=data_wrapper.load_full_landmarks(flag)

labels=np.argmax(labels,axis=-1)

data=data[:,:,:2].reshape((-1,305,2,18,2))

# new_data_list=[]
# count=0
# changes=0
# changed_list=[]
# for i,datum in enumerate(data):
#     new_datum=dejank(datum)
#     if not np.all(new_datum==datum):
#         changes+=1
#         changed_list.append(i)
#     count+=1
#     print(count,'done',changes,'changed',end='\r')
#     new_data_list.append(new_datum.reshape((305,2,36)))
# print(count,'done',changes,'changed')
#
#
# new_data=np.array(new_data_list)
# np.save(os.path.join(path,'both_body_full_{}_landmarks_dejanked.npy'.format(flag)), new_data)

# new_data=np.load(os.path.join(path,'both_body_full_{}_landmarks_dejanked.npy'.format(flag)))


LABEL, SAMPLE ,AMOUNT= 2, 20, 50

label_indices = np.where(labels == LABEL)[0]


for i in range(AMOUNT):
    sample_film=data[label_indices[SAMPLE+i]].reshape((305,2,18,2))
    visualize(sample_film,title='before: Label - '+str(LABEL)+' Sample - '+str(SAMPLE+i))
    # processed_sample_film, metrics, change_list= dejank(sample_film,return_metrics=True)
    # print(change_list)
    # visualize(processed_sample_film,title='after: Label - '+str(LABEL)+' Sample - '+str(SAMPLE+i)+' '+metrics)
