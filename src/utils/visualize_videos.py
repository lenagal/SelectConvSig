import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import src.algos.utils.SkeletonUtils as SkeletonUtils

def visualize(data,title='',dataset='UAVHuman',N_PERSONS=2,N_TIMESTEPS=305,AXES=[0,1]):

    x0, x1 = np.min(data[:, AXES[0], :, :]), np.max(data[:, AXES[0], :, :])
    y0, y1 = np.min(data[:, AXES[1], :, :]), np.max(data[:, AXES[1], :, :])

    ratio = (y1 - y0) / (x1 - x0)

    xh = 5
    yh = ratio * 5

    fig, ax = plt.subplots(figsize=(xh, yh))

    plt.xlim((x0, x1))
    plt.ylim((y0, y1))
    plt.title(title)

    if dataset=='UAVHuman':
        bones = SkeletonUtils.UAVHuman_Joints_graph_edges()
    elif dataset=='NTU':
        bones = SkeletonUtils.NTU_Joints_graph_edges()

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

def compare_videos(before,after,title=None,dataset='UAVHuman',N_PERSONS=2,N_TIMESTEPS=305,AXES=[0,1]):
    NUM_FILMS=len(before)
    xmin=[]
    xmax=[]
    ymax=[]
    ymin=[]
    for film in before:
        xmin.append(np.min(film[:, AXES[0], :, :]))
        xmax.append(np.max(film[:, AXES[0], :, :]))
        ymin.append(np.min(film[:, AXES[1], :, :]))
        ymax.append(np.max(film[:, AXES[1], :, :]))
    for film in after:
        xmin.append(np.min(film[:, AXES[0], :, :]))
        xmax.append(np.max(film[:, AXES[0], :, :]))
        ymin.append(np.min(film[:, AXES[1], :, :]))
        ymax.append(np.max(film[:, AXES[1], :, :]))

    ratio=[]
    for i in range(NUM_FILMS*2):
        ratio.append((ymax[i] - ymin[i]) / (xmax[i] - xmin[i]))

    fig=plt.figure(1)

    ax=[]
    for i in range(NUM_FILMS):
        ax.append([fig.add_subplot(2,NUM_FILMS,i+1),fig.add_subplot(2,NUM_FILMS,i+NUM_FILMS+1)])

    if title is not None:
        for i in range(NUM_FILMS):
            ax[i][0].set_title(title[i][0])
            ax[i][1].set_title(title[i][1])

    for i in range(NUM_FILMS):
        ax[i][0].set_xlim((xmin[i], xmax[i]))
        ax[i][0].set_ylim((ymin[i], ymax[i]))
        ax[i][1].set_xlim((xmin[i+NUM_FILMS], xmax[i+NUM_FILMS]))
        ax[i][1].set_ylim((ymin[i+NUM_FILMS], ymax[i+NUM_FILMS]))
        # ax[i][0].set_xlim((-1, 1))
        # ax[i][0].set_ylim((-1, 1))
        # ax[i][1].set_xlim((-1, 1))
        # ax[i][1].set_ylim((-1, 1))

    if dataset=='UAVHuman':
        bones = SkeletonUtils.UAVHuman_Joints_graph_edges()
    elif dataset=='NTU':
        bones = SkeletonUtils.NTU_Joints_graph_edges()

    p_type = ['b-', 'g-', 'g-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    pose = []
    frame_text = []

    for film in range(NUM_FILMS):
        ind_pose=[]
        for person in range(N_PERSONS):
            a = []
            b = []
            for bone_idx in range(len(bones)):
                a.append(ax[film][0].plot(np.zeros(2), np.zeros(2), p_type[person])[0])
                b.append(ax[film][1].plot(np.zeros(2), np.zeros(2), p_type[person])[0])
            ind_pose.append([a,b])
        pose.append(ind_pose)
        frame_text0= ax[film][0].text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax[film][0].transAxes)
        frame_text1= ax[film][1].text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax[film][1].transAxes)
        frame_text.append([frame_text0,frame_text1])

    def animate(t):
        for film_idx in range(NUM_FILMS):
            for person in range(N_PERSONS):

                for bone_idx, bone in enumerate(bones):
                    # x1 = data[t, AXES, bone[0], m]
                    # x2 = data[t, AXES, bone[1], m]
                    # if (x1.sum() != 0 and x2.sum() != 0):
                    pose[film_idx][person][0][bone_idx].set_xdata(before[film_idx][t, AXES[0], [bone[0], bone[1]], person])
                    pose[film_idx][person][0][bone_idx].set_ydata(before[film_idx][t, AXES[1], [bone[0], bone[1]], person])
                    pose[film_idx][person][1][bone_idx].set_xdata(after[film_idx][t, AXES[0], [bone[0], bone[1]], person])
                    pose[film_idx][person][1][bone_idx].set_ydata(after[film_idx][t, AXES[1], [bone[0], bone[1]], person])
            frame_text[film_idx][0].set_text(str(t))
            frame_text[film_idx][1].set_text(str(t))

        return np.concatenate([np.array(pose).flatten(),np.array(frame_text).flatten()])

    anim = animation.FuncAnimation(fig, animate, frames=N_TIMESTEPS, interval=20, blit=True)
    plt.show()
