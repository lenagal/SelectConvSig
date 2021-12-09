import numpy as np
import src.algos.utils.SkeletonUtils as SkeletonUtils
from sklearn.preprocessing import StandardScaler

N_PERSONS=2
CENTER_JOINT=17
SHOULDER_JOINT=5
N_JOINTS=18
THRESHOLDS=SkeletonUtils.UAVHuman_Joints_movement_threshold()
BONES=SkeletonUtils.UAVHuman_Joints_graph_edges()
ORDERED_BONES=[4,0,1,2,3,5,6,7,8,9,10,11,13,14,12,15,16]
N_BONES=len(BONES)
HEAD_BONES=[0,1,2,3]
PERSONS=[0,1]
BONE_THRESHOLDS=SkeletonUtils.UAVHuman_Bone_size_threshold()
MAX_JUMP_SIZE=20

def dejank(film,return_metrics=False,threshold_scale=1,
            fix_head=True,
            fix_jumps=True,
            fix_bones=True):
    '''
    film - Time x Axes x Joints x Persons
    '''

    N_TIMESTEPS, N_AXES, N_JOINTS, N_PERSONS = film.shape

    bone_size_limits=np.zeros((N_BONES,N_PERSONS))
    frame=0
    for person in PERSONS:
        for bone_idx in range(N_BONES):
            bone=BONES[bone_idx]
            xbone_size=film[frame,0,bone[0],person]-film[frame,0,bone[1],person]
            ybone_size=film[frame,1,bone[0],person]-film[frame,1,bone[1],person]
            bone_size=np.sqrt(xbone_size**2+ybone_size**2)
            bone_size_limits[bone_idx,person]=BONE_THRESHOLDS[bone_idx]*bone_size

    if fix_bones:
        BONES_TO_FIX=ORDERED_BONES
    elif fix_head:
        BONES_TO_FIX=HEAD_BONES
    else:
        BONES_TO_FIX=[]

    for frame in range(1,len(film)):
        for person in range(N_PERSONS):
            for bone_idx in BONES_TO_FIX:
                # print('dejankTestData debug fixing bone',bone_idx)
                bone=BONES[bone_idx]
                xbone_size=film[frame,0,bone[0],person]-film[frame,0,bone[1],person]
                ybone_size=film[frame,1,bone[0],person]-film[frame,1,bone[1],person]
                bone_size=np.sqrt(xbone_size**2+ybone_size**2)
                if bone_size>bone_size_limits[bone_idx,person]:
                    bone_vector = film[frame,:,bone[1],person]-film[frame,:,bone[0],person]
                    film[frame,:,bone[1],person]=film[frame,:,bone[0],person]+bone_vector*(bone_size_limits[bone_idx,person]/bone_size)

    start_position=film[0]

    def get_delta_film(film):
        shifted_right=np.pad(film,((1,0),(0,0),(0,0),(0,0)))
        shifted_left=np.pad(film,((0,1),(0,0),(0,0),(0,0)))
        delta_film=(shifted_left-shifted_right)[1:-1]
        return np.pad(delta_film,((1,0),(0,0),(0,0),(0,0)))

    delta_film = get_delta_film(film)

    max_delta=0
    change_list=[]
    # joint_last_jump=np.zeros((N_JOINTS,2),dtype='int32')
    # joint_last_jump_fixed=np.zeros((N_JOINTS,2),dtype='int32')
    last_in_place=[0,0]

    if fix_jumps:
        for frame in range(1,len(delta_film)):
            for person in range(N_PERSONS):
                xsize=film[last_in_place[person],0,SHOULDER_JOINT,person]-film[last_in_place[person],0,CENTER_JOINT,person]
                ysize=film[last_in_place[person],1,SHOULDER_JOINT,person]-film[last_in_place[person],1,CENTER_JOINT,person]
                size=np.sqrt(xsize**2+ysize**2)
                # for joint in range(N_JOINTS):
                #     xmotion=delta_film[frame,0,joint,person]
                #     ymotion=delta_film[frame,1,joint,person]
                #     motion=np.sqrt(xmotion**2+ymotion**2)
                #     if motion>size*THRESHOLDS[joint]*threshold_scale:
                #         if motion/size>max_delta:
                #             max_delta=motion/size
                #         if joint_last_jump[joint,person]!=0 and frame-joint_last_jump[joint,person]<=MAX_JUMP_SIZE:
                #             delta_film[joint_last_jump[joint,person]:frame+1,:,joint,person]=0
                #             # print('dejank motion debug 1',joint,joint_last_jump[joint,person],i)
                #             joint_last_jump[joint,person]=0
                #             change_list.append([joint,motion])
                #         else:
                #             joint_last_jump[joint,person]=frame
                xmotion=delta_film[frame,0,CENTER_JOINT,person]
                ymotion=delta_film[frame,1,CENTER_JOINT,person]
                motion=np.sqrt(xmotion**2+ymotion**2)
                if motion>size*THRESHOLDS[CENTER_JOINT]*threshold_scale:
                    if motion/size>max_delta:
                        max_delta=motion/size
                    if last_in_place[person]<frame-1:
                        xmotion=np.sum(delta_film[last_in_place[person]+1:frame+1,0,CENTER_JOINT,person])
                        ymotion=np.sum(delta_film[last_in_place[person]+1:frame+1,1,CENTER_JOINT,person])
                        total_motion=np.sqrt(xmotion**2+ymotion**2)
                        if total_motion<size*THRESHOLDS[CENTER_JOINT]*threshold_scale*2:
                            # delta_film[last_in_place[person]+1,0,:,person]=delta_film[last_in_place[person]+1,0,:,person]-delta_film[last_in_place[person]+1,0,CENTER_JOINT,person]
                            # delta_film[last_in_place[person]+1,1,:,person]=delta_film[last_in_place[person]+1,1,:,person]-delta_film[last_in_place[person]+1,1,CENTER_JOINT,person]
                            # delta_film[frame,0,:,person]=delta_film[frame,0,:,person]-delta_film[frame,0,CENTER_JOINT,person]
                            # delta_film[frame,1,:,person]=delta_film[frame,1,:,person]-delta_film[frame,1,CENTER_JOINT,person]
                            for frame_to_change in range(last_in_place[person]+1,frame+1):
                                delta_film[frame_to_change,0,:,person]=delta_film[frame_to_change,0,:,person]-delta_film[frame_to_change,0,CENTER_JOINT,person]
                                delta_film[frame_to_change,1,:,person]=delta_film[frame_to_change,1,:,person]-delta_film[frame_to_change,1,CENTER_JOINT,person]
                            last_in_place[person]=frame
                            change_list.append([CENTER_JOINT,motion])
                    else:
                        xmotion=np.sum(delta_film[1:frame+1,0,CENTER_JOINT,person])
                        ymotion=np.sum(delta_film[1:frame+1,1,CENTER_JOINT,person])
                        total_motion=np.sqrt(xmotion**2+ymotion**2)
                        if total_motion<size*THRESHOLDS[CENTER_JOINT]*threshold_scale:
                            last_in_place[person]=frame

                else:
                    if last_in_place[person]==frame-1:
                        last_in_place[person]=frame

    cumsum_delta=np.cumsum(delta_film,axis=0)
    new_film=start_position+cumsum_delta

    '''normalize:'''
    scaler=StandardScaler()

    new_film=new_film.transpose((0,2,3,1))
    new_film=new_film.reshape((-1,N_AXES))
    scaler.fit(new_film)
    new_film=scaler.transform(new_film)
    new_film=new_film.reshape((N_TIMESTEPS,N_JOINTS,N_PERSONS,N_AXES))
    new_film=new_film.transpose((0,3,1,2))



    if not return_metrics:
        return new_film
    else:
        metrics=' changes: {} max: {:.2f}'.format(len(change_list),max_delta)
        return new_film, metrics, change_list
