import os
import numpy as np
import pickle
from tqdm import tqdm
import glob
import re
import pandas
import src.algos.utils.SkeletonUtils as SkeletonUtils

class NTUDataGen:

    '''
        reads NTURGB+D skeleton data into a numpy file and
        generates an associated label file.
        The data array has shape Number X Time X Body X Joints X Axes
        The label array is a 1-D array of integers

        when generating or loading data use 'partial'/'all' flags for the 60/120
        datasets repectively.
    '''

    N_FRAMES_MAX=300
    N_JOINTS=25
    N_BODIES_MAX=2
    N_AXES=3

    BONES=len(SkeletonUtils.NTU_Joints_graph_edges())

    #TODO should receive data_path as argument
    def __init__(self,load_data=True,load_bones=True,dataset='all',debug=False):
        #self.dataset_folder_path = "".join(__file__.split("/")[:-1]) + "/raw_data/" --- OLD VERSION, had a bug
        #self.dataset_folder_path = os.path.dirname(__file__) + "/raw_data/"
        self.dataset_folder_path = "/scratch/gale/UAVHuman/NTURGBData/"
        self.path60='nturgb+d60'
        self.path120='nturgb+d120'
        self.data_filename='nturgb_data'
        self.label_filename='nturgb_label'
        self.supplemental_info_filename='nturgb_supplemental'
        self.bones_filename='nturgb_bones'
        self.example_file=os.path.join(self.dataset_folder_path,self.path60,'S017C003P020R002A060.skeleton')
        self.dataset=dataset

        if load_data:
            self.train_data, self.train_label, self.supplemental_info = self._load_data()
        elif not debug:
            self.train_data, self.train_label, self.supplemental_info = self._generate_ntu_data()

        if load_bones:
            self.bones_data = self._load_bones_data()
        else:
            self.bones_data = self._generate_bones_data()

    def get_bones(self,skeleton):
        '''
        input - 25x3 array
        output - bonesx3 array
        '''
        bones_list=[]
        for bone in self.BONES:
            bones_list.append(skeleton[bone[0]]-skeleton[bone[1]])
        return np.array(bones_list)

    def _load_bones_data(self):
        return np.load(os.path.join(self.dataset_folder_path,self.bones_filename+'_{}'.format(self.dataset)+'.npy'))

    def _generate_bones_data(self):
        bones_data_list=[]
        i=0
        for datum in self.train_data:
            bones_data_list.append(self.get_bones(datum.transpose([2,0,1,3])).transpose([1,2,0,3]))
            i+=1
            print('added',i,'bones',end='\r')

        print('added',i,'bones')
        bones_data=np.array(bones_data_list)
        np.save(os.path.join(self.dataset_folder_path,self.bones_filename+'_{}'.format(self.dataset)+'.npy'),bones_data)
        return bones_data

    def _generate_ntu_data(self):
        data_path=os.path.join(self.dataset_folder_path,self.path60)

        skeleton_filenames = [f for f in
        glob.glob(os.path.join(data_path, "**.skeleton"), recursive=True)]

        if self.dataset=='all':
            data_path=os.path.join(self.dataset_folder_path,self.path120)
            skeleton_filenames120 = [f for f in
            glob.glob(os.path.join(data_path, "**.skeleton"), recursive=True)]
            skeleton_filenames+=skeleton_filenames120

        data=np.zeros((len(skeleton_filenames),self.N_FRAMES_MAX,self.N_BODIES_MAX,self.N_JOINTS,self.N_AXES))
        labels=np.zeros((len(skeleton_filenames),),dtype=int)
        supplemental_info=np.zeros((len(skeleton_filenames),4),dtype=int)

        number=0
        for i,file in enumerate(skeleton_filenames):
            datum=self.read_file(file)
            data[i]=np.tile(datum,(int(300/len(datum))+1,1,1,1))[:300]
            label, supplemental = self.read_label(file)
            labels[i]= label
            supplemental_info[i]= np.array(supplemental)
            print(i,'files added',end='\r')
        print(len(skeleton_filenames),'files added')
        print('Saving files to disk')
        np.save(os.path.join(self.dataset_folder_path,self.data_filename+'_{}'.format(self.dataset)+'.npy'),data)
        np.save(os.path.join(self.dataset_folder_path,self.label_filename+'_{}'.format(self.dataset)+'.npy'),labels)
        np.save(os.path.join(self.dataset_folder_path,self.supplemental_info_filename+'_{}'.format(self.dataset)+'.npy'),supplemental_info)
        return data,labels, supplemental_info

    def _load_data(self):
        print('Loading',self.dataset,'NTU Data')
        data = np.load(os.path.join(self.dataset_folder_path,self.data_filename+'_{}'.format(self.dataset)+'.npy'))
        label = np.load(os.path.join(self.dataset_folder_path,self.label_filename+'_{}'.format(self.dataset)+'.npy'))
        supplemental_info = np.load(os.path.join(self.dataset_folder_path,self.supplemental_info_filename+'_{}'.format(self.dataset)+'.npy'))

        return data,label,supplemental_info

    def read_label(self,file):
        filename=file.split('/')[-1]
        camera_string=filename.split('C')[-1].split('P')[0]
        performer_string=filename.split('P')[-1].split('R')[0]
        replication_string=filename.split('R')[-1].split('A')[0]
        setup_string=filename.split('S')[-1].split('C')[0]
        label_string=filename.split('A')[-1].split('.')[0]

        return int(label_string),[int(setup_string),int(camera_string),int(performer_string),int(replication_string)]


    def read_file(self,file):
        with open(file) as f:
            lines = []
            for line in f:
                lines.append(line.split())

        framecount=int(lines[0][0])
        position=1
        frame_data=np.zeros((framecount,self.N_BODIES_MAX,self.N_JOINTS,self.N_AXES))
        max_bodies=0
        for frame in range(framecount):
            bodies=np.zeros((self.N_BODIES_MAX,self.N_JOINTS,self.N_AXES))
            bodycount=int(lines[position][0])
            position+=1
            for body in range(bodycount):
                bodyinfo={}
                # bodyinfo['id']=int(lines[position][0])
                # bodyinfo['clippedEdges'] = int(lines[position][1])
                # bodyinfo['handLeftConfidence'] = int(lines[position][2])
                # bodyinfo['handLeftState'] = int(lines[position][3])
                # bodyinfo['handRightConfidence'] = int(lines[position][4])
                # bodyinfo['handRightState'] = int(lines[position][5])
                # bodyinfo['isResticted'] = int(lines[position][6])
                # bodyinfo['leanXY']=[float(lines[position][7]),float(lines[position][8])]
                # bodyinfo['trackingState']=int(lines[position][9])
                position+=1
                bodyinfo['jointCount']= int(lines[position][0])
                position+=1
                bodyinfo['jointsXYZ']=[]
                for i in range(bodyinfo['jointCount']):
                    bodyinfo['jointsXYZ'].append([float(lines[position][0]),float(lines[position][1]),float(lines[position][2])])
                    position+=1
                if body<self.N_BODIES_MAX:
                    bodies[body]=np.array(bodyinfo['jointsXYZ'])
            frame_data[frame]=bodies
        return frame_data
