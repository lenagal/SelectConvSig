import numpy as np
from tqdm import tqdm
import math
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from src.data_grabbing.data_grabber import DataGrabber
from src.preprocessing.tool.validation_tools import *
from src.preprocessing.UAVHFileNames import UAVHFileNames
from src.algos.utils.misc import to_categorical
import src.algos.utils.SkeletonUtils as SkeletonUtils

class preNormaliser:
    '''
        A class for data prepropcessing. Parts of this class were developed in collaboration with members of SignedOB team
        ...for UAVHuman competition.
        The DataGrabber object requires data (np.array) of shape (N,C,T,V,M), see data_grabber.py.
        The preNormaliser saves data in files whose names are specified by a UAVHFileNames object initialized with data_path.
        The saved data is in shape (N,T,C,V*M).
        The saved raw labels are an integer array.
        The saved processed labels are in one-hot encoding using self.N_CLASSES for the number of classes.

        The Preprocessing consists of:
        switch bodies        this method goes through each frame, and tries to switch the two bodies in said frame in such a way, that both bodies move as continuous as possible
        scaling              scales a sample in such a way, that the maximum height/width of body number 0 is 1, while preserving the proportions WARNING: DOES NOT WORK FRAME-WISE YET
        eliminating Spikes   Removes frames, which move too fast compared too the next valid frames. Tries several start-frames as valid frames, and chooses the one which leads
                                  to the least amount of deleted frames. Deleted frames in the middle get replaced with convex combinations of the last and next valid frames. Deleted frames
                                  get marked with a 0 in the third coordinate (C = 2) of each vertex. Valid frames have a 1 in this coordinate. Also automatically pads the data.
        padding              If eliminateSpikes is True, this automatically happens in eliminate_spikes. Replaces zero-frames at the start with the next non-zero frame. Replaces zero
                                  frames in the middle with convex combinations of the next and last non-zero-frame. Replaces zero-frames at the end with the last non-zero frame.
                                  Replaced frames get marked with a 0 in C = 2 in each vertex, while original frames have a 1.
        centering            centers the center of mass (average between both shoulders and both hips) at (0,0) for each frame. The sample-wise version centers the first frame.
        scale again          eliminating spikes might have changed the scale, so we aplly it again.
        smoothing            applies a savgol filter too smoothen the data with parameters (Window = 9, degree = 2).
        adding joint in the middle of the torso    to facilitate definition of bones data
    '''

    def __init__(self, data_path, pad=True, centre=1, rotate=0, add_center_joint=True,
                switchBody =True, eliminateSpikes = True, scale = 2, parallel = False,
                smoothen = False, remove_extra_bodies=False,
                setPerson0 = 1, confidence = 0,debug=0):


        self.add_center_joint=add_center_joint      #True or False
        self.switchBody = switchBody                #True or False
        self.eliminateSpikes = eliminateSpikes      #True or False
        self.remove_extra_bodies = remove_extra_bodies      #True or False - not currently implemented
        self.isPadding = pad                        #True or False
        self.isScaling = scale                      #0 for doing nothing, 1 for frame-wise, 2 for sample-wise
        self.isCentering = centre                   #0 for doing nothing, 1 for frame-wise, 2 for sample-wise
        self.is3DRotating = rotate                  #0 for doing nothing, 1 for frame-wise, 2 for sample-wise
        self.smoothen = smoothen                    #True or False
        self.setPerson0 = setPerson0                #0 for doing nothing, 1 to set the more active person to be person 0, 2 to set the left person to be person 0, 3 to duplicate samples with 2 bodies
        self.confidence = confidence                #0 for visibility in the third coordinate, 1 for visibility in third, energy in fourth.
        self.N_TIMESTEPS = 305
        self.N_AXES = 2
        self.N_JOINTS = SkeletonUtils.UAVHuman_Joints()
        self.N_PERSONS = 2
        self.N_CLASSES=155
        self.file_names=UAVHFileNames(data_path)
        self.debug=debug
        if self.debug==True:
            self.debug=100

        self.isParallel = parallel

        self.data_grabber = DataGrabber()
        if debug:
            self.train_label = self.prenorm_labels(self.data_grabber.train_label[:debug],flag='train',debug=True)
            self.test_label = self.prenorm_labels(self.data_grabber.test_label[:debug],flag='test',debug=True)
            self.raw_train_data=self.data_grabber.train_data[:debug]
            self.prenorm_train_data = self.pre_normalization(self.raw_train_data,flag='train',debug=True)
            self.raw_test_data=self.data_grabber.test_data[:debug]
            self.prenorm_test_data = self.pre_normalization(self.raw_test_data,flag='test',debug=True)
        else:
            self.train_label = self.prenorm_labels(self.data_grabber.train_label,flag='train')
            self.test_label = self.prenorm_labels(self.data_grabber.test_label,flag='test')
            self.raw_train_data=self.data_grabber.train_data
            self.prenorm_train_data = self.pre_normalization(self.raw_train_data,flag='train')
            self.raw_test_data=self.data_grabber.test_data
            self.prenorm_test_data = self.pre_normalization(self.raw_test_data,flag='test')

    def prenorm_labels(self,label,flag='',debug=False):
        raw_label = label.copy()
        prenorm_label = to_categorical(raw_label,num_classes=self.N_CLASSES)
        prenorm_file_name = self.file_names.label_file_name(flag,debug=debug)
        print('Saving one-hot labels to',prenorm_file_name)
        np.save(prenorm_file_name,prenorm_label)
        return prenorm_label

    def pre_normalization(self, data, zaxis=[11, 5], xaxis=[6, 5],flag='',debug=False):
    # Remark ER: Is zaxis = [11,5] a good idea? It may reflect real people w.r.t. to the xy-plane. This is only an issue
    # for visualisation.
        '''
            Joint sequence same as COCO format + center joint: {
                0: nose,
                1: left_eye,
                2: right_eye,
                3: left_ear,
                4: right_ear,
                5: left_shoulder,
                6: right_shoulder,
                7: left_elbow,
                8: right_elbow,
                9: left_wrist,
                10: right_wrist,
                11: left_hip,
                12: right_hip,
                13: left_knee,
                14: right_knee,
                15: left_ankle,
                16: right_ankle
                17: center
            }

            Data format: {
            	N: sample
            	C: coordinate
            	T: frame ("time")
            	V: joint ("vertex")
            	M: person ID
            }
        '''
        raw_data=data.copy()
        data = data[:,:,:305,:,:]  # TO REDUCE DATA SIZE: MAXIMAL LENGTH 305

        N, C, T, V, M = data.shape

        s = np.transpose(data, [0, 4, 2, 3, 1])  # to (N, M, T, V, C)
        s[:,:,:,:,:2]=s[:,:,:,:,::2]

        print('N,M,T,V,C',s.shape)
        if self.switchBody:
            print("Switching bodies, if it reduces total energy.")
            skeletons = []
            if self.isParallel:
                skeletons = Parallel(n_jobs=-1)(delayed(switchPeople)(sample) for sample in tqdm(s))
            else:
                for i, sample in enumerate(tqdm(s)):
                    sample = switchPeople(sample)
                    skeletons.append(sample)
            s = np.stack(skeletons)

        if self.isScaling: #0 for not doing, 1 for frame-wise, 2 for sample-wise
            print('rescale each object sequence to the range [0,1] while conserving the high-width ratio')
            isSamplewise = True if self.isScaling == 2 else False
            ndim = min(C-1, 3) #the maximum num of coordinates for this operation is 3
            skeletons = []
            if self.isParallel:
                skeletons = Parallel(n_jobs=-1)(delayed(self.parallelScale)(skeleton,isSamplewise) for i,skeleton in enumerate(tqdm(s[...,:ndim])))
            else:
                for i,skeleton in enumerate(tqdm(s[...,:ndim])):
                    skeleton = self.parallelScale(skeleton,isSamplewise)
                    skeletons.append(skeleton)
            s[...,:ndim] = np.stack(skeletons)

        def eliminateSpikesSample(sample):
            for m in range(2):
                if sample[m, :, :, :].sum() == 0:
                    continue
                sample[m, :, :, :] = eliminate_spikes(sample[m, :, :, :])
            return sample

        def padNullFramesSample(sample):
            for m in range(2):
                if sample[m, :, :, :].sum() == 0:
                    continue
                sample[m, :, :, :] = padNullFrames(sample[m, :, :, :])
            return sample

        if self.eliminateSpikes:
            print("Eliminating spikes and padding null-frames.")
            skeletons = []
            if self.isParallel:
                skeletons = Parallel(n_jobs=-1)(delayed(eliminateSpikesSample)(sample) for sample in tqdm(s))
            else:
                for i, sample in enumerate(tqdm(s)):
                    sample = eliminateSpikesSample(sample)
                    skeletons.append(sample)
            s = np.stack(skeletons)
        elif self.isPadding:
            print("Padding null-frames.")
            skeletons = []
            if self.isParallel:
                skeletons = Parallel(n_jobs=-1)(delayed(padNullFramesSample)(sample) for sample in tqdm(s))
            else:
                for i, sample in enumerate(tqdm(s)):
                    sample = padNullFramesSample(sample)
                    skeletons.append(sample)
            s = np.stack(skeletons)

        """
        if self.isPadding:
            print("Shift non-zero nodes to beginning of frames, and then pad the null frames with the next valid frames")
            for i_s, skeleton in enumerate(tqdm(s)):  # Dimension N
                if skeleton.sum() == 0:
                    no_skeleton.append(i_s)
                # Shift non-zero nodes to beginning of frames by computing the valid range of all persons
                index_ranges = []
                for i_p, person in enumerate(skeleton):
                    #if i_p > 0: break # uncomment if we only use the first person to get the valid index range
                    # `index` of frames that have non-zero nodes
                    index = np.where(person.sum(-1).sum(-1) != 0)[0]
                    if len(index) > 0:
                        index_ranges.append([index[0],index[-1]])
                if len(index_ranges) > 0:
                    index_ranges = np.array(index_ranges)
                    index_start = index_ranges.min(0)[0]
                    index_end = index_ranges.max(0)[1]
                else:
                    index_start = 0
                    index_end = T-1
                tmp = skeleton[:,index_start:index_end+1].copy()
                skeleton *= 0
                length = index_end-index_start+1
                skeleton[:,:length] = tmp
                # pad the null frames with the next valid frames
                for i_p, person in enumerate(skeleton): # Dimension M (# person)
                    # `person` has shape (T, V, C)
                    if person.sum() == 0:
                        continue
                    isFirstIn = True
                    for i_f, frame in reversed(list(enumerate(person))):# in a reversed order
                        if isFirstIn:
                            isFirstIn = False
                            continue
                        if frame.sum() == 0:
                            s[i_s, i_p, i_f] = s[i_s, i_p, i_f+1]

                # pad the ending null frames with the previous valid frames
                for i_p, person in enumerate(skeleton): # Dimension M (# person)
                    # `person` has shape (T, V, C)
                    if person.sum() == 0:
                        continue
                    for i_f, frame in enumerate(person):
                        if i_f==0:
                            continue
                            # after prevous step, the first frame should be non-zero and valid now.
                        if frame.sum() == 0:
                            #if i_f < length: deactivated because we consider the frame for t in range(0,305)
                            s[i_s, i_p, i_f] = s[i_s, i_p, i_f-1]

        """

        no_skeleton = []
        for i_s, skeleton in enumerate(s):  # Dimension N
            if skeleton.sum() == 0:
                no_skeleton.append(i_s)

        print(no_skeleton,'have no skeleton.')

        """
            #print('skip the null frames')
            if person[0].sum() == 0:
                # `index` of frames that have non-zero nodes
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                # Shift non-zero nodes to beginning of frames
                person *= 0
                person[:len(tmp)] = tmp
            index_ranges = []
            for i_p, person in enumerate(skeleton):
                # `index` of frames that have non-zero nodes
                index = (person.sum(-1).sum(-1) != 0)
                index_ranges.append([index[0],index[-1]])
            index_ranges = np.array(index_ranges)
            index_start = index_ranges.min(0)[0]
            index_end = index_ranges.max(0)[1]
            tmp = skeleton[:,index_start:index_end].copy()
            # Shift non-zero nodes to beginning of frames
            skeleton *= 0
            skeleton[:,:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                # Each frame has shape (V, C)
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        # Repeat all the frames up to now (`i_f`) till the max seq len
                        rest = len(person) - i_f
                        reps = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[:i_f] for _ in range(reps)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

        if self.isCentering:
            print('sub the center joint of the first frame (spine joint in ntu and neck joint in kinetics)')
            index = np.array([5,6,11,12],dtype=np.int64)
            for i_s, skeleton in enumerate(tqdm(s[...,:3])):
                if skeleton.sum() == 0:
                    continue
                # Use the first skeleton's body center (index: hips, shoulder; 1:2: left eye)
                main_body_center = skeleton[0][:, index, :].mean(1,keepdims=True).copy()    # Shape (T, 4, C) -> Shape (T, 1, C)
                #main_body_center = skeleton[0][:, 1:2, :].copy()    # Shape (T, 1, C)
                #main_body_center = skeleton[0][:1, 1:2, :].copy()    # Shape (1, 1, C)
                for i_p, person in enumerate(skeleton):
                    if person.sum() == 0:
                        continue
                    # For all `person`, compute the `mask` which is the non-zero channel dimension
                    mask = (person.sum(-1) != 0).reshape(T, V, 1)
                    # Subtract the first skeleton's centre joint, s.shape = (N, M, T, V, C)
                    s[i_s, i_p, ..., :C] = (s[i_s, i_p, ..., :C] - main_body_center) * mask
        if self.is3DRotating:
            print('parallel the bone between (jpt {}) and (jpt {}) of the first person to the z axis'.format(zaxis[0],zaxis[1]))
            print('parallel the bone between right shoulder(jpt {}) and left shoulder(jpt {}) of the first person to the x axis'.format(xaxis[0],xaxis[1]))
            skeletons = Parallel(n_jobs=-1)(delayed(self.parallelRotation)(skeleton,zaxis,xaxis) for skeleton in tqdm(s[...,:3]))
            s = np.stack(skeletons)
        else:
            # eliminate the z-dimension which is zero
            s = s[:,:,:,:,:2]
            # C = 2
        """

        if self.setPerson0 == 1:
            print("setting more active person to position 0")
            skeletons = []
            if self.isParallel:
                skeletons = Parallel(n_jobs=-1)(delayed(setActivePerson0)(sample) for sample in tqdm(s))
            else:
                for i, sample in enumerate(tqdm(s)):
                    sample = setActivePerson0(sample)
                    skeletons.append(sample)
            s = np.stack(skeletons)

        if self.setPerson0 == 2:
            print("setting left person to position 0")
            skeletons = []
            if self.isParallel:
                skeletons = Parallel(n_jobs=-1)(delayed(setLeftPerson0)(sample) for sample in tqdm(s))
            else:
                for i, sample in enumerate(tqdm(s)):
                    sample = setLeftPerson0(sample)
                    skeletons.append(sample)
            s = np.stack(skeletons)

        if self.isCentering:  # 0 for not centering, 1 for frame-wise centering, 2 for sample-wise centering
            print('sub the center joint of {} frame (spine joint in ntu and neck joint in kinetics)'.format(
            'the first' if self.isCentering == 2 else 'each'))
            index = np.array([5, 6, 11, 12], dtype=np.int64)
            isSamplewise = True if self.isCentering == 2 else False
            ndim = min(C-1, 3)  # the maximum num of coordinates for this operation is 3
            skeletons = []
            if self.isParallel:
                # if padding with zero in previous steps, then the centering should only work on non-zero frames (use the length info)
                # skeletons = Parallel(n_jobs=-1)(delayed(parallelCenter)(skeleton,index,isSamplewise,lengths[i]) for i,skeleton in enumerate(tqdm(s[...,:ndim])))
                skeletons = Parallel(n_jobs=-1)(
                    delayed(self.parallelCenter)(skeleton, index, isSamplewise) for i, skeleton in
                    enumerate(tqdm(s[..., :ndim])))
            else:
                for i, skeleton in enumerate(tqdm(s[..., :ndim])):
                    # skeleton = parallelCenter(skeleton,index,isSamplewise,length[i])
                    skeleton = self.parallelCenter(skeleton, index, isSamplewise)
                    skeletons.append(skeleton)
            s[..., :ndim] = np.stack(skeletons)

        if self.is3DRotating:  # 0 for not doing, 1 for frame-wise, 2 for sample-wise
            if C > 3:  # only use for 3d
                print('parallel the bone between (jpt {}) and (jpt {}) of the person to the z axis'.format(zaxis[0],
                                                                                                       zaxis[1]))
                print(
                    'parallel the bone between right shoulder(jpt {}) and left shoulder(jpt {}) of the person to the x axis'.format(
                        xaxis[0], xaxis[1]))
                zaxis, xaxis = np.array(zaxis, dtype=np.int64), np.array(xaxis, dtype=np.int64)
                isSamplewise = True if self.is3DRotating == 2 else False
                ndim = 3  # the maximum num of coordinates for this operation is 3
                skeletons = []
                if self.isParallel:
                    skeletons = Parallel(n_jobs=-1)(
                        delayed(self.parallelRotation)(skeleton, zaxis, xaxis, isSamplewise) for i, skeleton in
                        enumerate(tqdm(s[..., :ndim])))
                else:
                    for i, skeleton in enumerate(tqdm(s[..., :ndim])):
                        skeleton = self.parallelRotation(skeleton, zaxis, xaxis, isSamplewise)
                        skeletons.append(skeleton)
                s[..., :ndim] = np.stack(skeletons)

        if self.isScaling:  # 0 for not doing, 1 for frame-wise, 2 for sample-wise
            print('rescale a second time')
            isSamplewise = True if self.isScaling == 2 else False
            ndim = min(C - 1, 3)  # the maximum num of coordinates for this operation is 3
            skeletons = []
            if self.isParallel:
                skeletons = Parallel(n_jobs=-1)(
                    delayed(self.parallelScale)(skeleton, isSamplewise) for i, skeleton in enumerate(tqdm(s[..., :ndim])))
            else:
                for i, skeleton in enumerate(tqdm(s[..., :ndim])):
                    skeleton = self.parallelScale(skeleton, isSamplewise)
                    skeletons.append(skeleton)
            s[..., :ndim] = np.stack(skeletons)

        if self.smoothen:
            print('apply Savgol filter')
            s[..., :2] = savgol_filter(s[..., :2], 9, 2, axis=2)

        # if self.flag == 'train':
        #     labels = self.train_prenorm_label.copy()
        # else:
        #     labels = self.test_filenames.copy()
        #
        # if self.setPerson0 == 3:
        #     print("duplicate samples with 2 bodies")
        #     swappedSamples = []
        #     newLabels = []
        #     for i, sample in enumerate(tqdm(s)):
        #         if numberBodies(sample) == 2:
        #             swappedSamples.append(np.flip(sample, axis=0))
        #             if self.flag == 'train':
        #                 newLabels.append(self.train_prenorm_label[i])
        #             else:
        #                 newLabels.append(self.test_filenames[i])
        #     if self.flag == 'train':
        #         labels = np.concatenate((self.train_prenorm_label, np.stack(newLabels)), axis = 0)
        #     else:
        #         labels.extend(newLabels)
        #     s = np.concatenate((s, np.stack(swappedSamples)), axis=0)

        def parallelGetEnergy(sample):
            energy0 = energyNodes(sample[0,...])
            energy1 = energyNodes(sample[1,...])
            return(np.array([energy0,energy1]))

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        if self.confidence != 0:
            print('calculating energy for each node')
            energy = []
            if self.isParallel:
                energy = Parallel(n_jobs=-1)(delayed(parallelGetEnergy)(sample) for sample in tqdm(s))
            else:
                for i, sample in enumerate(tqdm(s)):
                    energy_sample = parallelGetEnergy(sample)
                    energy.append(energy_sample)
            energy = np.stack(energy)

            for v in range(energy.shape[3]):
                maskedEnergy = np.ma.masked_equal(energy[...,v],0)
                std = np.std(maskedEnergy)
                energy[...,v] = energy[...,v]/std

            EN, EM, ET, EV = energy.shape
            energy = energy.reshape(EN, EM, ET, EV, 1)

            if self.confidence == 1:

                s = np.concatenate((s,energy), axis = 4)

            if self.confidence == 2:
                print("calculating confidence scores")
                nonZeroEnergies = (energy != 0)
                with np.errstate(divide='ignore'):
                    confidence = 1.5*(math.log(0.15) - np.log(energy))
                sigmoid_v = np.vectorize(sigmoid)
                confidence = nonZeroEnergies * sigmoid_v(confidence)
                s = np.concatenate((s,confidence), axis = 4)

        data = np.transpose(s, [0, 4, 2, 3, 1]) # to (N,C,T,V,M)

        def add_center_joint(data):
            new_data=np.zeros((data.shape[0],3,305,18,2))
            new_data[:,:,:,:17,:]=data
            JOINTS_AROUND_CENTER=[5,6,11,12]

            print('Adding center joint')
            for i in range(len(data)):
                print(i,'added',end='\r')
                for frame in range(self.N_TIMESTEPS):
                    for body in range(self.N_PERSONS):
                        center_coords=np.sum(data[i,:,frame,JOINTS_AROUND_CENTER,body],axis=0)/len(JOINTS_AROUND_CENTER)
                        new_data[i,:,frame,17,body]=center_coords

            return new_data

        if self.add_center_joint:
            data=add_center_joint(data)
            raw_data=add_center_joint(raw_data)

        print("-----------------Finished Preprocessing-----------------")

        raw_file_name=self.file_names.data_file_name('raw_'+flag,debug=debug)
        prenorm_file_name=self.file_names.data_file_name('prenorm_'+flag,debug=debug)
        #reshape data: from (N,C,T,V,M) to (N,T,C,V*M)
        raw_data=np.transpose(raw_data,axes=[0,2,1,3,4])
        raw_data=np.reshape(raw_data,(-1,self.N_TIMESTEPS,self.N_AXES+1,self.N_JOINTS*self.N_PERSONS))
        data=np.transpose(data,axes=[0,2,1,3,4])
        data=np.reshape(data,(-1,self.N_TIMESTEPS,self.N_AXES+1,self.N_JOINTS*self.N_PERSONS))
        print('Saving raw data to',raw_file_name)
        np.save(raw_file_name,raw_data)
        print('Saving processed data to',prenorm_file_name)
        np.save(prenorm_file_name,data)
        return data

    def parallelRotation(self, skeleton,zaxis,xaxis):
        if skeleton.sum() == 0:
            return skeleton
        # Shapes: (C,)

        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue

                joint_bottom = skeleton[0, i_f, zaxis[0]]
                joint_top = skeleton[0, i_f, zaxis[1]]
                axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
                angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
                matrix_z = self.rotation_matrix(axis, angle)

                joint_rshoulder = skeleton[0, i_f, xaxis[0]]
                joint_lshoulder = skeleton[0, i_f, xaxis[1]]
                axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
                angle = self.angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
                matrix_x = self.rotation_matrix(axis, angle)

                for i_j, joint in enumerate(frame):
                    skeleton[i_p, i_f, i_j, :3] = np.dot(np.dot(matrix_x, matrix_z), joint)
        return skeleton

    def rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @staticmethod
    def x_rotation(vector, theta):
        """Rotates 3-D vector around x-axis"""
        R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        return np.dot(R, vector)

    @staticmethod
    def y_rotation(vector, theta):
        """Rotates 3-D vector around y-axis"""
        R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        return np.dot(R, vector)

    @staticmethod
    def z_rotation(vector, theta):
        """Rotates 3-D vector around z-axis"""
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        return np.dot(R, vector)

    def parallel_get_duration(self, sample):#T,M,V,C
        T = sample.shape[0]
        length = T
        for t in range(T-1,-1,-1):
            if np.fabs(sample[t]).sum(-1).sum(-1).sum(-1) > 0.0001:
                length = t + 1
                break
        return length

    def get_data_duration(self, data_uav):#nctvm
        print('Get length info from data...')
        data_uav = np.transpose(data_uav,(0,2,4,3,1))
        N,T,M,V,C = data_uav.shape
        #print(N,T,M,V,C)
        lengths = Parallel(n_jobs=-1)(delayed(self.parallel_get_duration)(d) for d in data_uav)
        data_uav = np.transpose(data_uav,(0,4,1,3,2))
        lengths = np.array(lengths)
        return lengths

    def parallelScale(self,skeleton_origin,isSamplewise=True,length=-1,isCenter2Origin=False,isKeepHWRatio=True,scaleFactor=1.0):
        if skeleton_origin.sum() == 0:
            return skeleton_origin
        if length > 0:
            skeleton = skeleton_origin[:,:length,...]
        else:
            skeleton = skeleton_origin

        nonZeroFrames = findNonZeroFrames(skeleton[0,...])
        M, T, V, C = skeleton.shape
        ref_person_id = 0

        if isSamplewise:
            ma = np.mean((skeleton[ref_person_id,nonZeroFrames,:,:]).max(axis=1), axis = 0)
            mi = np.mean((skeleton[ref_person_id,nonZeroFrames,:,:]).min(axis=1), axis = 0)
        else:
            ma = (skeleton[ref_person_id].reshape(-1,C)).max(axis = 0)
            mi = (skeleton[ref_person_id].reshape(-1,C)).min(axis = 0)

        skeleton = (skeleton)*scaleFactor/((ma - mi).max() if isKeepHWRatio else (ma - mi))
        if isCenter2Origin: skeleton -= 0.5*scaleFactor

        skeleton_origin[:,:skeleton.shape[1],...] = skeleton
        return skeleton_origin

    def parallelCenter(self,skeleton_origin,index,isSamplewise=True,length=-1):
        if skeleton_origin.sum() == 0:
            return skeleton_origin
        if length > 0:
            skeleton = skeleton_origin[:,:length,...]
        else:
            skeleton = skeleton_origin

        M, T, V, C = skeleton.shape
        # Use the skeleton's body center
        ref_person_id = 0
        if isSamplewise:
            main_body_center = skeleton[ref_person_id][:1, index, :].mean(1,keepdims=True).copy()    # Shape (1, 4, C) -> Shape (1, 1, C)
        else:
            main_body_center = skeleton[ref_person_id][:, index, :].mean(1,keepdims=True).copy()    # Shape (T, 4, C) -> Shape (T, 1, C)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            # For all `person`, compute the `mask` which is the non-zero channel dimension
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            # Subtract the first skeleton's centre joint, s.shape = (N, M, T, V, C)
            skeleton[i_p, ..., :C] = (skeleton[i_p, ..., :C] - main_body_center) * mask
        skeleton_origin[:,:skeleton.shape[1],...] = skeleton
        return skeleton_origin

    def parallelRotation(self, skeleton_origin, zaxis, xaxis, isSamplewise=True, length=-1):
        if skeleton_origin.sum() == 0:
            return skeleton_origin
        if length > 0:
            skeleton = skeleton_origin[:, :length, ...]
        else:
            skeleton = skeleton_origin

        skeleton = np.transpose(skeleton, (1, 0, 2, 3))  # from (M, T, V, C) to (T, M, V, C)

        T, M, V, C = skeleton.shape


        if isSamplewise:
            matrix_all = None
            for i_f, frame in enumerate(skeleton):  # T, M, V, C = skeleton.shape
                if frame.sum() == 0:
                    continue
                ref_person_id = -1
                for i_p, person in enumerate(frame):
                    if person.sum() == 0:
                        continue
                    ref_person_id = i_p
                    break

                if ref_person_id >= 0:
                    joint_bottom = frame[ref_person_id, zaxis[0]].reshape(-1, C).mean(0)
                    joint_top = frame[ref_person_id, zaxis[1]].reshape(-1, C).mean(0)
                    axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
                    angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
                    matrix_z = self.rotation_matrix(axis, angle)

                    joint_right = frame[ref_person_id, xaxis[0]].reshape(-1, C).mean(0)
                    joint_left = frame[ref_person_id, xaxis[1]].reshape(-1, C).mean(0)
                    axis = np.cross(joint_right - joint_left, [1, 0, 0])
                    angle = self.angle_between(joint_right - joint_left, [1, 0, 0])
                    matrix_x = self.rotation_matrix(axis, angle)

                    matrix_all = np.dot(matrix_x, matrix_z)

                break
            if matrix_all is not None:
                skeleton = np.dot(matrix_all, skeleton.reshape(-1, C).T).T.reshape(T, M, V, C)

        else:  # frame-wise
            for i_f, frame in enumerate(skeleton):  # T, M, V, C = skeleton.shape
                if frame.sum() == 0:
                    continue

                ref_person_id = -1
                for i_p, person in enumerate(frame):
                    if person.sum() == 0:
                        continue
                    ref_person_id = i_p
                    break

                if ref_person_id >= 0:
                    joint_bottom = frame[ref_person_id, zaxis[0]].reshape(-1, C).mean(0)
                    joint_top = frame[ref_person_id, zaxis[1]].reshape(-1, C).mean(0)
                    axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
                    angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
                    matrix_z = self.rotation_matrix(axis, angle)

                    joint_right = frame[ref_person_id, xaxis[0]].reshape(-1, C).mean(0)
                    joint_left = frame[ref_person_id, xaxis[1]].reshape(-1, C).mean(0)
                    axis = np.cross(joint_right - joint_left, [1, 0, 0])
                    angle = self.angle_between(joint_right - joint_left, [1, 0, 0])
                    matrix_x = self.rotation_matrix(axis, angle)

                    matrix_all = np.dot(matrix_x, matrix_z)
                    skeleton[i_f] = np.dot(matrix_all, frame.reshape(-1, C).T).T.reshape(M, V, C)

        skeleton = np.transpose(skeleton, (1, 0, 2, 3))  # from (T, M, V, C) to (M, T, V, C)
        skeleton_origin[:, :skeleton.shape[1], ...] = skeleton
        return skeleton_origin
