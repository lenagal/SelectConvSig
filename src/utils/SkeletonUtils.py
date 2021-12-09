import numpy as np
import tensorflow as tf

class ConvGraph:
    def __init__(self,N_VERTICES,edges,neighbours):
        self.N_VERTICES=N_VERTICES
        self.edges=edges
        self.neighbours=neighbours
        self.N_NEIGHBOURS=len(neighbours[0])

    def neighbor_matrix(self,kernel):
        return neighbor_matrix(kernel,self.neighbours,self.N_VERTICES)

    def adj_matrix(self):
        return skeleton_adj_matrix(self.edges,self.N_VERTICES)

def UAVHuman_Bones_graph():
    return ConvGraph(
                    UAVHuman_Bones(),
                    UAVHuman_Bones_graph_edges(),
                    UAVHuman_Bones_graph_neighbors()
                    )

def UAVHuman_Joints_graph(with_center=True,Extended_neighbours=False,):
    return ConvGraph(
                UAVHuman_Joints(with_center=with_center),
                UAVHuman_Joints_graph_edges(with_center=with_center),
                UAVHuman_Joints_graph_neighbors(extended=Extended_neighbours)
                )
def NTU_Joints_graph():
    return ConvGraph(
                NTU_Joints(),
                NTU_Joints_graph_edges(),
                NTU_Joints_graph_neighbors()
                )

def NTU_Bones_graph():
    return ConvGraph(
                    len(NTU_Joints_graph_edges()),
                    NTU_Bones_graph_edges(),
                    NTU_Bones_graph_neighbors()
                    )
def NTU_Joints():
    '''
        0: pelvis
        1: stomach (center joint)
        2: neck
        3: head
        4: right_shoulder
        5: right_elbow
        6: right_wrist
        7: right_palm
        8: left_shoulder
        9: left_elbow
        10: left_wrist
        11: left_palm
        12: right_hip
        13: right_knee
        14: right_ankle
        15: right_foot
        16: left_hip
        17: left_knee
        18: left_ankle
        19: left_foot
        20: solar_plexus
        21: right_fingers
        22: right_thumb
        23: left_fingers
        24: left_thumb
    '''
    return 25

def NTU_Joints_graph_edges():
    '''24 Bones:'''
    BONES=[[0,1], #pelvis-stomach
            [1,20], #stomach-solar
            [20,2], #solar-neck
            [2,3], #neck-head
            [20,4], #solar-r shoulder
            [4,5], #r shoulder-r elbow
            [5,6], #r elbow-r wrist
            [6,7], #r wrist-r palm
            [7,22],#r palm - r thumb
            [7,21],#r palm -r fingers
            [20,8],#solar-l shoulder
            [8,9],#l shoulder-l elbow
            [9,10],#l elbow-l wrist
            [10,11],#l wrist-l palm
            [11,24],#l palm - l thumb
            [11,23],#l palm -l fingers
            [0,12],#pelvis-r hip
            [12,13],#r hip - r knee
            [13,14],#r knee - r ankle
            [14,15],#r ankle - r foot
            [0,16],#pelvis-l hip
            [16,17],#l hip - l knee
            [17,18],#l knee - l ankle
            [18,19],#l ankle - l foot
            ]
    return BONES

def NTU_Joints_graph_neighbors(Extended_neighbours=False):
    NEIGHBORS=[
        [[1],[0],[12,16]],
        [[0],[1],[20]],
        [[20],[2],[3]],
        [[2],[3],[]],
        [[20],[4],[5]],
        [[4],[5],[6]],
        [[5],[6],[7]],
        [[6],[7],[21,22]],
        [[20],[8],[9]],
        [[8],[9],[10]],
        [[9],[10],[11]],
        [[10],[11],[23,24]],
        [[0],[12],[13]],
        [[12],[13],[14]],
        [[13],[14],[15]],
        [[14],[15],[]],
        [[0],[16],[17]],
        [[16],[17],[18]],
        [[17],[18],[19]],
        [[18],[19],[]],
        [[1],[20],[2,8,4]],
        [[7],[21],[]],
        [[7],[22],[]],
        [[11],[23],[]],
        [[11],[24],[]]
    ]
    # if extended:
    #     NEIGHBORS=[
    #         [[1],[0],[12,16]],
    #         [[0],[1],[20]],
    #         [[20],[2],[3]],
    #         [[2],[3],[]],
    #         [[20],[4],[5]],
    #         [[4],[5],[6]],
    #         [[5],[6],[7]],
    #         [[6],[7],[21,22]],
    #         [[20],[8],[9]],
    #         [[8],[9],[10]],
    #         [[9],[10],[11]],
    #         [[10],[11],[23,24]],
    #         [[0],[12],[13]],
    #         [[12],[13],[14]],
    #         [[13],[14],[15]],
    #         [[14],[15],[]],
    #         [[0],[16],[17]],
    #         [[16],[17],[18]],
    #         [[17],[18],[19]],
    #         [[18],[19],[]],
    #         [[1],[20],[2,8,4]],
    #         [[7],[21],[]],
    #         [[7],[22],[]],
    #         [[11],[23],[]],
    #         [[11],[24],[]]
    #     ]
    return NEIGHBORS

def NTU_Bones_graph_edges():
    EDGES=[
        [0,1],
        [0,20],
        [0,16],
        [1,4],
        [1,10],
        [1,2],
        [2,3],
        [10,11],
        [11,12],
        [12,13],
        [13,14],
        [13,15],
        [4,5],
        [5,6],
        [6,7],
        [7,8],
        [7,9],
        [16,17],
        [17,18],
        [18,19],
        [20,21],
        [21,22],
        [22,23]
    ]
    return EDGES
def NTU_Bones_graph_neighbors():
    NEIGHBORS=[
        [[],[0],[20,16]],
        [[],[1],[4,10,2]],
        [[1],[2],[4,10]],
        [[2],[3],[]],
        [[1],[4],[10,2,5]],
        [[4],[5],[6]],
        [[5],[6],[7]],
        [[6],[7],[8,9]],
        [[7],[8],[]],
        [[7],[9],[]],
        [[1],[10],[4,2,11]],
        [[10],[11],[12]],
        [[11],[12],[13]],
        [[12],[13],[14,15]],
        [[13],[14],[]],
        [[13],[15],[]],
        [[0],[16],[20,17]],
        [[16],[17],[18]],
        [[17],[18],[19]],
        [[18],[19],[]],
        [[0],[20],[16,21]],
        [[20],[21],[22]],
        [[21],[22],[23]],
        [[22],[23],[]]
    ]
    return NEIGHBORS

def UAVHuman_Joints(with_center=True):
    '''
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
        17: center (average of 5,6,11,12)
    '''
    if with_center:
        return 18
    else:
        return 17


def UAVHuman_Joints_movement_threshold():
    '''
        Thresholds for movement relative to distance from center to neck
    '''
    head_threshold=0.5
    inner_joint_threshold=1
    middle_joint_threshold=2
    outer_joint_threshold=3
    center_threshold=0.5

    THRESHOLDS=[
        head_threshold, #0: nose,
        head_threshold, #1: left_eye,
        head_threshold, #2: right_eye,
        head_threshold, #3: left_ear,
        head_threshold, #4: right_ear,
        inner_joint_threshold, #5: left_shoulder,
        inner_joint_threshold, #6: right_shoulder,
        middle_joint_threshold, #7: left_elbow,
        middle_joint_threshold, #8: right_elbow,
        outer_joint_threshold, #9: left_wrist,
        outer_joint_threshold, #10: right_wrist,
        inner_joint_threshold, #11: left_hip,
        inner_joint_threshold, #12: right_hip,
        middle_joint_threshold, #13: left_knee,
        middle_joint_threshold, #14: right_knee,
        outer_joint_threshold, #15: left_ankle,
        outer_joint_threshold, #16: right_ankle
        center_threshold, #17: center (average of 5,6,11,12)
    ]
    return THRESHOLDS

def UAVHuman_Bone_size_threshold():
    head_threshold=1.2
    center_threshold=1.2
    small_limb_threshold=1.5
    big_limb_threshold=1.5

    THRESHOLDS=[
        head_threshold, #[0,1],#nose-eye
        head_threshold, #[0,2],#nose-eye
        head_threshold, #[1,3],#eye-ear
        head_threshold, #[2,4],#eye-ear
        center_threshold, #[17,0],#center-nose
        center_threshold, #[17,5],#center-shoulder
        small_limb_threshold, #[5,7],#shoulder-elbow
        small_limb_threshold, #[7,9],#elbow-wrist
        center_threshold, #[17,6],#center-shoulder
        small_limb_threshold, #[6,8],#shoulder-elbow
        small_limb_threshold, #[8,10],#elbow-wrist
        center_threshold, #[17,11],#center-hip
        big_limb_threshold, #[17,12],#hip-knee
        big_limb_threshold, #[11,13],#knee-foot
        center_threshold, #[13,15],#center-hip
        big_limb_threshold, #[12,14],#hip-knee
        big_limb_threshold, #[14,16]#knee-foot
    ]
    return THRESHOLDS
def UAVHuman_Joints_graph_edges(with_center=True):
    '''
        17 Bones:
    '''
    BONES=[
        [0,1],#0
        [0,2],#1
        [1,3],#2
        [2,4],#3
        [17,0],#4
        [17,5],#5
        [5,7],#6
        [7,9],#7
        [17,6],#8
        [6,8],#9
        [8,10],#10
        [17,11],#11
        [17,12],#12
        [11,13],#13
        [13,15],#14
        [12,14],#15
        [14,16]#16
    ]
    if not with_center:
        BONES=[
            (10, 8), (8, 6), (9, 7), (7, 5), # arms
            (15, 13), (13, 11), (16, 14), (14, 12), # legs
            (11, 5), (12, 6), (11, 12), (5, 6), # torso
            (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears
        ]
    return BONES

def UAVHuman_Bones():
    return 17

def UAVHuman_Bones_graph_neighbors():
    NEIGHBORS=[
    [[4],[0],[1,2]],
    [[4],[1],[0,3]],
    [[0],[2],[]],
    [[1],[3],[]],
    [[],[4],[0,1]],
    [[],[5],[6]],
    [[5],[6],[7]],
    [[6],[7],[]],
    [[],[8],[9]],
    [[8],[9],[10]],
    [[9],[10],[]],
    [[],[11],[13]],
    [[],[12],[15]],
    [[11],[13],[14]],
    [[13],[14],[]],
    [[12],[15],[16]],
    [[15],[16],[]],
    ]

    return NEIGHBORS

def UAVHuman_Bones_graph_edges():
    EDGES=[
        [0,2],
        [0,1],
        [1,3],
        [4,1],
        [5,6],
        [6,7],
        [8,9],
        [9,10],
        [11,13],
        [13,14],
        [12,15],
        [15,16]
    ]

    return EDGES

def UAVHuman_Joints_graph_neighbors(extended=False):
    NEIGHBORS=[
        [[17],[0],[1,2]],
        [[0],[1],[3]],
        [[0],[2],[4]],
        [[1],[3],[]],
        [[2],[4],[]],
        [[17],[5],[7,6,11]],
        [[17],[6],[8,5,12]],
        [[5],[7],[9]],
        [[6],[8],[10]],
        [[7],[9],[]],
        [[8],[10],[]],
        [[17],[11],[13,5,12]],
        [[17],[12],[14,6,11]],
        [[11],[13],[15]],
        [[12],[14],[16]],
        [[13],[15],[]],
        [[14],[16],[]]
    ]
    if extended:
        NEIGHBORS=[
            [[],[17],[0],[1,2],[3,4]],
            [[17],[0],[1],[3],[]],
            [[17],[0],[2],[4],[]],
            [[0],[1],[3],[],[]],
            [[0],[2],[4],[],[]],
            [[],[17],[5],[7],[9]],
            [[],[17],[6],[8],[10]],
            [[17],[5],[7],[9],[]],
            [[17],[6],[8],[10],[]],
            [[5],[7],[9],[],[]],
            [[6],[8],[10],[],[]],
            [[],[17],[11],[13],[15]],
            [[],[17],[12],[14],[16]],
            [[17],[11],[13],[15],[]],
            [[17],[12],[14],[16],[]],
            [[11],[13],[15],[],[]],
            [[12],[14],[16],[],[]]
        ]

    return NEIGHBORS

def neighbor_matrix_indices(neighbor_list,N_JOINTS):
    indices=[]
    multiplicities=[]
    for i,sequence in enumerate(neighbor_list):
        trip_mult=[]
        for list in sequence:
            trip_mult.append(len(list))
            for joint in list:
                indices.append([i,joint])
        multiplicities.append(trip_mult)
    return indices, multiplicities

def neighbor_matrix(kernel,neighbor_list,N_JOINTS):
    '''
        kernel must be of size N_NEIGHBOURS x extra dims
    '''
    N_NEIGHBOURS=len(neighbor_list[0])

    matrix=tf.zeros((N_JOINTS,N_JOINTS)+kernel.shape[1:],dtype=tf.float32)
    indices, multiplicities = neighbor_matrix_indices(neighbor_list,N_JOINTS)
    updates=[]
    for sequence in multiplicities:
        for i in range(N_NEIGHBOURS):
            for j in range(sequence[i]):
                updates.append(kernel[i]/sequence[i])
    matrix = tf.tensor_scatter_nd_update(matrix,indices,updates)
    return tf.transpose(matrix,perm=list(range(2,len(matrix.shape))) + [0,1])

def skeleton_adj_matrix(bones,N_JOINTS):
    matrix=np.zeros((N_JOINTS,N_JOINTS))
    for bone in bones:
         matrix[bone[0],bone[1]]=1
         matrix[bone[1],bone[0]]=1
    return matrix

def joints_to_bones_matrix(bones_list,N_JOINTS):
    matrix=np.zeros((len(bones_list),N_JOINTS))
    for i,bone in enumerate(bones_list):
        matrix[i,bone[0]]=-1
        matrix[i,bone[1]]=1
    return matrix
