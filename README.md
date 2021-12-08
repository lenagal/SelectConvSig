# SelectConvSig

This project contains several ANN classifiers for human action recognition (HAR) from videos using the skeletal data.

This project is motivated by a dataset of videos captured by UAV devices (UAVHuman) described in https://arxiv.org/pdf/2104.00946.pdf and available to download from https://github.com/SUTDCV/UAV-Human/tree/master/uavhumanactiontools. AlphaPose algorithm was used to extract skeletal data from the videos. The dataset is divided into a train and test subsets. We use this dataset as a challenge to produce a robust action recognition algorithm for working with skeletal data. We also benchmark performance of our methods on NTU RGB+D dataset, which is a standard benchmark in the field. 

Skeletal representation of the video data constains just the information of movements of certain key joints. This allows to strip away the redundant information inherent in the RGB presentation, while still retaining enough information to allow e.g. a human observer to classify many actions with reasonable accuracy. The group of algorithms based on this data presentation is called Skeletal Action Recognition Algorithms. In real-life applications the skeletal presentation can be further combined with other modalities to boost classification accuracy.

As a first step we implement some basic denoising and various data augmentation methods for the UAVHuman dataset, e.g. continuous rotations and zooming. This significantly improves performance of all our classification methods compared with the baseline. 

Our approach to skeletal action recognition is based on viewing the skeletal data as a set of spatio-temporal paths. An example of a spatial path of length two is a bone in a human skeleton. Extending this idea one can consider collections of paths of varying lenghts for any choice of joints. A crucial challenge is to pick out paths that best capture the meaningful information about the content of the video. This is similar to the human observer shifting attention from the whole skeleton to a subset of relevant joints as the video progresses to classify the action. Another challenge is to organize these paths into a structure that will enable application of convolutional kernel. For comparison the existing GC methods use paths of length 1 and 2 (these are just human joints and bones, as well as temporal connections of length 1 between same joints) and the spatial organization is mainly dictated by the physical skeleton and stays unchanged for the duration of the video. 

Signature is a mathematical invariant of paths introduced in the work of Terry Lyons that in the context of machine learning can be thought of as a method for finding linear approximations for functionals on curves. A signature of a curve in a vector space V is a vector in the tensor space T(V), so e.g. a planar curve is assigned a series of numbers. It is invariant with respect to parametrization and "similar" curves have "similar" signatures. (See eg https://arxiv.org/pdf/1405.4537.pdf). Signatures can be used to encode both spatial and temporal features of data.

Classifiers currently in this repository: 

SelectConvSigRNNBase - a basic algorithm that uses depthwise convolutio on all (unstructured) joints and signature+LSTM for temporal analysis. 

SelectConvSigRNNJB  - adds Graph Convolution on joints and bones spatially to achieve performance comparable with state-of-the-art methods mentioned in the article by the dataset authors.

SkeletalFeatureChoice- a group of algorithms that searches for additional features to aid in classification. TriplesChoiceRNN chooses triples of joints that are important for classification by using convolution on the set of all triples and l1-regularisation. ROCTriplesRNN uses AUC ROC metric on pairs of classes.

SelectConvSigRNNStreams - classifies using selected triples data.

TeacherStudent - UAVHuman train and test set are inherently problematic since they appear to represent different distributions (i.e mixing them and splitting test set off at random increases accuracy). We experiment with the teacher-student paradigm in an attempt to compensate for this.

