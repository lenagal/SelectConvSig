# SelectConvSig
Human activities recognition from videos is an important task with a wide range of possible real-life applications. Accuracy of various machine learning techniques in this field has a room for improvement, especially for noisy datasets. One of the succesful strategies for such datasets is to consider the skeletal representation of the data i.e. essentially just the coordinates of certain key joints of the subjects. This allows to strip away the redundant information inherent in the RGB presentation, while still retaining enough information to allow e.g. a human observer to classify many actions with reasonable accuracy. The group of algorithms based on this data presentation is called Skeletal Action Recognition Algorithms. 

The current state-of-the-art algorithms in this field are based on Graph Convolution method. The basic idea of this method can be understood as an attempt to apply convolutional methods to the dataset of body joints organized into a graph according to some principle, e.g. proximity on a physical skeleton. This graph replaces spatial organization that RGB images posses, enabling aplication of convolutional kernels. The major challenge for improvement of classification accuracy for this group of algorithms is finding a suitable graph for optimal learning. Proximity on physical skeleton is certainly an important feature, because it contains information about fixed distances between joints, but it is certainly not the only relation between joints that is important for classification of actions. For example to recognize an action of eating the important joints are mouth and hands, but this connection is not expressed by a graph based on physical skeleton. Some other shortcomings of Graph Convolution based algorithns are their inherent computational complexity and lack of invariance under small perturbations of coordinates.

This project is motivated by a dataset of videos captured by UAV devices described in https://arxiv.org/pdf/2104.00946.pdf and available to download from https://github.com/SUTDCV/UAV-Human/tree/master/uavhumanactiontools. AlphaPose algorithm was used to extract skeletal data from the videos. The dataset is divided into a train and test subsets. We use this dataset as a challenge to produce a robust action recognition algorithm for working with skeletal data. We implement various data augmentation methods, e.g. continuous rotations and zooming that improve performance of all classification methods from the baseline. 

We also benchmark performance of all our methods on NTU RGB+D dataset, which is a standard benchmark in the field. 

Broadly our approach is based on viewing the skeletal data as a set of spatio-temporal paths. An example of a spatial path of lenght two is a bone in a human skeleton. Extending this idea one can consider paths of varying lenghts for any choice of joints. Spatially, i.e. for every frame one crucial challenge is to pick out paths that best capture the meaningful information about the content of the video. This is of course dependent the temporal context, e.g. we should be focusing on the relevant joints as the video progresses. Another challenge is to dynamically organize these paths into a structure that will enable convolution. In this context the basic existing graph convolution methods use paths of length 1 and 2 (that are just human joints and bones) and the spatial organization is dictated by the physical skeleton and unchanged for the duration of the video. 

Signature is a mathematical invariant of paths introduced in the work of Terry Lyons that in the context of machine learning can be thought of as a method for finding linear approximations for functionals on curves. (References to articles and libraries). We use signatures to encode both spatial and temporal features of data.

Our basic algorithm (SelectConvSigRNNBase) uses just simple convolution on all joints and signature+LSTM for temporal analysis. SelectConvSigRNNJB uses joints and bones spatially and signature combined with LSTM classifier temporally to achieve performance comparable with state-of-the-art methods mentioned in the article by the dataset authors.

We explore different strategies to improve over these baseline models:

-Choose several triples of points to enhance separation between ambiguous classes... 
this performance then select a number of paths that allow for best separation between classes in the dataset according to the AUC ROC metric and use this as our base dataset. 

