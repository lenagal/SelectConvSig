# SelectConvSig
Human activities recognition from videos is an important task with a wide range of possible real-life applications. Success of application of various machine learning techniques for this task had so far been lagging way behind the their application to the image recognition task, by and large due to the fact that videos contain a magnitude more information than images.

One of the approaches to action recognition task is to classify activities according to the data encoded in the movement of joints and bones of subjects in the video. This strips away the redundant information inherent in the RGB presentation. The group of algorithms based on this data presentation is called Skeletal Action Recognition Algorithms.

The state-of-the-art method in this field is Graph Convolution. The basic idea of this method is to apply convolutional methods to the dataset of body joints organized into a graph according to some principle, e.g. proximity on a physical skeleton. This graph replaces spatial organization enabling aplication of convolutional kernels. The major challenge for improvement of classification accuracy for this group of algorithms is finding a suitable graph for optimal learning. For example to recognize an action of eating the important joints are mouth and hands, but this connection is not expressed by eg a graph based on physical skeleton. Another challenge is the computational complexitiy of these algorithms. 

Signature is a mathematical invariant introduced in the works of Terry Lyons that in the context of machine learning can be thought of as a method for finding linear approximations for functionals on curves. (References to articles and libraries). To connect this idea to the task in hand note that human joints and various graphs on them can be thought of as a set of spatio-temporal paths.  

We propose an approach to the Skeletal Action Recognition Task that uses paths etc
