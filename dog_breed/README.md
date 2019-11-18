## Dog breed classification

Welcome, I built this notebook while studying the AI nanodegree from Udacity. This notebook addresses a supervised learning problem, namely dog breed classification. Two CNN (Convolutional Neural Networks) are illustrated. The first is built from scratch. My implementation convolutional layer consists of 11 layers and an 11-layer FF (Feed Forward) neural network. Results were not fantastic as it only achieved test accuracy of 13%.  However, it illustrates the complete process for building a CNN; the second CNN is based on VGG16. I entirely replace the FF layer with my own and used VGG16 convolutional layer for feature extraction. The test results were quite surprising, test accuracy jump to 84%. That is nearly 6.5 fold increase in test accuracy using the same FF architecture, amazing! It shows how important feature extraction is in a learning process. 

You are welcome to use any of the trained models,
-	model_scratch2.pt.gz with 13% test precision.
-	model_scratch3.pt.gz with 84% test precision.
Unfortunately, you must run the training process to generate them. 

**NOTE** The datasets are quite large and are not stored alongside this repository. They can be downloaded from the [dog](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and [human](http://vis-www.cs.umass.edu/lfw/lfw.tgz) repositories. Unzip the dog dataset and place it at location `path/to/dog-project/dogImages`, similarly unzip the human dataset and place it at the location `path/to/dog-project/lfw`.

All the code was built with [Pytorch]( https://pytorch.org/). Models were trained using a GPU equipped architecture. Amazon web services provide EC2 GPU instances, for a cost.   

I want to express my appreciation to:
-	Yiwen(Owen) H (my mentor)
-	And to the Udacity community of reviewers.

