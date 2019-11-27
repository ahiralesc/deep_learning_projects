## Dog breed classification

Welcome, I built this notebook while studying the AI nanodegree from Udacity. This notebook addresses a supervised learning problem, namely the TV script generation problem. One RNN is trained using the Seinfeld TV scripts. The experimental model is based on Mikolovs (2011, 2012) recommendations for training large scale neural network language models. The experimental model hyper-parameters and argumentation are the following: “studies have shown that epochs of 10-50 are sufficient, all do for some cases seven are enough”. I chose 10 epochs; it is recommended that the vocabulary size is reduced using Goodman’s trick for speeding up maximum entropy models. Such, assigs non-domain specific words to a class. This optimization criterion was not applied; it has been shown that good performance is maintained using a hidden layer size of 100 units. This size is used as well. On average English sentence average length is of 15 to 20 words. Since the workload correspond to text dialogs of a zipcom, sentences tend to be shorter. Thus I used a sentence length of 10 words. Language models with embedding size of 128 have been show to capture semantical relations between words (Mikolov,2012) I used the same embedding size. Currently the RNN is not very accurate as it achieves a traing loss of 4.525680099010468. 

All the code was built with [Pytorch]( https://pytorch.org/). Models were trained using a GPU equipped architecture. Amazon web services provide EC2 GPU instances, for a cost.   

I want to express my appreciation to:
-	Yiwen(Owen) H (my mentor)
-	And to the Udacity community of reviewers.

