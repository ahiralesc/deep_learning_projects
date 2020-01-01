## TV script generation using RNN

Welcome, I built this notebook while studying the AI nanodegree from Udacity. This notebook addresses a supervised learning problem, namely the TV script generation problem. One RNN is trained using the Seinfeld TV scripts. The experimental model is based on Mikolovs (2011, 2012) recommendations for training large scale neural network language models. The experimental model hyper-parameters and argumentation are the following: “studies have shown that epochs of 10-50 are sufficient, all do for some cases seven are enough”. I chose 10 epochs; it is recommended that the vocabulary size is reduced using Goodman’s trick for speeding up maximum entropy models. Such, assigs non-domain specific words to a class. This optimization criterion was not applied; it has been shown that good performance is maintained using a hidden layer size of 100 units. This size is used as well. On average English sentence average length is of 15 to 20 words. Since the workload correspond to text dialogs of a zipcom, sentences tend to be shorter. Thus I used a sentence length of 10 words. Language models with embedding size of 128 have been show to capture semantical relations between words (Mikolov,2012) I used the same embedding size. Currently the RNN is not very accurate as it achieves a traing loss of 4.525680099010468. 

The model was retrained with the following hyper-parameters:

|Batch size|Embedding dimension|Learning rate|Loss|Example output |
|:----:|:----:|:----:|:----:|:----|
| 100 | 128 | 0.01 | 4.52| jerry: what about me?<br>jerry: i don't have to wait.<br>kramer:(to the sales table)<br>elaine:(to jerry) hey, look at this, i'm a good doctor.<br>newman:(to elaine) you think i have no idea of this...<br>elaine: oh, you better take the phone, and he was a little nervous.<br>kramer:(to the phone) hey, hey, jerry, i don't want to be a little bit.(to kramer and jerry) you can't.<br>jerry: oh, yeah. i don't even know, i know.<br>jerry:(to the phone) oh, i know.<br>kramer:(laughing) you know...(to jerry) you don't know.
| 256 | 200 | 0.001 | 4.05|jerry: i know, i think i have it, you know, i know, you don't know, i can't go.

jerry:(still to the phone) hey!

george: yeah. i don't know. i know, i can't have it.

george:(to the table) hey, you want to do?

kramer: i don't know, it's a big.

elaine:(to the phone) well, i don't know, but i know what you have.

jerry:(to george) yeah.

george: oh, well, i don't know, i'm not not gonna have to go to the way.

elaine: i know, i'm not gonna be the whole thing, i'm going to be the best. i don't know...(hangs up) i know, i'm not going out with a lot of one.

kramer: yeah.

kramer:(smiling) what happened to you?

kramer: well i think i have no idea.

jerry: i don't want to get up. i know. i know.

jerry:(to jerry) what is it?

kramer: yeah.

jerry:(to george) oh!

george:(to the bathroom) yeah.

jerry: well, you don't have to be a big....... i know what i have to do.(jerry leaves,)

jerry: well, i can't go.

george:(to elaine) i think i have a good thing.

george:(to the phone) yeah.

george: i don't know, i don't know, you know.

kramer: yeah. i know, i know i don't even have to get a few.

jerry: oh, i'm not going to go.

jerry: i know, i'm not gonna do the car, you know what i mean. i know you don't know.

george:(looking to george) i got this.|
| 512 | 200 | 0.001 | 0.0 | |


All the code was built with [Pytorch]( https://pytorch.org/). Models were trained using a GPU equipped architecture. Amazon web services provide EC2 GPU instances, for a cost.   

I want to express my appreciation to:
-	Yiwen(Owen) H (my mentor)
-	And to the Udacity community of reviewers.

