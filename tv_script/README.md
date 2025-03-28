## TV script generation using RNN

Welcome, I built this notebook while studying the AI nanodegree from Udacity. This notebook addresses a supervised learning problem, namely the TV script generation problem. One RNN is trained using the Seinfeld TV scripts. The experimental model is based on Mikolovs (2011, 2012) recommendations for training large scale neural network language models. The experimental model hyper-parameters and argumentation are the following: “studies have shown that epochs of 10-50 are sufficient, all do for some cases seven are enough”. I chose 10 epochs; it is recommended that the vocabulary size is reduced using Goodman’s trick for speeding up maximum entropy models. Such, assigs non-domain specific words to a class. This optimization criterion was not applied; it has been shown that good performance is maintained using a hidden layer size of 100 units. This size is used as well. On average English sentence average length is of 15 to 20 words. Since the workload correspond to text dialogs of a zipcom, sentences tend to be shorter. Thus I used a sentence length of 10 words. Language models with embedding size of 128 have been show to capture semantical relations between words (Mikolov,2012) I used the same embedding size. Currently the RNN is not very accurate as it achieves a traing loss of 4.525680099010468. 

The model was retrained with the following hyper-parameters:

|Batch size|Embedding dimension|Learning rate|Loss|Example output |
|:----:|:----:|:----:|:----:|:----|
| 100 | 128 | 0.01 | 4.52| |
| 256 | 200 | 0.001 | 4.05| jerry: i know, i think i have it, you know, i know, you don't know, i can't go.<br>jerry:(still to the phone) hey!<br>george: yeah. i don't know. i know, i can't have it.<br>george:(to the table) hey, you want to do?<br>kramer: i don't know, it's a big.<br>elaine:(to the phone) well, i don't know, but i know what you have.<br>jerry:(to george) yeah.<br>george: oh, well, i don't know, i'm not not gonna have to go to the way.<br>elaine: i know, i'm not gonna be the whole thing, i'm going to be the best. i don't know...(hangs up) i know, i'm not going out with a lot of one.<br>kramer: yeah.<br>kramer:(smiling) what happened to you?<br>kramer: well i think i have no idea.<br>jerry: i don't want to get up. i know. i know.<br>jerry:(to jerry) what is it?<br>kramer: yeah.<br>jerry:(to george) oh!<br>george:(to the bathroom) yeah.<br>jerry: well, you don't have to be a big....... i know what i have to do.(jerry leaves,)<br>jerry: well, i can't go.<br>george:(to elaine) i think i have a good thing.<br>george:(to the phone) yeah.<br>george: i don't know, i don't know, you know.<br>kramer: yeah. i know, i know i don't even have to get a few.<br>jerry: oh, i'm not going to go.<br>jerry: i know, i'm not gonna do the car, you know what i mean. i know you don't know.<br>george:(looking to george) i got this.|
| 512 | 200 | 0.001 | 4.03 |jerry:.<br>jerry:(to kramer) hey.<br>george:(pause) hey, i don't understand.<br>jerry: well, you know what, i don't know how you have to get a big little.<br>elaine: i think you can be in the car, the other thing is that.<br>elaine: well, i don't have to go in a way. i don't think i don't want a little thing.<br>jerry: oh, what? i can't believe i have to be...<br>jerry: well...<br>kramer: oh, i know, you want to see it.<br>jerry: oh, no, i got a lot of a little little, i can't go to the apartment.<br>elaine:(to kramer) what is the name?<br>george: well, i can't get the money.<br>elaine:(to jerry) what are you doing?<br>jerry: i can't believe you were gonna do you?<br>jerry: oh, no, no, i don't want to get the whole time. you know what i was in the bathroom!<br>jerry: oh, i think i don't want me to be a lot of the thing, and you know, you want me to have to go.<br>jerry: i can't believe i think i don't want a good guy.(to elaine) oh, what?<br>george: yeah.<br>george:(pointing) what is that?!<br>jerry: oh, i can't go.<br>jerry: i mean, i think i was a good time.<br>elaine: yeah, i'm not gonna be able to get up.<br>jerry: yeah.<br>elaine:(to the waitress, jerry) oh, you know you know....<br>george:(to kramer) i know, i don't know...<br>george: oh no, no, you got a big thing.<br>george:(looking at the door) : you think i mean, i know, |
| 512 | 200 | 0.001 |3.31|jerry:" the"""" suzanne,".<br>george:(to jerry) hey, you know what, are you sure? what is it?<br>kramer: well, i was thinking about it.<br>george: well, what did i say?<br>elaine: well, i'm not sure.<br>george: what?<br>elaine: oh, i don't understand.<br>elaine: well, what do i do?<br>jerry: i can't do it.<br>jerry: well, i was just curious, i just got a little bit of a few things.<br>george: i thought you had to be a good person.<br>george: what do you mean?(jerry nods.) what are you talking about?<br>kramer: well, it's a good idea.<br>george: i don't want to see it again.<br>elaine: oh, yeah, but.(picks up his coat)<br>kramer: hey, hey!(to kramer) so, what's the matter?<br>elaine: oh, i was going to see the muted.<br>elaine:(still laughing) oh, yeah.<br>elaine:(pointing at kramer) you know, i gotta go to the airport, i know, i was just trying to see her naked.<br>george: oh, you see, i know.<br>george: i don't know what this means.<br>george: well... i was in the middle of the month, it's like a while. i mean, you got any idea that we got a lot of yapping to him... and, it's not really bad..<br>kramer: yeah! well, i was just wondering....(jerry puts her head down.)<br>elaine: what?<br>elaine: yeah!(she exits)<br>jerry: oh, hi.<br>elaine: hey!<br>jerry: hey! hey!<br>jerry: hey.<br>jerry: what?<br>kramer:(to kramer) i know, i think|

The hyperparameters that produced the smallest error were the following

| Hyperparameter | Value |
|:-------------- |:-----:|
| Sequence length | 15 |
| Batch size | 512 |
| Number of Epochs | 12 |
| Learning rate | 0.001 |
| Vocabulary size | \| V \| |
| Output size | \| V \| |
| Embedding size | 300 |
| Hidden layer dimension | 256 |
| Number of RNN layers | 2 |

With \| V \| is the size of the vocabulary.


All the code was built with [Pytorch]( https://pytorch.org/). Models were trained using a GPU equipped architecture. Amazon web services provide EC2 GPU instances, for a cost.   

I want to express my appreciation to:
-	Yiwen(Owen) H (my mentor)
-	And to the Udacity community of reviewers.

