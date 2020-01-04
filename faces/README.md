## Deep Convolutional Generative Adversarial Network (DCGAN)

Welcome, I built this notebook while studying the AI nanodegree from Udacity. This notebook addresses training of an Deep Convolutional Generative Adversarial Network (DCGAN) using the [CelebFaces Attributes Dataset ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (CelebA). Image sizes are 64x64x3 RGB. The architecture that produced best results is summarized in Table I. 

<center> Table I. DCGAN architecture .</center> 

|Layer | Generator | Discriminator |
|:-------|:---------------|:---------------|
|1|Linear |Conv2d->BatchNorm2d | 
|2| Dropout | Conv2d->BatchNorm2d |
|3| ConvTranspose2d -> BatchNorm2d | Conv2d->BatchNorm2d |
|4| ConvTranspose2d -> BatchNorm2d | Linear|
|5| ConvTranspose2d -> BatchNorm2d | Dropout|

See fine grain details in the notebook. Hyper-parameter tuning was challenging. At first experiments took up to eight hours. The reviewer comets helped increased the quality of results. Leaky ReLU alleviates the problem of sparse gradients. Custom weight initialization accelerate the learning process. Dropout helped balanced dominance between the generator and discriminator. Final hyper-parameters are summarized in Table II.

<center> Table I. DCGAN hyper-parameters.</center> 

Hyperparameter tuning was challenging.

|Layer | Generator | Discriminator |
|:-------|:---------------|:---------------|
|Batch size|64| 64 | 
|Dropout| 0.5 | 0.5 |
|Convolution dimension| 32 | 32 |
|Linear layer size| 100 | 100 |
|Betas| (0.1, 0.999) | (0.1, 0.999) |
|Learning rate | 0.0003 | 0.0003 |
|Number of epochs |  30 |  30 |

An error of 1.14 and 2.53 was achieved for the discriminator and generator correspondingly. The generated images are shown next.

![Fig 1. First experiment learning error](img/exp4.png)

Previous networks did not include weight initialization, dropout, and use instance initialization instead batch normalization. I also used batch size of 20, 32, and 16. Finally, both the generator and discriminator included one more convolutional layer.
)



All the code was built with [Pytorch]( https://pytorch.org/). Models were trained using a GPU equipped architecture. Amazon web services provide EC2 GPU instances, for a cost.   

I want to express my appreciation to:
-	Yiwen(Owen) H (my mentor)
-	And to the Udacity community of reviewers.

