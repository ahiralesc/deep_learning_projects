## Bike sharing classification

Welcome, I built this notebook while studying the AI nanodegree from Udacity. This notebook addresses a supervised learning problem, namely the bike sharing prediction problem. A simple Feed Forward Neural Network is was built. The main learning outcome is to evaluate the effects of hyperparameter selection on the quality of the trained model. This problem is also referred to as *model assesment and selection*, see Hastie et al. (2009).

In this notebook, model selection is driven the the behaviour of the validation and training sample error. The model complexity is maintain fixed. The effect of the learning rate and its relation to the training dataset size is analyzed. The analysis also analyses the relation between the batch size and the neural network hiddent layer size. Stochastic gradient descent is applied over a normalized dataset.

All the code was built with [Python](https://www.python.org/) and normalization and preprocessing was done with [Pandas](https://pandas.pydata.org/). Some plots were produced with [bokenh](https://docs.bokeh.org/en/latest/). Models were trained using a CPU equipped architecture. 
I want to express my appreciation to:
-	Yiwen(Owen) H (my mentor)
-	And to the Udacity community of reviewers.
