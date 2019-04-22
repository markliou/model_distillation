fashing mnist distillation
===

This experiment used the fashin mnist to create a model (source) and then created a new model using only noise which will feed into the finely training model to get the output results.
Then the new created model is optimized using the result from finely training model.

# description

1. mnist.py
This script will train a simple fashin mnist model. The model is named "Source".

2. mnist_distillation.py
This script will train a CNN model using random generated noise, and be approxymated according to the result from Source model.


# notes
* The optimizer using gradients to regulized learning rate would not work well (such as RMSProp, Adma, etc.). This would be the input is always noise which usually give no information. The learning rates will be unstable if the gradient is small. (This will let the learning rate become very huge.) **The SGD with momentum is quite ideable optimizer for this task**. 

* Batch size will influence the loss. Too small size will cause large perturbation. Most time, small batch size would give no information due to the input is noise. Large batch size won't give the subtle information of source model. **Since most of time, the input only give noise, large size would keep the model crash due to learn the noise (This work use 128)**.

* The loss is from teach net. These answers are easy for student net to learn. This would cause the student overfitting. **Using dropout on student would help to keep away from this bad situation**.

* Perlin noise might be better than Laplace, Gaussian and Gumbel.

# references
1. https://arxiv.org/pdf/1711.01768.pdf
1. https://arxiv.org/abs/1503.02531


# License
MIT

