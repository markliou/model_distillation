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
*. The optimizer using gradients to regulized learning rate would not work well (such as RMSProp, Adma, etc.). This would be the input is always noise which usually give no information. The learning rates will be unstable if the gradient is small. (This will let the learning rate become very huge) 

*. Batch size will influence the loss. Too small size will cause large perturbation. Most time, small batch size would give no information due to the input is noise. Large batch size won't give the subtle information of source model. The batch size set to 16 would be suitable.

*. Gumbel noise would be better than Gaussian and Laplace.

# License
MIT

