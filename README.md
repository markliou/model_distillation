fashing mnist distillation
===

This experiment used the fashin mnist to create a model (source) and then created a new model using only noise which will feed into the finely training model to get the output results.
Then the new created model is optimized using the result from finely training model.

# description

1. mnist.py
This script will train a simple fashin mnist model. The model is named "Source".

2. mnist_distillation.py
This script will train a CNN model using random generated noise, and be approxymated according to the result from Source model.


# License
MIT

