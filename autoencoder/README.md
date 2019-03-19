Model_distillation
==
This small toy project aims to extract the information from a known model without any training information about the model (e.g. lacking the original training datasets).  
Three main steps will be applied:  

1. creating a new autoencoder using MNIST. After this step, a known model (a MNIST autoencoder) will be created.
2. feed a noise into the known model to encode the results.
3. training a new model using the nosie and its coorespong encoding results from the known model.

## The purpose for each scripts
Thie graph will describe the scripts used in this toy ground.

1. autoencoder.py  
This script will create a new autoencoder model by using MNIST.
2. info_extractor.py  
This will create noise and feed it into the known model which is generated from autoencoder.py. And then, the encode results will be stored. These results will be called as "noise-encode pair" below.
3. distillator.py  
The noise-encode pairs are used to train another new model.