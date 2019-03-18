""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 28, 28, 1])

# Building the encoder
def encoder(x):
    layer_1 = tf.layers.conv2d(x, 32, 5, strides = 2, padding = "SAME", activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.glorot_normal, name="en1") #14
    layer_2 = tf.layers.conv2d(layer_1, 32, 5, strides = 2, padding = "SAME", activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.glorot_normal, name="en2") #7
    layer_3 = tf.layers.conv2d(layer_2, 32, 5, strides = 2, padding = "SAME", activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.glorot_normal, name="en3") #4
    layer_4 = tf.layers.conv2d(layer_3, 32, 5, strides = 2, padding = "SAME", activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.glorot_normal, name="en4") #2
    
    return layer_4


# Building the decoder
def decoder(x):
    layer_1 = tf.layers.conv2d_transpose(x, 32, 5, strides = 2, padding = "VALID", activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.glorot_normal, name="de1") #7
    layer_2 = tf.layers.conv2d_transpose(layer_1, 32, 5, strides = 2, padding = "SAME", activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.glorot_normal, name="de2") #14
    layer_3 = tf.layers.conv2d_transpose(layer_2, 32, 5, strides = 2, padding = "SAME", activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.glorot_normal, name="de3") #28
    
    return layer_3

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: np.reshape(batch_x,[batch_size, 28, 28, 1])})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    save_path = saver.save(sess, "ae")

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: np.reshape(batch_x,[n, 28, 28, 1])})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
