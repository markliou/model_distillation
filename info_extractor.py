import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 
import PIL

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
GENERATING_NO = 10

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

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # Run the initializer
    saver.restore(sess,'ae')


