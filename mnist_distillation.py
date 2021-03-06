import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import cv2

# Training Parameters
learning_rate = 1E-4
num_steps = 5000000
batch_size = 64

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit


# Create the neural network
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out

# Create the neural network
def qconv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('qConvNet', reuse=reuse):

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.keras.layers.Conv2D(32, 5, activation=tf.nn.elu, kernel_initializer='glorot_normal')(x)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.elu, kernel_initializer='glorot_normal')(conv1)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.keras.layers.Dense(1024, kernel_initializer='glorot_normal')(fc1)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# entry point
MNIST_imgs = tf.placeholder(tf.float32, [None, 28 * 28])
MNIST_labels = tf.placeholder(tf.float32, [None])

# use TF.dataset to handle minst
MNIST_dataset = tf.data.Dataset.from_tensor_slices({'imgs':MNIST_imgs, 'labs':MNIST_labels})
MNIST_dataset = MNIST_dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
MNIST_dataset = MNIST_dataset.prefetch(buffer_size=100) # prefech
MNIST_dataset = MNIST_dataset.batch(batch_size)
MNIST_dataset_iter = MNIST_dataset.make_initializable_iterator()
MNIST_dataset_fetch = MNIST_dataset_iter.get_next()

logits_s =  conv_net(MNIST_dataset_fetch['imgs'], num_classes, 0, reuse=False, is_training=False)
logits_q = qconv_net(MNIST_dataset_fetch['imgs'], num_classes, 0, reuse=False, is_training=False)

logits_test = qconv_net(MNIST_imgs, num_classes, dropout=0, reuse=True, is_training=False)

# loss gate
f_gate = tf.clip_by_value((tf.reduce_max(tf.nn.softmax(logits_s), axis=-1) - .99),0,1) * 10
f_gate_count = tf.reduce_sum(tf.cast(tf.greater(f_gate,0), tf.float32))

# Predictions
pred_classes = tf.cast(tf.argmax(logits_test, axis=1), tf.float32)
pred_probas = tf.nn.softmax(logits_test)
acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(MNIST_labels,tf.shape(pred_classes)), pred_classes), tf.float32)) # accuracy

# loss 
# the performance comparing: logis MSE > softmax MSE >> JS >> KL
q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qConvNet')
dis_q = tf.distributions.Categorical(logits=logits_q)
dis_s = tf.distributions.Categorical(logits=logits_s)
dis_m = tf.distributions.Categorical(probs=(tf.nn.softmax(logits_q) + tf.nn.softmax(logits_s))/2)
#loss_op = tf.reduce_mean(
loss_op = tf.reduce_sum(
            # tf.pow((tf.nn.softmax(logits_s) - tf.nn.softmax(logits_q)) , 2)
            # tf.reduce_sum(tf.nn.softmax(logits_s + 1E-25) * tf.log(tf.nn.softmax(logits_s + 1E-25)/(tf.nn.softmax(logits_q) + 1E-25) + 1E-25), axis = -1) # KL-divergence give NaN error. this would be happened due to the float point computing
            tfp.distributions.kl_divergence(dis_s, dis_q) * f_gate # try to use the tensorflow probability module to get KL divergence
            
            # JS divergence https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
            # (tfp.distributions.kl_divergence(dis_s, dis_m) + tfp.distributions.kl_divergence(dis_q, dis_m))/2.
            
            #  tf.pow((logits_s - logits_q), 2) # MSE works well on logits, but softmax
          ) * (1/(f_gate_count + 1E-9))
# optimizer = tf.train.AdamOptimizer(learning_rate=1E-4)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1E-7, decay=.9, momentum=.0)
#optimizer = tf.contrib.opt.AdamWOptimizer(1E-4, learning_rate=1E-6)
#optimizer = tf.train.MomentumOptimizer(learning_rate=1E-4, momentum=.8)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate=1E-4)
# optimizer = tf.train.GradientDescentOptimizer(1E-4)
train_op = optimizer.minimize(loss_op, var_list=q_vars, global_step=tf.train.get_global_step())

# setting the device parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
s_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ConvNet')
saver = tf.train.Saver(allow_empty=True, var_list=s_vars)
sess.graph.finalize() 

# loading the weights
saver.restore(sess, 'source')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=False)

# training
sess.run(MNIST_dataset_iter.initializer, feed_dict={MNIST_imgs: np.random.gumbel(0, 1.5, [batch_size * 1000, 784]),
                                                    MNIST_labels: np.random.gumbel(0, .5, [batch_size * 1000])}) # initialize tf.data module
training_step = 0
highest_acc = 0
while(1):
    training_step += 1
    closs, _ = sess.run([loss_op, train_op])
    
    if training_step % 1000 == 0:
        update_noise = True
        print('step:{} loss:{}  '.format(training_step, closs), end='')
        
        # test
        acc = sess.run(acc_op, feed_dict={MNIST_imgs: np.array(mnist.test.images),
                                          MNIST_labels: np.array(mnist.test.labels)})
        if acc > highest_acc:
            highest_acc = acc
            update_noise = False

        print("Testing Accuracy:{} ({})".format(acc, highest_acc))

        # initialize new training data
        noise_o = np.random.gumbel(0. , 1., [batch_size * 100, 28 * 28])
        for noise_it in range(1):
            #mix_rate = np.random.random()
            mix_rate = .5
            # perlin-noise like noise http://physbam.stanford.edu/cs448x/old/Procedural_Noise(2f)Perlin_Noise.html 
            freqx = np.random.randint(3,6)
            freqy = np.random.randint(4,8)
            noise = np.random.gumbel(0. , .5, [batch_size * 100, freqx * freqy])
            noise_o = np.reshape(noise_o, [batch_size * 100, 28 * 28])
            noise_o = np.vstack([ np.array(x[:freqx * freqy]) for x in noise_o ])
            #noise = np.random.gumbel(0, 1., [500, 784])
            #noise = np.vstack([np.random.random([batch_size * 500, 784]), noise])
            noise = (np.random.gumbel(.25 , .25, [batch_size * 100, freqx * freqy]) * (1 - mix_rate) + noise * mix_rate )
            noise = (np.random.gumbel(.5 , .1, [batch_size * 100, freqx * freqy]) * (1 - mix_rate) + noise * mix_rate )
            #noise = (np.random.normal(.0 , 1., [batch_size * 500, 20 * 20]) * (1 - mix_rate) + noise * mix_rate )
            noise = (np.random.normal(0 , 1., [batch_size * 100, freqx * freqy]) * (1 - mix_rate) + noise * mix_rate )
            #noise = np.random.laplace(0, 1, [batch_size * 100, 784])
            noise_o = (noise_o + noise) * .5
            if np.random.random() > .5:
                noise_o = np.reshape(noise_o, [-1, freqx, freqy])
            else:
                noise_o = np.reshape(noise_o, [-1, freqy, freqx])
            noise_o = np.array([cv2.resize(i, dsize=(28,28),interpolation=cv2.INTER_CUBIC) for i in noise_o])
            noise_o = np.clip(noise_o, 0., 1.)
        noise_o = np.reshape(noise_o, [-1, 784])

        if update_noise or (training_step == 1000):
            sess.run(MNIST_dataset_iter.initializer, feed_dict={MNIST_imgs: noise_o ,
                                                                MNIST_labels: np.random.gumbel(0, 1., [batch_size * 100])}) # initialize tf.data module
            noise_m = noise_o
        else:
            sess.run(MNIST_dataset_iter.initializer, feed_dict={MNIST_imgs: noise_m ,
                                                                MNIST_labels: np.random.gumbel(0, 1., [batch_size * 100])})
        
        #acc = sess.run(acc_op, feed_dict={MNIST_imgs: np.array(mnist.test.images),
        #                                  MNIST_labels: np.array(mnist.test.labels)})


sess.close()



