import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os

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

# create noise gen
def noise_gen(x, reuse=False):
    with tf.variable_scope('noise_gen', reuse=reuse):
        fc1 = tf.keras.layers.Dense(1024, activation=tf.nn.tanh, kernel_initializer='random_uniform')(x)
        fc1 = tf.keras.layers.Dense(1024, activation=tf.nn.tanh, kernel_initializer='random_uniform')(fc1)
        fc1 = tf.keras.layers.Dense(1024, activation=tf.nn.tanh, kernel_initializer='random_uniform')(fc1)
        fc1 = tf.layers.dropout(fc1, rate=.5, training=True)
        out = tf.keras.layers.Dense(784, kernel_initializer='random_uniform')(fc1)
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

# noise gen
stimulate_tags = tf.random.uniform([batch_size,1])
stimulate_noise = noise_gen(tf.concat([MNIST_dataset_fetch['imgs'], stimulate_tags], axis=-1))

logits_s =  conv_net(stimulate_noise, num_classes, 0, reuse=False, is_training=False)
logits_q = qconv_net(stimulate_noise, num_classes, 0, reuse=False, is_training=True)

logits_test = qconv_net(MNIST_imgs, num_classes, dropout=0, reuse=True, is_training=False)

# Predictions
s_pred_classes = tf.cast(tf.reshape(stimulate_tags * 10, [-1]),dtype=tf.int32)
pred_classes = tf.cast(tf.argmax(logits_test, axis=1), tf.float32)
pred_probas = tf.nn.softmax(logits_test)
acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(MNIST_labels,tf.shape(pred_classes)), pred_classes), tf.float32)) # accuracy

# loss for noise gen
noise_gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='noise_gen')
noise_gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=s_pred_classes, logits=logits_s))
noise_opt = tf.train.RMSPropOptimizer(learning_rate=1E-3, decay=.9, momentum=.0)
#noise_opt = tf.contrib.opt.AdamWOptimizer(1E-5, learning_rate=learning_rate)
noise_train_op = noise_opt.minimize(noise_gen_loss, var_list=noise_gen_var)

# loss for q net
# the performance comparing: logis MSE > softmax MSE >> JS >> KL
q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qConvNet')
dis_q = tf.distributions.Categorical(logits=logits_q)
dis_s = tf.distributions.Categorical(logits=logits_s)
dis_m = tf.distributions.Categorical(probs=(tf.nn.softmax(logits_q) + tf.nn.softmax(logits_s))/2)
loss_op = tf.reduce_mean(
            # tf.pow((tf.nn.softmax(logits_s) - tf.nn.softmax(logits_q)) , 2)
            # tf.reduce_sum(tf.nn.softmax(logits_s + 1E-25) * tf.log(tf.nn.softmax(logits_s + 1E-25)/(tf.nn.softmax(logits_q) + 1E-25) + 1E-25), axis = -1) # KL-divergence give NaN error. this would be happened due to the float point computing
            tfp.distributions.kl_divergence(dis_s, dis_q) # try to use the tensorflow probability module to get KL divergence
            
            # JS divergence https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
            # (tfp.distributions.kl_divergence(dis_s, dis_m) + tfp.distributions.kl_divergence(dis_q, dis_m))/2.
            
            #  tf.pow((logits_s - logits_q), 2) # MSE works well on logits, but softmax
          )
# optimizer = tf.train.AdamOptimizer(learning_rate=1E-4)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1E-4, decay=.9, momentum=.0)
# optimizer = tf.contrib.opt.AdamWOptimizer(1E-4, learning_rate=learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate=1E-3, momentum=.8)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate=1E-4)
# optimizer = tf.train.GradientDescentOptimizer(1E-4)
train_op = optimizer.minimize(loss_op, var_list=q_vars, global_step=tf.train.get_global_step())

# setting the device parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
sess.run(MNIST_dataset_iter.initializer, feed_dict={MNIST_imgs: np.random.gumbel(0, .5, [batch_size * 1000, 784]),
                                                    MNIST_labels: np.random.gumbel(0, .5, [batch_size * 1000])}) # initialize tf.data module
training_step = 0
while(1):
    training_step += 1
    closs, _, _, noise_class, n_loss = sess.run([loss_op, train_op, noise_train_op, s_pred_classes, noise_gen_loss])
    if training_step % 1000 == 0:
        print('step:{} loss:{} nloss:{} '.format(training_step, closs, n_loss), end='')
        
        # test
        acc = sess.run(acc_op, feed_dict={MNIST_imgs: np.array(mnist.test.images),
                                          MNIST_labels: np.array(mnist.test.labels)})
        print("Testing Accuracy:",acc)
        print('class:{}'.format(noise_class))

        # initialize new training data
        #noise = np.random.gumbel(0, np.random.random() * 5, [batch_size * 500, 784])
        #noise = np.vstack([np.random.random([batch_size * 500, 784]), noise])
        noise = np.random.gumbel(0, 1.5, [5000, 784])
        #noise = (np.random.random([batch_size * 1000, 784]) + noise) * .5
        #noise = np.random.random([batch_size * 1000, 784])
        #noise = np.random.laplace(0, 1, [batch_size * 100, 784])
        sess.run(MNIST_dataset_iter.initializer, feed_dict={MNIST_imgs: noise ,
                                                            MNIST_labels: np.random.gumbel(0, 1., [5000])}) # initialize tf.data module
                                                            

sess.close()



