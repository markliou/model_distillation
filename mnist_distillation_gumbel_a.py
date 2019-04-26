import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import cv2

# Training Parameters
learning_rate = 1E-4
num_steps = 5000000
batch_size = 128

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

def generate_perlin_noise_2d(shape, res):
    def f(t):
       # return 6*t**5 - 15*t**4 + 10*t**3
        return 13*t**3 + 7*t**4 + 23*t**3 - 41*t**7
        #return np.random.gumbel(0,1)*t**2 + np.random.gumbel(0,1)*t**3 + np.random.gumbel(0,1)*t**7
        #return np.random.gumbel(0,.382,t.shape) 
    
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


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
        conv1 = tf.keras.layers.Conv2D(32, 5, activation=tf.nn.elu, kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l1(1e-12), activity_regularizer=tf.keras.regularizers.l2(1e-12))(x)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv1 = tf.keras.layers.SpatialDropout2D(rate=.1)(conv1, training=is_training)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.elu, kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l1(1e-12), activity_regularizer=tf.keras.regularizers.l2(1e-12))(conv1)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        conv2 = tf.keras.layers.SpatialDropout2D(rate=.2)(conv2, training=is_training)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.keras.layers.Dense(1024, kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l1(1e-16), activity_regularizer=tf.keras.regularizers.l2(1e-12))(fc1)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=.6, training=is_training)

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
logits_q = qconv_net(MNIST_dataset_fetch['imgs'], num_classes, 0, reuse=False, is_training=True)

logits_test = qconv_net(MNIST_imgs, num_classes, dropout=0, reuse=True, is_training=False)

# loss gate
f_gate = tf.pow(tf.clip_by_value((tf.reduce_max(tf.nn.softmax(logits_s), axis=-1) - .0), 0, 1)/1.0 , 1.)
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
#dis_s = tf.distributions.Categorical(logits=(logits_s * (1 + tf.random.normal(tf.shape(logits_s), 0, .618))))
dis_m = tf.distributions.Categorical(probs=(tf.nn.softmax(logits_q) + tf.nn.softmax(logits_s))/2)
#loss_op = tf.reduce_mean(
loss_op = tf.reduce_sum(
            # tf.pow((tf.nn.softmax(logits_s) - tf.nn.softmax(logits_q)) , 2)
            # tf.reduce_sum(tf.nn.softmax(logits_s + 1E-25) * tf.log(tf.nn.softmax(logits_s + 1E-25)/(tf.nn.softmax(logits_q) + 1E-25) + 1E-25), axis = -1) # KL-divergence give NaN error. this would be happened due to the float point computing
            tfp.distributions.kl_divergence(dis_s, dis_q) * f_gate # try to use the tensorflow probability module to get KL divergence
            
            # JS divergence https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
            #(tfp.distributions.kl_divergence(dis_s, dis_m) + tfp.distributions.kl_divergence(dis_q, dis_m))
            
            #tf.pow((logits_s - logits_q), 2) # MSE works well on logits, but softmax
          ) * (1/(f_gate_count + 1E-9))

#loss_op += tf.exp(tf.reduce_mean(tf.log(tf.nn.softmax(logits_q)+1e-9))) * 1e-2

# optimizer = tf.train.AdamOptimizer(learning_rate=1E-4)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=1E-6, decay=.9, momentum=.0)
optimizer = tf.contrib.opt.AdamWOptimizer(1E-4, learning_rate=1E-6)
#optimizer = tf.train.MomentumOptimizer(learning_rate=5E-6, momentum=.9)
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
sess.run(MNIST_dataset_iter.initializer, feed_dict={MNIST_imgs: np.random.laplace(0, 1., [batch_size * 1000, 784]),
                                                    MNIST_labels: np.random.gumbel(0, .5, [batch_size * 1000])}) # initialize tf.data module
training_step = 0
highest_acc = 0
cacc = 0
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
        #if acc > cacc :
        #    update_noise = False
        #cacc = acc

        print("Testing Accuracy:{} ({})".format(acc, highest_acc))

        # initialize new training data
        #freqx = np.random.randint(3,5)
        #freqy = np.random.randint(3,5)
        #freqn = np.random.randint(2,4)
        freqx, freqy, freqn = 2, 2, 14
        #noise_o = np.random.laplace(0, np.random.uniform() * 10., [batch_size * 1000, 784])
        noise_o = np.random.gumbel(0, .2, [batch_size * 1000, 784])
        #noise_o = np.vstack([cv2.resize(generate_perlin_noise_2d([freqx * freqn, freqy * freqn], [(freqx**np.random.randint(0,2)) * (freqn**np.random.randint(0,2)), (freqy**np.random.randint(0,2)) * (freqn**np.random.randint(0,2))]), dsize=(28, 28), interpolation=cv2.INTER_CUBIC) for x in range(batch_size * 1000)])
        #noise_o = np.vstack([cv2.resize(generate_perlin_noise_2d([freqx * freqn, freqy * freqn], [(freqn),(freqn)]), dsize=(28, 28), interpolation=cv2.INTER_CUBIC) for x in range(batch_size * 1000)])
        #noise_o = cv2.resize(noise_o, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
        noise_o = np.reshape(noise_o, [-1, 784])
        #noise_o = np.clip((noise_o - .5)/.5, -1., 1.) 
        #if update_noise or (training_step == 1000):
        if True:
            sess.run(MNIST_dataset_iter.initializer, feed_dict={MNIST_imgs: np.abs(noise_o) ,
                                                                MNIST_labels: np.random.gumbel(0, 1., [batch_size * 1000])}) # initialize tf.data module
            noise_m = noise_o
        else:
            sess.run(MNIST_dataset_iter.initializer, feed_dict={MNIST_imgs: noise_m ,
                                                                MNIST_labels: np.random.gumbel(0, 1., [batch_size * 500])})
        
        #acc = sess.run(acc_op, feed_dict={MNIST_imgs: np.array(mnist.test.images),
        #                                  MNIST_labels: np.array(mnist.test.labels)})


sess.close()



