# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import os
import random

import tensorflow as tf
import numpy as np
from scipy import misc
import glob
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split

bs = 32
epochs = 20
image_mode = "L"
saved_model = "1layer_mlp_2objects_RGB.ckpt"
RANDOM_SEED = 42
train_test_ratio = 0.8
input_shape = [250, 250, 1]

random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

image_folder = os.path.join("./images/2objects/")

def init_weights(shape, name):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name=name)

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_data():
    """ Read the data set and split them into training and test sets """
    X = []
    Label = []
    fns = []
    for image_path in glob.glob(os.path.join(image_folder, "*.png")):
        fns.append(os.path.basename(image_path))
        Label.append(int(os.path.basename(image_path).split("_")[0]))
        image = X.append(misc.imread(image_path, mode=image_mode).flatten())
    X = np.array(X) / 255
    Label = np.array(Label)
    print X.shape
    print Label.shape

    num_exps, img_dim = X.shape
    all_X = X

    # Convert into one-hot vectors
    num_labels = len(np.unique(Label))

    all_Y = np.eye(num_labels)[Label]  # One liner trick!

    all_index = range(X.shape[0])
    random.shuffle(all_index)

    all_X = all_X[all_index]
    all_Y = all_Y[all_index]

    index_cutoff = int(X.shape[0] * train_test_ratio)

    return all_X[0:index_cutoff, :], all_X[index_cutoff:, :], \
           all_Y[0:index_cutoff, :], all_Y[index_cutoff:, :], \
           fns[0:index_cutoff], fns[index_cutoff:], \

def main():
    train_dir = './results/'

    # load the data
    train_X, test_X, train_y, test_y, train_fn, test_fn = get_data()

    # Layer's sizes
    # x_size = train_X.shape[1] # Number of input nodes
    # y_size = train_y.shape[1]  # Number of outcomes

    sess = tf.InteractiveSession()

    with tf.device("/gpu:0"):
        x = tf.placeholder(tf.float32, shape=[None, 250*250])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

        # reshape the input image
        # x_image = tf.reshape(x, [-1, 250, 250, 1])
        # first layer
        # W_conv1 = weight_variable([250*250, 2])
        # b_conv1 = bias_variable([2])

        # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # h_pool1 = max_pool_2x2(h_conv1)

        # h_pool1_flat = tf.reshape(h_conv1, [-1, 246 * 246 * 32])
        # h_pool1_flat = tf.matmul(x, W_conv1) + b_conv1

        # dropout
        # keep_prob = tf.placeholder(tf.float32)
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # softmax
        W_fc2 = weight_variable([250*250, 2])
        b_fc2 = bias_variable([2])

        y_conv = tf.nn.softmax(tf.matmul(x, W_fc2) + b_fc2)

        # setup training
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Add a scalar summary for the snapshot loss.
        # tf.summary.scalar(cross_entropy.op.name, cross_entropy)
        # Build the summary operation based on the TF collection of Summaries.
        # summary_op = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    # summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)

    # run the training
    for epoch in range(epochs):
        for i in range(len(train_X) / bs):
            if i % 1 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: train_X[bs * i: bs * i + bs], y_: train_y[bs * i: bs * i + bs]})
                print("step %d, training accuracy %g" % (epoch*len(train_X) / bs + i, train_accuracy))

                # Update the events file.
                # summary_str = sess.run(summary_op, feed_dict={x: train_X[bs * i: bs * i + bs], y_: train_y[bs * i: bs * i + bs]})
                # summary_writer.add_summary(summary_str, i)
                # summary_writer.flush()

            if i % 1100 == 0:
                checkpoint_file = os.path.join(train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=i)

            train_step.run(feed_dict={x: train_X[bs * i: bs * i + bs], y_: train_y[bs * i: bs * i + bs]})

        print("test accuracy at epoach %d: %g" % (epoch, accuracy.eval(feed_dict={x: test_X, y_: test_y})))

    # print test error
    print("Final test accuracy %g" % accuracy.eval(feed_dict={x: test_X, y_: test_y}))

if __name__ == '__main__':
    main()
