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
epochs = 5
image_mode = "L"
saved_model = "1layer_conv_2objects_RGB.ckpt"
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
    initial = tf.random_normal(shape, stddev=0.001)
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
    x_size = train_X.shape[1] # Number of input nodes
    y_size = train_y.shape[1]  # Number of outcomes

    with tf.device("/gpu:0"):
        X = tf.placeholder(tf.float32, shape=[None, 250*250])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

        # softmax
        W_fc2 = weight_variable([250*250, 2])
        b_fc2 = bias_variable([2])

        # y_conv = tf.nn.softmax(tf.matmul(x, W_fc2) + b_fc2)
        yhat = tf.matmul(X, W_fc2) + b_fc2
        predict = tf.argmax(yhat, axis=1)

        # setup training
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=yhat))
        updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    saver = tf.train.Saver()

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # sess.run(Op_record_init)
    for epoch in range(epochs):
        # Train with each example
        for i in range(int(len(train_X) / bs)):
            sess.run(updates, feed_dict={X: train_X[bs * i: bs * i + bs], y: train_y[bs * i: bs * i + bs]})
            train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                     sess.run(predict, feed_dict={X: train_X, y: train_y}))
            test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: test_X, y: test_y}))

            print("Epoch = %d, batch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, i + 1, 100. * train_accuracy, 100. * test_accuracy))

    # sess.run(Op_diff)
    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")

    save_path = saver.save(sess, os.path.join("saved_model", saved_model))
    print("Model saved in file: %s" % save_path)

    sess.close()

if __name__ == '__main__':
    main()
