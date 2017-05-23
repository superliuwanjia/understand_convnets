# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import os

import tensorflow as tf
import numpy as np
from scipy import misc
import glob
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

image_folder = os.path.join("images/2objects/")

def init_weights(shape, name):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name=name)

def forwardprop(X, w_hidden, w_soft):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h = tf.nn.relu(tf.matmul(X, w_hidden))  # The \sigma function
    yhat = tf.matmul(h, w_soft)  # The \varphi function
    return yhat, h

def get_data():
    """ Read the data set and split them into training and test sets """
    X = []
    Label = []
    for image_path in glob.glob(os.path.join(image_folder,"*.png")):
        label = os.basename(image_path).split("_")[0]
        image = X.append(misc.imread(image_path, mode="RGB").flatten())
    X = np.asarray(X)
    Laebl = np.asarray(Label)
    print X.shape
    print Label.shape

    # Prepend the column of 1s for bias
    num_exps, img_dim  = X.shape
    all_X = np.ones((num_exps, img_dim + 1))
    all_X[:, 1:] = X

    # Convert into one-hot vectors
    num_labels = len(np.unique(Label))
    all_Y = np.eye(num_labels)[Label]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
    train_X, test_X, train_y, test_y = get_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes
    h_size = 100                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size], name="x")
    y = tf.placeholder("float", shape=[None, y_size], name="y")

    # Weight initializations
    w_hidden = init_weights((x_size, h_size), "w_hidden")
    w_soft = init_weights((h_size, y_size), "w_softmax")

    # Forward propagation
    yhat, h    = forwardprop(X, w_hidden, w_soft)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Saver
    saver = tf.train.Saver()

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    if not os.path.exist("saved_model"):
        os.mkdir("saved_model")

    save_path = saver.save(sess, os.path.join("saved_model","1layer_mlp_2objects.ckpt")
    print("Model saved in file: %s" % save_path)

    sess.close()


if __name__ == '__main__':
    main()
