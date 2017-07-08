import os
import random
import tensorflow as tf
import numpy as np
from scipy import misc
import glob
import time

import data_loader
bs = 32
epochs = 50
num_hidden = 1000
saved_model = "one_hidden_2objects_translation_RGB_1e-4.ckpt"
image_mode = "RGB"
init_std = 1e-4
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

image_folder = os.path.join("./images/2objects_translation/")

def forwardprop(X, w_hidden, w_soft, soft_bias):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h_before_relu = tf.matmul(X, w_hidden)
    h = tf.nn.relu(h_before_relu)
    yhat = tf.matmul(h, w_soft) + soft_bias
    return yhat, h, h_before_relu


def main():

    train_X, test_X, train_y, test_y, train_fn, test_fn = \
            data_loader.read_image_data(image_folder, image_mode)

    # Layer's sizes
    input_size = train_X.shape[1]
    hidden_size = num_hidden
    output_size = train_y.shape[1]
    
    # Symbols
    X = tf.placeholder("float", shape=[None, train_X.shape[1]], name="X")
    y = tf.placeholder("float", shape=[None, train_y.shape[1]], name="y")

    # Weight initializations
    w_hidden = tf.Variable(tf.random_normal((train_X.shape[1], num_hidden), stddev=init_std),
                      name="w_hidden", trainable=True)
    w_soft = tf.Variable(tf.random_normal((num_hidden, train_y.shape[1]), stddev=init_std),
                      name="w_soft", trainable=True)
	
    # bia initializations
    soft_bias = tf.Variable(0.*tf.random_normal((1, train_y.shape[1]), stddev=init_std), name="soft_bias", trainable=True)
	
    # Forward propagation
    yhat, h, h_before_relu = forwardprop(X, w_hidden, w_soft, soft_bias)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    # Saver
    saver = tf.train.Saver()

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")


    for epoch in range(epochs):
        for i in range(int(len(train_X)/bs)):
            sess.run(updates, feed_dict={X: train_X[bs * i: bs * i + bs], y: train_y[bs * i: bs * i + bs]})
            
            train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                     sess.run(predict, feed_dict={X: train_X, y: train_y}))
            test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: test_X, y: test_y}))

            print("Epoch = %d, batch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, i + 1, 100. * train_accuracy, 100. * test_accuracy))

        save_path = saver.save(sess, os.path.join("saved_model", saved_model))
        print("Model saved in file: %s" % save_path)

    sess.close()


if __name__ == '__main__':
    main()
