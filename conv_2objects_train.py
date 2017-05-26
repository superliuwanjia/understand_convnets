import os
import random
import tensorflow as tf
import numpy as np
from scipy import misc
import glob

bs = 32
epochs = 10
image_mode = "RGB"
saved_model = "conv_ks_32_nf_64_2objects_RGB_random_init.ckpt"
saved_model_best = "conv_ks_32_nf_64_2objects_RGB_random_init_best.ckpt"
RANDOM_SEED = 42
train_test_ratio = 0.8
input_shape = [250, 250, 3]

random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

image_folder = os.path.join("./images/2objects/")


def init_weights(shape, name):
    """ Weight initialization """
    # weights = tf.ones(shape)
    weights = tf.random_normal(shape, stddev=1e-3)
    return tf.Variable(weights, name=name), weights

def init_bias(shape, name):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name=name)


def forwardprop(X, w_soft):
    """
    Forward-propagation.
    """
    yhat = tf.matmul(X, w_soft)
    return yhat


def decision_boundary(X, Label):
    """
    average(class0) - average(class1)
	"""
    accu_0 = np.zeros(X[0, :].shape)
    accu_1 = np.zeros(X[0, :].shape)
    for i in range(Label.shape[0]):
        if Label[i] == 0:
            accu_0 += X[i, :]
        else:
            accu_1 += X[i, :]
    accu_0 = accu_0 / float(360)
    accu_1 = accu_1 / float(360)
    return accu_0 - accu_1


def get_data():
    """ Read the data set and split them into training and test sets """
    X = []
    Label = []
    fns = []
    for image_path in glob.glob(os.path.join(image_folder, "*.png")):
        fns.append(os.path.basename(image_path))
        Label.append(int(os.path.basename(image_path).split("_")[0]))
        X.append(misc.imread(image_path, mode=image_mode).flatten())

    X = np.array(X) / 225.

    Label = np.array(Label)

    print (X.shape)
    print (Label.shape)

    dec_b = decision_boundary(X, Label)

    # Prepend the column of 1s for bias
    num_exps, img_dim = X.shape
    X_bias = X

    # Convert into one-hot vectors
    num_labels = len(np.unique(Label))
    Y_onehot = np.eye(num_labels)[Label]

    # shuffle X
    all_index = np.arange(X.shape[0])
    np.random.shuffle(all_index)
    X_bias = X_bias[all_index]
    Y_onehot = Y_onehot[all_index]

    index_cutoff = int(X.shape[0] * train_test_ratio)
    return X_bias[0:index_cutoff, :], X_bias[index_cutoff:, :], \
           Y_onehot[0:index_cutoff, :], Y_onehot[index_cutoff:, :], \
           fns[0:index_cutoff], fns[index_cutoff:], \
           dec_b


def main():
    train_X, test_X, train_y, test_y, train_fn, test_fn, dec_b = get_data()
    # Layer's sizes
    x_size = train_X.shape[1]  # Number of input nodes
    y_size = train_y.shape[1]  # Number of outcomes

    with tf.device("/gpu:0"):
        # Symbols
        X = tf.placeholder("float", shape=[None, x_size], name="x")
        y = tf.placeholder("float", shape=[None, y_size], name="y")

        # reshape the input image
        X_image = tf.reshape(X, [-1, input_shape[0], input_shape[1], input_shape[2]])
        # first layer
        ks1 = [32, 32, input_shape[2]]
        nf1 = 64
        h_size = nf1 * (input_shape[0] - ks1[0] + 1) * (input_shape[1] - ks1[1] + 1)  # Number of hidden nodes
        w_conv1, w_conv1_init_val = init_weights([ks1[0], ks1[1], ks1[2], nf1], name="w1")
        w_conv1_init = tf.Variable(w_conv1_init_val, name='w1_init')
        # b_conv1 = init_bias([nf1], name="b1")

        u1 = tf.nn.conv2d(X_image, w_conv1, strides=[1, 1, 1, 1], padding='VALID')
        act1 = tf.nn.relu(u1)
        # h1 = tf.nn.max_pool(act1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        h1 = tf.reshape(act1, [-1, h_size])

        # Weight initializations
        w_soft, w_soft_init_val = init_weights([h_size, y_size], "w_soft")
        w_soft_init = tf.Variable(w_soft_init_val, name='w_soft_init')
        b_soft = init_bias([y_size], name="b_soft")

        # Forward propagation
        u_soft = tf.matmul(h1, w_soft) + b_soft
        yhat = tf.nn.softmax(u_soft)
        predict = tf.argmax(yhat, axis=1)

        tf.add_to_collection("u1", u1)
        tf.add_to_collection("act1", act1)
        tf.add_to_collection("u_soft", u_soft)
        tf.add_to_collection("yhat", yhat)
        tf.add_to_collection("predict", predict)

        # Backward propagation
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(yhat), reduction_indices=[1]))
        updates = tf.train.AdamOptimizer(1e-4).minimize(cost)

    # Saver
    saver = tf.train.Saver()

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    test_accu_best = 0.
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
        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                sess.run(predict, feed_dict={X: test_X, y: test_y}))
        print("Test accuracy at epoch %d = %.2f%%" % (epoch + 1, 100. * test_accuracy))
        if test_accuracy >= test_accu_best:
            if not os.path.exists("saved_model"):
                os.mkdir("saved_model")
            save_path_best = saver.save(sess, os.path.join("saved_model", saved_model_best))
            test_accu_best = test_accuracy

    # sess.run(Op_diff)
    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")

    save_path = saver.save(sess, os.path.join("saved_model", saved_model))
    print("Model saved in file: %s" % save_path)

    sess.close()


if __name__ == '__main__':
    main()