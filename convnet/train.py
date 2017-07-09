import os
import random
import sys
import tensorflow as tf
import numpy as np
from scipy import misc
import glob

sys.path.append("../")
import data_loader

bs = 32
epochs = 100
image_mode = "RGB"
init_std = 1e-4
saved_model = "conv_ks_7_nf_64_2objects_RGB_"+str(init_std)+".ckpt"
saved_model_best = "conv_ks_7_nf_64_2objects_RGB_"+str(init_std)+"_best.ckpt"
image_folder = os.path.join("../images/2objects/")
RANDOM_SEED = 42
train_test_ratio = 0.8
input_shape = [250, 250, 3]
ks1 = [7, 7, input_shape[2]]
nf1 = 64
activation = tf.nn.tanh
viz_path = os.path.join("visualizations", "conv_ks_nf_64_2objects_RGB_"+str(init_std)+"_with_bias_tanh")


random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def normalize_contrast(matrix):
    return ((matrix - matrix.min())/np.ptp(matrix)*255).astype(np.uint8)

def save_images(images, fns, path, dim=input_shape):
    if not os.path.exists(path):
        os.mkdir(path)
    for i, (image,fn) in enumerate(zip(images, fns)):
        if not dim == None:
            image = np.reshape(image, dim)
        #scipy.misc.imsave(os.path.join(path, fn), image,vmin=0,vmax=255)
        scipy.misc.toimage(image, cmin=0,cmax=255).save(os.path.join(path,fn))

def init_weights(shape, name):
    """ Weight initialization """
    # weights = tf.ones(shape)
    weights = tf.random_normal(shape, stddev=init_std)
    return tf.Variable(weights, name=name)

def init_bias(shape, name):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name=name)


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

def main():
    train_X, test_X, train_y, test_y, train_fn, test_fn = \
            data_loader.read_image_data(image_folder, image_mode)
            
    # Layer's sizes
    x_size = train_X.shape[1]  # Number of input nodes
    y_size = train_y.shape[1]  # Number of outcomes

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size], name="x")
    y = tf.placeholder("float", shape=[None, y_size], name="y")

    # reshape the input image
    X_image = tf.reshape(X, [-1, input_shape[0], input_shape[1], input_shape[2]])

    # first layer
    h_size = nf1 * (input_shape[0] - ks1[0] + 1) * (input_shape[1] - ks1[1] + 1)  # Number of hidden nodes
    w_conv1 = init_weights([ks1[0], ks1[1], ks1[2], nf1], name="w1")

    u1 = tf.nn.conv2d(X_image, w_conv1, strides=[1, 1, 1, 1], padding='VALID')
    act1 = activation(u1)
    # h1 = tf.nn.max_pool(act1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h1 = tf.reshape(act1, [-1, h_size])

    # Weight initializations
    w_soft = init_weights([h_size, y_size], "w_soft")
    b_soft = init_bias([y_size], name="b_soft")

    # Forward propagation
    u_soft = tf.matmul(h1, w_soft) + b_soft
    yhat = tf.nn.softmax(u_soft)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=u_soft))
    updates = tf.train.AdamOptimizer(1e-4).minimize(cost)

    # Saver
    saver = tf.train.Saver()

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    test_accu_best = 0.

    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")
    if not os.path.exists(viz_path):
        os.mkdir(viz_path)

    # sess.run(Op_record_init)
    for epoch in range(epochs):
        # Train with each example
        for i in range(int(len(train_X) / bs)):

            # all sorts of visualizations, once per epoch
                    
            viz_path_current_epoch = os.path.join(viz_path, str(epoch*(int(len(train_X)/bs))+i).zfill(6))
            if not os.path.exists(viz_path_current_epoch):
                os.mkdir(viz_path_current_epoch)

            w_conv1_value = sess.run(w_conv1)

            # visualize weights

            sess.run(updates, feed_dict={X: train_X[bs * i: bs * i + bs], y: train_y[bs * i: bs * i + bs]})
            train_accuracy = np.mean(np.argmax(train_y[bs * i: bs * i + bs], axis=1) ==
                                     sess.run(predict, feed_dict={X: train_X[bs * i: bs * i + bs], y: train_y[bs * i: bs * i + bs]}))
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
