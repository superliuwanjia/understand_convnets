import os
import sys
import random
import tensorflow as tf
import numpy as np
import scipy
import argparse
import logging
from scipy import misc

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from data_loader import read_image_data

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', '--dataset', type=str, required=True)
args = parser.parse_args()

# Training parameters
bs = 128
epochs = 15
num_hidden = 100
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
dataset = args.dataset
image_folder = os.path.join("/mnt/group3/ucnn/understand_convnet/data/", dataset)
image_mode = "RGB"
init_std = 1e-4
lr = 1e-4
RANDOM_SEED = 42
num_to_viz = 10
is_viz = False

# number of shuffles applied on the training set
shuffle = 0
# hidden layer activation type
activation = tf.nn.relu
# number of hidden layers
num_hidden_layers = 1

viz_dimension = (10, 10)
img_dim = (64, 64, 3)
viz_path = os.path.join("visualizations", "fc_layers{}_neurons{}_bs{}_{}".
                        format(num_hidden_layers, num_hidden, bs, dataset))
log_dir = os.path.join("logs", "fc_layers{}_neurons{}_bs{}_{}.log".
                        format(num_hidden_layers, num_hidden, bs, dataset))
saved_model = "fc_layers{}_neurons{}_bs{}.ckpt".format(num_hidden_layers, num_hidden, bs)

random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def normalize_contrast(matrix):
    shifted = matrix - matrix.min()
    return (shifted / np.ptp(shifted) * 255).astype(np.uint8)


def save_images(images, fns, path, dim=img_dim):
    if not os.path.exists(path):
        os.mkdir(path)
    for i, (image, fn) in enumerate(zip(images, fns)):
        if not dim == None:
            image = np.reshape(image, dim)
        scipy.misc.toimage(image, cmin=0, cmax=255).save(os.path.join(path, fn))


def forwardprop(X, w_vars, b_vars, activation):
    """
    Forward-propagation.
    """
    h_before = tf.matmul(X, w_vars[0]) + b_vars[0]
    h = activation(h_before)

    h_before_vars = [h_before]
    h_vars = [h]

    for i in range(num_hidden_layers - 1):

        h_before = tf.matmul(h, w_vars[i + 1]) + b_vars[i + 1]
        h = activation(h_before)

        h_before_vars += [h_before]
        h_vars += [h]

    # softmax
    yhat = tf.matmul(h, w_vars[-1]) + b_vars[-1]

    return yhat, h_vars, h_before_vars

def act_multi(image, w_vars, b_vars, activation):
    """
    Record the firing of a given input each layer and do the weight matrices multi
    """
    image = tf.reshape(image, [1, -1])
    h = activation(tf.matmul(image, w_vars[0]) + b_vars[0])
    act = tf.sign(tf.reshape(h, [-1]))
    A = tf.diag(act) # the mask
    multi = tf.matmul(tf.matmul(w_vars[0], A), w_vars[1])

    for i in range(num_hidden_layers - 1):
        h = activation(tf.matmul(h, w_vars[i + 1]) + b_vars[i + 1])
        act = tf.sign(tf.reshape(h, [-1]))
        A = tf.diag(act) # the mask
        multi = tf.matmul(tf.matmul(multi, A), w_vars[i + 2])
    return multi

def main():
    train_X, test_X, train_y, test_y, train_fn, test_fn = read_image_data(image_folder, image_mode)

    # Layer's sizes
    input_size = train_X.shape[1]
    hidden_size = num_hidden
    output_size = train_y.shape[1]

    # Symbols
    X = tf.placeholder(tf.float32, shape=[None, input_size], name="X")
    y = tf.placeholder(tf.float32, shape=[None, output_size], name="y")

    # Weight initializations
    w1_hidden = tf.Variable(tf.random_normal((input_size, num_hidden), stddev=init_std),
                            dtype=tf.float32, name="w1_hidden")
    w_vars = [w1_hidden]
    for i in range(num_hidden_layers - 1):
        w_vars += [tf.Variable(tf.random_normal((num_hidden, num_hidden), stddev=init_std),
                               dtype=tf.float32, name="w{}_hidden".format(i + 2))]
    w_soft = tf.Variable(tf.random_normal((num_hidden, output_size), stddev=init_std),
                         dtype=tf.float32, name="w_soft")
    w_vars += [w_soft]

    # bias initializations
    b1_hidden = tf.Variable(tf.zeros([1, hidden_size]), dtype=tf.float32, name="b1_hidden")
    b_vars = [b1_hidden]
    for i in range(num_hidden_layers - 1):
        b_vars += [tf.Variable(tf.zeros([1, hidden_size]), dtype=tf.float32, name="b{}_hidden".format(i))]
    b_soft = tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name="soft_bias")
    b_vars += [b_soft]

    # Forward propagation
    yhat, h_vars, h_before_vars = forwardprop(X, w_vars, b_vars, activation)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # just to pick a few to vizualize. image is huge
    to_viz = np.random.choice(range(train_X.shape[0]), num_to_viz)

    train_X_to_viz = train_X[to_viz, :]
    train_y_to_viz = train_y[to_viz, :]
    train_fn_to_viz = train_fn[to_viz]

    for _ in range(shuffle):
        shuffle_index = range(train_X.shape[0])
        random.shuffle(shuffle_index)
        train_X = train_X[shuffle_index, :]
        train_y = train_y[shuffle_index, :]
        train_fn = train_fn[shuffle_index]

    # Saver
    saver = tf.train.Saver()

    # Run SGD
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")
    if not os.path.exists(viz_path):
        os.mkdir(viz_path)

    # logging info
    logging.basicConfig(filename=log_dir, level=logging.DEBUG)

    # train & visualization
    for epoch in range(epochs):
        for b in range(int(len(train_X) / bs)):
            # all sorts of visualizations, once per epoch
            test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: test_X, y: test_y}))

            if is_viz:
                viz_path_current_epoch = os.path.join(viz_path, str(epoch * (int(len(train_X) / bs)) + b).zfill(6))
                if not os.path.exists(viz_path_current_epoch):
                    os.mkdir(viz_path_current_epoch)

                h_vars_value = sess.run(h_vars, feed_dict={X: train_X_to_viz, y: train_y_to_viz})

                w_muls = [w_vars[0]]
                for i in range(num_hidden_layers):
                    w_muls += [tf.matmul(w_muls[-1], w_vars[i + 1])]
                w_muls_value = sess.run(w_muls)

                w_act_muls = []
                for i in range(num_to_viz):
                    w_act_muls += [act_multi(train_X_to_viz[i], w_vars, b_vars, activation)]
                w_act_muls_value = sess.run(w_act_muls)

                viz_weights_fc(w_act_muls_value, w_muls_value, test_accuracy,
                               h_vars_value, viz_path_current_epoch, train_fn_to_viz)

            # training
            sess.run(updates, feed_dict={X: train_X[bs * b: bs * b + bs], y: train_y[bs * b: bs * b + bs]})

            train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                     sess.run(predict, feed_dict={X: train_X, y: train_y}))

            msg = "Epoch = {}, batch = {}, train accuracy = {:.4}, test accuracy = {:.4f}".\
                format(epoch + 1, b + 1, train_accuracy, test_accuracy)
            print(msg)

        logging.info(msg)

        # save model
        save_path = saver.save(sess, os.path.join("saved_model", saved_model))
        print("Model saved in file: %s" % save_path)

    sess.close()


def viz_weights_fc(w_act_muls_value, w_muls_value, test_accuracy, h_vars_value,
                   viz_path_current_epoch, train_fn_to_viz):
    test_accuracy_pixel = int(test_accuracy * 255)

    # weights multi with masking matrix
    # k is looping the viz images
    # i is looping the classes
    for k in range(num_to_viz):
        w_mul_soft = w_act_muls_value[k]
        save_images([np.concatenate([normalize_contrast(w_mul_soft[:, i]).reshape(img_dim), np.ones(img_dim) *
                                     test_accuracy_pixel], axis=1)
                     for i in range(w_mul_soft.shape[1])],
                    [str(k) + "weights_multi_with_masking" + str(i) + ".png" for i in range(w_mul_soft.shape[1])],
                    os.path.join(viz_path_current_epoch, "weights_multi_with_masking"), dim=None)

    # visualize filter weights per image

    # make sure visualization dimension matches number of hidden neurons
    assert np.prod(viz_dimension) == num_hidden

    for idx in range(num_hidden_layers):
        w_mul_value = w_muls_value[idx]
        h_var_value = h_vars_value[idx]
        w_stacked = np.concatenate(
            [np.concatenate(
                [normalize_contrast(w_mul_value[:, viz_dimension[0] * i + j].reshape(img_dim))
                 for j in range(viz_dimension[1])], axis=1)
             for i in range(viz_dimension[0])], axis=0)

        # put filter weights on the right side of stacked filters
        # filter weights are first scaled to match matrix dimension
        filter_weights = h_var_value.reshape((h_var_value.shape[0], viz_dimension[0], viz_dimension[1]))

        # draw each filter against filter weight image, too large to process as a whole
        for i, filter_weight in enumerate(filter_weights):

            # normalize contrast for ez view
            filter_weight = normalize_contrast(filter_weight)

            # match filter weight dimension with stacked filter matrix dimension
            filter_weight = np.repeat(np.repeat(filter_weight, img_dim[0], axis=0), img_dim[1], axis=1)

            # match image channels
            if len(img_dim) == 3 and img_dim[2] == 3:
                filter_weight = np.concatenate([np.expand_dims(filter_weight, 2)] * img_dim[2], axis=2)

            save_images([np.concatenate([w_stacked, filter_weight, np.ones(filter_weight.shape) *
                                         test_accuracy_pixel], axis=1)], [train_fn_to_viz[i]],
                        os.path.join(viz_path_current_epoch, "weight{}_of_filters_per_image".format(idx)), dim=None)


if __name__ == '__main__':
    main()