# updates
# 7/31/2017 when calculating the "diff" and "diff_exsoft" feedforwards, set bias to zeros

import os
import sys
import random
import tensorflow as tf
import numpy as np
from config_fc import get_config
import logging
from utils import save_saliency_img
import scipy
from scipy import misc

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from data_loader import read_image_data

# Basic model parameters as external flags.
FLAGS = None
# activation function to use
activation = tf.nn.relu


def normalize_contrast(matrix):
    shifted = matrix - matrix.min()
    return (shifted / np.ptp(shifted) * 255).astype(np.uint8)


# def save_images(images, fns, path, dim=img_dim):
#     if not os.path.exists(path):
#         os.mkdir(path)
#     for i, (image, fn) in enumerate(zip(images, fns)):
#         if not dim == None:
#             image = np.reshape(image, dim)
#         scipy.misc.toimage(image, cmin=0, cmax=255).save(os.path.join(path, fn))


def forwardprop(X, w_vars, b_vars, activation, name='Forwardprop'):
    with tf.name_scope(name):
        with tf.name_scope('FC1'):
            h_before = tf.matmul(X, w_vars[0]) + b_vars[0]
            h = activation(h_before)

        h_before_vars = [h_before]
        h_vars = [h]

        for i in range(FLAGS.num_layers - 1):
            with tf.name_scope('FC{}'.format(i + 2)):
                h_before = tf.matmul(h, w_vars[i + 1]) + b_vars[i + 1]
                h = activation(h_before)

            h_before_vars += [h_before]
            h_vars += [h]

        with tf.name_scope('Softmax'):
            yhat = tf.matmul(h, w_vars[-1]) + b_vars[-1]

    return yhat, h_vars, h_before_vars


# def act_multi(image, w_vars, b_vars, activation):
#     """
#     Record the firing of a given input each layer and do the weight matrices multi
#     """
#     image = tf.reshape(image, [1, -1])
#     h = activation(tf.matmul(image, w_vars[0]) + b_vars[0])
#     act = tf.sign(tf.reshape(h, [-1]))
#     A = tf.diag(act) # the mask
#     multi = tf.matmul(tf.matmul(w_vars[0], A), w_vars[1])
#
#     for i in range(num_hidden_layers - 1):
#         h = activation(tf.matmul(h, w_vars[i + 1]) + b_vars[i + 1])
#         act = tf.sign(tf.reshape(h, [-1]))
#         A = tf.diag(act) # the mask
#         multi = tf.matmul(tf.matmul(multi, A), w_vars[i + 2])
#     return multi

def placeholder_inputs(input_size, output_size, name='Inputs'):
    with tf.name_scope(name):
        img_ph = tf.placeholder(tf.float32, shape=(None, input_size), name='X')
        lbl_ph = tf.placeholder(tf.int32, shape=(None, output_size), name='y')

    # check what the inputs look like
    images = tf.reshape(img_ph, [-1, 64, 64, 3])
    tf.summary.image('Inputs', images, max_outputs=2, collections=None)

    return img_ph, lbl_ph


def weights_and_bias(input_size, output_size):
    with tf.name_scope('hidden1/'):
        w1_hidden = tf.Variable(tf.truncated_normal((input_size, FLAGS.num_neurons), stddev=FLAGS.std),
                                dtype=tf.float32, name="w1_hidden")
        tf.summary.histogram('W', w1_hidden)

    w_vars = [w1_hidden]

    for i in range(FLAGS.num_layers - 1):
        with tf.name_scope('hidden{}/'.format(i + 2)):
            wi_hidden = tf.Variable(tf.truncated_normal((FLAGS.num_neurons, FLAGS.num_neurons), stddev=FLAGS.std),
                                    dtype=tf.float32, name="w{}_hidden".format(i + 2))
            tf.summary.histogram('W'.format(i), wi_hidden)

        w_vars += [wi_hidden]

    with tf.name_scope('soft/'):
        w_soft = tf.Variable(tf.truncated_normal((FLAGS.num_neurons, output_size), stddev=FLAGS.std),
                             dtype=tf.float32, name="w_soft")
        tf.summary.histogram('W', w_soft)

    w_vars += [w_soft]

    # store the init values so that we can calculate the learned "diff" later
    w_vars_init = [tf.Variable(w_vars[i].initialized_value(), name='w_init_{}'.format(i))
                   for i in range(len(w_vars))]

    # init bias either to all zeros or to small positive constant 0.1
    if not FLAGS.pb:

        with tf.name_scope('hidden1/'):
            b1_hidden = tf.Variable(tf.zeros([1, FLAGS.num_neurons]), dtype=tf.float32, name="b1_hidden")
            tf.summary.histogram('B', b1_hidden)

        b_vars = [b1_hidden]

        for i in range(FLAGS.num_layers - 1):
            with tf.name_scope('hidden{}/'.format(i + 2)):
                bi_hidden = tf.Variable(tf.zeros([1, FLAGS.num_neurons]), dtype=tf.float32, name="b{}_hidden".format(i))
                tf.summary.histogram('B'.format(i), bi_hidden)

            b_vars += [bi_hidden]

        with tf.name_scope('soft/'):
            b_soft = tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name="soft_bias")
            tf.summary.histogram('B', b_soft)

        b_vars += [b_soft]

    else:

        with tf.name_scope('hidden1/'):
            b1_hidden = tf.Variable(tf.constant(0.1, shape=[1, FLAGS.num_neurons]), dtype=tf.float32, name="b1_hidden")
            tf.summary.histogram('B', b1_hidden)

        b_vars = [b1_hidden]

        for i in range(FLAGS.num_layers - 1):
            with tf.name_scope('hidden{}/'.format(i + 2)):
                bi_hidden = tf.Variable(tf.constant(0.1, shape=[1, FLAGS.num_neurons]), dtype=tf.float32,
                                        name="b{}_hidden".format(i))
                tf.summary.histogram('B'.format(i), bi_hidden)

            b_vars += [bi_hidden]

        with tf.name_scope('soft/'):
            b_soft = tf.Variable(tf.constant(0.1, shape=[1, output_size]), dtype=tf.float32, name="soft_bias")
            tf.summary.histogram('B', b_soft)

        b_vars += [b_soft]

    # we will need this zero bias when calculating the diff and diff_exsoft forwardprob
    b1_hidden_zero = tf.Variable(tf.zeros([1, FLAGS.num_neurons]), dtype=tf.float32, name="b1_hidden_zero")
    b_vars_zero = [b1_hidden_zero]
    for i in range(FLAGS.num_layers - 1):
        b_vars_zero += [
            tf.Variable(tf.zeros([1, FLAGS.num_neurons]), dtype=tf.float32, name="b{}_hidden_zero".format(i))]
    b_soft_zero = tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name="soft_bias_zero")
    b_vars_zero += [b_soft_zero]

    return w_vars, w_vars_init, b_vars, b_vars_zero


def entropy_loss(logits, labels, name='Entropy_loss'):
    with tf.name_scope(name):
        with tf.name_scope('Cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')

        with tf.name_scope('Cost'):
            cost = tf.reduce_mean(cross_entropy, name='entropy_mean')

    return cost


def train(loss, name='Train'):
    with tf.name_scope(name):
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        train_op = optimizer.minimize(loss)

    return train_op


def evaluation(logits, labels):
    predict = tf.argmax(logits, axis=1)
    correct = tf.argmax(labels, axis=1)
    accu = tf.reduce_mean(tf.to_float(tf.equal(correct, predict)))
    return accu


def diff(X, w_vars, w_vars_init, b_vars, b_vars_zero):
    # Calculate diff_weights and use it get logits_diff
    w_vars_diff = [w_vars[i] - w_vars_init[i] for i in range(len(w_vars))]

    logits_diff, _, _ = forwardprop(X, w_vars_diff, b_vars_zero, activation)

    # Calculate diff_exsoft_weights and use it to get logits_diff_exsoft
    w_vars_diff_exsoft = [w_vars[i] - w_vars_init[i] for i in range(len(w_vars) - 1)]
    w_vars_diff_exsoft += [w_vars[-1]]

    # We also need to exclude the softmax bias
    b_vars_zero_exsoft = [b_vars_zero[i] for i in range(len(b_vars_zero) - 1)]
    b_vars_zero_exsoft += [b_vars[-1]]

    logits_diff_exsoft, _, _ = forwardprop(X, w_vars_diff_exsoft, b_vars_zero_exsoft, activation)

    return logits_diff, logits_diff_exsoft


def saliency_map(x_in, w_vars, b_vars, act=activation):
    logits, _, _ = forwardprop(x_in, w_vars, b_vars, act)
    max_logits = tf.reduce_max(logits, axis=1)
    saliency = tf.gradients(max_logits, x_in)
    max_class = tf.argmax(logits,axis=1)
    return saliency, max_class


def run_training(viz_dimension, img_dim, image_folder, summary_name, log_dir, saved_model):
    train_X, test_X, train_y, test_y, train_fn, test_fn = read_image_data(image_folder, 'RGB')

    # just to pick a few to visualize. image is huge
    to_viz = np.random.choice(range(train_X.shape[0]), 10)

    train_X_to_viz = train_X[to_viz, :]
    train_y_to_viz = train_y[to_viz, :]
    train_fn_to_viz = train_fn[to_viz]

    # Layer's sizes
    input_size = train_X.shape[1]
    output_size = train_y.shape[1]

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # Placeholders for the images and labels.
        img_ph, lbl_ph = placeholder_inputs(input_size, output_size)

        # Define weights and bias
        w_vars, w_vars_init, b_vars, b_vars_zero = weights_and_bias(input_size, output_size)

        # Forward propagation
        logits, h_vars, h_before_vars = forwardprop(img_ph, w_vars, b_vars, activation)

        # loss
        loss = entropy_loss(logits, lbl_ph)
        tf.summary.scalar('train/loss', loss)

        # training operation
        train_op = train(loss)

        # accuracy
        accuracy = evaluation(logits, lbl_ph)
        tf.summary.scalar('accu/accu', accuracy)

        # "diff" related forward propagation
        logits_diff, logits_diff_exsoft = diff(img_ph, w_vars, w_vars_init, b_vars, b_vars_zero)

        # "diff" related accuracy
        diff_accuracy = evaluation(logits_diff, lbl_ph)
        tf.summary.scalar('accu/diff_accu', diff_accuracy)
        diff_exsoft_accuracy = evaluation(logits_diff_exsoft, lbl_ph)
        tf.summary.scalar('accu/diff_accu_exsoft', diff_exsoft_accuracy)

        # saliency map
        saliency, max_class = saliency_map(img_ph, w_vars, b_vars)

        # Saver
        saver = tf.train.Saver()

        # create sess
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # merge all the summaries
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summary_name + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(summary_name + '/test')

        # init
        sess.run(tf.global_variables_initializer())

        if not os.path.exists("saved_model"):
            os.mkdir("saved_model")
        if not os.path.exists("logs"):
            os.mkdir("logs")

        if FLAGS.is_viz:
            if not os.path.exists("visualizations"):
                os.mkdir("visualizations")

        # logging info
        logging.basicConfig(filename=log_dir, level=logging.DEBUG)

        # train & visualization
        step = 0
        for epoch in range(FLAGS.epochs):
            for b in range(int(len(train_X) / FLAGS.bs)):

                # if FLAGS.is_viz:
                #     viz_path_current_epoch = os.path.join(
                #         viz_path, str(epoch * (int(len(train_X) / bs)) + b).zfill(6))
                #     if not os.path.exists(viz_path_current_epoch):
                #         os.mkdir(viz_path_current_epoch)
                #
                #     h_vars_value = sess.run(h_vars, feed_dict={X: train_X_to_viz, y: train_y_to_viz})
                #
                #     if is_viz_weight_diff:
                #         w_muls = [w_vars_diff[0]]
                #         for i in range(num_hidden_layers):
                #             w_muls += [tf.matmul(w_muls[-1], w_vars_diff[i + 1])]
                #         w_muls_value = sess.run(w_muls)
                #     else:
                #         w_muls = [w_vars[0]]
                #         for i in range(num_hidden_layers):
                #             w_muls += [tf.matmul(w_muls[-1], w_vars[i + 1])]
                #         w_muls_value = sess.run(w_muls)
                #
                #     w_act_muls = []
                #     for i in range(num_to_viz):
                #         w_act_muls += [act_multi(train_X_to_viz[i], w_vars, b_vars, activation)]
                #     w_act_muls_value = sess.run(w_act_muls)
                #
                #     viz_weights_fc(w_act_muls_value, w_muls_value, test_accu_val,
                #                    h_vars_value, viz_path_current_epoch, train_fn_to_viz)

                # save saliency map (only use one datapoint for now)
                saliency_val, max_class_val = sess.run([saliency, max_class],
                                                feed_dict={img_ph: train_X_to_viz[0], lbl_ph: train_y_to_viz[0]})
                save_saliency_img(train_X_to_viz, saliency_val, max_class_val)

                # testing
                sum_str_test, test_accu, test_diff_accu, test_diff_exsoft_accu = \
                    sess.run([merged, accuracy, diff_accuracy, diff_exsoft_accuracy],
                             feed_dict={img_ph: test_X,
                                        lbl_ph: test_y})

                test_writer.add_summary(sum_str_test, step)

                # training
                _, sum_str_train, train_accu, train_diff_accu, train_diff_exsoft_accu = \
                    sess.run([train_op, merged, accuracy, diff_accuracy, diff_exsoft_accuracy],
                             feed_dict={img_ph: train_X[FLAGS.bs * b: FLAGS.bs * b + FLAGS.bs],
                                        lbl_ph: train_y[FLAGS.bs * b: FLAGS.bs * b + FLAGS.bs]})

                train_writer.add_summary(sum_str_train, step)

                msg = "epoch = {}, batch = {}, " \
                      "train accu = {:.4f}, test accu = {:.4f}, " \
                      "train diff accu = {:.4f}, test diff accu = {:.4f}, " \
                      "train diff exsoft accu = {:.4f}, test diff exsoft accu = {:.4f}" \
                    .format(epoch, b,
                            train_accu, test_accu,
                            train_diff_accu, test_diff_accu,
                            train_diff_exsoft_accu, train_diff_exsoft_accu)
                print(msg)
                logging.info(msg)

                step += 1

        # save model
        save_path = saver.save(sess, os.path.join("saved_model", saved_model))
        print("Model saved in file: %s" % save_path)

        sess.close()


# def viz_weights_fc(w_act_muls_value, w_muls_value, test_accuracy, h_vars_value,
#                    viz_path_current_epoch, train_fn_to_viz):
#     test_accuracy_pixel = int(test_accuracy * 255)
#
#     # weights multi with masking matrix
#     # k is looping the viz images, i is looping the classes
#     for k in range(num_to_viz):
#         w_mul_soft = w_act_muls_value[k]
#         save_images([np.concatenate([normalize_contrast(w_mul_soft[:, i]).reshape(img_dim), np.ones(img_dim) *
#                                      test_accuracy_pixel], axis=1)
#                      for i in range(w_mul_soft.shape[1])],
#                     [str(k) + "weights_multi_with_masking" + str(i) + ".png" for i in range(w_mul_soft.shape[1])],
#                     os.path.join(viz_path_current_epoch, "weights_multi_with_masking"), dim=None)
#
#     # visualize filter weights per image
#
#     # make sure visualization dimension matches number of hidden neurons
#     assert np.prod(viz_dimension) == num_hidden
#
#     for idx in range(num_hidden_layers):
#         w_mul_value = w_muls_value[idx]
#         h_var_value = h_vars_value[idx]
#         w_stacked = np.concatenate(
#             [np.concatenate(
#                 [normalize_contrast(w_mul_value[:, viz_dimension[0] * i + j].reshape(img_dim))
#                  for j in range(viz_dimension[1])], axis=1)
#              for i in range(viz_dimension[0])], axis=0)
#
#         # put filter weights on the right side of stacked filters
#         # filter weights are first scaled to match matrix dimension
#         filter_weights = h_var_value.reshape((h_var_value.shape[0], viz_dimension[0], viz_dimension[1]))
#
#         # draw each filter against filter weight image, too large to process as a whole
#         for i, filter_weight in enumerate(filter_weights):
#
#             # normalize contrast for ez view
#             filter_weight = normalize_contrast(filter_weight)
#
#             # match filter weight dimension with stacked filter matrix dimension
#             filter_weight = np.repeat(np.repeat(filter_weight, img_dim[0], axis=0), img_dim[1], axis=1)
#
#             # match image channels
#             if len(img_dim) == 3 and img_dim[2] == 3:
#                 filter_weight = np.concatenate([np.expand_dims(filter_weight, 2)] * img_dim[2], axis=2)
#
#             save_images([np.concatenate([w_stacked, filter_weight, np.ones(filter_weight.shape) *
#                                          test_accuracy_pixel], axis=1)], [train_fn_to_viz[i]],
#                         os.path.join(viz_path_current_epoch, "weight{}_of_filters_per_image".format(idx)), dim=None)

def main(FLAGS):
    # some dimensions
    # Note: the product of viz_dimension elements has to be equal to the num of hidden neurons
    viz_dimension = (10, 10)
    img_dim = (64, 64, 3)

    # set the random seed
    random.seed(FLAGS.rs)
    tf.set_random_seed(FLAGS.rs)
    np.random.seed(FLAGS.rs)

    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

    # image dataset path
    image_folder = os.path.join("data/", FLAGS.dataset)

    # summary path and name
    summary_path = os.path.join("summaries", FLAGS.s_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    summary_name = os.path.join(summary_path, "fc_num_layers{}_num_neurons{}_bs{}_lr{}_std{}_{}".
                                format(FLAGS.num_layers, FLAGS.num_neurons, FLAGS.bs, FLAGS.lr, FLAGS.std,
                                       FLAGS.dataset))

    # always save the training log
    log_dir = os.path.join("logs", "fc_num_layers{}_num_neurons{}_bs{}_lr{}_std{}_{}.log".
                           format(FLAGS.num_layers, FLAGS.num_neurons, FLAGS.bs, FLAGS.lr, FLAGS.std, FLAGS.dataset))

    # always save the trained model
    saved_model = "fc_num_layers{}_num_neurons{}_bs{}_lr{}_std{}_{}.ckpt". \
        format(FLAGS.num_layers, FLAGS.num_neurons, FLAGS.bs, FLAGS.lr, FLAGS.std, FLAGS.dataset)

    # start the training
    run_training(viz_dimension,
                 img_dim,
                 image_folder,
                 summary_name,
                 log_dir,
                 saved_model
                 )


if __name__ == '__main__':
    FLAGS, unparsed = get_config()
    main(FLAGS)
