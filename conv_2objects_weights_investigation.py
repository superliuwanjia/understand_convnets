import os
import scipy.misc

import tensorflow as tf
import numpy as np

import conv_2objects_train

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

model = os.path.join("saved_model", conv_2objects_train.saved_model_best)
graph = os.path.join("saved_model",conv_2objects_train.saved_model_best + ".meta")
viz_root = "visualizations"
viz_folder = conv_2objects_train.saved_model_best.split('.')[0]
viz_path = os.path.join(viz_root, viz_folder)
if not os.path.exists(viz_root):
    os.mkdir(viz_root)
if not os.path.exists(viz_path):
    os.mkdir(viz_path)

def save_images(images, fns, path, dim=(250, 250, 3)):
    if not os.path.exists(path):
        os.mkdir(path)

    for i, (image, fn) in enumerate(zip(images, fns)):
        image = image.reshape(dim)
        if dim[-1] == 1:
            scipy.misc.imsave(os.path.join(path, fn), image[:,:,0])
        else:
            scipy.misc.imsave(os.path.join(path, fn), image)

def main():
    # restore session with variables

    saver = tf.train.import_meta_graph(graph)
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            saver.restore(sess, model)
            print "Session loaded."

            train_X, test_X, train_y, test_y, train_fn, test_fn, dec_b = conv_2objects_train.get_data()
            # Layer's sizes
            x_size = train_X.shape[1]  # Number of input nodes
            y_size = train_y.shape[1]  # Number of outcomes

            # Weight initializations
            w1 = tf.get_collection(tf.GraphKeys.VARIABLES, "w1")[0]
            w_soft = tf.get_collection(tf.GraphKeys.VARIABLES, "w_soft")[0]
            w1_init = tf.get_collection(tf.GraphKeys.VARIABLES, "w1_init")[0]
            w_soft_init = tf.get_collection(tf.GraphKeys.VARIABLES, "w_soft_init")[0]

            # Forward propagation
            u1 = tf.get_collection("u1")[0]
            act1 = tf.get_collection("act1")[0]
            u_soft = tf.get_collection("u_soft")[0]
            yhat = tf.get_collection("yhat")[0]
            predict = tf.get_collection("predict")[0]

            net = tf.get_default_graph()
            X = net.get_tensor_by_name("x:0")
            y = net.get_tensor_by_name("y:0")

            # Reconstruct the input image
            I_hat_from_u1 = tf.nn.conv2d_transpose(u1, w1,
                                                   output_shape=[conv_2objects_train.bs,
                                                                 conv_2objects_train.input_shape[0],
                                                                 conv_2objects_train.input_shape[1],
                                                                 conv_2objects_train.input_shape[2]],
                                                   strides=[1,1,1,1], padding='VALID')
            I_hat_from_act1 = tf.nn.conv2d_transpose(act1, w1,
                                                     output_shape=[conv_2objects_train.bs,
                                                                   conv_2objects_train.input_shape[0],
                                                                   conv_2objects_train.input_shape[1],
                                                                   conv_2objects_train.input_shape[2]],
                                                    strides=[1, 1, 1, 1], padding='VALID')

        # visualize weights of layer 1
        w1_val = sess.run(w1)
        w1_init_val = sess.run(w1_init)
        if not os.path.exists(os.path.join(viz_path, "w1")):
            os.mkdir(os.path.join(viz_path, "w1"))
        save_images([w1_val[:,:,:,i] - w1_init_val[:,:,:,i] for i in range(w1_val.shape[-1])], \
                    [str(i) + ".png" for i in range(w1_val.shape[-1])], os.path.join(viz_path, "w1"), dim=w1_val.shape[0:-1])

        # softmax filter
        w_soft_val = sess.run(w_soft)
        w_soft_init_val = sess.run(w_soft_init)

        # visualize I
        I = train_X[0:  conv_2objects_train.bs]
        I = I.reshape([-1, 250, 250, 1])
        save_images([I[i,:,:,:] for i in range(I.shape[0])], \
                    [str(i) + ".png" for i in range(I.shape[0])],
                    os.path.join(viz_path, "I"), dim=conv_2objects_train.input_shape)

        # visualize reconstruction from u1
        I_hat_from_u1_val = sess.run(I_hat_from_u1, feed_dict={X: train_X[0:  conv_2objects_train.bs], y: train_y[0:  conv_2objects_train.bs]})
        save_images([I_hat_from_u1_val[i,:,:,:] for i in range(I_hat_from_u1_val.shape[0])], \
                    [str(i) + ".png" for i in range(I_hat_from_u1_val.shape[0])], os.path.join(viz_path, "I_hat_from_u1"),
                    dim=conv_2objects_train.input_shape)

        # visualize reconstruction from act1
        I_hat_from_act1_val = sess.run(I_hat_from_act1, feed_dict={X: train_X[0:  conv_2objects_train.bs],
                                                               y: train_y[0:  conv_2objects_train.bs]})
        save_images([I_hat_from_act1_val[i, :, :, :] for i in range(I_hat_from_act1_val.shape[0])],
                    [str(i) + ".png" for i in range(I_hat_from_act1_val.shape[0])],
                    os.path.join(viz_path, "I_hat_from_act1"), dim=conv_2objects_train.input_shape)

        # save_images([np.matmul(w, soft)[:, i][0:w.shape[0] - 1, ] for i in range(soft.shape[1])], \
        #             [str(i) + ".png" for i in range(soft.shape[1])], os.path.join(viz_path, "w*s"))

        # # visualize weights * I
        # h_materialized = sess.run(h, feed_dict={X: train_X, y: train_y})
        # save_images(h_materialized, train_fn, os.path.join(viz_path, "weights_of_filters_per_image"), dim=(10, 10))
        #
        # yhat_p = tf.placeholder("float", shape=[None, y_size], name="yhap_p")
        #
        # # w_hidden^T * a^hat * w_soft^T * yhat
        # y_type = "u_sm"
        # h_hat = tf.matmul(yhat, tf.transpose(w_soft))
        #
        # # Relu state
        # relu_mask = tf.to_float(tf.greater(h_hat, tf.zeros_like(h_hat)))
        # negative_relu_mask = tf.to_float(tf.equal(h_hat, tf.zeros_like(h_hat)))
        # all_pass_mask = tf.to_float(tf.ones_like(h_hat))
        #
        # # reconstruct using forward relu state
        # pos_relu_X = tf.matmul(tf.multiply(h_hat, relu_mask), tf.transpose(w_hidden))
        # pos_relu_X = sess.run(pos_relu_X, feed_dict={X: train_X, y: train_y})
        #
        # # reconstruct using negative forward relu state
        # neg_relu_X = tf.matmul(tf.multiply(h_hat, negative_relu_mask), tf.transpose(w_hidden))
        # neg_relu_X = sess.run(neg_relu_X, feed_dict={X: train_X, y: train_y})
        #
        # # reconstruct regardless of relu state
        # all_pass_X = tf.matmul(tf.multiply(h_hat, all_pass_mask), tf.transpose(w_hidden))
        # all_pass_X = sess.run(all_pass_X, feed_dict={X: train_X, y: train_y})
        #
        # save all reconstructed images
        # save_images(pos_relu_X[:, 0:pos_relu_X.shape[1] - 1], train_fn, \
        #             os.path.join(viz_path, "sigma_dot_a_" + y_type))
        # save_images(neg_relu_X[:, 0:neg_relu_X.shape[1] - 1], train_fn, \
        #             os.path.join(viz_path, "sigma_dot_1-a_" + y_type))
        # save_images(all_pass_X[:, 0:all_pass_X.shape[1] - 1], train_fn, \
        #             os.path.join(viz_path, "sigma_" + y_type))
        #
        # # w_hidden^T * a^hat * w_soft^T * train_y
        # y_type = "miu_cg"
        # h_hat_p = tf.matmul(yhat_p, tf.transpose(w_soft))
        #
        # # Relu state
        # relu_mask = tf.to_float(tf.greater(h_hat_p, tf.zeros_like(h_hat_p)))
        # negative_relu_mask = tf.to_float(tf.equal(h_hat_p, tf.zeros_like(h_hat_p)))
        # all_pass_mask = tf.to_float(tf.ones_like(h_hat_p))
        #
        # # reconstruct using forward relu state
        # pos_relu_X = tf.matmul(tf.multiply(h_hat_p, relu_mask), tf.transpose(w_hidden))
        # pos_relu_X = sess.run(pos_relu_X, feed_dict={X: train_X, y: train_y, yhat_p: train_y})
        #
        # # reconstruct using negative forward relu state
        # neg_relu_X = tf.matmul(tf.multiply(h_hat_p, negative_relu_mask), tf.transpose(w_hidden))
        # neg_relu_X = sess.run(neg_relu_X, feed_dict={X: train_X, y: train_y, yhat_p: train_y})
        #
        # # reconstruct regardless of relu state
        # all_pass_X = tf.matmul(tf.multiply(h_hat_p, all_pass_mask), tf.transpose(w_hidden))
        # all_pass_X = sess.run(all_pass_X, feed_dict={X: train_X, y: train_y, yhat_p: train_y})
        #
        # # save all reconstructed images
        # save_images(pos_relu_X[:, 0:pos_relu_X.shape[1] - 1], train_fn, \
        #             os.path.join(viz_path, "sigma_dot_a_" + y_type))
        # save_images(neg_relu_X[:, 0:neg_relu_X.shape[1] - 1], train_fn, \
        #             os.path.join(viz_path, "sigma_dot_1-a_" + y_type))
        # save_images(all_pass_X[:, 0:all_pass_X.shape[1] - 1], train_fn, \
        #             os.path.join(viz_path, "sigma_" + y_type))
        #
        # # w_hidden^T * a^hat * u
        # # Relu state
        # relu_mask = tf.to_float(tf.greater(u, tf.zeros_like(h)))
        # negative_relu_mask = tf.to_float(tf.equal(u, tf.zeros_like(h)))
        # all_pass_mask = tf.to_float(tf.ones_like(u))
        #
        # # reconstruct using forward relu state
        # pos_relu_X = tf.matmul(tf.multiply(u, relu_mask), tf.transpose(w_hidden))
        # pos_relu_X = sess.run(pos_relu_X, feed_dict={X: train_X, y: train_y})
        #
        # # reconstruct using negative forward relu state
        # neg_relu_X = tf.matmul(tf.multiply(u, negative_relu_mask), tf.transpose(w_hidden))
        # neg_relu_X = sess.run(neg_relu_X, feed_dict={X: train_X, y: train_y})
        #
        # # reconstruct regardless of relu state
        # all_pass_X = tf.matmul(tf.multiply(u, all_pass_mask), tf.transpose(w_hidden))
        # all_pass_X = sess.run(all_pass_X, feed_dict={X: train_X, y: train_y})
        #
        # # save all reconstructed images
        # save_images(pos_relu_X[:, 0:pos_relu_X.shape[1] - 1], train_fn, \
        #             os.path.join(viz_path, "sigma_dot_a_u"))
        # save_images(neg_relu_X[:, 0:neg_relu_X.shape[1] - 1], train_fn, \
        #             os.path.join(viz_path, "sigma_dot_1-a_u"))
        # save_images(all_pass_X[:, 0:all_pass_X.shape[1] - 1], train_fn, \
        #             os.path.join(viz_path, "sigma_dot_u"))

if __name__ == "__main__":
    main()
