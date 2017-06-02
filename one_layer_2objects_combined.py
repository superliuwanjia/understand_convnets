import os
import random
import tensorflow as tf
import numpy as np

import scipy
from scipy import misc
import glob
import time

import data_loader
bs = 1
epochs = 25
num_hidden = 4
saved_model = "one_hidden_2objects_RGB_1e-4.ckpt"
image_folder = os.path.join("./images/2objects/")
image_mode = "RGB"
init_std = 1e-4
RANDOM_SEED = 42

viz_dimention =(2, 2)
img_dim = (250, 250,3)
viz_path = os.path.join("visualizations", "rgb_epoch_1e-4_batch_1")
num_to_viz = 1

random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def normalize_contrast(matrix):
    return ((matrix - matrix.min())/np.ptp(matrix)*255).astype(np.uint8)

def save_images(images, fns, path, dim=img_dim):
    if not os.path.exists(path):
        os.mkdir(path)
    for i, (image,fn) in enumerate(zip(images, fns)):
        if not dim == None:
            image = np.reshape(image, dim)
        scipy.misc.imsave(os.path.join(path, fn), image)
 
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
    w_mul = tf.matmul(w_hidden, w_soft) + soft_bias
    yhat, h, u = forwardprop(X, w_hidden, w_soft, soft_bias)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    # just to pick a few to vizualize. image is huge
    to_viz = np.random.choice(range(train_X.shape[0]), num_to_viz)
    train_X_to_viz = train_X[to_viz,:]
    train_y_to_viz = train_y[to_viz,:]	
    train_fn_to_viz = train_fn[to_viz]


    # Saver
    saver = tf.train.Saver()

    # Run SGD
    sess = tf.Session()
    sess.as_default()      

    init = tf.global_variables_initializer()
    sess.run(init)

    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")
    if not os.path.exists(viz_path):
        os.mkdir(viz_path)

    # train & vizualization
    for epoch in range(epochs):
        for b in range(int(len(train_X)/bs)):
            # all sorts of visualizations, once per epoch
        
            viz_path_current_epoch = os.path.join(viz_path, str(epoch*(int(len(train_X)/bs))+b).zfill(6))
            if not os.path.exists(viz_path_current_epoch):
                os.mkdir(viz_path_current_epoch)

    
        
            # read out weights
            w_hidden_value = sess.run(w_hidden)	
            w_soft_value = sess.run(w_soft)
            w_mul_value = sess.run(w_mul)
            """ 
            # hidden weights 
            save_images([w_hidden_value[:,i] for i in range(w_hidden_value.shape[1])], \
                        ["hidden_weights_" + str(i) + ".png" for i in range(w_hidden_value.shape[1])], \
                        os.path.join(viz_path_current_epoch, "hidden_weights"))  
            """
                        
            # hidden_weights * soft_weights
            save_images([w_mul_value[:,i] for i in range(w_mul_value.shape[1])], \
                        ["hidden_multi_soft_" + str(i) + ".png" for i in range(w_mul_value.shape[1])], \
                        os.path.join(viz_path_current_epoch, "hidden_multi_soft_"))  

            # hidden weights in a huge image

            # make sure vizualization dimention matches number of hidden neurons
            assert np.prod(viz_dimention) == num_hidden

            w_stacked = np.concatenate(
                        [np.concatenate( \
                            [normalize_contrast(w_hidden_value[:,viz_dimention[0]*i+j].reshape(img_dim)) \
                            for j in range(viz_dimention[1])], axis=1) \
                        for i in range(viz_dimention[0])], axis=0)
            print w_stacked.shape
            # visualize filter weights per image
            filter_weights = sess.run(h, feed_dict={X:train_X_to_viz, y:train_y_to_viz})

            # put filter weights on the right side of stacked filters
            # filter weights are first scaled to match matrix dimention
            filter_weights = filter_weights.reshape((filter_weights.shape[0], viz_dimention[0], \
                viz_dimention[1]))

            # draw each filter against filter weight image, too large to process as a whole
            for i, filter_weight in enumerate(filter_weights):

                # normalize contrast for ez view
                filter_weight = normalize_contrast(filter_weight)

                # match filter wieght dimention with stacked filter matrix dimention
                filter_weight = np.repeat(np.repeat(filter_weight, 250, axis=0), 250, axis=1)

                # match image channels
                if len(img_dim) == 3 and img_dim[2] == 3:
                    filter_weight = np.concatenate([np.expand_dims(filter_weight,2)] * img_dim[2],\
                        axis=2)
             
                save_images([np.concatenate([w_stacked,filter_weight], axis=1)], [train_fn[i]], \
                    os.path.join(viz_path_current_epoch, "weights_of_filters_per_image"),dim=None)


            yhat_p = tf.placeholder("float", shape=[None, output_size], name="yhap_p")
            """ 
            # w_hidden^T * a^hat * w_soft^T * yhat
            y_type = "u_sm"
            h_hat = tf.matmul(yhat, tf.transpose(w_soft))

            # Relu state
            relu_mask = tf.to_float(tf.greater(h_hat, tf.zeros_like(h_hat)))
            negative_relu_mask = tf.to_float(tf.equal(h_hat, tf.zeros_like(h_hat)))
            all_pass_mask = tf.to_float(tf.ones_like(h_hat))
     
            # reconstruct using forward relu state
            pos_relu_X = tf.matmul(tf.multiply(h_hat, relu_mask), tf.transpose(w_hidden))
            pos_relu_X = sess.run(pos_relu_X, feed_dict={X:train_X_to_viz, y:train_y_to_viz})      
           
            # reconstruct using negative forward relu state
            neg_relu_X = tf.matmul(tf.multiply(h_hat, negative_relu_mask), tf.transpose(w_hidden))
            neg_relu_X = sess.run(neg_relu_X, feed_dict={X:train_X_to_viz, y:train_y_to_viz})      
            
            # reconstruct regardless of relu state
            all_pass_X = tf.matmul(tf.multiply(h_hat, all_pass_mask), tf.transpose(w_hidden))
            all_pass_X = sess.run(all_pass_X, feed_dict={X:train_X_to_viz, y:train_y_to_viz})      
           
            # save all reconstructed images
            save_images(pos_relu_X,train_fn, \
                os.path.join(viz_path_current_epoch, "sigma_dot_a_"+y_type)) 
            save_images(neg_relu_X,train_fn, \
                os.path.join(viz_path_current_epoch, "sigma_dot_1-a_"+y_type)) 
            save_images(all_pass_X,train_fn, \
                os.path.join(viz_path_current_epoch, "sigma_"+y_type)) 
            
           
            # w_hidden^T * a^hat * w_soft^T * train_y 
            y_type = "miu_cg"
            h_hat_p = tf.matmul(yhat_p, tf.transpose(w_soft))

            # Relu state
            relu_mask = tf.to_float(tf.greater(h_hat_p, tf.zeros_like(h_hat_p)))
            negative_relu_mask = tf.to_float(tf.equal(h_hat_p, tf.zeros_like(h_hat_p)))
            all_pass_mask = tf.to_float(tf.ones_like(h_hat_p))
     
            # reconstruct using forward relu state
            pos_relu_X = tf.matmul(tf.multiply(h_hat_p, relu_mask), tf.transpose(w_hidden))
            pos_relu_X = sess.run(pos_relu_X, feed_dict={X:train_X_to_viz, y:train_y_to_viz, yhat_p:train_y_to_viz})      
           
            # reconstruct using negative forward relu state
            neg_relu_X = tf.matmul(tf.multiply(h_hat_p, negative_relu_mask), tf.transpose(w_hidden))
            neg_relu_X = sess.run(neg_relu_X, feed_dict={X:train_X_to_viz, y:train_y_to_viz, yhat_p:train_y_to_viz})      
            
            # reconstruct regardless of relu state
            all_pass_X = tf.matmul(tf.multiply(h_hat_p, all_pass_mask), tf.transpose(w_hidden))
            all_pass_X = sess.run(all_pass_X, feed_dict={X:train_X_to_viz, y:train_y_to_viz, yhat_p:train_y_to_viz})      
           
            # save all reconstructed images
            save_images(pos_relu_X,train_fn, \
                os.path.join(viz_path_current_epoch, "sigma_dot_a_"+y_type)) 
            save_images(neg_relu_X,train_fn, \
                os.path.join(viz_path_current_epoch, "sigma_dot_1-a_"+y_type)) 
            save_images(all_pass_X,train_fn, \
                os.path.join(viz_path_current_epoch, "sigma_"+y_type)) 
     

            # w_hidden^T * a^hat * u
            # Relu state
            relu_mask = tf.to_float(tf.greater(u, tf.zeros_like(h)))
            negative_relu_mask = tf.to_float(tf.equal(u, tf.zeros_like(h)))
            all_pass_mask = tf.to_float(tf.ones_like(u))
     
            # reconstruct using forward relu state
            pos_relu_X = tf.matmul(tf.multiply(u, relu_mask), tf.transpose(w_hidden))
            pos_relu_X = sess.run(pos_relu_X, feed_dict={X:train_X_to_viz, y:train_y_to_viz})      
           
            # reconstruct using negative forward relu state
            neg_relu_X = tf.matmul(tf.multiply(u, negative_relu_mask), tf.transpose(w_hidden))
            neg_relu_X = sess.run(neg_relu_X, feed_dict={X:train_X_to_viz, y:train_y_to_viz})      
            
            # reconstruct regardless of relu state
            all_pass_X = tf.matmul(tf.multiply(u, all_pass_mask), tf.transpose(w_hidden))
            all_pass_X = sess.run(all_pass_X, feed_dict={X:train_X_to_viz, y:train_y_to_viz})      
           
            # save all reconstructed images
            save_images(pos_relu_X,train_fn, \
                os.path.join(viz_path_current_epoch, "sigma_dot_a_u")) 
            save_images(neg_relu_X,train_fn, \
                os.path.join(viz_path_current_epoch, "sigma_dot_1-a_u")) 
            save_images(all_pass_X,train_fn, \
                os.path.join(viz_path_current_epoch, "sigma_dot_u")) 
            """
            # training 
            sess.run(updates, feed_dict={X: train_X[bs * b: bs * b + bs], y: train_y[bs * b: bs * b + bs]})
            
            train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                     sess.run(predict, feed_dict={X: train_X, y: train_y}))
            test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: test_X, y: test_y}))

            print("Epoch = %d, batch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, b + 1, 100. * train_accuracy, 100. * test_accuracy))

 
        # save model
        save_path = saver.save(sess, os.path.join("saved_model", saved_model))
        print("Model saved in file: %s" % save_path)

    sess.close()


if __name__ == '__main__':
    main()
