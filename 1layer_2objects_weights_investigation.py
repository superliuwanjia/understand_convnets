import os
import scipy.misc

import tensorflow as tf
import numpy as np

import 1layer_2objects_train

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

model = os.path.join("saved_model","1layer_mlp_2objects.ckpt")
save_path = "reconstructed_images"

def save_images(images, path, dim=(250,250,3)):
    for i, image in enumerate(images):
        scipy.misc.imsave(os.path.join(path, str(i), ".png"), image.reshape(dim), \
            format="RGB")
        
def main():
    # restore session with variables

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model)
        print "Session loaded."
    
        train_X, test_X, train_y, test_y = 1layer_2objects_train.get_data()
        # Layer's sizes
        x_size = train_X.shape[1]   # Number of input nodes
        h_size = 100                # Number of hidden nodes
        y_size = train_y.shape[1]   # Number of outcomes

        # Symbols
        X = tf.placeholder("float", shape=[None, x_size], name="x")
        y = tf.placeholder("float", shape=[None, y_size], name="y")

        # Weight initializations
        w_hidden = tf.get_variable("w_hidden")
        w_soft = tf.get_variable("w_softmax")

        # Forward propagation
        yhat, h = 1layer_2objects_train.forwardprop(X, w_hidden, w_soft)
        predict = tf.argmax(yhat, axis=1)
    
        # Relu state
        relu_mask = tf.greater(h, tf.zeros_like(h)).to_int32()
 
        reconstructed_X = tf.matmul(tf.multiply(h, relu_mask), tf.tranpose(w_hidden))
        reconstructed_images = sess.run(reconstruct_X, feed_dict={X:train_X, y:train_y})      
       
        save_images(reconstructed_images, save_path) 
        
if __name__ == "__main__":
    main()
