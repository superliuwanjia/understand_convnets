import os
import scipy.misc

import tensorflow as tf
import numpy as np

import one_layer_2objects_train

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

model = os.path.join("saved_model","1layer_mlp_2objects_RGB.ckpt")
graph = os.path.join("saved_model","1layer_mlp_2objects_RGB.ckpt.meta")
viz_path = "visualizations"
def save_images(images, fns, path, dim=(250,250, 3)):
    if not os.path.exists(path):
        os.mkdir(path)

    for i, (image,fn) in enumerate(zip(images, fns)):
        image = image.reshape(dim)
        scipy.misc.imsave(os.path.join(path, fn), image)
        
def main():
    # restore session with variables

    saver = tf.train.import_meta_graph(graph)
    with tf.Session() as sess:
        saver.restore(sess, model)
        print "Session loaded."
        
 
        train_X, test_X, train_y, test_y, train_fn, test_fn = one_layer_2objects_train.get_data()
        # Layer's sizes
        x_size = train_X.shape[1]   # Number of input nodes
        h_size = 100                # Number of hidden nodes
        y_size = train_y.shape[1]   # Number of outcomes

        # Symbols
        X = tf.placeholder("float", shape=[None, x_size], name="x")
        y = tf.placeholder("float", shape=[None, y_size], name="y")

        # Weight initializations
        w_hidden = tf.get_collection(tf.GraphKeys.VARIABLES,"w_hidden")[0]
        w_soft = tf.get_collection(tf.GraphKeys.VARIABLES,"w_softmax")[0]

        # Forward propagation
        yhat, h = one_layer_2objects_train.forwardprop(X, w_hidden, w_soft)
        predict = tf.argmax(yhat, axis=1)
    
        # Relu state
        relu_mask = tf.to_float(tf.greater(h, tf.zeros_like(h)))
        negative_relu_mask = tf.to_float(tf.less(h, tf.zeros_like(h)))
        all_pass_mask = tf.to_float(tf.ones_like(h))
 
        # reconstruct using forward relu state
        pos_relu_X = tf.matmul(tf.multiply(h, relu_mask), tf.transpose(w_hidden))
        pos_relu_X = sess.run(pos_relu_X, feed_dict={X:train_X, y:train_y})      
       
        # reconstruct using negative forward relu state
        neg_relu_X = tf.matmul(tf.multiply(h, relu_mask), tf.transpose(w_hidden))
        neg_relu_X = sess.run(neg_relu_X, feed_dict={X:train_X, y:train_y})      
        
        # reconstruct regardless of relu state
        all_pass_X = tf.matmul(tf.multiply(h, relu_mask), tf.transpose(w_hidden))
        all_pass_X = sess.run(all_pass_X, feed_dict={X:train_X, y:train_y})      
       
        # save all reconstructed images
        save_images(pos_relu_X[:,0:pos_relu_X.shape[1]-1],train_fn, \
            os.path.join(viz_path, "pos_relu")) 
        save_images(neg_relu_X[:,0:neg_relu_X.shape[1]-1],train_fn, \
            os.path.join(viz_path, "neg_relu")) 
        save_images(all_pass_X[:,0:all_pass_X.shape[1]-1],train_fn, \
            os.path.join(viz_path, "all_pass")) 
        
        # visualize weights
        w = sess.run(w_hidden)
        save_images([w[:i][0:w.shape[0]-1] for i in w.shape[1]], \
                    [str(i) for i in range(w.shape[1]), os.path.join(viz_path, "weights"))  

        # visualize class templates
  
if __name__ == "__main__":
    main()
