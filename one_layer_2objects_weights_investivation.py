import os
import scipy.misc
import tensorflow as tf
import numpy as np
import one_layer_2objects_train

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

num_hidden = 400
viz_dimention =(10, 40)
img_dim = (250, 250)

model = os.path.join("saved_model","one_hidden_2objects_L_400.ckpt")
graph = os.path.join("saved_model","one_hidden_2objects_L_400.ckpt.meta")

viz_path = os.path.join("visualizations", "L_400")

def normalize_contrast(matrix):
    return ((matrix - matrix.min())/np.ptp(matrix)*255).astype(np.uint8)

def save_images(images, fns, path, dim=img_dim):
    if not os.path.exists(path):
        os.mkdir(path)
    for i, (image,fn) in enumerate(zip(images, fns)):
        if not dim == None:
            image = np.asarray(tf.reshape(image, dim).eval())
        scipy.misc.imsave(os.path.join(path, fn), image)
        
def main():
    # restore session with variables
    if not os.path.exists(viz_path):
        os.mkdir(viz_path)

    saver = tf.train.import_meta_graph(graph)
    with tf.Session() as sess:
        saver.restore(sess, model)
        print "Session loaded."
		
        # restore the weights
        w_hidden = tf.get_collection(tf.GraphKeys.VARIABLES,"w_hidden")[0]
        w_soft = tf.get_collection(tf.GraphKeys.VARIABLES,"w_soft")[0]
        soft_bias = tf.get_collection(tf.GraphKeys.VARIABLES,"soft_bias")[0]
        w_mul = tf.matmul(w_hidden, w_soft) + soft_bias
		
        print (w_hidden.shape)
        print (w_soft.shape)
        
        # restore the training data
        train_X, test_X, train_y, test_y, train_fn, test_fn = one_layer_2objects_train.get_data()
 
        # just to pick a few to vizualize. image is huge
        to_viz = np.random.choice(range(train_X.shape[0]), 5)
        train_X = train_X[to_viz,:]
        train_y = train_y[to_viz,:]	
        train_fn = train_fn[to_viz]
        # Layer's sizes
        input_size = train_X.shape[1]
        hidden_size = num_hidden         
        output_size = train_y.shape[1]

        # Symbols
        X = tf.placeholder("float", shape=[None, input_size], name="X")
        y = tf.placeholder("float", shape=[None, output_size], name="y")

        # Forward propagation
        yhat, h, u = one_layer_2objects_train.forwardprop(X, w_hidden, w_soft, soft_bias)
        predict = tf.argmax(yhat, axis=1)
    
        # read out weights
        w_hidden_value = sess.run(w_hidden)	
        w_soft_value = sess.run(w_soft)
        w_mul_value = sess.run(w_mul)
       
		# hidden weights 
        save_images([w_hidden_value[:,i] for i in range(w_hidden_value.shape[1])], \
                    ["hidden_weights_" + str(i) + ".png" for i in range(w_hidden_value.shape[1])], \
                    os.path.join(viz_path, "hidden_weights"))  
					
        # hidden_weights * soft_weights
        save_images([w_mul_value[:,i] for i in range(w_mul_value.shape[1])], \
                    ["hidden_multi_soft_" + str(i) + ".png" for i in range(w_mul_value.shape[1])], \
                    os.path.join(viz_path, "hidden_multi_soft_"))  

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
        filter_weights = sess.run(h, feed_dict={X:train_X, y:train_y})

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
         
            save_images([np.concatenate([w_stacked,filter_weight], axis=1)], train_fn, \
                os.path.join(viz_path, "weights_of_filters_per_image"),dim=None)


        yhat_p = tf.placeholder("float", shape=[None, output_size], name="yhap_p")
        
        # w_hidden^T * a^hat * w_soft^T * yhat
        y_type = "u_sm"
        h_hat = tf.matmul(yhat, tf.transpose(w_soft))

        # Relu state
        relu_mask = tf.to_float(tf.greater(h_hat, tf.zeros_like(h_hat)))
        negative_relu_mask = tf.to_float(tf.equal(h_hat, tf.zeros_like(h_hat)))
        all_pass_mask = tf.to_float(tf.ones_like(h_hat))
 
        # reconstruct using forward relu state
        pos_relu_X = tf.matmul(tf.multiply(h_hat, relu_mask), tf.transpose(w_hidden))
        pos_relu_X = sess.run(pos_relu_X, feed_dict={X:train_X, y:train_y})      
       
        # reconstruct using negative forward relu state
        neg_relu_X = tf.matmul(tf.multiply(h_hat, negative_relu_mask), tf.transpose(w_hidden))
        neg_relu_X = sess.run(neg_relu_X, feed_dict={X:train_X, y:train_y})      
        
        # reconstruct regardless of relu state
        all_pass_X = tf.matmul(tf.multiply(h_hat, all_pass_mask), tf.transpose(w_hidden))
        all_pass_X = sess.run(all_pass_X, feed_dict={X:train_X, y:train_y})      
       
        # save all reconstructed images
        save_images(pos_relu_X,train_fn, \
            os.path.join(viz_path, "sigma_dot_a_"+y_type)) 
        save_images(neg_relu_X,train_fn, \
            os.path.join(viz_path, "sigma_dot_1-a_"+y_type)) 
        save_images(all_pass_X,train_fn, \
            os.path.join(viz_path, "sigma_"+y_type)) 
        
       
        # w_hidden^T * a^hat * w_soft^T * train_y 
        y_type = "miu_cg"
        h_hat_p = tf.matmul(yhat_p, tf.transpose(w_soft))

        # Relu state
        relu_mask = tf.to_float(tf.greater(h_hat_p, tf.zeros_like(h_hat_p)))
        negative_relu_mask = tf.to_float(tf.equal(h_hat_p, tf.zeros_like(h_hat_p)))
        all_pass_mask = tf.to_float(tf.ones_like(h_hat_p))
 
        # reconstruct using forward relu state
        pos_relu_X = tf.matmul(tf.multiply(h_hat_p, relu_mask), tf.transpose(w_hidden))
        pos_relu_X = sess.run(pos_relu_X, feed_dict={X:train_X, y:train_y, yhat_p:train_y})      
       
        # reconstruct using negative forward relu state
        neg_relu_X = tf.matmul(tf.multiply(h_hat_p, negative_relu_mask), tf.transpose(w_hidden))
        neg_relu_X = sess.run(neg_relu_X, feed_dict={X:train_X, y:train_y, yhat_p:train_y})      
        
        # reconstruct regardless of relu state
        all_pass_X = tf.matmul(tf.multiply(h_hat_p, all_pass_mask), tf.transpose(w_hidden))
        all_pass_X = sess.run(all_pass_X, feed_dict={X:train_X, y:train_y, yhat_p:train_y})      
       
        # save all reconstructed images
        save_images(pos_relu_X,train_fn, \
            os.path.join(viz_path, "sigma_dot_a_"+y_type)) 
        save_images(neg_relu_X,train_fn, \
            os.path.join(viz_path, "sigma_dot_1-a_"+y_type)) 
        save_images(all_pass_X,train_fn, \
            os.path.join(viz_path, "sigma_"+y_type)) 
 

        # w_hidden^T * a^hat * u
        # Relu state
        relu_mask = tf.to_float(tf.greater(u, tf.zeros_like(h)))
        negative_relu_mask = tf.to_float(tf.equal(u, tf.zeros_like(h)))
        all_pass_mask = tf.to_float(tf.ones_like(u))
 
        # reconstruct using forward relu state
        pos_relu_X = tf.matmul(tf.multiply(u, relu_mask), tf.transpose(w_hidden))
        pos_relu_X = sess.run(pos_relu_X, feed_dict={X:train_X, y:train_y})      
       
        # reconstruct using negative forward relu state
        neg_relu_X = tf.matmul(tf.multiply(u, negative_relu_mask), tf.transpose(w_hidden))
        neg_relu_X = sess.run(neg_relu_X, feed_dict={X:train_X, y:train_y})      
        
        # reconstruct regardless of relu state
        all_pass_X = tf.matmul(tf.multiply(u, all_pass_mask), tf.transpose(w_hidden))
        all_pass_X = sess.run(all_pass_X, feed_dict={X:train_X, y:train_y})      
       
        # save all reconstructed images
        save_images(pos_relu_X,train_fn, \
            os.path.join(viz_path, "sigma_dot_a_u")) 
        save_images(neg_relu_X,train_fn, \
            os.path.join(viz_path, "sigma_dot_1-a_u")) 
        save_images(all_pass_X,train_fn, \
            os.path.join(viz_path, "sigma_dot_u")) 
                
 
if __name__ == "__main__":
    main()
