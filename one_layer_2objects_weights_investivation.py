import os
import scipy.misc
import tensorflow as tf
import numpy as np
import one_layer_2objects_train

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

model = os.path.join("saved_model","one_hidden_2objects_RGB.ckpt")
graph = os.path.join("saved_model","one_hidden_2objects_RGB.ckpt.meta")

viz_path = "visualizations"

def save_images(images, fns, path, dim=(250,250, 3)):
    if not os.path.exists(path):
        os.mkdir(path)

    for i, (image,fn) in enumerate(zip(images, fns)):
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
        w_mul = tf.matmul(w_hidden, w_soft)
		
        print (w_hidden.shape)
        print (w_soft.shape)
        
        # restore the training data
        #train_X, test_X, train_y, test_y, train_fn, test_fn = one_layer_2objects_train.get_data()
		
        # Layer's sizes
        #input_size = train_X.shape[1]
        #hidden_size = 100            
        #output_size = train_y.shape[1]

        # Symbols
        #X = tf.placeholder("float", shape=[None, input_size], name="X")
        #y = tf.placeholder("float", shape=[None, output_size], name="y")

        # Forward propagation
        #yhat, h, h_before_relu = one_layer_2objects_train.forwardprop(X, w_hidden, w_soft)
        #predict = tf.argmax(yhat, axis=1)
    
        # visualize weights
		
		
		# hidden weights 
        #save_images([w_hidden[:,i] for i in range(w_hidden.shape[1])], \
                    #["hidden_weights_" + str(i) + ".png" for i in range(w_hidden.shape[1])], viz_path)  
					
        # hidden_weights * soft_weights
        save_images([w_mul[:,i] for i in range(w_mul.shape[1])], \
                    ["hidden_multi_soft_" + str(i) + ".png" for i in range(w_mul.shape[1])], viz_path)  

'''
        # visualize weights * I
        h_materialized = sess.run(h, feed_dict={X:train_X, y:train_y})
        save_images(h_materialized, train_fn, os.path.join(viz_path, "weights_of_filters_per_image"),dim=(10,10))

        yhat_p = tf.placeholder("float", shape=[None, y_size], name="yhap_p")

            
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
        save_images(pos_relu_X[:,0:pos_relu_X.shape[1]-1],train_fn, \
            os.path.join(viz_path, "sigma_dot_a_"+y_type)) 
        save_images(neg_relu_X[:,0:neg_relu_X.shape[1]-1],train_fn, \
            os.path.join(viz_path, "sigma_dot_1-a_"+y_type)) 
        save_images(all_pass_X[:,0:all_pass_X.shape[1]-1],train_fn, \
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
        save_images(pos_relu_X[:,0:pos_relu_X.shape[1]-1],train_fn, \
            os.path.join(viz_path, "sigma_dot_a_"+y_type)) 
        save_images(neg_relu_X[:,0:neg_relu_X.shape[1]-1],train_fn, \
            os.path.join(viz_path, "sigma_dot_1-a_"+y_type)) 
        save_images(all_pass_X[:,0:all_pass_X.shape[1]-1],train_fn, \
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
        save_images(pos_relu_X[:,0:pos_relu_X.shape[1]-1],train_fn, \
            os.path.join(viz_path, "sigma_dot_a_u")) 
        save_images(neg_relu_X[:,0:neg_relu_X.shape[1]-1],train_fn, \
            os.path.join(viz_path, "sigma_dot_1-a_u")) 
        save_images(all_pass_X[:,0:all_pass_X.shape[1]-1],train_fn, \
            os.path.join(viz_path, "sigma_dot_u")) 
'''       
                
 
if __name__ == "__main__":
    main()
