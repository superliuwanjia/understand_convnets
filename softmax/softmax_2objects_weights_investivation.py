import os
import scipy.misc
import tensorflow as tf
import numpy as np
import softmax_2objects_train

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

model = os.path.join("saved_model","softmax_2objects_RGB.ckpt")
graph = os.path.join("saved_model","softmax_2objects_RGB.ckpt.meta")
save_path = "softmax_weights"

def save_images(images, path, dim=(250,250,3)):
    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(images.shape[1]):
        name = ['class0.png', 'class1.png']
        image = np.asarray(tf.reshape(images[:,i], dim).eval())
        print (image.shape)
        scipy.misc.imsave(os.path.join(path, name[i]), image)
        
def main():
    # restore session with variables
    saver = tf.train.import_meta_graph(graph)
    with tf.Session() as sess:
        saver.restore(sess, model)
        print "Session loaded."
		
        w_soft = tf.get_collection(tf.GraphKeys.VARIABLES,"w_soft")[0]
        w_soft = w_soft[0:w_soft.shape[0]-1,:]
        print (w_soft.shape)

        save_images(w_soft, save_path) 
        
if __name__ == "__main__":
    main()
