import os
import glob
import random

import numpy as np
from scipy import misc

import keras.datasets.mnist

def to_one_hot(label):
    num_labels = len(np.unique(label))
    Y_onehot = np.eye(num_labels)[label]
    return Y_onehot

def from_one_hot(one_hot):
    return np.argmax(one_hot,axis=1)

def read_image_data(image_folder, image_mode, train_test_ratio=0.8, shuffle=1):
    """ Read the data set and split them into training and test sets """
    X = []
    Label = []
    fns = []
	
    for image_path in glob.glob(os.path.join(image_folder, "*.png")):
        fns.append(os.path.basename(image_path))
        Label.append(int(os.path.basename(image_path).split("_")[0]))
        image = X.append(misc.imread(image_path, mode=image_mode).flatten())
    X = np.array(X) / 255.
    Label = np.array(Label)
    fns = np.array(fns)
    
    print X.shape	
    # Convert into one-hot vectors
    Y_onehot = to_one_hot(Label)
    
    all_index = np.arange(X.shape[0])
    for _ in range(shuffle):
        np.random.shuffle(all_index)
    X = X[all_index, :]
    Y_onehot = Y_onehot[all_index, :]
    fns = fns[all_index]

    index_cutoff = int(X.shape[0] * train_test_ratio)

    return X[0:index_cutoff, :], X[index_cutoff:, :], \
           Y_onehot[0:index_cutoff, :], Y_onehot[index_cutoff:, :], \
           fns[0:index_cutoff], fns[index_cutoff:]

def mnist_2_class():
    """ Read mnist dataset with only 0 and 1s """
    X_train, X_test, Y_train, Y_test, fn_train, fn_test = mnist()

    index_train = np.where((from_one_hot(Y_train).flatten() == 0) | 
                           (from_one_hot(Y_train).flatten() == 1))
    index_test = np.where((from_one_hot(Y_test).flatten() == 0) | 
                           (from_one_hot(Y_test).flatten() == 1))
 
    return X_train[index_train,:], X_test[index_test,:], \
           Y_train[index_train,:], Y_test[index_test,:], \
           fn_train[index_train], fn_test[index_test]



def mnist():
    """ mnist dataset """
    # read raw data in
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((X_test.shape[0], np.prod(X_test.shape[1:])))

    # pretend file names, class_index.png
    fn_train = y_train.flatten().tolist()
    fn_train = np.array([str(label)+"_"+str(i)+".png" for i, label in enumerate(fn_train)])

    fn_test = y_test.flatten().tolist()
    fn_test = np.array([str(label)+"_"+str(i)+".png" for i, label in enumerate(fn_test)])

    # to one hot encoding
    Y_train = to_one_hot(y_train)
    Y_test = to_one_hot(y_test)

    # shuffle
    index_train = range(X_train.shape[0])
    index_test = range(X_test.shape[0])

    random.shuffle(index_train)
    random.shuffle(index_test)
    
    return X_train[index_train,:], X_test[index_test,:], \
           Y_train[index_train,:], Y_test[index_test,:], \
           fn_train[index_train], fn_test[index_test]

