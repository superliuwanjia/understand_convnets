import os
import glob
import random

import numpy as np
from scipy import misc


def to_one_hot(label):
    num_labels = len(np.unique(label))
    Y_onehot = np.eye(num_labels)[label]
    return Y_onehot


def from_one_hot(one_hot):
    return np.argmax(one_hot,axis=1)


def read_image_data(image_folder, image_mode, train_test_ratio=0.8, shuffle=1, rand_label=False):
    """ Read the data set and split them into training and test sets """
    X = []
    Label = []
    fns = []

    for image_path in glob.glob(os.path.join(image_folder, "*.png")):
        fns.append(os.path.basename(image_path))
        if rand_label:
            Label.append(np.random.choice(2))
        else:
            Label.append(int(os.path.basename(image_path).split("_")[0]))
        X.append(misc.imread(image_path, mode=image_mode).flatten())
    X = (np.array(X) / 255.).astype(np.float32)
    Label = np.array(Label)
    fns = np.array(fns)
    
    print(X.shape)
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
