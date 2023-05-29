# reader.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Mahir Morshed for the spring 2021 semester
"""
This file is responsible for providing functions for reading the files
"""
from os import listdir
import numpy as np
import pickle
import random
import torch

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_dataset(filename, class1, class2):
    A = unpickle(filename)
    coarse_labels = np.array(A[b'coarse_labels'])
    desired_indices = np.nonzero((coarse_labels == class1) | (coarse_labels == class2))[0]
    X = A[b'data'][desired_indices]
    Y = coarse_labels[desired_indices]
    test_size = int(0.25 * len(X)) # set aside 25% for testing
    X_test = X[:test_size]
    Y_test = Y[:test_size]
    X = X[test_size:]
    Y = Y[test_size:]

    Y = np.array([ float(Y[i] == class2) for i in range(len(Y))])
    Y_test = np.array([float(Y_test[i] == class2) for i in range(len(Y_test))])

    return X, Y, X_test, Y_test

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
