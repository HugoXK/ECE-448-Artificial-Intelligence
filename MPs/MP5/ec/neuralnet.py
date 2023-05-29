# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques for the fall 2021 semester
# Modified by Kaiwen Hong for the Spring 2022 semester

"""
This is the main entry point for part 2. You should only modify code
within this file and neuralnet.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict
from torch.utils.data import DataLoader

global class1
global class2
class1 = 0
class2 = 19


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @param l(x,y) an () tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 2 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size

        self.h_size = 32
        # self.f2=nn.Sigmoid()#
        # self.f=nn.ReLU(inplace=True)#

        # self.f_in = nn.Linear(self.in_size,32)
        # self.relu = nn.ReLU(inplace=True)
        # self.f_out = nn.Linear(32,self.out_size)
        # self.logsoftmax = nn.LogSoftmax()
        # self.f1=nn.Linear(32,128)
        # self.f2=nn.Linear(128,32)

        self.net = nn.Sequential(
            nn.Linear(self.in_size, self.h_size),
            nn.ReLU(inplace=True),
            # self.f1,
            # self.relu,
            # self.f2,
            # self.relu,
            # # self.conv,
            # # self.mp,
            nn.Linear(self.h_size, self.out_size)
        )

        self.optimizer = optim.SGD(self.parameters(), lr=self.lrate)
    # def set_parameters(self, params):
    #     """ Sets the parameters of your network.

    #     @param params: a list of tensors containing all parameters of the network
    #     """
    #     raise NotImplementedError("You need to write this part!")

    # def get_parameters(self):
    #     """ Gets the parameters of your network.

    #     @return params: a list of tensors containing all parameters of the network
    #     """
    #     raise NotImplementedError("You need to write this part!")

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)
        y = self.net(x)
        # return self.logsoftmax(y)
        return y

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        # raise NotImplementedError("You need to write this part!")
        # return 0.0
        self.optimizer.zero_grad()
        loss_value = self.loss_fn(self.forward(x), y)
        loss_value.backward()
        self.optimizer.step()

        # loss_value.detach().cpu().numpy()
        return loss_value.item()


def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Fit a neural net. Use the full batch size.

    @param train_set: an (N, in_size) Tensors
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of epoches of training
    @param batch_size: size of each batch to train on. (default 100)

    NOTE: This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # raise NotImplementedError("You need to write this part!")
    # return [], [], None

    lrate = 1e-2
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = F.cross_entropy
    print(train_set.size())
    print(dev_set.size())
    in_size = train_set.size()[1]  # RGB image size 32*32*3 = 3072
    out_size = 2  # Number of Labels
    losses = []

    # normalized train set x = (z-mean)/std
    train_mean = torch.mean(train_set)
    train_std = torch.std(train_set)
    train_set = (train_set - train_mean) / train_std

    # normalized test set y = (z-mean)/std
    dev_mean = torch.mean(dev_set)
    dev_std = torch.std(dev_set)
    dev_set = (dev_set - dev_mean) / dev_std

    net = NeuralNet_ec(lrate, loss_fn, in_size, out_size)

    # train n_iter (number of batches)
    for i in range(n_iter):
        batch = train_set[i*batch_size:(i+1)*batch_size]
        label_batch = train_labels[i*batch_size:(i+1)*batch_size]
        # separate by batch, update parameters
        losses.append(net.step(batch, label_batch))

    yhats = np.zeros(len(dev_set))
    Fw = net.forward(dev_set).detach().numpy()
    # print(type(Fw))
    for i, r in enumerate(Fw):
        # taking the index of the maximum of the two outputs (argmax).
        yhats[i] = np.argmax(Fw[i])
    return losses, yhats, net


class NeuralNet_ec(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @param l(x,y) an () tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 2 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet_ec, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size  # 32*32*3
        self.out_size = out_size

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_size[0], 64, kernel_size = 3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64,128, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),     # inplace=True        
            nn.MaxPool2d(2, 2)
            
            )

        self.f=nn.ReLU()
        # nn.ReLU(inplace=True),nn.RReLU(),nn.Sigmoid(),nn.PReLU(),nn.Softplus(),nn.Tanh(),nn.Sigmoid(),nn.ReLU6()

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            self.f,
            

            nn.Conv2d(128, 128, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            self.f,
            
            
            nn.Conv2d(128, 128, kernel_size = 3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            self.f,
            
            
            # nn.Conv2d(128, 128, kernel_size = 3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # nn.Conv2d(64, 64, kernel_size = 3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            
            # # nn.Conv2d(64, 64, kernel_size = 3, stride=1, padding=1),
            # # nn.BatchNorm2d(64),
            # # nn.ReLU(inplace=True),

            # nn.Conv2d(64, 32, kernel_size = 3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2)
            )

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size = 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(64, 64, kernel_size = 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
            
        #     nn.Conv2d(64, 64, kernel_size = 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
            
        #     nn.Conv2d(64, 64, kernel_size = 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(64, 32, kernel_size = 3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
            
        #     nn.MaxPool2d(2, 2)
        #     )


        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            # self.layer4,
            )
        
        
        self.net = nn.Sequential(    
            self.conv,    
            nn.Flatten(),          
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
    
            nn.Linear(128, self.out_size)
            )
        
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.lrate)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lrate)

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        return self.net(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        # raise NotImplementedError("You need to write this part!")
        # return 0.0
        self.optimizer.zero_grad()
        loss_value = self.loss_fn(self.forward(x), y)
        loss_value.backward()
        self.optimizer.step()

        # loss_value.detach().cpu().numpy()
        return loss_value.item()


def fit_ec(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Fit a neural net. Use the full batch size.

    @param train_set: an (N, in_size) Tensors
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of epoches of training
    @param batch_size: size of each batch to train on. (default 100)

    NOTE: This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # raise NotImplementedError("You need to write this part!")
    # return [], [], None
    
    lrate = 0.1
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = F.cross_entropy

    # print(train_set)
    train_set=train_set.reshape([train_set.size()[0],3,32,32])
    dev_set=dev_set.reshape([dev_set.size()[0],3,32,32])

    # print(train_set)
    print(train_set.size())
    print(dev_set.size())
    # print(train_set)
    in_size = train_set.size()[1:3]  # RGB image size 32*32*3 = 3072
    out_size = 2  # Number of Labels
    losses = []

    # normalized train set x = (z-mean)/std
    train_mean = torch.mean(train_set)
    train_std = torch.std(train_set)
    train_set = (train_set - train_mean) / train_std

    # normalized test set y = (z-mean)/std
    dev_mean = torch.mean(dev_set)
    dev_std = torch.std(dev_set)
    dev_set = (dev_set - dev_mean) / dev_std

   
    net = NeuralNet_ec(lrate, loss_fn, in_size, out_size)

    # train n_iter (number of batches)
    for i in range(n_iter):
        batch = train_set[i*batch_size:(i+1)*batch_size]
        label_batch = train_labels[i*batch_size:(i+1)*batch_size]

        # separate by batch, update parameters
        losses.append(net.step(batch, label_batch))

    yhats = np.zeros(len(dev_set))
    Fw = net.forward(dev_set).detach().numpy()
    # print(type(Fw))
    for i, r in enumerate(Fw):
        # taking the index of the maximum of the two outputs (argmax).
        yhats[i] = np.argmax(Fw[i])
    return losses, yhats, net
