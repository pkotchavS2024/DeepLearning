# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys
import torch

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        # 1st layer takes 24 channel input and output 8 channel; kernel width of 8 since it is simple scan that with 8 neurons in this layer
        self.conv1_in, self.conv1_out, self.conv1_kernel = 24, 8, 8
        # 2nd layer takes 8 channel input (== last conv layer's output channel) and output 16 channel adn kernel is 1 because it is a simple layer with 1 8 neuron block
        self.conv2_in, self.conv2_out, self.conv2_kernel = 8, 16, 1
        # 3rd layer takes 16 channel input (== last conv layer's out channel) and output 4 channel and kernel is 1 for similar reason above
        self.conv3_in, self.conv3_out, self.conv3_kernel = 16, 4, 1
        self.conv1 = Conv1d(in_channels = self.conv1_in , out_channels = self.conv1_out, kernel_size = self.conv1_kernel, stride =4)
        self.conv2 = Conv1d(in_channels = self.conv2_in , out_channels = self.conv2_out, kernel_size = self.conv2_kernel, stride =1)
        self.conv3 = Conv1d(in_channels = self.conv3_in , out_channels = self.conv3_out, kernel_size = self.conv3_kernel, stride =1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()] 

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        
        w1, w2, w3 = weights
        print(w1)
        self.conv1.conv1d_stride1.W = w1.transpose().reshape(self.conv1_out, self.conv1_kernel, self.conv1_in)
        self.conv1.conv1d_stride1.W = self.conv1.conv1d_stride1.W.transpose(0,2,1)
        self.conv2.conv1d_stride1.W = w2.transpose().reshape(self.conv2_out, self.conv2_kernel, self.conv2_in)
        self.conv2.conv1d_stride1.W = self.conv2.conv1d_stride1.W.transpose(0,2,1)
        self.conv3.conv1d_stride1.W = w3.transpose().reshape(self.conv3_out, self.conv3_kernel, self.conv3_in)
        self.conv3.conv1d_stride1.W = self.conv3.conv1d_stride1.W.transpose(0,2,1)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # 1st layer takes 24 channel input and output 2 channel; kernel width of 2 since it considers 2 input instances
        self.conv1_in, self.conv1_out, self.conv1_kernel = 24, 2, 2
        # 2nd layer takes 2 channel input and output 8 channel; kernel width of 2 since it takes in 2 blocks from prev
        self.conv2_in, self.conv2_out, self.conv2_kernel = 2, 8, 2
        # 3rd layer takes 8 channel input and output 4 channel; kernel width of 2 since it takes in 2 blocks from prev
        self.conv3_in, self.conv3_out, self.conv3_kernel = 8, 4, 2
        # stride of 2, 2, 1 = 4 --> total stride of the MLP
        self.conv1 = Conv1d(in_channels = self.conv1_in , out_channels = self.conv1_out, kernel_size = self.conv1_kernel, stride =2)
        self.conv2 = Conv1d(in_channels = self.conv2_in , out_channels = self.conv2_out, kernel_size = self.conv2_kernel, stride =2)
        self.conv3 = Conv1d(in_channels = self.conv3_in , out_channels = self.conv3_out, kernel_size = self.conv3_kernel, stride =1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()] 

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        # for all weights w1, w2, w3, since parameters are shared, carve out the block "top left" block for example. 
        # The dimension should correspond to input_channel * kernal size by output_channel
        # These are the weights that when multiply the input to each layer, would yield an output of appropriate output channel 
        w1_sliced = w1[:self.conv1_in*self.conv1_kernel,:self.conv1_out]
        self.conv1.conv1d_stride1.W = w1_sliced.transpose().reshape(self.conv1_out, self.conv1_kernel, self.conv1_in)
        self.conv1.conv1d_stride1.W = self.conv1.conv1d_stride1.W.transpose(0,2,1)

        w2_sliced = w2[:self.conv2_in*self.conv2_kernel,:self.conv2_out]
        self.conv2.conv1d_stride1.W = w2_sliced.transpose().reshape(self.conv2_out, self.conv2_kernel, self.conv2_in)
        self.conv2.conv1d_stride1.W = self.conv2.conv1d_stride1.W.transpose(0,2,1)

        w3_sliced = w3[:self.conv3_in*self.conv3_kernel,:self.conv3_out]
        self.conv3.conv1d_stride1.W = w3_sliced.transpose().reshape(self.conv3_out, self.conv3_kernel, self.conv3_in)
        self.conv3.conv1d_stride1.W = self.conv3.conv1d_stride1.W.transpose(0,2,1)


    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
