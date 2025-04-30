# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1

        

        # Initialize output Z
        Z = np.zeros((batch_size, self.out_channels, output_size))

        
        for i in range(output_size): 
            A_sub = A[:, :, i:i+self.kernel_size]  # (batch_size, in_channels, kernel_size)

            # Convolve with the filter for this output channel (over in_channels)
            Z[:, :, i] = np.tensordot(A_sub, self.W, axes=([1,2],[1,2])) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        batch_size, in_channels, input_size = self.A.shape
        _, out_channels, output_size = dLdZ.shape
        
        # Initialize gradients
        dLdA = np.zeros_like(self.A)  # Gradient with respect to input
        # dLdW = np.zeros_like(self.W)  # Gradient with respect to weights
        self.dLdb = np.sum(dLdZ, axis=(0, 2))


       
        for i in range(output_size):
            A_slice = self.A[:, :, i:i+self.kernel_size]  # Input slice (batch_size, in_channels, kernel_size)

            self.dLdW += np.tensordot(dLdZ[:, :, i], A_slice, axes=(0, 0))
            # Gradient with respect to input (dLdA)

        # Flip W 
        W_flip = np.flip(self.W, axis=2)

        # Padding is kernel size - 1
        padding = self.kernel_size - 1
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (padding, padding)), mode='constant', constant_values=0)

        # convolve padded dLdZ with the flipped weights 
        # calculate dLdA for each instance at each input at a time but across all channels
        for b in range(batch_size):
            for i in range(input_size):
                dLdA[b, :, i] = np.tensordot(dLdZ_padded[b, :, i:i + self.kernel_size], W_flip, axes=([0, 1], [0, 2]))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,weight_init_fn,bias_init_fn) 
        self.downsample1d = Downsample1d(downsampling_factor=stride) 

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant')

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)

        # Unpad the gradient 
        # account for 0 dimension with dLdA.shape[-1]-self.pad
        dLdA = dLdA[:, :, self.pad:dLdA.shape[-1]-self.pad]

        return dLdA
