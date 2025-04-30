import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        output_height = input_height - self.kernel + 1 
        output_width = input_width - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        # store A so that it can be used to determine location which max value was pulled from
        self.A = A
        # calculate and populate Z for all batches and channels at some 2d input index (i,j) at a time 
        # Element (:,:,i,j) is simply the max value in that patch of A given the kernel size
        for i in range(output_height):
            for j in range(output_width):
                Z[:, :, i, j] = np.max(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))
                
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        A = self.A
        kernel = self.kernel
        batch_size, in_channels, input_width, input_height = A.shape
        _, _, output_width, output_height = dLdZ.shape

        dLdA = np.zeros_like(A)

        for i in range(output_width):
            for j in range(output_height):
                # get the patches in A that contributed to this index (i,j) in Z
                curr_patch = A[:, :, i:i+kernel, j:j+kernel]
                # create a mask that specify which location in each patch the max value was drawn from 
                mask = (curr_patch == np.max(curr_patch, axis=(2, 3)).reshape(batch_size, in_channels,1,1))
                # pass the gradient appropriately
                dLdA[:, :, i:i+kernel, j:j+kernel] += mask * dLdZ[:, :, i, j].reshape(batch_size, in_channels,1,1)

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                # Calculate the mean for the current kernel
                Z[:, :, i, j] = np.mean(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_height = output_height + self.kernel - 1
        input_width = output_width + self.kernel - 1
        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        for i in range(output_height):
            for j in range(output_width):
                # The contribution to each patch in the input A; identical everywhere for meanpool
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += dLdZ[:, :, i, j].reshape(batch_size, in_channels, 1, 1) / (self.kernel * self.kernel)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        maxpool = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(maxpool)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        downsam_backward = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(downsam_backward)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        pooling = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(pooling)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        backward_downsample = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(backward_downsample)
        return dLdA
