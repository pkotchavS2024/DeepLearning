import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        N, C, W_in = A.shape

        # Z is initially all zeros.
        Z= np.zeros((N,C, self.upsampling_factor*(W_in-1) + 1))
        '''
        Z is the same as A in the first 2 dimensions
        but in the 3rd dimension (output), every kth+1 element starting from and including 0th in Z, 
        corresponds to the elements in A's input

        k is the upsampling factor
        '''
        
        Z[:,:, 0:Z.shape[2]:self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        '''
        dLdA is the same as dLdZ in the first 2 dimensions
        but in the 3rd dimension, we subsample only the non-padding values, 
        which is the value every kth+1 element starting from and including 0th.

        k is the upsampling factor
        '''
        
        dLdA = dLdZ[:,:,0::self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        '''
        Z is the same as A in the first 2 dimensions
        but in the 3rd dimension, we subsample only the non-padding values, 
        which is the value every kth+1 element starting from and including 0th.

        k is the downsampling factor
        '''
        self.Win = A.shape[2]
        Z = A[:,:,0::self.downsampling_factor]
        assert Z.shape[2] == (self.Win - 1)//self.downsampling_factor + 1
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        '''
        dLdA is the same as dLdZ in the first 2 dimensions
        but in the 3rd dimension (output), every kth+1 element starting from and including 0th in dLdA, 
        corresponds to the elements in dLdZ's output

        k is the downsampling factor
        '''
        N,C,Wout = dLdZ.shape
        dLdA = np.zeros((N,C,self.Win))
        dLdA[:,:, 0:self.Win:self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        '''
        Similar to 1d version, Z is the same as A in the first 2 dimensions
        but in the 3rd, 4th dimension, the 2d array is padded row- and column-wise, 
        resulting in Hout x Wout 2d array with interweaved rows and columns of 0's 
        determined by the upsampling factor. The non-padding elements in Z 
        corresponds to A's input, and is filled by traversing the pre-allocated array
        of 0's using the upsampling factor to determine the index that should be populated
        with A's input

        k is the upsampling factor
        '''
        N, C, Hin, Win = A.shape
        Hout = self.upsampling_factor * (Hin - 1) + 1
        Wout = self.upsampling_factor * (Win - 1) + 1

        Z = np.zeros((N,C,Hout,Wout))
        Z[:,:, 0:Hout:self.upsampling_factor, 0:Wout:self.upsampling_factor] = A
       
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        '''
        Similar to 1d version, dLdA is the same as dLdZ in the first 2 dimensions
        but in the 3rd, 4th dimension, the 2d array is reversing the padding 
        resulting in Hin x Win 2d array. Using the upsampling factor k, we traverse dLdZ
        and subsample only the non-padding element (should be every kth+1 element starting from
        and including 0).

        '''
        N, C, Hout, Wout = dLdZ.shape
        dLdA = dLdZ[:,:,0:Hout:self.upsampling_factor, 0:Wout:self.upsampling_factor] 

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        '''
        Similar to 1d version, Z is the same as A in the first 2 dimensions
        but in the 3rd, 4th dimension, the 2d array is subsampling from the input 2d array.
        The elements that remain are determined by the downsampling factor (should be every kth+1 element starting from
        and including 0 for both row and col). i.e row: (0,0) (0,k) (0,2*k) and col: (0,0) (k,0) (2*k,0)  remains

        k is the downsampling factor
        '''
        N, C, self.Hin, self.Win = A.shape
        Z = A[:,:,0::self.downsampling_factor, 0::self.downsampling_factor]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        '''
        Similar to 1d version, dLdA is the same as dLdZ in the first 2 dimensions
        but in the 3rd, 4th dimension (output), every kth+1 element (row- and col- wise) starting from and including 0th elem in 
        pre-allocated zeros array dLdA, corresponds to the elements in dLdZ's output. i.e. dLdA's (0,0) -> dLdZ's (0,0),
        dLdA's (0,k) -> dLdZ's (0,1), dLdA's (k,0) -> dLdZ's (1,0) etc.

        k is the downsampling factor
        '''

        N,C = dLdZ.shape[0], dLdZ.shape[1]
        dLdA = np.zeros((N,C,self.Hin,self.Win))
        dLdA[:,:, 0:self.Hin:self.downsampling_factor, 0:self.Win:self.downsampling_factor] = dLdZ
        

        return dLdA
