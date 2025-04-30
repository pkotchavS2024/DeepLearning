import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features,in_features))
        self.b = np.zeros((out_features,1))

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = A.shape[0]
        # self.Ones help with creating the bias matrix by doing the outer product of bias vector and vector of ones
        self.Ones = np.ones((self.N,1))
        Z = self.A @ self.W.T + self.Ones * self.b.T 

        # Z should have dimension of N by C_out
        assert Z.shape == (self.N, self.W.shape[0])
        return Z

    def backward(self, dLdZ):
        """
        :dLdZ: Change in Loss w.r.t small changes in the output of immediate layer 
        :dLdW: Change in Loss w.r.t small changes in Weight
        :dLdb: Change in Loss w.r.t small changes in bias
        :return dLdA: Change in Loss w.r.t input
        """
        dLdA = dLdZ @ self.W  
        self.dLdW = dLdZ.T @ self.A  
        self.dLdb = dLdZ.T @ self.Ones

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA
