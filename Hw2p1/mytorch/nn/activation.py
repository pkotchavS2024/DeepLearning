import numpy as np

# Copy your Linear class from HW1P1 here
class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:
    """
    Sigmoid function of a scalar value is a = 1 / (1 + exp(-z))

    :forward: forward method takes in a batch of data Z of shape N × C (representing N samples
    where each sample has C features), and applies the activation function to Z to compute output
    A of shape N × C.

    :backward: backward method takes in dLdA, a measure of how the post-activations (output) affect
    the loss. Using this and the derivative of the activation function itself, the method calculates and
    returns dLdZ, how changes in pre-activation features (input) Z affect the loss L.
    """
    # Z is a matrix of size N x C
    # A is a matrix of size N x C
    def forward(self, Z):

        self.A = 1 / ( 1 + np.exp(-Z))

        return self.A

    # dLdA should be a matrix size N x C
    def backward(self, dLdA):

        dAdZ = self.A - (self.A * self.A)
        dLdZ = dLdA * dAdZ

        return dLdZ


class Tanh:

    def forward(self, Z):

        self.A = np.tanh(Z)

        return self.A

    def backward(self, dLdA):

        dAdZ = 1 - self.A * self.A
        dLdZ = dLdA * dAdZ

        return dLdZ


class ReLU:
    """
    ReLU activation for scalar value z is a = max(0,z)
    """

    # Z is a matrix of size N x C
    # A is a matrix of size N x C
    def forward(self, Z):

        self.A = Z * (Z > 0)

        return self.A
    
    # dLdA should be a matrix size N x C
    def backward(self, dLdA):

        dAdZ = 1 * (self.A > 0)
        dLdZ = dLdA * dAdZ

        return dLdZ
    
