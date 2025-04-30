import numpy as np
import scipy
import scipy.special

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


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
    """
    Tanh activation for scalar value z is a = (exp(z) - exp(-z))/(exp(z) + exp(-z))
    """
    # Z is a matrix of size N x C
    # A is a matrix of size N x C
    def forward(self, Z):

        self.A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))

        return self.A

    # dLdA should be a matrix size N x C
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

class GELU:
    """
    GELU activation for input z is a = 0.5 * z * (1 + erf(z/sqrt(2)))
    """

    # Z is a matrix of size N x C
    # A is a matrix of size N x C
    def forward(self, Z):

        self.A = 0.5 * Z * (1 + scipy.special.erf(Z/np.sqrt(2)))
        self.Z = Z
        return self.A
    
    # dLdA should be a matrix size N x C
    def backward(self, dLdA):

        dAdZ = 0.5 * (1 + scipy.special.erf(self.Z/np.sqrt(2))) + (self.Z/np.sqrt(2*np.pi)) * (np.exp(-0.5*np.power(self.Z,2)))
        dLdZ = dLdA * dAdZ

        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = e_Z / np.sum(e_Z, axis=1, keepdims=True)

        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N = dLdA.shape[0] 
        C = dLdA.shape[1] 

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros_like(dLdA) 

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = self.A[i, m] * (1 - self.A[i, m])
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = dLdA[i, :] @ J

        return dLdZ