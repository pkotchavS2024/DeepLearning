import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0]  # TODO
        self.M = np.sum(Z,axis=0) / self.N  # TODO
        self.V = np.sum(np.power((Z - self.M),2) / self.N, axis=0)

        if eval == False:
            # training mode
            self.NZ = (Z - np.ones((self.N,1)) * self.M) / (np.ones((self.N,1)) * np.sqrt(self.V + self.eps))
            self.BZ = self.NZ * (np.ones((self.N,1)) * self.BW) + np.ones((self.N,1)) * self.Bb 

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
            
            return self.BZ
        else:
            # inference mode
            NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            BZ = NZ * self.BW + self.Bb

        return BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0)
        self.dLdBb = np.sum(dLdBZ, axis=0)

        dLdNZ = dLdBZ * self.BW
        dLdV = np.sum(dLdNZ * (self.Z - self.M) * (-0.5) * np.power(self.V + self.eps, -3/2), axis=0)
        dLdM = np.sum(dLdNZ * (-1 / np.sqrt(self.V + self.eps)), axis=0) + dLdV * np.sum(-2 * (self.Z - self.M) / self.N, axis=0)


        dLdZ = (dLdNZ / np.sqrt(self.V + self.eps)) + (dLdV * 2 * (self.Z - self.M) / self.N) + (dLdM / self.N)


        return dLdZ
