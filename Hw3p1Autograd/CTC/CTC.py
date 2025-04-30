import numpy as np
import sys

sys.path.append("./")
from mytorch.autograd_engine import *
from mytorch.functional import *


class CTC(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        skip_connect = []
        for i in range(len(target)):
            skip_connect.append(0)

            if (i > 0 and target[i] != target[i-1]):
                skip_connect.append(1)
            else:
                skip_connect.append(0)
            
        skip_connect.append(0)
            
        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect
        

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------
        alpha[0][0] = logits[0, extended_symbols[0]]
        alpha[0][1] = logits[0, extended_symbols[1]]
        alpha[0][2:S] = 0
        for t in range(1,T):
            alpha[t][0] = alpha[t-1][0] * logits[t, extended_symbols[0]]
            for i in range(1,S):
                alpha[t][i] = alpha[t-1][i] + alpha[t-1][i-1]
                if(skip_connect[i]):
                    alpha[t][i] += alpha[t-1][i-2]
                alpha[t][i] *= logits[t, extended_symbols[i]]
        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        beta[T-1][S-1] = 1
        beta[T-1][S-2] = 1
        beta[T-1][0:S-2] = 0

        # Iterate from T-2 or (second to last index) until 0 but range is exclusive on the right so we put -1
        for t in range(T-2, -1, -1):
            beta[t][S-1] = beta[t+1][S-1] * logits[t+1, extended_symbols[S-1]]

            # Iterate from S-2 or (second to last index) until 0 but range is exclusive on the right so we put -1
            for i in range(S-2, -1, -1):
                beta[t][i] = beta[t+1][i] * logits[t+1, extended_symbols[i]] + beta[t+1][i+1] * logits[t+1, extended_symbols[i+1]]
                if i < S -3 and skip_connect[i+2]:
                    beta[t][i] += beta[t+1][i+2] * logits[t+1, extended_symbols[i+2]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        for t in range(T):
            sumgamma[t] = 0
            for i in range(S):
                gamma[t][i] = alpha[t][i] * beta[t][i]
                sumgamma[t] += gamma[t][i]

            for i in range(S):
                gamma[t][i] = gamma[t][i] / sumgamma[t]

        return gamma


class CTCLoss(object):

    def __init__(self, autograd_engine, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.

        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()
        self.autograd_engine = autograd_engine

        self.BLANK = BLANK
        self.ctc = CTC()

        # NOTE: Toggle using ctc_loss_backward version
        # or a version using more primitive operations
        self.USE_PRIMITIVE = True
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

                Computes the CTC Loss by calculating forward, backward, and
                posterior proabilites, and then calculating the avg. loss between
                targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
                        log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = np.empty(B, dtype=object)
        self.gammas = np.empty(B, dtype=object)


        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # Truncate
            trunc_target = self.target[batch_itr][:self.target_lengths[batch_itr]]
            trunc_logit = self.logits[:self.input_lengths[batch_itr], batch_itr, :]

            # Extend symbols
            target_extend, skip_connect = self.ctc.extend_target_with_blank(trunc_target)
            self.extended_symbols[batch_itr] = target_extend
            
            # Calcualte probability and posterior
            forward_prob = self.ctc.get_forward_probs(trunc_logit,target_extend,skip_connect)
            backward_prob = self.ctc.get_backward_probs(trunc_logit,target_extend,skip_connect)
            gamma = self.ctc.get_posterior_probs(forward_prob,backward_prob)
            # T = self.input_lengths[batch_itr]
            # S = len(target_extend)
            self.gammas[batch_itr] = gamma
            
            # compute loss
            for i in range(gamma.shape[1]):
                total_loss[batch_itr] -= np.sum(gamma[0:, i] * np.log(trunc_logit[:, target_extend[i]]))
        
           
        total_loss = np.sum(total_loss) / B
        print("Types and shapes of inputs:")
        print(f"logits: type={type(self.logits)}, shape={self.logits.shape}")
        print(f"input_lengths: type={type(input_lengths)}, shape={input_lengths.shape}")
        print(f"gammas: type={type(self.gammas)}, shape={self.gammas.shape}")
        print(f"extended_symbols: type={type(self.extended_symbols)}, length={len(self.extended_symbols)}")
        print("First extended_symbols element type:", type(self.extended_symbols[0]))
        print("First extended_symbols element shape:", self.extended_symbols[0].shape)
        # TODO: You must implement ctc_loss_backward
        self.autograd_engine.add_operation(
            inputs=[self.logits, input_lengths, self.gammas, self.extended_symbols],
            output=total_loss,
            gradients_to_update=[None, None, None, None],
            backward_operation=ctc_loss_backward,
        )
        return total_loss
       
