import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        prev_symbol = None 
        decoded_path = []  

        # Iterate over each time step
        for t in range(y_probs.shape[1]):
            this_prob = y_probs[:,t,0]
            max_prob_index = np.argmax(this_prob)  
            max_prob = this_prob[max_prob_index]

            path_prob *= max_prob

            if max_prob_index != blank:
                # Check if current symbol does not equate current symbol, add to path
                if self.symbol_set[max_prob_index - 1] != prev_symbol:
                    decoded_path.append(self.symbol_set[max_prob_index - 1])
                # Re-assign prev_symbol
                prev_symbol = self.symbol_set[max_prob_index - 1]
            else:
                # Reset if see blank
                prev_symbol = None 

        # Join the list to form the final decoded path string
        decoded_path = ''.join(decoded_path)
        
        return decoded_path, path_prob
        


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def initialize_paths(self, symbol_set, y_probs):
        # Initialize paths ending with symbol and their scores
        paths_symbol = []
        paths_symbol_score = {}
        for i, symb in enumerate(symbol_set):
            prob = y_probs[i + 1, 0, 0]
            path = symb
            paths_symbol.append(path)
            paths_symbol_score[path] = prob
        
        return paths_symbol, paths_symbol_score
    
    def prune(self,paths_blank, paths_symbol, paths_blank_score, paths_symbol_score, beam_width):
        scores = list(paths_blank_score.values()) + list(paths_symbol_score.values())
        scores.sort(reverse=True)
        cutoff = scores[beam_width - 1] if len(scores) >= beam_width else scores[-1]

        # Prune paths_blank
        pruned_paths_blank = []
        pruned_paths_blank_score = {}
        for path in paths_blank:
            if paths_blank_score[path] >= cutoff:
                pruned_paths_blank.append(path)
                pruned_paths_blank_score[path] = paths_blank_score[path]

        # Prune paths_symbol
        pruned_paths_symbol = []
        pruned_paths_symbol_score = {}
        for path in paths_symbol:
            if paths_symbol_score[path] >= cutoff:
                pruned_paths_symbol.append(path)
                pruned_paths_symbol_score[path] = paths_symbol_score[path]
        

        return pruned_paths_blank, pruned_paths_symbol, pruned_paths_blank_score, pruned_paths_symbol_score
    
    def extend_with_blank(self, pruned_paths_blank, pruned_paths_blank_score, y_probs, pruned_paths_symbol,pruned_paths_symbol_score,t):
        updated_blank_paths = {}
        blank = 0
        for path in pruned_paths_blank:
                score = pruned_paths_blank_score[path]
                # Extend with blank
                new_score = score * y_probs[blank, t, 0]
                updated_blank_paths[path] = updated_blank_paths.get(path, 0) + new_score

        for path in pruned_paths_symbol:
            score = pruned_paths_symbol_score[path]
            # Extend with blank
            new_score = score * y_probs[blank, t, 0]
            updated_blank_paths[path] = updated_blank_paths.get(path, 0) + new_score

        return updated_blank_paths
    
    def extend_with_symbols(self, pruned_paths_blank, pruned_paths_blank_score, y_probs, pruned_paths_symbol,pruned_paths_symbol_score,t):
        updated_symbol_paths = {}
        # Extend paths ending with symbol
        for path in pruned_paths_blank:
            score = pruned_paths_blank_score[path]
            for idx, symbol in enumerate(self.symbol_set):
                new_path = path + symbol
                new_score = score * y_probs[idx + 1, t, 0]
                updated_symbol_paths[new_path] = updated_symbol_paths.get(new_path, 0) + new_score

        for path in pruned_paths_symbol:
            score = pruned_paths_symbol_score[path]
            last_character = path[-1]
            for idx, symbol in enumerate(self.symbol_set):
                new_path = path if symbol == last_character else path + symbol
                new_score = score * y_probs[idx + 1, t, 0]
                updated_symbol_paths[new_path] = updated_symbol_paths.get(new_path, 0) + new_score

        return updated_symbol_paths
    
    def merge(self, blank_paths, blank_scores, paths_with_symbol, symbol_scores):
        # Merge paths and compute final scores
        final_scores = {}
        for path in blank_paths:
            final_scores[path] = blank_scores[path]
        for path in paths_with_symbol:
            final_scores[path] = final_scores.get(path, 0) + symbol_scores[path]

        return final_scores


    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
                        batch size for part 1 will remain 1, but if you plan to use your
                        implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None

        blank = 0

        # Initialize paths and their corresponding scores
        blank_paths = ['']
        blank_scores = {'': y_probs[blank, 0, 0]}

        paths_with_symbol, symbol_scores = self.initialize_paths(self.symbol_set, y_probs)

        for t in range(1, T):
            # prune
            pruned_paths_blank, pruned_paths_symbol, pruned_paths_blank_score, pruned_paths_symbol_score = self.prune(blank_paths,paths_with_symbol,blank_scores,symbol_scores,self.beam_width)
            
            # Extend paths ending with blank
            updated_blank_paths = self.extend_with_blank(pruned_paths_blank, pruned_paths_blank_score, y_probs, pruned_paths_symbol, pruned_paths_symbol_score,t)

            # Extend paths ending with symbol
            updated_symbol_paths = self.extend_with_symbols(pruned_paths_blank, pruned_paths_blank_score, y_probs, pruned_paths_symbol, pruned_paths_symbol_score,t)

            # Update paths and scores
            blank_paths = list(updated_blank_paths.keys())
            blank_scores = updated_blank_paths
            paths_with_symbol = list(updated_symbol_paths.keys())
            symbol_scores = updated_symbol_paths

        # Merge paths and compute final scores
        final_scores = self.merge(blank_paths, blank_scores, paths_with_symbol, symbol_scores)

        # Determine the best path
        best_path = max(final_scores, key=final_scores.get)
        merged_scores = final_scores

        return best_path, merged_scores
