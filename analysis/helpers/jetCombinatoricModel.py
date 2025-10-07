import logging

import awkward as ak
import numpy as np
import yaml
from src.math_tools.random import Squares
from scipy.special import comb


class jetCombinatoricModel:
    def __init__(self, filename, cut='passPreSel', zero_npt=False, nbt=3, maxPseudoTags=12):
        """
        Initialize the jet combinatoric model with parameters from a file.
        :param filename: Path to the parameter file (txt or yaml format).
        :param
        cut: The cut to apply for the model (default is 'passPreSel').
        :param zero_npt: If True, will return zero pseudo-tags for all events.
        :param nbt: Number of required b-tags (default is 3).
        :param maxPseudoTags: Maximum number of pseudo-tags (default is 12).
        """
        self.filename = filename
        self.cut = cut
        self.zero_npt = zero_npt
        self.nbt = nbt  # number of required b-tags
        self.maxPseudoTags = maxPseudoTags
        self.read_parameter_file()
        self._rng = Squares(("JCM", "pseudo tag"))

    def read_parameter_file(self):

        if self.filename.endswith('txt'):
            self.data = {}
            with open(self.filename, 'r') as lines:
                for line in lines:
                    words = line.split()
                    if not len(words): continue
                    if len(words) == 2:
                        self.data[words[0]] = float(words[1])
                    else:
                        self.data[words[0]] = ' '.join(words[1:])

            self.p = self.data[f'pseudoTagProb_{self.cut}']
            self.e = self.data[f'pairEnhancement_{self.cut}']
            self.d = self.data[f'pairEnhancementDecay_{self.cut}']
            self.t = self.data[f'threeTightTagFraction_{self.cut}']

        else:
            self.data = yaml.safe_load(open(self.filename, 'r'))
            try:
                self.p = self.data[f'pseudoTagProb_{self.cut}']
                self.e = self.data[f'pairEnhancement_{self.cut}']
                self.d = self.data[f'pairEnhancementDecay_{self.cut}']
                self.t = self.data[f'threeTightTagFraction_{self.cut}']
                if "JCM_weights" in self.data:
                    self.JCM_weights = self.data[f'JCM_weights']

            except KeyError:
                logging.error(f'No {self.cut} key in JCM file. Keys are {self.data.keys()}')

    def __call__(self, untagged_jets, event=None):
        
        nEvent = len(untagged_jets)
        maxPseudoTags = self.maxPseudoTags
        nbt = self.nbt  # number of required b-tags
        nlt = ak.to_numpy(ak.num(untagged_jets, axis=1))  # number of light jets
        
        # Pre-compute pseudo-tag probability table for all possible light jet counts
        # Use np.max with default value for empty arrays
        max_nlt = np.max(nlt, initial=0) if nlt.size > 0 else 0
        
        # Arrays to hold probabilities and cumulative probabilities
        # shape: (max_nlt+1, maxPseudoTags+1)
        all_probs = np.zeros((max_nlt+1, maxPseudoTags+1))
        all_cumulative_probs = np.zeros((max_nlt+1, maxPseudoTags+1))
        
        # Compute for all possible light jet counts (1 to max_nlt)
        for n_light in range(1, max_nlt + 1):
            # Calculate probability of zero pseudo-tags
            all_probs[n_light, 0] = self.t * (1-self.p)**n_light

            # Calculate for each number of pseudo-tags
            for npt in range(1, min(n_light + 1, maxPseudoTags + 1)):
                nt = nbt + npt  # total tagged jets
                nnt = n_light - npt  # non-tagged jets
                
                # Calculate binomial coefficient directly
                ncr = comb(n_light, npt, exact=True)
                
                # Calculate the probability
                w_npt = self.t * ncr * self.p**npt * (1-self.p)**nnt
                
                # Apply enhancement for even number of tags
                if (nt % 2) == 0:
                    w_npt *= 1 + self.e/n_light**self.d
                        
                all_probs[n_light, npt] = w_npt
            
            # Pre-compute cumulative probabilities for each number of light jets
            # Exclude zero-tag probability from cumulative sums
            all_cumulative_probs[n_light, 0] = 0  # Start with zero
            all_cumulative_probs[n_light, 1:] = np.cumsum(all_probs[n_light, 1:])
        
        # Calculate total weights (sum of all probabilities except zero pseudo-tags)
        total_weights = np.sum(all_probs[:, 1:], axis=1)
        
        # Vectorized lookup for each event
        w = total_weights[nlt]

        # Get zero pseudo-tag probabilities for each event
        zero_pt_probs = all_probs[nlt, 0]
        if self.zero_npt:
            w = zero_pt_probs
        
        # Get the appropriate cumulative probabilities for each event
        # This avoids recalculating sums for each event
        event_cumulative_probs = all_cumulative_probs[nlt]  # shape: (nEvent, maxPseudoTags+1)
        
        # Generate random numbers and determine number of pseudo-tags
        if event is None:
            prob = np.random.uniform(0, 1, size=nEvent)
        else:
            prob = self._rng.uniform(event, 0, 1)
        
        # Calculate total probability (zero + non-zero)
        total_prob = zero_pt_probs + w
        
        # Scale random numbers to total probability space
        r = prob * total_prob
        
        # If random value is less than zero-tag probability, assign 0 tags
        # Otherwise compare with cumulative probabilities
        npt = np.zeros(nEvent, dtype=int)
        
        # Handle events where r is greater than zero-tag probability
        nonzero_mask = r > zero_pt_probs
        
        if np.any(nonzero_mask):
            # For these events, compare with the cumulative probabilities to determine npt
            r_nonzero = r[nonzero_mask].reshape(-1, 1)
            cumprobs_nonzero = event_cumulative_probs[nonzero_mask]
            comparison = r_nonzero > cumprobs_nonzero
            npt[nonzero_mask] = np.sum(comparison, axis=1)

        # Check if we have JCM_weights stored and compare with calculated values
        if hasattr(self, 'JCM_weights'):
            # JCM_weights is a fixed list of 14 elements
            logging.debug(f"Comparing calculated weights with stored JCM_weights (fixed length={len(self.JCM_weights)})")
            
            # Only compare up to the minimum length or max_nlt, whichever is smaller
            compare_len = min(len(self.JCM_weights), len(total_weights[1:]))
            
            # Loop through relevant indices (starting from 1, since 0 is for jets with 0 light jets)
            for i in range(1, compare_len):
                if total_weights[i+1] > 0 and self.JCM_weights[i] > 0:
                    rel_diff = abs(total_weights[i+1] - self.JCM_weights[i]) / self.JCM_weights[i]
                    if rel_diff > 0.01:  # 1% threshold
                        logging.warning(f"Calculated weight for {i} light jets ({total_weights[i+1]:.6f}) "
                                    f"differs from stored weight ({self.JCM_weights[i]:.6f}) "
                                    f"by {rel_diff*100:.2f}%")
            
            # Check if we need more weights than what's stored
            if max_nlt >= len(self.JCM_weights):
                logging.warning(f"Some events have more light jets ({max_nlt}) than available in "
                            f"JCM_weights (length={len(self.JCM_weights)}). "
                            f"Using calculated weights for these events.")

        return w, npt