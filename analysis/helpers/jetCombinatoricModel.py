import logging

import awkward as ak
import fsspec
import numpy as np
import yaml
from src.math_tools.random import Squares
from scipy.special import comb

class jetCombinatoricModel:
    def __init__(self, filename, cut='passPreSel', zero_npt=False, nbt=3, maxPseudoTags=12, lowpt_mode=False, used_stored_weights=False):
        """
        Initialize the jet combinatoric model with parameters from a file.

        :param filename: Path to the parameter file (txt or yaml format).
        :param cut: The cut to apply for the model (default is 'passPreSel').
        :param zero_npt: If True, will return zero pseudo-tags for all events.
        :param nbt: Number of baseline b-tags (default is 3).
        :param maxPseudoTags: Maximum number of pseudo-tags (default is 12).
        :param lowpt_mode: If True, model lowpt jets becoming lowpt tags (default is False).

        Standard mode: Models light jets becoming b-tags (3tag → 4tag)
        Lowpt mode: Models lowpt jets becoming lowpt tags (3tag+0lowpt → 3tag+≥1lowpt)

        Both modes use the same enhancement logic: enhance when total tags (nbt + npt) is even.
        """
        self.filename = filename
        self.cut = cut
        self.zero_npt = zero_npt
        self.nbt = nbt  # number of baseline b-tags (3 regular tags in both modes)
        self.maxPseudoTags = maxPseudoTags
        self.lowpt_mode = lowpt_mode
        self.used_stored_weights = used_stored_weights
        self.read_parameter_file()
        self._rng = Squares(("JCM", "pseudo tag"))

        logging.debug(f"JCM initialized in {'lowpt' if lowpt_mode else 'standard'} mode with cut={cut}")

    def read_parameter_file(self):
        # fsspec so the JCM can be read straight from EOS (root://), e.g. a production
        # roast's handoff copy; local paths behave exactly as with open().

        # f(nJet) x r^n JCM: pseudoTagProbSlope (alpha) and looseVetoFactor (r). Files without
        # these keys keep alpha = 0, r = 1, i.e. the default model.
        self.alpha = 0.0
        self.r = 1.0

        if self.filename.endswith('txt'):
            self.data = {}
            with fsspec.open(self.filename, 'rt') as lines:
                for line in lines:
                    words = line.split()
                    if not len(words): continue
                    if len(words) == 2:
                        self.data[words[0]] = float(words[1])
                    else:
                        self.data[words[0]] = ' '.join(words[1:])
            cut_suffix = f'_{self.cut}' if self.cut else ''
            self.p = self.data[f'pseudoTagProb{cut_suffix}']
            self.e = self.data[f'pairEnhancement{cut_suffix}']
            self.d = self.data[f'pairEnhancementDecay{cut_suffix}']
            self.t = self.data[f'threeTightTagFraction{cut_suffix}']
            self.alpha = float(self.data.get(f'pseudoTagProbSlope{cut_suffix}', self.data.get('pseudoTagProbSlope', 0.0)))
            self.r = float(self.data.get(f'looseVetoFactor{cut_suffix}', self.data.get('looseVetoFactor', 1.0)))

        else:
            with fsspec.open(self.filename, 'rt') as f:
                self.data = yaml.safe_load(f)
            try:
                cut_suffix = f'_{self.cut}' if self.cut else ''
                self.p = self.data.get(f'pseudoTagProb{cut_suffix}', self.data.get('pseudoTagProb'))
                self.e = self.data.get(f'pairEnhancement{cut_suffix}', self.data.get('pairEnhancement'))
                self.d = self.data.get(f'pairEnhancementDecay{cut_suffix}', self.data.get('pairEnhancementDecay'))
                self.t = self.data.get(f'threeTightTagFraction{cut_suffix}', self.data.get('threeTightTagFraction'))
                if self.p is None or self.e is None or self.d is None or self.t is None:
                    raise KeyError(f'pseudoTagProb{cut_suffix}')
                self.alpha = float(self.data.get(f'pseudoTagProbSlope{cut_suffix}', self.data.get('pseudoTagProbSlope', 0.0)))
                self.r = float(self.data.get(f'looseVetoFactor{cut_suffix}', self.data.get('looseVetoFactor', 1.0)))
                if "JCM_weights" in self.data:
                    self.JCM_weights = self.data[f'JCM_weights']

            except KeyError:
                logging.error(f'No {self.cut} key in JCM file {self.filename}. Keys are {list(self.data.keys()) if isinstance(self.data, dict) else "not a dict"}')
                raise

        logging.debug(f"JCM {self.filename}: pseudoTagProbSlope (alpha) = {self.alpha}, looseVetoFactor (r) = {self.r}")

    def __call__(self, num_untagged_jets, event=None):
        """
        Apply JCM weights to events.

        :param jets:
            - Standard mode: num_untagged_jets (jets that aren't b-tagged)
            - Lowpt mode: lowpt_jets (jets that could become lowpt tags)
        :param event: Optional event number for reproducible random generation
        :return: (w, npt) where w is the event weight and npt is number of pseudo-tags

        Physics:
        - Standard: 3 regular tags + pseudo-tags from light jets
        - Lowpt: 3 regular tags + lowpt tags from lowpt jets
        - Enhancement applies when total tags (3 + npt) is even in both cases
        """
        nEvent = len(num_untagged_jets)
        maxPseudoTags = self.maxPseudoTags
        nbt = self.nbt  # number of baseline b-tags (always 3)

        # Number of jets that could become pseudo-tags
        # Standard mode: light jets (nlt)
        # Lowpt mode: lowpt jets (also called nlt internally)
        # nlt = ak.to_numpy(ak.num(jets, axis=1))
        nlt = ak.to_numpy(num_untagged_jets)  # number of light jets

        # Pre-compute pseudo-tag probability table for all possible light jet counts
        # Use np.max with default value for empty arrays
        max_nlt = np.max(nlt, initial=0) if nlt.size > 0 else 0

        # Arrays to hold probabilities and cumulative probabilities
        # shape: (max_nlt+1, maxPseudoTags+1)
        all_probs = np.zeros((max_nlt+1, maxPseudoTags+1))
        all_cumulative_probs = np.zeros((max_nlt+1, maxPseudoTags+1))

        # Compute for all possible light jet counts (1 to max_nlt)
        for n_light in range(1, max_nlt + 1):
            # f(nJet): p(nSelJets) = p * (nSelJets/4)**alpha with nSelJets = n_light + nbt (alpha = 0: constant p)
            p = self.p * ((n_light + nbt) / 4.0) ** self.alpha if self.alpha else self.p

            # Calculate probability of zero pseudo-tags
            all_probs[n_light, 0] = self.t * (1-p)**n_light

            # Calculate for each number of pseudo-tags
            for npt in range(1, min(n_light + 1, maxPseudoTags + 1)):
                nt = nbt + npt  # total tagged jets
                nnt = n_light - npt  # non-tagged jets

                # Calculate binomial coefficient directly
                ncr = comb(n_light, npt, exact=True)

                # Calculate the probability
                w_npt = self.t * ncr * p**npt * (1-p)**nnt

                # Apply enhancement for even number of tags
                if (nt % 2) == 0:
                    w_npt *= 1 + self.e/n_light**self.d

                all_probs[n_light, npt] = w_npt

            # Pre-compute cumulative probabilities for each number of light jets
            # Exclude zero-tag probability from cumulative sums
            all_cumulative_probs[n_light] = np.cumsum(all_probs[n_light])


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

        # random number between nPseudoTagProb[0] and nPseudoTagProb.sum(axis=0)
        r = prob * w + zero_pt_probs
        r_reshaped  = r.reshape(1,nEvent).repeat(maxPseudoTags+1,0).T

        cumprobs_nonzero = event_cumulative_probs
        comparison = r_reshaped > cumprobs_nonzero
        npt = np.sum(comparison, axis=1)

        # Loose-veto factor: the returned weight times r**n_light. Applied after the npt sampling, which
        # uses the per-npt probabilities only (r = 1: weight unchanged; not applied to the zero_npt weight).
        if self.r != 1.0 and not self.zero_npt:
            w = w * self.r ** nlt

        # Check if we have JCM_weights stored and compare with calculated values
        if hasattr(self, 'JCM_weights') and self.used_stored_weights:
            logging.debug(f"Comparing calculated weights with stored JCM_weights for {'lowpt' if self.lowpt_mode else 'standard'} (length={len(self.JCM_weights)})")

            # JCM_weights[i] is the weight for nSelJets = i + 4, i.e. n_light = i + 1; compare every
            # stored entry up to max_nlt, including the r**n_light factor
            weight_table = total_weights * self.r ** np.arange(len(total_weights)) if self.r != 1.0 else total_weights
            compare_len = min(len(self.JCM_weights), max_nlt)

            for i in range(compare_len):
                if self.JCM_weights[i] > 0:
                    rel_diff = abs(weight_table[i+1] - self.JCM_weights[i]) / self.JCM_weights[i]
                    if rel_diff > 1e-9:
                        logging.error(f"Calculated weight for {i+1} light jets ({weight_table[i+1]:.12g}) "
                                      f"differs from stored weight ({self.JCM_weights[i]:.12g}) "
                                      f"by a relative {rel_diff:.3g}")
                    else:
                        logging.debug(f"Calculated weight for {i+1} light jets matches stored weight to 1e-9 for {'lowpt' if self.lowpt_mode else 'standard'}.")

            # Check if we need more weights than what's stored
            if max_nlt > len(self.JCM_weights):
                logging.warning(f"Some events have more light jets ({max_nlt}) than available in "
                            f"JCM_weights (length={len(self.JCM_weights)}). "
                            f"Using calculated weights for these events.")

        return w, npt
