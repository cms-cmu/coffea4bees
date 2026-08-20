import numpy as np
import logging

class HLTHtEmulator:
    def __init__(self, high_bin_edge, eff, eff_err):
        self.m_highBinEdge = high_bin_edge
        self.m_eff = eff
        self.m_effErr = eff_err
        self.m_rand = np.random.default_rng()  # Initialize a random number generator
        self.m_nBins = len(self.m_highBinEdge)

    def passHt(self, ht, seedOffset=1.0, smearFactor=0.0):
        # Optionally set the seed, similar to the C++ code (commented out here)
        # seed = int(ht * seedOffset + ht)
        # np.random.seed(seed)  # Set seed for reproducibility, if needed

        htRand = self.m_rand.random()  # Generate a random number in [0, 1)
        return self.passHtThreshold(ht, htRand, smearFactor)

    def passHtThreshold(self, ht, htRand, smearFactor=0.0, debug=False):

        eff = -99
        effErr = -99
        if debug: print(f" HLTHtEmulator.passHtThreshold m_nBins = {self.m_nBins}")
        for iBin in range(self.m_nBins):
            if debug:
                print(f"{iBin} comparing {ht} to {self.m_highBinEdge[iBin]} ")

            if ht < self.m_highBinEdge[iBin]:
                eff = self.m_eff[iBin]
                effErr = self.m_effErr[iBin]
                if debug:
                    logging.debug(f"eff is {eff}")
                break

        if eff < 0:
            eff = self.m_eff[-1]
            effErr = self.m_effErr[-1]

        assert eff >= 0, "ERROR: eff < 0"

        thisTagEff = eff + effErr * smearFactor
        if debug:
            logging.debug(f"thisTagEff {thisTagEff} for ht = {ht}")
        return thisTagEff > htRand

    def get_eff_vectorized(self, ht, smearFactor=0.0):
        """Vectorized efficiency lookup for an array of HT values."""
        ht_arr = np.asarray(ht)
        orig_shape = ht_arr.shape
        ht_flat = ht_arr.reshape(-1)

        idx = np.searchsorted(self.m_highBinEdge, ht_flat, side='right')
        idx = np.clip(idx, 0, self.m_nBins - 1)

        eff_arr = np.array(self.m_eff, dtype=np.float64)
        eff = np.maximum(eff_arr[idx], 0.0)

        if smearFactor != 0.0:
            err_arr = np.array(self.m_effErr, dtype=np.float64)
            eff = np.clip(eff + err_arr[idx] * smearFactor, 0.0, 1.0)

        return eff.reshape(orig_shape)

