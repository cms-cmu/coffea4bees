import numpy as np


class HLTBTagEmulator:
    def __init__(self, high_bin_edge, eff, eff_err):
        self.m_highBinEdge = high_bin_edge
        self.m_eff = eff
        self.m_effErr = eff_err
        self.m_rand = np.random.default_rng()  # Initialize a random number generator
        self.m_nBins = len(self.m_highBinEdge)

    def passJetThreshold(self, pt, bTagRand, smearFactor=0.0):
        eff = -99
        effErr = -99

        for iBin in range(self.m_nBins):
            if pt < self.m_highBinEdge[iBin]:
                eff = self.m_eff[iBin]
                effErr = self.m_effErr[iBin]
                break

        if eff < 0:
            eff = self.m_eff[-1]
            effErr = self.m_effErr[-1]
        if eff < 0:
            eff = 0

        thisTagEff = eff + effErr * smearFactor
        if thisTagEff > bTagRand:
            return True

        return False

    def passJet(self, pt, seedOffset=1.0, smearFactor=0.0):
        # Optionally set the seed, similar to the C++ code (commented out here)
        # seed = int(pt * seedOffset + pt)
        # np.random.seed(seed)  # Set seed for reproducibility, if needed

        bTagRand = self.m_rand.random()  # Generate a random number in [0, 1)
        return self.passJetThreshold(pt, bTagRand, smearFactor)

    def get_eff_vectorized(self, pt, smearFactor=0.0):
        """Vectorized efficiency lookup for an array of b-jet pTs."""
        pt_arr = np.asarray(pt)
        orig_shape = pt_arr.shape
        pt_flat = pt_arr.reshape(-1)

        idx = np.searchsorted(self.m_highBinEdge, pt_flat, side='right')
        idx = np.clip(idx, 0, self.m_nBins - 1)

        eff_arr = np.array(self.m_eff, dtype=np.float64)
        eff = np.maximum(eff_arr[idx], 0.0)

        if smearFactor != 0.0:
            err_arr = np.array(self.m_effErr, dtype=np.float64)
            eff = np.clip(eff + err_arr[idx] * smearFactor, 0.0, 1.0)

        return eff.reshape(orig_shape)

