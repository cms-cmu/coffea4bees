# import logging

class TrigEmulator:
    def __init__(self, ht_thresholds, jet_thresholds, jet_multiplicities, btag_op_points, btag_multiplicities, nToys=100):
        self.m_htThresholds = ht_thresholds
        self.m_jetThresholds = jet_thresholds
        self.m_jetMultiplicities = jet_multiplicities
        self.m_bTagOpPoints = btag_op_points
        self.m_bTagMultiplicities = btag_multiplicities
        self.m_nToys = nToys

    def passTrig(self, offline_jet_pts, offline_btagged_jet_pts, ht=-1, seedOffset=1.0):
        # Ht Cut
        for iThres in range(len(self.m_htThresholds)):
            HLTHtCut = self.m_htThresholds[iThres]

            if ht > 0 and HLTHtCut:
                if not HLTHtCut.passHt(ht, seedOffset):
                    return False

        # Loop on all thresholds
        for iThres in range(len(self.m_jetThresholds)):
            HLTJet = self.m_jetThresholds[iThres]
            nJetsPassed = 0

            # Count passing jets
            for jet_pt in offline_jet_pts:
                if HLTJet.passJet(jet_pt, seedOffset):
                    nJetsPassed += 1

            # Impose trigger cut
            if nJetsPassed < self.m_jetMultiplicities[iThres]:
                return False

        # Apply BTag Operating Points
        for iThres in range(len(self.m_bTagOpPoints)):
            HLTBTag = self.m_bTagOpPoints[iThres]
            nJetsPassBTag = 0

            # Count passing jets
            for bjet_pt in offline_btagged_jet_pts:
                if HLTBTag.passJet(bjet_pt, seedOffset):
                    nJetsPassBTag += 1

            # Impose trigger cut
            if nJetsPassBTag < self.m_bTagMultiplicities[iThres]:
                return False

        return True

    # Used for calculating correlated decisions with input (ht and btagging) weights
    def passTrigCorrelated(self, offline_jet_pts, offline_btagged_jet_pts, ht, btag_rand, ht_rand, seedOffset=1.0, debug=False):

        # Ht Cut
        for iThres in range(len(self.m_htThresholds)):
            HLTHtCut = self.m_htThresholds[iThres]
            if debug: print(f" (TrigEmulator.passTrigCorrelated) ht={ht} ")
            if ht > 0 and HLTHtCut:
                if not HLTHtCut.passHtThreshold(ht, ht_rand[iThres]):
                    if debug: print(" (TrigEmulator.passTrigCorrelated) fail HLTHtCut")
                    return False

        # Loop on all thresholds
        for iThres in range(len(self.m_jetThresholds)):
            HLTJet = self.m_jetThresholds[iThres]
            nJetsPassed = 0

            # Count passing jets
            for jet_pt in offline_jet_pts:
                if HLTJet.passJet(jet_pt, seedOffset):
                    nJetsPassed += 1

            # Impose trigger cut
            if nJetsPassed < self.m_jetMultiplicities[iThres]:
                if debug: print(" (TrigEmulator.passTrigCorrelated) fail jetMultiplcities")
                return False

        # Apply BTag Operating Points

        for iThres in range(len(self.m_bTagOpPoints)):
            HLTBTag = self.m_bTagOpPoints[iThres]
            nJetsPassBTag = 0

            # Count passing jets
            for iBJet in range(len(offline_btagged_jet_pts)):
                bjet_pt = offline_btagged_jet_pts[iBJet]
                if HLTBTag.passJetThreshold(bjet_pt, btag_rand[iBJet][iThres]):
                    nJetsPassBTag += 1

            # Impose trigger cut
            if nJetsPassBTag < self.m_bTagMultiplicities[iThres]:
                if debug: print(" (TrigEmulator.passTrigCorrelated) fail btag multiplcicite")
                return False

        return True

    #  Calculate weight for trigger, average nPass over nToys
    def calcWeight(self, offline_jet_pts, offline_btagged_jet_pts, ht=-1):
        nPass = 0

        for iToy in range(self.m_nToys):
            # Count all events
            if self.passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, iToy):
                nPass += 1

        weight = float(nPass) / self.m_nToys
        # logging.info(f"TrigEmulator::calcWeight is {weight}")
        return weight

    def passTrigVectorized(
        self,
        jet_pts,
        jet_mask,
        bjet_pts,
        bjet_mask,
        ht_array,
        btag_rand,
        ht_rand,
        rng,
        smearFactor=0.0,
    ):
        """
        Vectorized evaluation of trigger decision across nToys for N events.

        Parameters
        ----------
        jet_pts : np.ndarray (N, n_jets)
        jet_mask : np.ndarray (N, n_jets)
        bjet_pts : np.ndarray (N, n_bjets)
        bjet_mask : np.ndarray (N, n_bjets)
        ht_array : np.ndarray (N,)
        btag_rand : np.ndarray (nToys, N, n_bjets, max_bcuts)
        ht_rand : np.ndarray (nToys, N, max_htcuts)
        rng : np.random.Generator
        smearFactor : float

        Returns
        -------
        pass_trig : np.ndarray (nToys, N) of bool
        """
        import numpy as np

        nToys, nEvents = ht_rand.shape[0], ht_rand.shape[1]
        pass_trig = np.ones((nToys, nEvents), dtype=bool)

        # 1. HT Cuts
        for iThres, HLTHtCut in enumerate(self.m_htThresholds):
            if HLTHtCut is not None:
                eff_ht = HLTHtCut.get_eff_vectorized(ht_array, smearFactor=smearFactor)
                pass_ht = eff_ht[np.newaxis, :] > ht_rand[:, :, iThres]
                ht_active = (ht_array > 0)[np.newaxis, :]
                pass_trig = pass_trig & np.where(ht_active, pass_ht, True)

        # 2. Jet Pt Cuts
        for iThres, HLTJet in enumerate(self.m_jetThresholds):
            mult = self.m_jetMultiplicities[iThres]
            eff_jets = HLTJet.get_eff_vectorized(jet_pts, smearFactor=smearFactor)
            rand_jets = rng.random((nToys, nEvents, jet_pts.shape[1]))
            passed_jets = (eff_jets[np.newaxis, :, :] > rand_jets) & jet_mask[np.newaxis, :, :]
            nJetsPassed = np.sum(passed_jets, axis=2)
            pass_trig = pass_trig & (nJetsPassed >= mult)

        # 3. BTag Operating Points
        for iThres, HLTBTag in enumerate(self.m_bTagOpPoints):
            mult = self.m_bTagMultiplicities[iThres]
            eff_bjets = HLTBTag.get_eff_vectorized(bjet_pts, smearFactor=smearFactor)
            passed_bjets = (eff_bjets[np.newaxis, :, :] > btag_rand[:, :, :, iThres]) & bjet_mask[np.newaxis, :, :]
            nJetsPassBTag = np.sum(passed_bjets, axis=2)
            pass_trig = pass_trig & (nJetsPassBTag >= mult)

        return pass_trig

