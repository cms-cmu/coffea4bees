import numpy as np
from src.physics.common import drClean
import awkward as ak

# For trigger emulation
def compute_emulation_vars(event, useOnlyTop4=False):

    event['Jet', 'muon_cleaned'] = drClean(event.Jet, event.selMuon)[1]
    event['Jet', 'ht_selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) < 2.4) & event.Jet.muon_cleaned
    event['Jet', 'pfht_selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) < 2.4)

    all_jets = event.Jet
    pfjetht   = ak.sum(all_jets[all_jets.pfht_selected].pt, axis=1)
    calojetht = ak.sum(all_jets[all_jets.ht_selected  ].pt, axis=1)


    # 2. Sort Jets for Trigger Checks
    # By b-tag (assume 'btagDeepFlavB' exists)
    # Marina: "sorted(all_jets, key=lambda x: x.bdisc, reverse=True)"
    # Note: Marina uses 'pn_b' for > 2018.
    #b_score_name = "btagDeepFlavB" if self.year <= 2018 else "pn_b"
    # Adjust if column name is different in your NANOAOD
    #if b_score_name not in all_jets.fields:
    #    # Fallback
    #    print("falling back to btagDeepFlavB\n")
    #    b_score_name = "btagDeepFlavB"
    #print("calculating SF with ",b_score_name,"\n")
    jets_by_b = all_jets[ak.argsort(all_jets.btagScore, ascending=False)]


    # Top 4 by b-tag are selected (for <= 2018 logic in Marina), or all (for > 2018)
    # Marina: "if self.year <= 2018: selected_jets = btag_sorted_jets[0:4] else: all_jets"
    # Then "sort the jets in pt"

    if useOnlyTop4: #year <= 2018:
        selected_jets = jets_by_b[:, :4] # Takes top 4
    else:
        selected_jets = all_jets

    jets_by_pt = selected_jets[ak.argsort(selected_jets.pt, ascending=False)]

    # We need at least 4 jets? trigger usually requires 4.
    # Handle events with fewer than 4 jets by padding or masking?
    # For SF calculation, usually we run on events passing selection (>=4 jets)
    # We'll use ak.pad_none to be safe
    jets_by_pt = ak.pad_none(jets_by_pt, 4, clip=True)

    # We need to fill None values (missing jets) with a value that yields 0 efficiency (e.g. 0.0)
    # to ensure events with < 4 jets have 0 trigger efficiency
    pt1 = ak.fill_none(jets_by_pt[:, 0].pt, 0.0)
    pt2 = ak.fill_none(jets_by_pt[:, 1].pt, 0.0)
    pt3 = ak.fill_none(jets_by_pt[:, 2].pt, 0.0)
    pt4 = ak.fill_none(jets_by_pt[:, 3].pt, 0.0)

    scores_sorted = jets_by_b.btagScore
    scores_sorted = ak.pad_none(scores_sorted, 4, clip=True)
    # Fill missing b-tag scores with -1 or 0 (assuming low score = low efficiency)
    b1 = ak.fill_none(scores_sorted[:, 0], 0.0)
    b2 = ak.fill_none(scores_sorted[:, 1], 0.0)
    b3 = ak.fill_none(scores_sorted[:, 2], 0.0)
    b4 = ak.fill_none(scores_sorted[:, 3], 0.0)

    btagTMean = np.arctanh(0.5*(b1 + b2)) #ak.zeros_like(b1) # Placeholder for post-run2

    event['trigEm'] = ak.zip({"pfjetht"   : pfjetht,
                              "calojetht" : calojetht,
                              "pt1"       : pt1,
                              "pt2"       : pt2,
                              "pt3"       : pt3,
                              "pt4"       : pt4,
                              "b1"        : b1,
                              "b2"        : b2,
                              "b3"        : b3,
                              "b4"        : b4,
                              "btagTMean" : btagTMean,
                              })
