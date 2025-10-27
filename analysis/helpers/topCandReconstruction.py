import awkward as ak
import numpy as np
import numba
# from memory_profiler import profile


mW, mt = 80.4, 173.0

@numba.njit
def find_tops_kernel(events_jets, builder):
    """Search for top quarks

       All quadjet events will have well defined xWt0, a top candidate where all three jets are allowed to be candidate jets.
    """

    for jets in events_jets:
        nJets = len(jets)

        if nJets < 3: continue

        builder.begin_list()

        # Pre-calculate sets for faster checks
        valid_pair_indices = [(ib, ij) for ib in range(0, 3) for ij in range(2, nJets) if ib != ij]
        valid_triplet_indices = [(ib, ij, il) for ib in range(0, 3) for ij in range(2, nJets) for il in range(2, nJets) if len({ib, ij, il}) == 3]

        for ib in range(0, 3):
            for ij in range(2, nJets):
                if (ib, ij) not in valid_pair_indices:
                    continue

                if jets[ib].btagScore < jets[ij].btagScore:
                    continue

                for il in range(2, nJets):
                    if (ib, ij, il) not in valid_triplet_indices:
                        continue

                    # don't consider W pairs where j is more b-like than b.
                    if jets[ij].btagScore < jets[il].btagScore:
                        continue

                    builder.begin_tuple(3)
                    builder.index(0).integer(ib)
                    builder.index(1).integer(ij)
                    builder.index(2).integer(il)
                    builder.end_tuple()

        builder.end_list()

    return builder

# @profile
def find_tops_kernel_slow(events_jets, builder):
    """Search for valid 4-lepton combinations from an array of events * leptons {charge, ...}

    A valid candidate has two pairs of leptons that each have balanced charge
    Outputs an array of events * candidates {indices 0..3} corresponding to all valid
    permutations of all valid combinations of unique leptons in each event
    (omitting permutations of the pairs)
    """
    for jets in events_jets:
        nJets = len(jets)

        if nJets < 3: continue

        builder.begin_list()

        # Pre-calculate sets for faster checks
        valid_pair_indices = [(ib, ij) for ib in range(0, 3) for ij in range(2, nJets) if ib != ij]
        valid_triplet_indices = [(ib, ij, il) for ib in range(0, 3) for ij in range(2, nJets) for il in range(2, nJets) if len({ib, ij, il}) == 3]

        for ib in range(0, 3):
            for ij in range(2, nJets):
                if (ib, ij) not in valid_pair_indices:
                    continue

                if jets[ib].btagScore < jets[ij].btagScore:
                    continue

                for il in range(2, nJets):
                    if (ib, ij, il) not in valid_triplet_indices:
                        continue

                    # don't consider W pairs where j is more b-like than b.
                    if jets[ij].btagScore < jets[il].btagScore:
                        continue

                    builder.begin_tuple(3)
                    builder.index(0).integer(ib)
                    builder.index(1).integer(ij)
                    builder.index(2).integer(il)
                    builder.end_tuple()

        builder.end_list()

    return builder

def find_tops(events_jets):

    # if ak.backend(events_jets) == "typetracer":
    #    raise Exception("typetracer")
    #    # here we fake the output of find_4lep_kernel since
    #    # operating on length-zero data returns the wrong layout!
    #    ak.typetracer.length_zero_if_typetracer(events_jets.btagScore) # force touching of the necessary data
    #    return ak.Array(ak.Array([[(0,0,0)]]).layout.to_typetracer(forget_length=True))
    return find_tops_kernel(events_jets, ak.ArrayBuilder()).snapshot()


def find_tops_slow(events_jets):
    # if ak.backend(events_leptons) == "typetracer":
    #    # here we fake the output of find_4lep_kernel since
    #    # operating on length-zero data returns the wrong layout!
    #    ak.typetracer.length_zero_if_typetracer(events_leptons.charge) # force touching of the necessary data
    #    return ak.Array(ak.Array([[(0,0,0,0)]]).layout.to_typetracer(forget_length=True))
    return find_tops_kernel_slow(events_jets, ak.ArrayBuilder()).snapshot()


def find_tops_no_numba(events_jest):
    from src.math_tools.partition import Partition
    sizes = ak.num(events_jest)
    combs = []
    for i in range(ak.max(sizes) + 1):
        parts = Partition(i, 1, 3).combination[0]
        if len(parts) > 0:
            parts = parts[(parts[:, :, 0] < 3) & (parts[:, :, 1] > 1), :]
        combs.append(parts)
    combs = ak.Array(combs)[sizes]
    j0 = events_jest[combs[:, :, 0]]
    j1 = events_jest[combs[:, :, 1]]
    j2 = events_jest[combs[:, :, 2]]
    combs = combs[
        (j0.btagScore >= j1.btagScore)
        & (j1.btagScore >= j2.btagScore),
        :,
    ]
    return combs

def dumpTopCandidateTestVectors(event, logging, chunk, nEvent):

#    logging.info(f'{chunk}\n\n')
#    logging.info(f'{chunk} self.input_jet_pt  = {[event[iE].Jet[event[iE].Jet.selected].pt.tolist() for iE in range(nEvent)]}')
#    logging.info(f'{chunk} self.input_jet_eta = {[event[iE].Jet[event[iE].Jet.selected].eta.tolist() for iE in range(nEvent)]}')
#    logging.info(f'{chunk} self.input_jet_phi = {[event[iE].Jet[event[iE].Jet.selected].phi.tolist() for iE in range(nEvent)]}')
#    logging.info(f'{chunk} self.input_jet_mass = {[event[iE].Jet[event[iE].Jet.selected].mass.tolist() for iE in range(nEvent)]}')
#    logging.info(f'{chunk} self.input_jet_btagScore = {[event[iE].Jet[event[iE].Jet.selected].btagScore.tolist() for iE in range(nEvent)]}')
#    logging.info(f'{chunk} self.input_jet_bRegCorr = {[event[iE].Jet[event[iE].Jet.selected].bRegCorr.tolist() for iE in range(nEvent)]}')
#    logging.info(f'{chunk} self.output_xbW = {[event[iE].xbW for iE in range(nEvent)]}')
#    logging.info(f'{chunk} self.output_xW = {[event[iE].xW for iE in range(nEvent)]}')
#    logging.info(f'{chunk}\n\n')

    print(f'{chunk}\n\n')
    print(f'{chunk} self.input_jet_pt            = {[event[iE].selJet.pt  .tolist() for iE in range(nEvent)]}')
    print(f'{chunk} self.input_jet_eta           = {[event[iE].selJet.eta .tolist() for iE in range(nEvent)]}')
    print(f'{chunk} self.input_jet_phi           = {[event[iE].selJet.phi .tolist() for iE in range(nEvent)]}')
    print(f'{chunk} self.input_jet_mass          = {[event[iE].selJet.mass.tolist() for iE in range(nEvent)]}')
    print(f'{chunk} self.input_jet_btagScore = {[event[iE].selJet.btagScore.tolist() for iE in range(nEvent)]}')
    print(f'{chunk} self.input_jet_bRegCorr      = {[event[iE].selJet.bRegCorr.tolist() for iE in range(nEvent)]}')
    print(f'{chunk} self.output_xbW              = {[event[iE].xbW for iE in range(nEvent)]}')
    print(f'{chunk} self.output_xW               = {[event[iE].xW for iE in range(nEvent)]}')
    print(f'{chunk}\n\n')



def buildTop(input_jets, top_cand_idx):
    """ Takes indices of jets and returns reconstructed top candidate
    """

    # Extract jets based on indices
    try:
        b, j, l = input_jets[top_cand_idx["0"]], input_jets[top_cand_idx["1"]], input_jets[top_cand_idx["2"]]
    except IndexError as e:
        raise ValueError(f"Index error while accessing input_jets: {e}")


    # Compute W properties
    W_p = j + l
    xW = (W_p.mass - mW) / (0.10 * W_p.mass)
    pW = W_p * (mW / W_p.mass)

    mbW = (b + pW).mass

    # smaller resolution term because there are fewer degrees of freedom. FWHM=25GeV, about the same as mW
    xbW = (mbW - mt) / (0.05 * mbW)

    rec_top_cands = ak.zip({
        "b": b,
        "j": j,
        "l": l,
        "xW": xW,
        "xbW": xbW,
        "mbW": mbW,
        "W": ak.zip({
            "p": W_p,
            "pW": pW,
            "j": j,
            "l": l
        })
    })

    # Sort and select the best candidate
    rec_top_cands = rec_top_cands[ak.argsort(rec_top_cands.xW ** 2 + rec_top_cands.xbW ** 2, axis=1, ascending=True)]

    top_cand = rec_top_cands[:,0]
    top_cand["p"] = top_cand.b + top_cand.j + top_cand.l
    top_cand["xt"] = (top_cand.p.mass - mt) / (0.10 * top_cand.p.mass)
    top_cand["xWt"] = np.sqrt(top_cand.xW ** 2 + top_cand.xt ** 2)
    top_cand["xWbW"] = np.sqrt(top_cand.xW ** 2 + top_cand.xbW ** 2)
    # after minimizing, the ttbar distribution is centered around ~(0.5, 0.25) with surfaces of constant density approximiately constant radii
    top_cand["rWbW"] = np.sqrt((top_cand.xbW - 0.25) ** 2 + (top_cand.xW - 0.5) ** 2)
    top_cand["xbW_reco"] = top_cand.xbW
    top_cand["xW_reco"] = top_cand.xW

    return top_cand, rec_top_cands

def adding_top_reco_to_event(event, top_cand):
    """dictionary to convert friend trees back to event variables
    """

    event['top_cand'] = ak.zip({
        "p": ak.zip({
            "pt" : top_cand.p_pt,
            "eta" : top_cand.p_eta,
            "phi" : top_cand.p_phi,
            "mass" : top_cand.p_mass,
            }),
        "b": ak.zip({
            "pt" : top_cand.b_pt,
            "eta" : top_cand.b_eta,
            "phi" : top_cand.b_phi,
            "mass" : top_cand.b_mass,
            "puId" : top_cand.b_puId,
            "jetId" : top_cand.b_jetId,
            'btagScore' : top_cand.b_btagScore,

            }),
        'W' : ak.zip({
            "p" : ak.zip({
                "pt" : top_cand.W_p_pt,
                "eta" : top_cand.W_p_eta,
                "phi" : top_cand.W_p_phi,
                "mass" : top_cand.W_p_mass,
                }),
            "pW" : ak.zip({
                "pt" : top_cand.W_pW_pt,
                "eta" : top_cand.W_pW_eta,
                "phi" : top_cand.W_pW_phi,
                "mass" : top_cand.W_pW_mass,
                }),
                "l": ak.zip({
                    "pt" : top_cand.W_l_pt,
                    "eta" : top_cand.W_l_eta,
                    "phi" : top_cand.W_l_phi,
                    "mass" : top_cand.W_l_mass,
                    "puId" : top_cand.W_l_puId,
                    "jetId" : top_cand.W_l_jetId,
                    'btagScore' : top_cand.W_l_btagScore,
                    }),
                "j": ak.zip({
                    "pt" : top_cand.W_j_pt,
                    "eta" : top_cand.W_j_eta,
                    "phi" : top_cand.W_j_phi,
                    "mass" : top_cand.W_j_mass,
                    "puId" : top_cand.W_j_puId,
                    "jetId" : top_cand.W_j_jetId,
                    'btagScore' : top_cand.W_j_btagScore,
                    }),
        }),
        "mbW": top_cand.mbW,
        "xW": top_cand.xW,
        "xt": top_cand.xt,
        "xbW": top_cand.xbW,
        "xWt": top_cand.xWt,
        "xWbW": top_cand.xWbW,
        "rWbW": top_cand.rWbW,
    })
    event['xbW'] = top_cand['xbW']
    event['xW'] = top_cand['xW']
