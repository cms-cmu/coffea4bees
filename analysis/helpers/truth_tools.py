import numpy as np
import awkward as ak
from functools import reduce

#### from https://github.com/aebid/HHbbWW_Run3/blob/main/python/genparticles.py#L42
def find_genpart(genpart, pdgid, ancestors):
    """
    Find gen level particles given pdgId (and ancestors ids)

    Parameters:
    genpart (GenPart): NanoAOD GenPart collection.
    pdgid (list): pdgIds for the target particles.
    idmother (list): pdgIds for the ancestors of the target particles.

    Returns:
    NanoAOD GenPart collection
    """

    def check_id(p):
        return np.abs(genpart.pdgId) == p

    pid = reduce(np.logical_or, map(check_id, pdgid))

    if ancestors:
        ancs, ancs_idx = [], []
        for i, mother_id in enumerate(ancestors):
            if i == 0:
                mother_idx = genpart[pid].genPartIdxMother
            else:
                mother_idx = genpart[ancs_idx[i-1]].genPartIdxMother
            ancs.append(np.abs(genpart[mother_idx].pdgId) == mother_id)
            ancs_idx.append(mother_idx)

        decaymatch =  reduce(np.logical_or, ancs)
        return genpart[pid][decaymatch]

    return genpart[pid]


def _flag(genpart, bit):
    return ((genpart.statusFlags >> bit) & 1) == 1


def label_jet_truth_parent(jets, genpart, dr_max=0.4, max_depth=20):
    """
    Label each jet 'H' or 't' only if it is matched to a truth b quark from the Higgs
    or from one of the two highest-pT top quarks; every other jet (including ones
    matched to a light quark from a W->qq decay, or unmatched) is labeled 'o'.

    Hard-process b quarks (|pdgId| == 5, isHardProcess) are traced up the mother chain to
    tag which come from a last-copy Higgs ('H') vs. from one of the two highest-pT
    last-copy top quarks ('t'). Each such b quark is matched to its closest jet with
    dR < dr_max; if a jet is claimed by b quarks from both, 'H' takes precedence.

    Returns:
        dict of per-jet arrays: truthParent ('H'/'t'/'o'), truthFromH, truthFromTop, truthB
        (truthFromH/truthFromTop/truthB are all equivalent to the b-jet condition here.)
    """
    pdg = abs(genpart.pdgId)
    mother = genpart.genPartIdxMother
    gen_idx = ak.local_index(genpart, axis=1)
    last_copy = _flag(genpart, 13)

    is_H = (pdg == 25) & last_copy
    tops = genpart[(pdg == 6) & last_copy]
    top2_idx = gen_idx[(pdg == 6) & last_copy][ak.argsort(tops.pt, axis=1, ascending=False)][:, :2]
    is_top2 = ak.any(gen_idx[:, :, np.newaxis] == top2_idx[:, np.newaxis, :], axis=2)

    parton_mask = (pdg == 5) & _flag(genpart, 7) & (mother >= 0)
    partons = genpart[parton_mask]

    # Walk up the mother chain; negative indices must not wrap around
    parton_from_H = ak.zeros_like(partons.pdgId, dtype=bool)
    parton_from_top = ak.zeros_like(partons.pdgId, dtype=bool)
    cur = partons.genPartIdxMother
    for _ in range(max_depth):
        valid = cur >= 0
        cur_safe = ak.where(valid, cur, 0)
        parton_from_H = parton_from_H | (valid & is_H[cur_safe])
        parton_from_top = parton_from_top | (valid & is_top2[cur_safe])
        cur = ak.where(valid, mother[cur_safe], -1)
        if not ak.any(cur >= 0):
            break
    parton_from_top = parton_from_top & ~parton_from_H

    # Only b quarks from H/top are matched to jets (e.g. a hard-process b from g->bb is dropped).
    keep = parton_from_H | parton_from_top
    partons = partons[keep]
    parton_from_H = parton_from_H[keep]
    parton_from_top = parton_from_top[keep]

    # Parton -> closest jet
    dr = partons.metric_table(jets)                       # [evt][parton][jet]
    closest = ak.fill_none(ak.argmin(dr, axis=2), -1)
    matched = ak.fill_none(ak.min(dr, axis=2) < dr_max, False)
    jet_idx = ak.local_index(jets, axis=1)
    hit = (closest[:, :, np.newaxis] == jet_idx[:, np.newaxis, :]) & matched[:, :, np.newaxis]

    jet_from_H = ak.any(hit & parton_from_H[:, :, np.newaxis], axis=1)
    jet_from_top = ak.any(hit & parton_from_top[:, :, np.newaxis], axis=1) & ~jet_from_H
    jet_is_b = jet_from_H | jet_from_top

    code = ak.values_astype(jet_from_H, np.int8) * 2 + ak.values_astype(jet_from_top, np.int8)
    labels = np.array(["o", "t", "H"])[ak.to_numpy(ak.flatten(code))]

    return {
        "truthParent": ak.unflatten(labels, ak.num(code)),
        "truthFromH": jet_from_H,
        "truthFromTop": jet_from_top,
        "truthB": jet_is_b,
    }
