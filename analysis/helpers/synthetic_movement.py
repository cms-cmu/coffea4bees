"""Declustering of the two selected fat jets and movement of their constituents.

build_two_jet_containers -> decluster_two_jets -> move_pfcands gives, per selected fat jet, the
original subjets (i0, i1), the declustered subjets (i0_new, i1_new) and the PF candidates moved
with their nearest subjet. events_with_moved_inputs writes the moved FatJet / PFCands / SV
kinematics back into the event (for a tagger run on the synthetic jets).
"""

import logging

import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector as coffea_vector

from coffea4bees.jet_clustering.clustering import comb_jet_flavor
from coffea4bees.jet_clustering.declustering import make_synthetic_event


def wrap_phi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def delta_phi(a, b):
    return wrap_phi(a - b)


def str_where(condition, if_true, if_false):
    """ak.where(condition, "b", "bj") for strings on a jagged array."""
    counts = ak.num(condition)
    flat = ak.flatten(condition)
    none_mask = ak.is_none(flat)
    out = ak.Array(np.where(ak.to_numpy(ak.fill_none(flat, False)), if_true, if_false))
    if ak.any(none_mask):
        out = ak.mask(out, ~none_mask)
    return ak.unflatten(out, counts)


#
#  Validity checks
#
def subjet_validity_report(events, jet1, jet2):
    """Subjet indices of the two selected jets: in range of events.SubJet, distinct within a jet
    and not shared by the two jets.  Returns (valid_mask, counts of each failure)."""
    nsub = ak.num(events.SubJet, axis=1)

    def _in_bounds(idx):
        return (idx >= 0) & (idx < nsub)

    bounds_ok = (
        _in_bounds(jet1.subJetIdx1) & _in_bounds(jet1.subJetIdx2)
        & _in_bounds(jet2.subJetIdx1) & _in_bounds(jet2.subJetIdx2)
    )
    distinct_ok = (jet1.subJetIdx1 != jet1.subJetIdx2) & (jet2.subJetIdx1 != jet2.subJetIdx2)
    disjoint_ok = (
        (jet1.subJetIdx1 != jet2.subJetIdx1) & (jet1.subJetIdx1 != jet2.subJetIdx2)
        & (jet1.subJetIdx2 != jet2.subJetIdx1) & (jet1.subJetIdx2 != jet2.subJetIdx2)
    )
    valid = ak.to_numpy(bounds_ok & distinct_ok & disjoint_ok)
    counts = {
        "subjet_bounds_fail": int(ak.sum(~bounds_ok)),
        "subjet_duplicate_fail": int(ak.sum(bounds_ok & ~distinct_ok)),
        "subjet_shared_fail": int(ak.sum(bounds_ok & distinct_ok & ~disjoint_ok)),
    }
    return valid, counts


def parent_contract_report(declustered_sorted, n_parents):
    """The declustered children, sorted by btagScore (the parent label), must be exactly two per
    parent with labels [0, 0, 1, 1, ...], so that no child is regrouped with the wrong fat jet.
    Returns (ok_mask, counts of each failure)."""
    n_children = ak.to_numpy(ak.num(declustered_sorted, axis=1))
    n_par = ak.to_numpy(n_parents)
    count_ok = n_children == 2 * n_par

    # compare the float labels (an int cast would accept e.g. [0.5, 0.5, 1.5, 1.5])
    labels = ak.values_astype(declustered_sorted.btagScore, np.float64)
    expected = ak.unflatten(
        np.repeat(
            ak.flatten(ak.Array([np.arange(n) for n in n_par])).to_numpy()
            if len(n_par) else np.zeros(0, dtype=np.int64),
            2,
        ),
        2 * n_par,
    )
    label_ok = np.zeros(len(n_par), dtype=bool)
    if len(n_par):
        comparable = count_ok
        if comparable.any():
            eq = ak.all(labels[comparable] == ak.values_astype(expected[comparable], np.float64), axis=1)
            label_ok[comparable] = ak.to_numpy(eq)
    ok = count_ok & label_ok
    counts = {
        "parent_count_fail": int((~count_ok).sum()),
        "parent_label_fail": int((count_ok & ~label_ok).sum()),
    }
    return ok, counts


#
#  Two-jet container: pT-sorted subjets i0/i1, their sum p and the PF candidates of each jet
#
def build_two_jet_containers(selev):
    indices_str = []
    for arr in selev.selFatJet.pt:
        indices_str.append([f"({i},{i})" for i in range(len(arr))])
    selev["selFatJet", "btag_string"] = indices_str

    sorted_sub_jets = selev.selFatJet.subjets
    sorted_sub_jets = sorted_sub_jets[
        ak.argsort(sorted_sub_jets.pt, axis=2, ascending=False)
    ]
    if "tau1" not in sorted_sub_jets.fields:
        sorted_sub_jets = ak.with_field(sorted_sub_jets, sorted_sub_jets.pt, "tau1")
        sorted_sub_jets = ak.with_field(sorted_sub_jets, sorted_sub_jets.pt, "tau2")
        sorted_sub_jets = ak.with_field(sorted_sub_jets, sorted_sub_jets.pt, "tau3")

    pf0_mask = selev.FatJetPFCands.jetIdx == selev.selFatJet.orig_idx[:, 0]
    pf1_mask = selev.FatJetPFCands.jetIdx == selev.selFatJet.orig_idx[:, 1]
    pf_idx0 = selev.FatJetPFCands[pf0_mask].pFCandsIdx
    pf_idx1 = selev.FatJetPFCands[pf1_mask].pFCandsIdx
    pf_per_jet = ak.Array(
        [[a, b] for a, b in zip(selev.PFCands[pf_idx0], selev.PFCands[pf_idx1])]
    )

    fatjets = ak.zip(
        {
            "p": sorted_sub_jets[:, :, 0] + sorted_sub_jets[:, :, 1],
            "i0": sorted_sub_jets[:, :, 0],
            "i1": sorted_sub_jets[:, :, 1],
            "PFCands": pf_per_jet,
        },
        depth_limit=1,
    )
    # Item access: with the Lorentz-vector behaviour the attribute .tau2 is the vector property
    # (mass**2), not the SubJet_tau2 branch
    fatjets["i0", "tau21"] = fatjets.i0["tau2"] / (fatjets.i0["tau1"] + 1e-3)
    fatjets["i1", "tau21"] = fatjets.i1["tau2"] / (fatjets.i1["tau1"] + 1e-3)
    fatjets["i0", "tau32"] = fatjets.i0["tau3"] / (fatjets.i0["tau2"] + 1e-3)
    fatjets["i1", "tau32"] = fatjets.i1["tau3"] / (fatjets.i1["tau2"] + 1e-3)
    fatjets["i0", "btag_string"] = "0.001"
    fatjets["i1", "btag_string"] = "0.001"

    fatjets["i0", "jet_flavor"] = str_where(fatjets.i0.tau21 > 0.5, "b", "bj")
    fatjets["i1", "jet_flavor"] = str_where(fatjets.i1.tau21 > 0.5, "b", "bj")
    i1_more_complex = (fatjets.i1.jet_flavor == "bj") & (fatjets.i0.jet_flavor == "b")
    fatjets["iA"] = ak.where(i1_more_complex, fatjets.i1, fatjets.i0)
    fatjets["iB"] = ak.where(i1_more_complex, fatjets.i0, fatjets.i1)

    C = [
        comb_jet_flavor(a, b)
        for a, b in zip(ak.flatten(fatjets.iA.jet_flavor), ak.flatten(fatjets.iB.jet_flavor))
    ]
    selev["selFatJet", "jet_flavor"] = ak.unflatten(
        np.array(C), ak.num(selev.selFatJet)
    )
    return selev, fatjets, (pf_idx0, pf_idx1)


#
#  Declustering of the subjet sums
#
def decluster_two_jets(selev, fatjets, clustering_pdfs, rand_seed, chunk, strict=True):
    """Decluster the subjet sum of each selected jet, check that the children regroup with their
    own parent and attach i0_new / i1_new / p_new.  Returns (selev, fatjets, ok_mask, counts).
    strict=True raises on any event that fails the parent check, strict=False drops it."""
    clustered_jets = ak.zip(
        {
            "pt": ak.values_astype((selev.selFatJet.subjets[:, :, 0] + selev.selFatJet.subjets[:, :, 1]).pt, np.float64),
            "eta": ak.values_astype((selev.selFatJet.subjets[:, :, 0] + selev.selFatJet.subjets[:, :, 1]).eta, np.float64),
            "phi": ak.values_astype((selev.selFatJet.subjets[:, :, 0] + selev.selFatJet.subjets[:, :, 1]).phi, np.float64),
            "mass": ak.values_astype((selev.selFatJet.subjets[:, :, 0] + selev.selFatJet.subjets[:, :, 1]).mass, np.float64),
            "jet_flavor": selev.selFatJet.jet_flavor,
            "btag_string": selev.selFatJet.btag_string,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=coffea_vector.behavior,
    )

    declustered_jets = make_synthetic_event(
        clustered_jets,
        clustering_pdfs,
        declustering_rand_seed=rand_seed,
        b_pt_threshold=20,
        dr_threshold=0,
        chunk=chunk,
        debug=False,
        splitting_types_to_ignore=[("bj", "b")],
    )
    declustered_jets = declustered_jets[
        ak.argsort(declustered_jets.btagScore, axis=1, ascending=True)
    ]

    ok, counts = parent_contract_report(declustered_jets, ak.num(selev.selFatJet))

    if strict and not ok.all():
        bad = np.nonzero(~ok)[0]
        for i in bad[:20]:
            logging.error(
                f"{chunk}parent check failed: event {i} "
                f"n_children={ak.num(declustered_jets, axis=1)[i]} "
                f"sorted_btagScore={declustered_jets.btagScore[i].tolist()}"
            )
        raise RuntimeError(
            f"{chunk}{len(bad)} events fail the parent check "
            f"(count_fail={counts['parent_count_fail']}, label_fail={counts['parent_label_fail']}): "
            f"the declustered children can not be assigned to their fat jet"
        )

    selev = selev[ok]
    fatjets = fatjets[ok]
    declustered_jets = declustered_jets[ok]
    pairs = ak.unflatten(declustered_jets, 2, axis=1)
    fatjets = ak.with_field(fatjets, pairs[:, :, 0], "i0_new")
    fatjets = ak.with_field(fatjets, pairs[:, :, 1], "i1_new")
    fatjets = ak.with_field(fatjets, pairs[:, :, 0] + pairs[:, :, 1], "p_new")
    return selev, fatjets, ok, counts


#
#  PF candidates move with their nearest subjet: pT scaled by the subjet pT ratio,
#  eta and phi relative to the subjet kept
#
def move_pfcands(fatjets):
    sj0, sj1, pf = fatjets.i0, fatjets.i1, fatjets.PFCands

    dr2_0 = (pf.eta - sj0.eta) ** 2 + wrap_phi(pf.phi - sj0.phi) ** 2
    dr2_1 = (pf.eta - sj1.eta) ** 2 + wrap_phi(pf.phi - sj1.phi) ** 2
    choice = ak.where(dr2_0 <= dr2_1, 0, 1)
    pf_s0 = pf[choice == 0]
    pf_s1 = pf[choice == 1]

    def _move(pf_in, old_fj, new_fj):
        dEta = pf_in.eta - old_fj.eta
        dPhi = delta_phi(pf_in.phi, old_fj.phi)
        pf_pt_b, old_pt_b, new_pt_b, dEta_b, dPhi_b, new_eta_b, new_phi_b = ak.broadcast_arrays(
            pf_in.pt, old_fj.pt, new_fj.pt, dEta, dPhi, new_fj.eta, new_fj.phi
        )
        rPt = ak.where(old_pt_b > 0, pf_pt_b / old_pt_b, 0.0)
        out = ak.with_field(pf_in, rPt * new_pt_b, "pt")
        out = ak.with_field(out, new_eta_b + dEta_b, "eta")
        out = ak.with_field(out, wrap_phi(new_phi_b + dPhi_b), "phi")
        return out

    moved0 = _move(pf_s0, fatjets.i0, fatjets.i0_new)
    moved1 = _move(pf_s1, fatjets.i1, fatjets.i1_new)

    pf_idx = ak.local_index(fatjets.PFCands, axis=2)
    concat = ak.concatenate([moved0, moved1], axis=2)
    idx_concat = ak.concatenate([pf_idx[choice == 0], pf_idx[choice == 1]], axis=2)
    restore = ak.argsort(idx_concat, axis=2)
    fatjets = ak.with_field(fatjets, concat[restore], "PFCands_moved")
    return fatjets


#
#  Event patching, for a tagger run on the moved inputs
#
def replace_jagged_field_values(collection, field, indices, values):
    """collection[field][event][indices] = values, with per-event indices."""
    counts = ak.to_numpy(ak.num(collection[field], axis=1))
    if len(counts) == 0:
        return collection
    flat = np.array(ak.to_numpy(ak.flatten(collection[field], axis=1)), copy=True)
    idx_counts = ak.to_numpy(ak.num(indices, axis=1))
    if np.sum(idx_counts) == 0:
        return collection
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    event_offsets = ak.unflatten(np.repeat(offsets[:-1], idx_counts), idx_counts)
    global_indices = indices + event_offsets
    gflat = ak.to_numpy(ak.flatten(global_indices, axis=None)).astype(np.int64)
    vflat = ak.to_numpy(ak.flatten(values, axis=None))
    # an index outside its event (e.g. -1) would write into the neighbouring event
    starts = np.repeat(offsets[:-1], idx_counts)
    stops = np.repeat(offsets[1:], idx_counts)
    if not ((gflat >= starts) & (gflat < stops)).all():
        bad = np.nonzero(~((gflat >= starts) & (gflat < stops)))[0][:10]
        raise RuntimeError(
            f"replace_jagged_field_values: {len(bad)}+ indices outside their event "
            f"(first at flat positions {bad.tolist()})"
        )
    flat[gflat] = vflat
    return ak.with_field(collection, ak.unflatten(flat, counts), field)


def events_with_patched_fatjet(events_in, fj_idx, new_fatjet):
    """Replace only the pt/eta/phi/mass of the selected FatJet (PF candidates and SVs untouched).
    The tagger computes the axis-relative inputs (pfcand_etarel/phirel, sv_etarel/phirel)
    against events.FatJet[fj_idx]."""
    fj = events_in.FatJet
    fj = replace_jagged_field_values(fj, "pt", fj_idx, new_fatjet.pt[:, None])
    fj = replace_jagged_field_values(fj, "eta", fj_idx, new_fatjet.eta[:, None])
    fj = replace_jagged_field_values(fj, "phi", fj_idx, new_fatjet.phi[:, None])
    fj = replace_jagged_field_values(fj, "mass", fj_idx, new_fatjet.mass[:, None])
    return ak.with_field(events_in, fj, "FatJet")


def events_with_moved_inputs(
    events_in, fj_idx, pf_idx_for_fatjet, sv_idx_for_fatjet,
    old_fatjet, new_fatjet, moved_pfcands,
    old_sj0, old_sj1, new_sj0, new_sj1,
    recompute_btag_rel=True,
):
    """Write the moved kinematics of ONE selected jet into the event: the FatJet axis, its PF
    candidates and its SVs (each SV moves with its nearest subjet, like the PF candidates).
    With recompute_btag_rel the FatJetPFCands btagPtRatio / btagPParRatio / btagEtaRel, which are
    defined relative to the jet axis, are recomputed for the new axis."""
    fj = events_in.FatJet
    fj = replace_jagged_field_values(fj, "pt", fj_idx, new_fatjet.pt[:, None])
    fj = replace_jagged_field_values(fj, "eta", fj_idx, new_fatjet.eta[:, None])
    fj = replace_jagged_field_values(fj, "phi", fj_idx, new_fatjet.phi[:, None])
    fj = replace_jagged_field_values(fj, "mass", fj_idx, new_fatjet.mass[:, None])

    pf = events_in.PFCands
    pf = replace_jagged_field_values(pf, "pt", pf_idx_for_fatjet, moved_pfcands.pt)
    pf = replace_jagged_field_values(pf, "eta", pf_idx_for_fatjet, moved_pfcands.eta)
    pf = replace_jagged_field_values(pf, "phi", pf_idx_for_fatjet, moved_pfcands.phi)

    selected_svs = events_in.SV[sv_idx_for_fatjet]
    # Read the SV momentum branches by item: with a four-vector mixin on SV (mixins["SV"] = "PFCand")
    # the attributes sv.pt/eta/phi are the vertex position (x, y, z), not the momentum
    _sv_pt = selected_svs["pt"]
    _sv_eta = selected_svs["eta"]
    _sv_phi = selected_svs["phi"]
    dr2_sv0 = (_sv_eta - old_sj0.eta[:, None]) ** 2 + wrap_phi(_sv_phi - old_sj0.phi[:, None]) ** 2
    dr2_sv1 = (_sv_eta - old_sj1.eta[:, None]) ** 2 + wrap_phi(_sv_phi - old_sj1.phi[:, None]) ** 2
    sv_to_sj0 = dr2_sv0 <= dr2_sv1
    sj0_rpt = ak.where(old_sj0.pt > 0, new_sj0.pt / old_sj0.pt, 1.0)
    sj1_rpt = ak.where(old_sj1.pt > 0, new_sj1.pt / old_sj1.pt, 1.0)
    sj0_deta, sj1_deta = new_sj0.eta - old_sj0.eta, new_sj1.eta - old_sj1.eta
    sj0_dphi, sj1_dphi = delta_phi(new_sj0.phi, old_sj0.phi), delta_phi(new_sj1.phi, old_sj1.phi)
    moved_sv_pt = ak.where(sv_to_sj0, _sv_pt * sj0_rpt[:, None], _sv_pt * sj1_rpt[:, None])
    moved_sv_eta = ak.where(sv_to_sj0, _sv_eta + sj0_deta[:, None], _sv_eta + sj1_deta[:, None])
    moved_sv_phi = ak.where(sv_to_sj0, wrap_phi(_sv_phi + sj0_dphi[:, None]), wrap_phi(_sv_phi + sj1_dphi[:, None]))

    sv = events_in.SV
    sv = replace_jagged_field_values(sv, "pt", sv_idx_for_fatjet, moved_sv_pt)
    sv = replace_jagged_field_values(sv, "eta", sv_idx_for_fatjet, moved_sv_eta)
    sv = replace_jagged_field_values(sv, "phi", sv_idx_for_fatjet, moved_sv_phi)

    out = ak.with_field(events_in, fj, "FatJet")
    out = ak.with_field(out, pf, "PFCands")
    out = ak.with_field(out, sv, "SV")

    #
    #  The stored values are kept and only the axis change is applied, so the original jet is unchanged
    #  (old == new gives factors of exactly 1):
    #     ptRatio'   = orig * sin(theta_new) / sin(theta_old)
    #     pparRatio' = orig * cos(theta_new) / cos(theta_old)
    #     etaRel'    = orig + etaRel(new) - etaRel(old),  etaRel = 0.5 ln((E + p_par) / (E - p_par)), pion mass
    #  theta = angle between the candidate and the jet axis.  Neutral candidates (stored 0) are left at 0.
    #
    if recompute_btag_rel:
        fjpf = events_in.FatJetPFCands
        _slot = ak.firsts(fj_idx)
        _rowmask = fjpf.jetIdx == _slot
        _rows = ak.local_index(fjpf.jetIdx, axis=1)[_rowmask]
        _pidx = fjpf.pFCandsIdx[_rowmask]
        if not ak.all(ak.fill_none(ak.flatten(_pidx == pf_idx_for_fatjet, axis=None), True)):
            raise RuntimeError("FatJetPFCands rows do not align with pf_idx_for_fatjet")

        _mpi2 = 0.13957039 ** 2

        def _axis_angles(cand_pt, cand_eta, cand_phi, ax_pt, ax_eta, ax_phi):
            axx = np.cos(ax_phi); axy = np.sin(ax_phi); axz = np.sinh(ax_eta)
            an = np.sqrt(axx * axx + axy * axy + axz * axz)
            axx, axy, axz = axx / an, axy / an, axz / an
            px = cand_pt * np.cos(cand_phi)
            py = cand_pt * np.sin(cand_phi)
            pz = cand_pt * np.sinh(cand_eta)
            pmag = np.sqrt(px * px + py * py + pz * pz)
            ppar = px * axx[:, None] + py * axy[:, None] + pz * axz[:, None]
            cos_t = ppar / pmag
            cos_t = ak.where(cos_t > 1.0, 1.0, ak.where(cos_t < -1.0, -1.0, cos_t))
            sin_t = np.sqrt(np.maximum(1.0 - cos_t * cos_t, 0.0))
            e = np.sqrt(pmag * pmag + _mpi2)
            _num = e + ppar
            _den = e - ppar
            _den = ak.where(_den < 1e-8, 1e-8, _den)
            eta_rel = 0.5 * np.log(_num / _den)
            return sin_t, cos_t, eta_rel

        _pf_o = events_in.PFCands[pf_idx_for_fatjet]
        so, co, ro = _axis_angles(_pf_o.pt, _pf_o.eta, _pf_o.phi,
                                  old_fatjet.pt, old_fatjet.eta, old_fatjet.phi)
        sn, cn, rn = _axis_angles(moved_pfcands.pt, moved_pfcands.eta, moved_pfcands.phi,
                                  new_fatjet.pt, new_fatjet.eta, new_fatjet.phi)

        _f_sin = ak.where(so > 1e-4, sn / ak.where(so > 1e-4, so, 1.0), 1.0)
        _f_cos = ak.where(np.abs(co) > 1e-3, cn / ak.where(np.abs(co) > 1e-3, co, 1.0), 1.0)
        _d_rel = rn - ro

        for _field, _kind in (("btagPtRatio", "sin"), ("btagPParRatio", "cos"),
                              ("btagEtaRel", "shift")):
            _orig = fjpf[_field][_rowmask]
            _live = _orig != 0
            if _kind == "sin":
                _new = ak.where(_live, _orig * _f_sin, _orig)
                _new = ak.where(_new > 1.0, 1.0, ak.where(_new < 0.0, 0.0, _new))
            elif _kind == "cos":
                _new = ak.where(_live, _orig * _f_cos, _orig)
                _new = ak.where(_new > 1.0, 1.0, ak.where(_new < -1.0, -1.0, _new))
            else:
                _new = ak.where(_live, _orig + _d_rel, _orig)
            fjpf = replace_jagged_field_values(fjpf, _field, _rows, _new)
        out = ak.with_field(out, fjpf, "FatJetPFCands")
    return out
