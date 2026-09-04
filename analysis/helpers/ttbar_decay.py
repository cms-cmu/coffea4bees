from __future__ import annotations

import logging
import awkward as ak
import numpy as np

# Canonical category definitions
GROUPED_CATEGORIES = [
    "Hadronic",
    "Semileptonic (e/mu)",
    "Dileptonic (e/mu)",
    "Tau decays",
    "Other",
]

DETAILED_CATEGORIES = [
    "Hadronic",
    "Semileptonic (e)",
    "Semileptonic (mu)",
    "Dileptonic (ee)",
    "Dileptonic (mumu)",
    "Dileptonic (emu)",
    "Semileptonic (tau)",
    "Dileptonic (tau+tau)",
    "Dileptonic (tau+lep)",
    "Other",
]

# Z decay category definitions
Z_DECAY_CATEGORIES = [
    "Z -> qq (hadronic)",
    "Z -> ee",
    "Z -> mumu",
    "Z -> tautau",
    "Z -> nunu (invisible)",
    "Other",
]

# Detailed hadronic Z decay category definitions (quark-flavor split)
Z_DECAY_HADRONIC_CATEGORIES = [
    "Z -> bb",
    "Z -> cc",
    "Z -> ss",
    "Z -> uu",
    "Z -> dd",
    "Other hadronic",
]

def classify_ttbar_decays(genpart: ak.Array) -> dict[str, ak.Array]:
    """Classify ttbar decay modes from GenPart collection using generator truth.

    Identifies prompt charged leptons (e, mu, tau) whose direct parent is a W boson (pdgId 24)
    originating from top quarks.

    Args:
        genpart: NanoAOD / picoAOD GenPart collection (jagged array).

    Returns:
        dict containing:
            - 'n_e': count of prompt electrons from W per event
            - 'n_mu': count of prompt muons from W per event
            - 'n_tau': count of prompt taus from W per event
            - 'n_lep': count of prompt (e + mu) from W per event
            - 'n_tot': count of all prompt leptons (e + mu + tau) from W per event
            - 'grouped_masks': dict of category_name -> boolean mask (shape: len(events))
            - 'detailed_masks': dict of detailed_category_name -> boolean mask (shape: len(events))
    """
    if genpart is None or len(genpart) == 0:
        empty_mask = np.zeros(len(genpart) if genpart is not None else 0, dtype=bool)
        return {
            "n_e": ak.Array([]),
            "n_mu": ak.Array([]),
            "n_tau": ak.Array([]),
            "n_lep": ak.Array([]),
            "n_tot": ak.Array([]),
            "grouped_masks": {cat: empty_mask for cat in GROUPED_CATEGORIES},
            "detailed_masks": {cat: empty_mask for cat in DETAILED_CATEGORIES},
        }

    mother_idx = genpart.genPartIdxMother
    num_particles = ak.num(genpart, axis=1)
    valid_mother = (mother_idx >= 0) & (mother_idx < num_particles)
    safe_mother_idx = ak.where(valid_mother, mother_idx, 0)

    # Lookup mother pdgId safely
    pdg_array = genpart.pdgId
    mother_pdg = ak.where(valid_mother, np.abs(pdg_array[safe_mother_idx]), 0)

    # Mother must be W boson (|pdgId| == 24)
    is_from_W = mother_pdg == 24

    abs_pdg = np.abs(genpart.pdgId)
    is_e = is_from_W & (abs_pdg == 11)
    is_mu = is_from_W & (abs_pdg == 13)
    is_tau = is_from_W & (abs_pdg == 15)

    n_e = ak.sum(is_e, axis=1)
    n_mu = ak.sum(is_mu, axis=1)
    n_tau = ak.sum(is_tau, axis=1)
    n_lep = n_e + n_mu
    n_tot = n_e + n_mu + n_tau

    # --- Grouped Categories ---
    is_had = (n_tot == 0)
    is_semilep_emu = (n_tau == 0) & (n_lep == 1)
    is_dilep_emu = (n_tau == 0) & (n_lep == 2)
    is_tau_decay = (n_tau >= 1) & (n_tot <= 2)
    is_other_grouped = ~(is_had | is_semilep_emu | is_dilep_emu | is_tau_decay)

    grouped_masks = {
        "Hadronic": is_had,
        "Semileptonic (e/mu)": is_semilep_emu,
        "Dileptonic (e/mu)": is_dilep_emu,
        "Tau decays": is_tau_decay,
        "Other": is_other_grouped,
    }

    # --- Detailed Subcategories ---
    is_semi_e = (n_tau == 0) & (n_e == 1) & (n_mu == 0)
    is_semi_mu = (n_tau == 0) & (n_mu == 1) & (n_e == 0)
    is_dilep_ee = (n_tau == 0) & (n_e == 2) & (n_mu == 0)
    is_dilep_mumu = (n_tau == 0) & (n_mu == 2) & (n_e == 0)
    is_dilep_emu_mix = (n_tau == 0) & (n_e == 1) & (n_mu == 1)
    is_semi_tau = (n_tau == 1) & (n_lep == 0)
    is_dilep_tautau = (n_tau == 2) & (n_lep == 0)
    is_dilep_taulep = (n_tau == 1) & (n_lep == 1)
    is_other_detailed = ~(
        is_had | is_semi_e | is_semi_mu | is_dilep_ee | is_dilep_mumu |
        is_dilep_emu_mix | is_semi_tau | is_dilep_tautau | is_dilep_taulep
    )

    detailed_masks = {
        "Hadronic": is_had,
        "Semileptonic (e)": is_semi_e,
        "Semileptonic (mu)": is_semi_mu,
        "Dileptonic (ee)": is_dilep_ee,
        "Dileptonic (mumu)": is_dilep_mumu,
        "Dileptonic (emu)": is_dilep_emu_mix,
        "Semileptonic (tau)": is_semi_tau,
        "Dileptonic (tau+tau)": is_dilep_tautau,
        "Dileptonic (tau+lep)": is_dilep_taulep,
        "Other": is_other_detailed,
    }

    return {
        "n_e": n_e,
        "n_mu": n_mu,
        "n_tau": n_tau,
        "n_lep": n_lep,
        "n_tot": n_tot,
        "grouped_masks": grouped_masks,
        "detailed_masks": detailed_masks,
    }


def classify_z_decays(genpart: ak.Array) -> dict[str, ak.Array]:
    """Classify Z decay modes from GenPart collection using generator truth.

    Identifies prompt fermions (quarks, e, mu, tau, neutrinos) whose direct parent is a
    Z boson (pdgId 23), mirroring the W-boson ancestry logic in classify_ttbar_decays.

    Args:
        genpart: NanoAOD / picoAOD GenPart collection (jagged array).

    Returns:
        dict containing:
            - 'n_q': count of prompt quarks from Z per event
            - 'n_d', 'n_u', 'n_s', 'n_c', 'n_b': per-flavor prompt quark counts from Z per event
            - 'n_e': count of prompt electrons from Z per event
            - 'n_mu': count of prompt muons from Z per event
            - 'n_tau': count of prompt taus from Z per event
            - 'n_nu': count of prompt neutrinos from Z per event
            - 'masks': dict of category_name -> boolean mask (shape: len(events))
            - 'hadronic_masks': dict of quark-flavor category_name -> boolean mask,
              defined only within the "Z -> qq (hadronic)" subset (see Z_DECAY_HADRONIC_CATEGORIES)
    """
    if genpart is None or len(genpart) == 0:
        empty_mask = np.zeros(len(genpart) if genpart is not None else 0, dtype=bool)
        return {
            "n_q": ak.Array([]),
            "n_d": ak.Array([]),
            "n_u": ak.Array([]),
            "n_s": ak.Array([]),
            "n_c": ak.Array([]),
            "n_b": ak.Array([]),
            "n_e": ak.Array([]),
            "n_mu": ak.Array([]),
            "n_tau": ak.Array([]),
            "n_nu": ak.Array([]),
            "masks": {cat: empty_mask for cat in Z_DECAY_CATEGORIES},
            "hadronic_masks": {cat: empty_mask for cat in Z_DECAY_HADRONIC_CATEGORIES},
        }

    mother_idx = genpart.genPartIdxMother
    num_particles = ak.num(genpart, axis=1)
    valid_mother = (mother_idx >= 0) & (mother_idx < num_particles)
    safe_mother_idx = ak.where(valid_mother, mother_idx, 0)

    # Lookup mother pdgId safely
    pdg_array = genpart.pdgId
    mother_pdg = ak.where(valid_mother, np.abs(pdg_array[safe_mother_idx]), 0)

    # Mother must be Z boson (|pdgId| == 23)
    is_from_Z = mother_pdg == 23

    abs_pdg = np.abs(genpart.pdgId)
    is_q = is_from_Z & (abs_pdg >= 1) & (abs_pdg <= 5)
    is_d = is_from_Z & (abs_pdg == 1)
    is_u = is_from_Z & (abs_pdg == 2)
    is_s = is_from_Z & (abs_pdg == 3)
    is_c = is_from_Z & (abs_pdg == 4)
    is_b = is_from_Z & (abs_pdg == 5)
    is_e = is_from_Z & (abs_pdg == 11)
    is_mu = is_from_Z & (abs_pdg == 13)
    is_tau = is_from_Z & (abs_pdg == 15)
    is_nu = is_from_Z & ((abs_pdg == 12) | (abs_pdg == 14) | (abs_pdg == 16))

    n_q = ak.sum(is_q, axis=1)
    n_d = ak.sum(is_d, axis=1)
    n_u = ak.sum(is_u, axis=1)
    n_s = ak.sum(is_s, axis=1)
    n_c = ak.sum(is_c, axis=1)
    n_b = ak.sum(is_b, axis=1)
    n_e = ak.sum(is_e, axis=1)
    n_mu = ak.sum(is_mu, axis=1)
    n_tau = ak.sum(is_tau, axis=1)
    n_nu = ak.sum(is_nu, axis=1)

    is_qq = n_q >= 2
    is_ee = n_e >= 2
    is_mumu = n_mu >= 2
    is_tautau = n_tau >= 2
    is_nunu = n_nu >= 2
    is_other = ~(is_qq | is_ee | is_mumu | is_tautau | is_nunu)

    masks = {
        "Z -> qq (hadronic)": is_qq,
        "Z -> ee": is_ee,
        "Z -> mumu": is_mumu,
        "Z -> tautau": is_tautau,
        "Z -> nunu (invisible)": is_nunu,
        "Other": is_other,
    }

    # --- Quark-flavor breakdown within the hadronic (Z -> qq) subset ---
    is_bb = is_qq & (n_b >= 2)
    is_cc = is_qq & (n_c >= 2)
    is_ss = is_qq & (n_s >= 2)
    is_uu = is_qq & (n_u >= 2)
    is_dd = is_qq & (n_d >= 2)
    is_other_hadronic = is_qq & ~(is_bb | is_cc | is_ss | is_uu | is_dd)

    hadronic_masks = {
        "Z -> bb": is_bb,
        "Z -> cc": is_cc,
        "Z -> ss": is_ss,
        "Z -> uu": is_uu,
        "Z -> dd": is_dd,
        "Other hadronic": is_other_hadronic,
    }

    return {
        "n_q": n_q,
        "n_d": n_d,
        "n_u": n_u,
        "n_s": n_s,
        "n_c": n_c,
        "n_b": n_b,
        "n_e": n_e,
        "n_mu": n_mu,
        "n_tau": n_tau,
        "n_nu": n_nu,
        "masks": masks,
        "hadronic_masks": hadronic_masks,
    }


def compute_decay_yields(
    masks: dict[str, ak.Array],
    weights: ak.Array | np.ndarray | None = None,
    filter_mask: ak.Array | np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Calculate raw and weighted event yields for each decay category mask.

    Args:
        masks: dictionary of category_name -> boolean array.
        weights: array of event weights (if None, raw counts only).
        filter_mask: optional boolean mask (e.g. selection cut) applied before counting.

    Returns:
        dict of category_name -> {'raw': int, 'weighted': float}
    """
    yields = {}
    for cat_name, cat_mask in masks.items():
        if filter_mask is not None:
            active_mask = cat_mask & filter_mask
        else:
            active_mask = cat_mask

        raw_count = int(ak.sum(active_mask))
        if weights is not None:
            weighted_val = float(ak.sum(weights[active_mask]))
        else:
            weighted_val = float(raw_count)

        yields[cat_name] = {
            "raw": raw_count,
            "weighted": weighted_val,
        }
    return yields
