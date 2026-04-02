import numpy as np
import awkward as ak
import logging
import yaml
from src.math_tools.random import Squares
from coffea4bees.analysis.helpers.SvB_helpers import compute_SvB, compute_SvB_FeynNet
from coffea4bees.analysis.helpers.FvT_helpers import compute_FvT
from coffea.nanoevents.methods import vector
from coffea.analysis_tools import Weights


def load_candidates_selection_config(path: str) -> dict:
    """Load candidates selection thresholds from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Dictionary of thresholds, suitable for passing as ``cand_cfg``
        to :func:`cand_jet_selection` and :func:`create_cand_jet_dijet_quadjet`.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cand_jet_selection(
    selev,
    include_lowptjets=False,
    cand_cfg: dict = None,
):
    """
    Creates candidate jets

    Parameters:
    -----------
    selev : ak.Array
        The selected events.
    cand_cfg : dict, optional
        Thresholds loaded from candidates_selection_thresholds.yml.
        If None, hard-coded defaults are used.
    """

    cfg = (cand_cfg or {}).get('canJet', {})
    isSelJet_pt_min  = cfg.get('isSelJet_pt_min',  40)
    isSelJet_eta_max = cfg.get('isSelJet_eta_max', 2.4)

    #
    # Build and select boson candidate jets with bRegCorr applied
    #
    sorted_idx = ak.argsort( selev.Jet.btagScore * selev.Jet.selected, axis=1, ascending=False )
    logging.debug(f"all jets {selev.Jet.pt[:2]}")
    if include_lowptjets:
        sorted_idx_lowpt = ak.argsort( selev.Jet.btagScore * selev.Jet.selected_lowpt, axis=1, ascending=False )
        canJet_idx = ak.concatenate([sorted_idx[:, 0:3], sorted_idx_lowpt[:, :1]], axis=1)
        logging.debug(f"jet lowpt selected {(selev.Jet.selected_lowpt)[:2]}")
        logging.debug(f"canJet_idx with lowpt {canJet_idx[:2]}")

    else:
        canJet_idx = sorted_idx[:, 0:4]

    # Exclude canJet_idx from sorted_idx
    is_canJet = ak.zeros_like(sorted_idx, dtype=bool)
    for i in range(4):
        is_canJet = is_canJet | (sorted_idx == canJet_idx[:, i])
    notCanJet_idx = sorted_idx[~is_canJet]
    del is_canJet

    notCanJet = selev.Jet[notCanJet_idx]
    logging.debug(f"all notCanJet {notCanJet.pt[:2]}")
    notCanJet = notCanJet[notCanJet.selected_loose | (notCanJet.selected_lowpt if include_lowptjets else False)]
    logging.debug(f"notCanJet selected_loose {notCanJet.pt[:2]}")

    notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]
    logging.debug(f"notCanJet sorted {notCanJet.pt[:2]}")

    logging.debug(f"canJet_idx {canJet_idx[:2]}")
    logging.debug(f"notCanJet_idx {notCanJet_idx[:2]}\n\n")

    # apply bJES to canJets
    logging.debug(f"canJet before bReg {selev.Jet[canJet_idx].pt[:2]}\n")
    canJet_raw = selev.Jet[canJet_idx]
    canJet = canJet_raw * canJet_raw.bRegCorr
    canJet["bRegCorr"]  = canJet_raw.bRegCorr
    canJet["btagScore"] = canJet_raw.btagScore
    canJet["puId"]      = canJet_raw.puId
    canJet["jetId"]     = canJet_raw.jetId
    if "hadronFlavour" in selev.Jet.fields:
        canJet["hadronFlavour"] = canJet_raw.hadronFlavour
    del canJet_raw

    #
    # pt sort canJets
    #
    canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
    selev["canJet"] = canJet
    for i in range(4):
        selev[f"canJet{i}"] = selev["canJet"][:, i]

    notCanJet["isSelJet"] = 1 * ( (notCanJet.pt >= isSelJet_pt_min) & (np.abs(notCanJet.eta) < isSelJet_eta_max) )
    selev["notCanJet_coffea"] = notCanJet
    selev["nNotCanJet"] = ak.num(selev.notCanJet_coffea)

    # Release indexing intermediates
    del sorted_idx, canJet_idx, notCanJet_idx, notCanJet
    if include_lowptjets:
        del sorted_idx_lowpt

    return selev


# ---------------------------------------------------------------------------
# Private helpers for create_cand_jet_dijet_quadjet
# ---------------------------------------------------------------------------

def _compute_vbf_variables(selev, cand_cfg):
    """Compute VBF jet variables and passVBFSel on selev (in-place)."""
    vbfJets = ak.pad_none(selev.notCanJet_coffea, 2)
    mask_two_notCanJets = ak.num(selev.notCanJet_coffea) >= 2

    selev["vbfJets_mjj"] = ak.where(
        mask_two_notCanJets,
        (vbfJets[:, 0] + vbfJets[:, 1]).mass,
        -1.0,
    )
    selev["vbfJets_detajj"] = ak.where(
        mask_two_notCanJets,
        np.abs(vbfJets[:, 0].eta - vbfJets[:, 1].eta),
        -1.0,
    )

    vbf_cfg = (cand_cfg or {}).get('vbf', {})
    selev['passVBFSel'] = (
        (selev.vbfJets_mjj    > vbf_cfg.get('mjj_min',    400)) &
        (selev.vbfJets_detajj > vbf_cfg.get('detajj_min', 3.5))
    )


def _build_dijets(selev, cand_cfg, isRun3):
    """Build dijet pairs and compute dijet-level variables.

    Returns
    -------
    diJet : ak.Array
        Dijet array indexed by [event, pairing, lead/subl].
    diJetDr : ak.Array
        Same pairings sorted by dR (close/other).
    sr2_cfg : dict
        sr_run2 thresholds (re-used by _build_quadjets for xZZ/xZH/xHH).
    """
    canJet  = selev["canJet"]
    pairing = [([0, 2], [0, 1], [0, 1]), ([1, 3], [2, 3], [3, 2])]
    diJet   = canJet[:, pairing[0]] + canJet[:, pairing[1]]
    diJet["lead"] = canJet[:, pairing[0]]
    diJet["subl"] = canJet[:, pairing[1]]
    diJet["st"]   = diJet["lead"].pt + diJet["subl"].pt
    diJet["dr"]   = diJet["lead"].delta_r(diJet["subl"])
    diJet["dphi"] = diJet["lead"].delta_phi(diJet["subl"])

    # Sort diJets within views to be lead/subl by st (Run 2) or pt (Run 3)
    if isRun3:
        diJet = diJet[ak.argsort(diJet.pt, axis=2, ascending=False)]
    else:
        diJet = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
    diJetDr = diJet[ak.argsort(diJet.dr, axis=2, ascending=True)]

    # Dijet mass window
    dj_cfg = (cand_cfg or {}).get('dijet', {})
    mass_min = dj_cfg.get('mass_min', [52,  50])
    mass_max = dj_cfg.get('mass_max', [180, 173])
    minDiJetMass = np.array([[[mass_min[0], mass_min[1]]]])
    maxDiJetMass = np.array([[[mass_max[0], mass_max[1]]]])
    diJet["passDiJetMass"] = (minDiJetMass < diJet.mass) & (diJet.mass < maxDiJetMass)

    # Mass-to-dR ratio (MDR) cuts
    mdr_cfg = (cand_cfg or {}).get('mdr', {})
    min_m4j_scale = np.array([mdr_cfg.get('min_m4j_scale', [360, 235])])
    min_dr_offset = np.array([mdr_cfg.get('min_dr_offset', [-0.5, 0.0])])
    max_m4j_scale = np.array([mdr_cfg.get('max_m4j_scale', [650, 650])])
    max_dr_offset = np.array([mdr_cfg.get('max_dr_offset', [0.5,  0.7])])
    max_dr        = np.array([mdr_cfg.get('max_dr',        [1.5,  1.5])])
    m4j = selev["v4j"].mass[:, np.newaxis, np.newaxis]
    diJet["passMDR"] = (
        (min_m4j_scale / m4j + min_dr_offset < diJet.dr) &
        (diJet.dr < np.maximum(max_m4j_scale / m4j + max_dr_offset, max_dr))
    )

    # Consistency with Z/H masses (used for Run 2 SR; also computed in Run 3 for monitoring)
    sr2_cfg = (cand_cfg or {}).get('sr_run2', {})
    mZ      = sr2_cfg.get('mZ',      91.0)
    mH      = sr2_cfg.get('mH',      125.0)
    st_bias = sr2_cfg.get('st_bias', [1.02, 0.98])
    st_bias = np.array([[[st_bias[0], st_bias[1]]]])
    diJet["xZ"] = (diJet.mass - mZ * st_bias) / (0.1 * diJet.mass)
    diJet["xH"] = (diJet.mass - mH * st_bias) / (0.1 * diJet.mass)

    del canJet, pairing
    return diJet, diJetDr, sr2_cfg


def _select_quadjet_run2(quadJet):
    """Pick best quadjet pairing and assign Run 2 SR/SB regions."""
    quadJet["SR"] = quadJet.ZZSR | quadJet.ZHSR | quadJet.HHSR
    quadJet["SB"] = quadJet.passDiJetMass & ~quadJet.SR
    quadJet["rank"] = (
        10 * quadJet.passDiJetMass
        + quadJet.lead.passMDR
        + quadJet.subl.passMDR
        + quadJet.random
    )
    quadJet["selected"] = quadJet.rank == np.max(quadJet.rank, axis=1)


def _select_quadjet_run3(selev, quadJet, cand_cfg):
    """Pick best quadjet pairing and assign Run 3 SR/SB regions (rhh-based)."""
    sr3_cfg = (cand_cfg or {}).get('sr_run3', {})
    diagonalXoYo = sr3_cfg.get('diagonalXoYo', 1.04)
    delta_dhh_max = sr3_cfg.get('delta_dhh_max', 30)
    cLead     = sr3_cfg.get('cLead',     125)
    cSubl     = sr3_cfg.get('cSubl',     120)
    SR_radius = sr3_cfg.get('SR_radius', 30)
    CR_radius = sr3_cfg.get('CR_radius', 55)

    # Distance to the diagonal in the (m_lead, m_subl) plane
    # https://gitlab.cern.ch/mkolosov/hh4b_run3/-/blob/run2/python/producers/hh4bTreeProducer.py#L3386
    quadJet["dhh"] = (1.0 / np.sqrt(1 + diagonalXoYo**2)) * abs(
        quadJet["lead"].mass - diagonalXoYo * quadJet["subl"].mass
    )

    dhh_sorted = np.sort(quadJet["dhh"], axis=1)
    delta_dhh  = abs(dhh_sorted[:, 1] - dhh_sorted[:, 0])

    quadJet_min_dhh_mask  = dhh_sorted[:, 0] == quadJet.dhh
    quadJet_min_dhh       = quadJet[quadJet_min_dhh_mask]
    quadJet_min2_dhh_mask = dhh_sorted[:, 1] == quadJet.dhh
    quadJet_min2_dhh      = quadJet[quadJet_min2_dhh_mask]

    # Boost to CM frame to break ties when the two closest pairings are similar
    boost_vec_v4j = ak.zip(
        {
            "x": selev.v4j.px / selev.v4j.energy,
            "y": selev.v4j.py / selev.v4j.energy,
            "z": selev.v4j.pz / selev.v4j.energy,
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )
    quadJet_min_dhh_lead_CM  = quadJet_min_dhh.lead[:, 0].boost(-boost_vec_v4j)
    quadJet_min2_dhh_lead_CM = quadJet_min2_dhh.lead[:, 0].boost(-boost_vec_v4j)
    use_dhh2_mask = (
        (delta_dhh < delta_dhh_max) &
        (quadJet_min2_dhh_lead_CM.pt > quadJet_min_dhh_lead_CM.pt)
    )
    quadJet["selected"] = ak.where(use_dhh2_mask, quadJet_min2_dhh_mask, quadJet_min_dhh_mask)

    # Radial distance to the HH hypothesis in the dijet mass plane
    quadJet["rhh"] = np.sqrt(
        (quadJet["lead"].mass - cLead)**2 + (quadJet["subl"].mass - cSubl)**2
    )
    quadJet["SR"]  = quadJet.rhh < SR_radius
    quadJet["SB"]  = (~quadJet.SR) & (quadJet.rhh < CR_radius)
    quadJet["passDiJetMass"] = quadJet.SR | quadJet.SB


def _build_quadjets(selev, diJet, diJetDr, sr2_cfg, cand_cfg, isRun3):
    """Build quadjet candidates and assign signal regions.

    Returns
    -------
    quadJet : ak.Array
        Quadjet array indexed by [event, pairing].
    """
    rng_0 = Squares("quadJetSelection")
    rng_1 = rng_0.shift(1)
    rng_2 = rng_0.shift(2)
    counter = selev.event

    quadJet = ak.zip({
        "lead":  diJet[:, :, 0],
        "subl":  diJet[:, :, 1],
        "close": diJetDr[:, :, 0],
        "other": diJetDr[:, :, 1],
        "passDiJetMass": ak.all(diJet.passDiJetMass, axis=2),
        "random": np.concatenate([
            rng_0.uniform(counter, low=0.1, high=0.9)[:, np.newaxis],
            rng_1.uniform(counter, low=0.1, high=0.9)[:, np.newaxis],
            rng_2.uniform(counter, low=0.1, high=0.9)[:, np.newaxis],
        ], axis=1),
    })

    quadJet["dr"]      = quadJet["lead"].delta_r(quadJet["subl"])
    quadJet["dphi"]    = quadJet["lead"].delta_phi(quadJet["subl"])
    quadJet["deta"]    = quadJet["lead"].eta - quadJet["subl"].eta
    quadJet["v4jmass"] = selev["v4j"].mass

    # Run 2-style signal regions (also computed in Run 3 for monitoring)
    max_xZZ = sr2_cfg.get('max_xZZ', 2.6)
    max_xZH = sr2_cfg.get('max_xZH', 1.9)
    max_xHH = sr2_cfg.get('max_xHH', 1.9)
    quadJet["xZZ"] = np.sqrt(quadJet.lead.xZ**2 + quadJet.subl.xZ**2)
    quadJet["xHH"] = np.sqrt(quadJet.lead.xH**2 + quadJet.subl.xH**2)
    quadJet["xZH"] = np.sqrt(np.minimum(
        quadJet.lead.xH**2 + quadJet.subl.xZ**2,
        quadJet.lead.xZ**2 + quadJet.subl.xH**2,
    ))
    quadJet["ZZSR"] = quadJet.xZZ < max_xZZ
    quadJet["ZHSR"] = quadJet.xZH < max_xZH
    quadJet["HHSR"] = (
        (quadJet.xHH < max_xHH) & selev.notInBoostedSel
        if 'notInBoostedSel' in selev.fields
        else (quadJet.xHH < max_xHH)
    )

    if isRun3:
        _select_quadjet_run3(selev, quadJet, cand_cfg)
    else:
        _select_quadjet_run2(quadJet)

    return quadJet


def _apply_ml_scores(
    selev,
    quadJet,
    apply_FvT,
    classifier_FvT,
    run_SvB,
    run_systematics,
    classifier_SvB,
    classifier_SvB_MA,
    classifier_SvB_FeynNet,
    weights,
    list_weight_names,
    analysis_selections,
    label3b,
):
    """Apply FvT and SvB ML scores to quadjet candidates.

    Returns updated ``apply_FvT`` flag (set True if classifier_FvT is provided).
    """
    if classifier_FvT is not None:
        logging.info("Computing FvT scores with classifier")
        compute_FvT(selev, selev[label3b], FvT=classifier_FvT)
        weight_FvT = np.ones(len(weights.weight()), dtype=float)
        weight_FvT[analysis_selections] *= ak.to_numpy(selev.FvT.FvT)
        weights.add("FvT", weight_FvT)
        list_weight_names.append("FvT")
        logging.debug(f"FvT {weights.partial_weight(include=['FvT'])[:10]}\n")
        apply_FvT = True

    if apply_FvT and ("FvT" in selev.fields):
        quadJet["FvT_q_score"] = np.concatenate([
            selev.FvT.q_1234[:, np.newaxis],
            selev.FvT.q_1324[:, np.newaxis],
            selev.FvT.q_1423[:, np.newaxis],
        ], axis=1)

    if run_SvB:
        if (classifier_SvB is not None) or (classifier_SvB_MA is not None):
            tmp_mask = (
                (selev.fourTag & quadJet[quadJet.selected][:, 0].SR)
                if run_systematics
                else np.full(len(selev), True)
            )
            compute_SvB(selev, tmp_mask, SvB=classifier_SvB, SvB_MA=classifier_SvB_MA, doCheck=False)

        if "SvB" in selev.fields:
            quadJet["SvB_q_score"] = np.concatenate([
                selev.SvB.q_1234[:, np.newaxis],
                selev.SvB.q_1324[:, np.newaxis],
                selev.SvB.q_1423[:, np.newaxis],
            ], axis=1)
        if "SvB_MA" in selev.fields:
            quadJet["SvB_MA_q_score"] = np.concatenate([
                selev.SvB_MA.q_1234[:, np.newaxis],
                selev.SvB_MA.q_1324[:, np.newaxis],
                selev.SvB_MA.q_1423[:, np.newaxis],
            ], axis=1)

    if run_SvB and classifier_SvB_FeynNet is not None:
        tmp_mask_fn = (
            (selev.fourTag & quadJet[quadJet.selected][:, 0].SR)
            if run_systematics
            else np.full(len(selev), True)
        )
        # _higgs_cand_flags needs quadJet_selected; set it temporarily here since
        # _assign_output_vars hasn't run yet.
        selev["quadJet_selected"] = quadJet[quadJet.selected][:, 0]
        compute_SvB_FeynNet(selev, tmp_mask_fn, SvB_FeynNet=classifier_SvB_FeynNet)

    if "SvB_FeynNet" in selev.fields:
        quadJet["SvB_FeynNet_reweight"] = selev.SvB_FeynNet.reweight

    return apply_FvT


def _assign_output_vars(selev, diJet, quadJet, run_SvB, cand_cfg):
    """Assign all derived fields to selev."""
    selev["diJet"]            = diJet
    selev["quadJet"]          = quadJet
    selev["quadJet_selected"] = quadJet[quadJet.selected][:, 0]
    selev["passDiJetMass"]    = ak.any(quadJet.passDiJetMass, axis=1)

    arg_min_close_dr = np.argmin(quadJet.close.dr, axis=1).to_numpy()
    selev["quadJet_min_dr"] = quadJet[np.array(range(len(quadJet))), arg_min_close_dr]

    selev["m4j"]      = selev.v4j.mass
    selev["m4j_HHSR"] = ak.where(~selev.quadJet_selected.HHSR, -2, selev.m4j)
    selev["m4j_ZHSR"] = ak.where(~selev.quadJet_selected.ZHSR, -2, selev.m4j)
    selev["m4j_ZZSR"] = ak.where(~selev.quadJet_selected.ZZSR, -2, selev.m4j)

    selev['leadStM_selected'] = selev.quadJet_selected.lead.mass
    selev['sublStM_selected'] = selev.quadJet_selected.subl.mass

    selev['dijet_HHSR'] = ak.zip({
        "lead_m": ak.where(~selev.quadJet_selected.HHSR, -2, selev.leadStM_selected),
        "subl_m": ak.where(~selev.quadJet_selected.HHSR, -2, selev.sublStM_selected),
    })
    selev['dijet_ZHSR'] = ak.zip({
        "lead_m": ak.where(~selev.quadJet_selected.ZHSR, -2, selev.leadStM_selected),
        "subl_m": ak.where(~selev.quadJet_selected.ZHSR, -2, selev.sublStM_selected),
    })
    selev['dijet_ZZSR'] = ak.zip({
        "lead_m": ak.where(~selev.quadJet_selected.ZZSR, -2, selev.leadStM_selected),
        "subl_m": ak.where(~selev.quadJet_selected.ZZSR, -2, selev.sublStM_selected),
    })

    selev["region"] = ak.zip({
        "SR": selev["quadJet_selected"].SR,
        "SB": selev["quadJet_selected"].SB,
    })

    svb_cfg = (cand_cfg or {}).get('svb', {})
    if run_SvB:
        if "SvB_MA" in selev.fields:
            svb_ps = selev["SvB_MA"].ps
        elif "SvB_FeynNet" in selev.fields:
            svb_ps = selev["SvB_FeynNet"].ps
        else:
            svb_ps = None
        if svb_ps is not None:
            selev["passSvB"] = svb_ps > svb_cfg.get('passSvB_min', 0.80)
            selev["failSvB"] = svb_ps < svb_cfg.get('failSvB_max', 0.05)


# ---------------------------------------------------------------------------

def create_cand_jet_dijet_quadjet(
    selev,
    apply_FvT: bool = False,
    classifier_FvT=None,
    run_SvB: bool = False,
    run_systematics: bool = False,
    classifier_SvB=None,
    classifier_SvB_MA=None,
    classifier_SvB_FeynNet=None,
    processOutput=None,
    isRun3=False,
    include_lowptjets=False,
    label3b: str = "threeTag",
    weights: Weights = None,
    list_weight_names: list[str] = None,
    analysis_selections: ak.Array = None,
    cand_cfg: dict = None,
):
    """
    Creates candidate jets, dijets, and quadjets for event selection.

    Parameters:
    -----------
    selev : ak.Array
        The selected events.
    apply_FvT : bool, optional
        Whether to apply FvT weights. Defaults to False.
    run_SvB : bool, optional
        Whether to run SvB classification. Defaults to False.
    run_systematics : bool, optional
        Whether to run systematics. Defaults to False.
    classifier_SvB : optional
        The SvB classifier. Defaults to None.
    classifier_SvB_MA : optional
        The SvB_MA classifier. Defaults to None.
    processOutput : optional
        Output dictionary for processing. Defaults to None.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.
        Overridden by ``cand_cfg['quadjet_selection']['mode']`` when present.
    cand_cfg : dict, optional
        Thresholds loaded from candidates_selection_thresholds.yml.
        If None, hard-coded defaults are used.
        ``quadjet_selection.mode`` ('run2' or 'run3') controls which quadjet
        selection algorithm is used, taking precedence over ``isRun3``.

    Returns:
    --------
    selev : ak.Array
        Events with candidate jet, dijet, quadjet, and region fields added.
    """
    # Resolve quadjet selection mode: YAML key takes precedence over isRun3 flag
    _mode = (cand_cfg or {}).get('quadjet_selection', {}).get('mode')
    if _mode == 'run3':
        isRun3 = True
    elif _mode == 'run2':
        isRun3 = False

    selev = cand_jet_selection(selev, include_lowptjets, cand_cfg=cand_cfg)
    selev["v4j"] = selev.canJet.sum(axis=1)

    _compute_vbf_variables(selev, cand_cfg)

    diJet, diJetDr, sr2_cfg = _build_dijets(selev, cand_cfg, isRun3)
    quadJet = _build_quadjets(selev, diJet, diJetDr, sr2_cfg, cand_cfg, isRun3)
    del diJetDr

    apply_FvT = _apply_ml_scores(
        selev, quadJet, apply_FvT, classifier_FvT,
        run_SvB, run_systematics, classifier_SvB, classifier_SvB_MA, classifier_SvB_FeynNet,
        weights, list_weight_names, analysis_selections, label3b,
    )

    _assign_output_vars(selev, diJet, quadJet, run_SvB, cand_cfg)
    del diJet, quadJet

    return selev
