import numpy as np
import awkward as ak
import logging
import yaml
from src.math_tools.random import Squares
from coffea.nanoevents.methods import vector
from coffea.analysis_tools import Weights

from coffea4bees.analysis.helpers.candidates_selection import (
    cand_jet_selection,
    _compute_vbf_variables,
    _apply_ml_scores,
)


def _build_dijets_ttHbb(selev, cand_cfg=None, isRun3=False):
    """Build dijet pairs and compute dijet-level variables for ttHbb.
    
    Removes the [50, 170] GeV hard dijet mass window to allow exploring the
    entire 2D mass phase space.
    """
    canJet = selev["canJet"]
    pairing = [np.array([[0, 2], [0, 1], [0, 1]]), np.array([[1, 3], [2, 3], [3, 2]])]
    diJet = canJet[:, pairing[0]] + canJet[:, pairing[1]]
    diJet["lead"] = canJet[:, pairing[0]]
    diJet["subl"] = canJet[:, pairing[1]]
    diJet["st"] = diJet["lead"].pt + diJet["subl"].pt
    diJet["dr"] = diJet["lead"].delta_r(diJet["subl"])
    diJet["dphi"] = diJet["lead"].delta_phi(diJet["subl"])

    # Sort diJets within views to be lead/subl by st (Run 2) or pt (Run 3)
    if isRun3:
        diJet = diJet[ak.argsort(diJet.pt, axis=2, ascending=False)]
    else:
        diJet = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
    diJetDr = diJet[ak.argsort(diJet.dr, axis=2, ascending=True)]

    # Unconstrained dijet mass window for full phase space (0 to 2000 GeV)
    minDiJetMass = np.array([[[0.0, 0.0]]])
    maxDiJetMass = np.array([[[2000.0, 2000.0]]])
    diJet["passDiJetMass"] = (minDiJetMass < diJet.mass) & (diJet.mass < maxDiJetMass)

    # Mass-to-dR ratio (MDR) cuts
    mdr_cfg = (cand_cfg or {}).get('mdr', {})
    min_m4j_scale = np.array([mdr_cfg.get('min_m4j_scale', [360, 235])])
    min_dr_offset = np.array([mdr_cfg.get('min_dr_offset', [-0.5, 0.0])])
    max_m4j_scale = np.array([mdr_cfg.get('max_m4j_scale', [650, 650])])
    max_dr_offset = np.array([mdr_cfg.get('max_dr_offset', [0.5,  0.7])])
    max_dr = np.array([mdr_cfg.get('max_dr', [1.5, 1.5])])
    m4j = selev["v4j"].mass[:, np.newaxis, np.newaxis]
    diJet["passMDR"] = (
        (min_m4j_scale / m4j + min_dr_offset < diJet.dr) &
        (diJet.dr < np.maximum(max_m4j_scale / m4j + max_dr_offset, max_dr))
    )

    # Consistency with Higgs mass (mH = 125 GeV)
    mH = 125.0
    diJet["xH"] = (diJet.mass - mH) / (0.1 * diJet.mass)

    del canJet, pairing
    return diJet, diJetDr


def _select_quadjet_ttHbb(quadJet, cand_cfg=None):
    """Pick best quadjet pairing and assign ttHbb SR/SB regions.
    
    SR: Shape 2 (Resonant Single-Higgs L-Shape up to 1000 GeV):
        Horizontal arm: m_subl in [85, 185] GeV, m_lead in [25, 1000] GeV
        Vertical arm:   m_lead in [90, 185] GeV, m_subl in [25, 1000] GeV
        SR = in_h_arm | in_v_arm
        
    SB: Full preselection phase space excluding SR:
        m_lead in [25, 1000] GeV and m_subl in [25, 1000] GeV and (~SR)
    """
    m_lead = quadJet["lead"].mass
    m_subl = quadJet["subl"].mass

    in_h_arm = (m_subl >= 85.0) & (m_subl <= 185.0) & (m_lead >= 25.0) & (m_lead <= 1000.0)
    in_v_arm = (m_lead >= 90.0) & (m_lead <= 185.0) & (m_subl >= 25.0) & (m_subl <= 1000.0)
    quadJet["SR"] = in_h_arm | in_v_arm

    in_analysis_box = (m_lead >= 25.0) & (m_lead <= 1000.0) & (m_subl >= 25.0) & (m_subl <= 1000.0)
    quadJet["SB"] = in_analysis_box & (~quadJet["SR"])

    # Compute Euclidean radial distance for monitoring
    quadJet["rH"] = np.sqrt((m_lead - 125.0)**2 + (m_subl - 125.0)**2)

    # Ranking: prioritize MDR passing pairings, with random tie-breaker
    quadJet["rank"] = (
        10 * quadJet.passDiJetMass
        + quadJet.lead.passMDR
        + quadJet.subl.passMDR
        + quadJet.random
    )
    quadJet["selected"] = quadJet.rank == np.max(quadJet.rank, axis=1)


def _build_quadjets_ttHbb(selev, diJet, diJetDr, cand_cfg=None, isRun3=False):
    """Build quadjet candidates and assign signal regions for ttHbb."""
    rng_0 = Squares("quadJetSelection")
    rng_1 = rng_0.shift(1)
    rng_2 = rng_0.shift(2)
    counter = selev.event

    quadJet = ak.zip({
        "lead": diJet[:, :, 0],
        "subl": diJet[:, :, 1],
        "close": diJetDr[:, :, 0],
        "other": diJetDr[:, :, 1],
        "passDiJetMass": ak.all(diJet.passDiJetMass, axis=2),
        "random": np.concatenate([
            rng_0.uniform(counter, low=0.1, high=0.9)[:, np.newaxis],
            rng_1.uniform(counter, low=0.1, high=0.9)[:, np.newaxis],
            rng_2.uniform(counter, low=0.1, high=0.9)[:, np.newaxis],
        ], axis=1),
    })

    quadJet["dr"] = quadJet["lead"].delta_r(quadJet["subl"])
    quadJet["dphi"] = quadJet["lead"].delta_phi(quadJet["subl"])
    quadJet["deta"] = quadJet["lead"].eta - quadJet["subl"].eta
    quadJet["v4jmass"] = selev["v4j"].mass

    _select_quadjet_ttHbb(quadJet, cand_cfg)

    return quadJet


def _assign_output_vars_ttHbb(selev, diJet, quadJet, run_SvB=False, cand_cfg=None):
    """Assign all derived candidate fields to selev."""
    selev["diJet"] = diJet
    selev["quadJet"] = quadJet
    selev["quadJet_selected"] = quadJet[quadJet.selected][:, 0]
    selev["passDiJetMass"] = ak.any(quadJet.passDiJetMass, axis=1)

    arg_min_close_dr = ak.argmin(quadJet.close.dr, axis=1, keepdims=True)
    selev["quadJet_min_dr"] = quadJet[arg_min_close_dr][:, 0]

    selev["m4j"] = selev.v4j.mass
    selev['leadStM_selected'] = selev.quadJet_selected.lead.mass
    selev['sublStM_selected'] = selev.quadJet_selected.subl.mass

    # Single-Higgs SR dijet mass monitoring
    selev['dijet_SR'] = ak.zip({
        "lead_m": ak.where(~selev.quadJet_selected.SR, -2, selev.leadStM_selected),
        "subl_m": ak.where(~selev.quadJet_selected.SR, -2, selev.sublStM_selected),
    })

    selev["region"] = ak.zip({
        "inclusive": np.full(len(selev.event), True),
        "SR": selev["quadJet_selected"].SR,
        "SB": selev["quadJet_selected"].SB,
    })

    svb_cfg = (cand_cfg or {}).get('svb', {})
    if run_SvB:
        if "SvB_MA" in selev.fields:
            if "ps_ttHbb" in selev["SvB_MA"].fields:
                svb_ps = selev["SvB_MA"].ps_ttHbb
            elif "pttHbb" in selev["SvB_MA"].fields:
                svb_ps = selev["SvB_MA"].pttHbb
            elif "ps" in selev["SvB_MA"].fields:
                svb_ps = selev["SvB_MA"].ps
            else:
                svb_ps = None
        elif "SvB_FeynNet" in selev.fields:
            svb_ps = 1.0 - selev["SvB_FeynNet"].p_bkg
        else:
            svb_ps = None
        if svb_ps is not None:
            selev["passSvB"] = svb_ps > svb_cfg.get('passSvB_min', 0.80)
            selev["failSvB"] = svb_ps < svb_cfg.get('failSvB_max', 0.05)


def create_cand_jet_dijet_quadjet_ttHbb(
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
    """Creates candidate jets, dijets, and quadjets unconstrained for ttHbb analysis."""
    selev = cand_jet_selection(selev, include_lowptjets, cand_cfg=cand_cfg)
    selev["v4j"] = selev.canJet.sum(axis=1)

    _compute_vbf_variables(selev, cand_cfg)

    diJet, diJetDr = _build_dijets_ttHbb(selev, cand_cfg, isRun3)
    quadJet = _build_quadjets_ttHbb(selev, diJet, diJetDr, cand_cfg, isRun3)
    del diJetDr

    apply_FvT = _apply_ml_scores(
        selev, quadJet, apply_FvT, classifier_FvT,
        run_SvB, run_systematics, classifier_SvB, classifier_SvB_MA, classifier_SvB_FeynNet,
        weights, list_weight_names, analysis_selections, label3b,
    )

    _assign_output_vars_ttHbb(selev, diJet, quadJet, run_SvB, cand_cfg)
    del diJet, quadJet

    return selev
