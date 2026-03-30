import numpy as np
import awkward as ak
import logging
import yaml
import os
from src.physics.common import drClean, compute_puid
from src.physics.objects.jet_corrections import apply_jet_veto_maps, apply_jerc_corrections, apply_jerc_corrections_jsonpog
from src.physics.objects.jet_tools import compute_jet_id
from coffea4bees.analysis.trigger_emulator.helpers import compute_emulation_vars
from copy import copy
from typing import Dict, Any


def load_object_selection_config(path: str) -> dict:
    """
    Load object selection thresholds from a YAML file.

    Parameters:
    -----------
    path : str
        Path to the YAML file containing threshold definitions.

    Returns:
    --------
    dict
        Dictionary of thresholds, suitable for passing as ``sel_cfg``
        to the selection functions.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def muon_selection(muon: ak.Array, isRun3: bool = False, sel_cfg: dict = None) -> ak.Array:
    """
    Selects muons based on kinematic, isolation, and identification criteria.

    Parameters:
    -----------
    muon : ak.Array
        The muon collection containing fields such as `pt`, `eta`, `pfRelIso04_all`, `looseId`, `dz`, and `dxy`.
    isRun3 : bool, optional
        Whether to apply Run 3 selection criteria. Defaults to False.
    sel_cfg : dict, optional
        Dictionary of threshold overrides loaded from ``object_selection_thresholds.yml``.
        When ``None`` all thresholds fall back to their hardcoded defaults.

    Returns:
    --------
    ak.Array
        A collection of selected muons.
    """
    cfg = (sel_cfg or {}).get('muon', {})
    pt_min    = cfg.get('pt_min', 10)
    eta_max   = cfg.get('eta_max_run3' if isRun3 else 'eta_max_run2', 2.4 if isRun3 else 2.5)
    iso_max   = cfg.get('iso_max', 0.15)
    ip        = cfg.get('ip_cuts', {})
    barrel_eta = ip.get('barrel_eta_boundary', 1.479)
    barrel_dz  = ip.get('barrel_dz_max', 0.1)
    barrel_dxy = ip.get('barrel_dxy_max', 0.05)
    endcap_dz  = ip.get('endcap_dz_max', 0.2)
    endcap_dxy = ip.get('endcap_dxy_max', 0.1)

    muon_kin    = (muon.pt > pt_min) & (abs(muon.eta) < eta_max)
    muon_iso_ID = (muon.pfRelIso04_all < iso_max) & muon.looseId

    if isRun3:
        muon_IP = (
            ((abs(muon.eta) < barrel_eta) & (abs(muon.dz) < barrel_dz) & (abs(muon.dxy) < barrel_dxy)) |
            ((abs(muon.eta) >= barrel_eta) & (abs(muon.dz) < endcap_dz) & (abs(muon.dxy) < endcap_dxy))
        )
    else:
        muon_IP = True

    muon['selected'] = muon_kin & muon_iso_ID & muon_IP

    return muon[muon.selected]


def electron_selection(electron: ak.Array, isRun3: bool = False, sel_cfg: dict = None) -> ak.Array:
    """
    Selects electrons based on kinematic, isolation, and identification criteria.

    Parameters:
    -----------
    electron : ak.Array
        The electron collection containing fields such as `pt`, `eta`, `pfRelIso03_all`, `mvaNoIso_WP90`, `mvaFall17V2Iso_WP90`, `dz`, and `dxy`.
    isRun3 : bool, optional
        Whether to apply Run 3 selection criteria. Defaults to False.
    sel_cfg : dict, optional
        Dictionary of threshold overrides loaded from ``object_selection_thresholds.yml``.
        When ``None`` all thresholds fall back to their hardcoded defaults.

    Returns:
    --------
    ak.Array
        A boolean mask indicating selected electrons.
    """
    cfg = (sel_cfg or {}).get('electron', {})
    pt_min  = cfg.get('pt_min', 15)
    eta_max = cfg.get('eta_max', 2.5)
    iso_max = cfg.get('iso_max', 0.15)
    ip      = cfg.get('ip_cuts', {})
    barrel_eta = ip.get('barrel_eta_boundary', 1.479)
    barrel_dz  = ip.get('barrel_dz_max', 0.1)
    barrel_dxy = ip.get('barrel_dxy_max', 0.05)
    endcap_dz  = ip.get('endcap_dz_max', 0.2)
    endcap_dxy = ip.get('endcap_dxy_max', 0.1)

    electron_kin    = (electron.pt > pt_min) & (abs(electron.eta) < eta_max)
    electron_iso_ID = (electron.pfRelIso03_all < iso_max) & (
        getattr(electron, 'mvaNoIso_WP90') if isRun3 else getattr(electron, 'mvaFall17V2Iso_WP90')
    )

    electron_IP = (
        ((abs(electron.eta) < barrel_eta) & (abs(electron.dz) < barrel_dz) & (abs(electron.dxy) < barrel_dxy)) |
        ((abs(electron.eta) >= barrel_eta) & (abs(electron.dz) < endcap_dz) & (abs(electron.dxy) < endcap_dxy))
    ) if isRun3 else True

    electron['selected'] = electron_kin & electron_iso_ID & electron_IP

    return electron[electron.selected]

def lepton_selection(event: ak.Array, isRun3: bool = False, sel_cfg: dict = None) -> ak.Array:
    """
    Selects leptons (muons and electrons) and adds them to the event.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Muon` and `Electron`.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.
    sel_cfg : dict, optional
        Dictionary of threshold overrides. Passed through to ``muon_selection``
        and ``electron_selection``.

    Returns:
    --------
    ak.Array
        The input event data with additional fields:
        - `selMuon`: Selected muons.
        - `selElec`: Selected electrons (if present).
    """
    # Select muons
    event['selMuon'] = muon_selection(event.Muon, isRun3, sel_cfg)

    # Select electrons if present
    if 'Electron' in event.fields:
        event['selElec'] = electron_selection(event.Electron, isRun3, sel_cfg)
        event['selLepton'] = ak.concatenate([event.selElec, event.selMuon], axis=1)
    else:
        event['selLepton'] = event.selMuon

    return event

def jet_selection(
    event: ak.Array,
    corrections_metadata: dict,
    isRun3: bool = False,
    isMC: bool = False,
    isSyntheticData: bool = False,
    isSyntheticMC: bool = False,
    dataset: str = '',
    doLeptonRemoval: bool = True,
    do_jet_veto_maps: bool = False,
    apply_mixeddata_sel: bool = False,
    override_selected_with_flavor_bit: bool = False,
    sel_cfg: dict = None
) -> ak.Array:
    """
    Applies jet selection criteria and creates new variables for the event data.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Jet`.
    corrections_metadata : dict
        Metadata containing corrections and configuration information, such as b-tagging working points.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.
    isMC : bool, optional
        Whether the data is Monte Carlo simulation. Defaults to False.
    isSyntheticData : bool, optional
        Whether the data is synthetic. Defaults to False.
    isSyntheticMC : bool, optional
        Whether the Monte Carlo data is synthetic. Defaults to False.
    dataset : str, optional
        The dataset name. Defaults to an empty string.
    doLeptonRemoval : bool, optional
        Whether to perform lepton removal. Defaults to True.
    do_jet_veto_maps : bool, optional
        Whether to apply jet veto maps. Defaults to False.
    mixeddata_sel : bool, optional
        Whether to apply mixeddata selection as in HIG-22-011. Defaults to False.
    override_selected_with_flavor_bit : bool, optional
        Whether to override selected jets with flavor bit. Defaults to False.
    sel_cfg : dict, optional
        Dictionary of threshold overrides loaded from ``object_selection_thresholds.yml``.
        When ``None`` all thresholds fall back to their hardcoded defaults.

    Returns:
    --------
    ak.Array
        The input event data with additional fields for jet selection and tagging:
        - `Jet['lepton_cleaned']`: Boolean mask for jets cleaned of leptons.
        - `Jet['jet_veto_maps']`: Boolean mask for jets passing veto maps (if applied).
        - `Jet['bRegCorr']`: Regression correction factor for jets (Run3 only).
        - `Jet['btagScore']`: B-tagging score for jets.
        - `Jet['pileup']`: Boolean mask for pileup jets.
        - `Jet['selected_loose']`: Boolean mask for loosely selected jets.
        - `Jet['selected']`: Boolean mask for selected jets.
        - `Jet['tagged']`: Boolean mask for b-tagged jets.
        - `Jet['tagged_loose']`: Boolean mask for loosely b-tagged jets.
        - `nJet_selected`: Number of selected jets.
        - `selJet_no_bRegCorr`: Jets selected without bRegCorr applied.
        - `selJet`: Jets selected with bRegCorr applied.
        - `tagJet`: Jets tagged as b-jets.
        - `tagJet_loose`: Jets loosely tagged as b-jets.
        - `nJet_tagged`: Number of b-tagged jets.
        - `nJet_tagged_loose`: Number of loosely b-tagged jets.
    """
    # Initialize lepton-cleaned jets
    event['Jet', 'lepton_cleaned'] = np.full(len(event), True) if not doLeptonRemoval else drClean(event.Jet, event['selLepton'])[1]

    # Apply jet veto maps if required
    if do_jet_veto_maps:
        event['Jet', 'jet_veto_maps'] = apply_jet_veto_maps(corrections_metadata['jet_veto_maps'], event.Jet)
        event['Jet'] = event['Jet'][event['Jet', 'jet_veto_maps']]

    # Resolve jet thresholds from sel_cfg (or fall back to hardcoded defaults)
    jet_cfg = (sel_cfg or {}).get('jet', {})

    # Run3-specific jet selection
    if isRun3:
        r3 = jet_cfg.get('run3', {})
        r3_pu = r3.get('pileup', {})
        r3_sl = r3.get('selected_loose', {})
        r3_s  = r3.get('selected', {})

        r3_puId_thr      = r3_pu.get('puId_threshold', 7)
        r3_pu_pt_thr     = r3_pu.get('pt_threshold', 50)
        r3_fwd_eta_min   = r3_pu.get('forward_eta_min', 2.4)
        r3_fwd_pt_thr    = r3_pu.get('forward_pt_threshold', 40)
        r3_sl_pt_min     = r3_sl.get('pt_min', 20)
        r3_sl_eta_max    = r3_sl.get('eta_max', 4.7)
        r3_sl_jetId_tag  = r3_sl.get('jetId_tag', 'AK4PUPPI_TightLeptonVeto')
        r3_s_pt_min      = r3_s.get('pt_min', 30)
        r3_s_eta_max     = r3_s.get('eta_max', 2.4)
        r3_s_jetId_tag   = r3_s.get('jetId_tag', 'AK4PUPPI_TightLeptonVeto')
        r3_s_run2        = r3.get('selected_run2', {})
        r2_s_pt_min      = r3_s_run2.get('pt_min', 40)

        event['Jet', 'bRegCorr'] = 1.0
        event['Jet', 'btagScore'] = event.Jet.btagPNetB

        if not isSyntheticData:
            #### temporary hack
            if '2024' in dataset:
                event['Jet'] = apply_jerc_corrections_jsonpog(
                    event,
                    corrections_metadata=corrections_metadata,
                    isMC=isMC,
                    dataset=dataset,
                    run_systematics=False,
                    jet_type="AK4PFPuppi"
                )
            else:
                event['Jet'] = ak.where(
                    event.Jet.btagScore >= corrections_metadata['btagWP']['L'],
                    apply_jerc_corrections(
                        event,
                        corrections_metadata=corrections_metadata,
                        isMC=isMC,
                        run_systematics=False,
                        dataset=dataset,
                        jet_corr_factor=event.Jet.PNetRegPtRawCorr * event.Jet.PNetRegPtRawCorrNeutrino,
                        jet_type="AK4PFPuppiPNetRegressionPlusNeutrino"
                    ),
                    apply_jerc_corrections_jsonpog(
                        event,
                        corrections_metadata=corrections_metadata,
                        isMC=isMC,
                        dataset=dataset,
                        run_systematics=False,
                        jet_type="AK4PFPuppi"
                    )
                )

        event['Jet', 'puId'] = 10
        if 'jetId' in event.Jet.fields: ###### temporary hack before using nanoV15
            event['Jet', 'passJetId_loose'] = event.Jet.jetId >= 2
            event['Jet', 'passJetId'] = event.Jet.jetId >= 2
        else:
            event['Jet', 'passJetId_loose'] = compute_jet_id(event.Jet, corrections_metadata['jet_id'], r3_sl_jetId_tag)
            event['Jet', 'jetId'] = event['Jet', 'passJetId_loose']
            if r3_s_jetId_tag == r3_sl_jetId_tag:
                event['Jet', 'passJetId'] = event['Jet', 'passJetId_loose']
            else:
                event['Jet', 'passJetId'] = compute_jet_id(event.Jet, corrections_metadata['jet_id'], r3_s_jetId_tag)
        # event['Jet', 'pileup'] = ((event.Jet.puId < r3_puId_thr) & (event.Jet.pt < r3_pu_pt_thr)) | ((np.abs(event.Jet.eta) > r3_fwd_eta_min) & (event.Jet.pt < r3_fwd_pt_thr))
        event['Jet', 'selected_loose'] = (event.Jet.pt >= r3_sl_pt_min) & event.Jet.passJetId_loose & event.Jet.lepton_cleaned & (np.abs(event.Jet.eta) <= r3_sl_eta_max)
        event['Jet', 'selected'] = (event.Jet.pt >= r3_s_pt_min) & (np.abs(event.Jet.eta) <= r3_s_eta_max) & event.Jet.passJetId & event.Jet.lepton_cleaned
        event['Jet', 'selected_run2'] = (event.Jet.pt >= r2_s_pt_min) & (np.abs(event.Jet.eta) <= r3_s_eta_max) & event.Jet.passJetId & event.Jet.lepton_cleaned

    # Non-Run3 jet selection
    else:
        event['Jet', 'calibration'] = event.Jet.pt / (event.Jet.pt_raw if 'pt_raw' in event.Jet.fields else ak.full_like(event.Jet.pt, 1))
        event['Jet', 'btagScore'] = event.Jet.btagDeepFlavB

        if apply_mixeddata_sel:
            r2 = jet_cfg.get('run2', {}).get('mixeddata', {})
            r2_pu = r2.get('pileup', {})
            r2_sl = r2.get('selected_loose', {})
            r2_s  = r2.get('selected', {})

            r2_puId_thr  = r2_pu.get('puId_threshold', 6)
            r2_pu_pt_thr = r2_pu.get('pt_threshold', 50)
            r2_sl_pt_min = r2_sl.get('pt_min', 20)
            r2_s_pt_min  = r2_s.get('pt_min', 40)
            r2_s_eta_max = r2_s.get('eta_max', 2.4)

            event['Jet', 'pileup'] = ((event.Jet.puId < r2_puId_thr) & (event.Jet.pt < r2_pu_pt_thr))
            event['Jet', 'selected_loose'] = (event.Jet.pt >= r2_sl_pt_min) & ~event.Jet.pileup
            event['Jet', 'selected'] = (event.Jet.pt >= r2_s_pt_min) & (np.abs(event.Jet.eta) <= r2_s_eta_max) & ~event.Jet.pileup
            event['Jet', 'selected_run2'] = (event.Jet.pt >= r2_s_pt_min) & (np.abs(event.Jet.eta) <= r2_s_eta_max) & ~event.Jet.pileup

        else:
            r2 = jet_cfg.get('run2', {}).get('default', {})
            r2_pu = r2.get('pileup', {})
            r2_sl = r2.get('selected_loose', {})
            r2_s  = r2.get('selected', {})

            r2_pu_pt_thr     = r2_pu.get('pt_threshold', 50)
            r2_fwd_eta_min   = r2_pu.get('forward_eta_min', 2.4)
            r2_fwd_pt_thr    = r2_pu.get('forward_pt_threshold', 40)
            r2_sl_pt_min     = r2_sl.get('pt_min', 20)
            r2_sl_jetId_min  = r2_sl.get('jetId_min', 2)
            r2_s_pt_min      = r2_s.get('pt_min', 40)
            r2_s_eta_max     = r2_s.get('eta_max', 2.4)
            r2_s_jetId_min   = r2_s.get('jetId_min', 2)

            if ('GluGlu' in dataset) and not isSyntheticMC:
                event['Jet', 'corrPuId'] = compute_puid(event.Jet, dataset)
            else:
                event['Jet', 'corrPuId'] = ak.where(event.Jet.puId < 7, True, False)

            event['Jet', 'pileup'] = ((event.Jet.corrPuId) & (event.Jet.pt < r2_pu_pt_thr)) | ((np.abs(event.Jet.eta) > r2_fwd_eta_min) & (event.Jet.pt < r2_fwd_pt_thr))
            event['Jet', 'selected_loose'] = (event.Jet.pt >= r2_sl_pt_min) & ~event.Jet.pileup & (event.Jet.jetId >= r2_sl_jetId_min) & event.Jet.lepton_cleaned
            event['Jet', 'selected'] = (event.Jet.pt >= r2_s_pt_min) & (np.abs(event.Jet.eta) <= r2_s_eta_max) & ~event.Jet.pileup & (event.Jet.jetId >= r2_s_jetId_min) & event.Jet.lepton_cleaned
            event['Jet', 'selected_run2'] = event.Jet.selected

    # Tagging jets
    event['Jet', 'tagged'] = event.Jet.selected & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
    event['Jet', 'tagged_loose'] = event.Jet.selected & (event.Jet.btagScore >= corrections_metadata['btagWP']['L'])
    event['Jet', 'tagged_run2'] = event.Jet.selected_run2 & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
    event['Jet', 'tagged_loose_run2'] = event.Jet.selected_run2 & (event.Jet.btagScore >= corrections_metadata['btagWP']['L'])

    # Override selected jets with flavor bit if required
    if override_selected_with_flavor_bit and "jet_flavor_bit" in event.Jet.fields:
        event['Jet', 'selected'] = (event.Jet.selected) | (event.Jet.jet_flavor_bit == 1)
        event['Jet', 'selected_loose'] = True

    # Count selected jets
    event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)

    # Additional variables
    event['selJet_no_bRegCorr'] = event.Jet[event.Jet.selected]
    event['selJet'] = apply_bRegCorr(event.Jet)
    event['selJetRun2'] = apply_bRegCorr(event.Jet, selected_label='selected_run2', tagged_label='tagged_run2', tagged_loose_label='tagged_loose_run2')
    event['tagJet'] = event.selJet[event.selJet.tagged]
    event['tagJetRun2'] = event.selJetRun2[event.selJetRun2.tagged_run2]
    event['tagJet_loose'] = event.selJet[event.selJet.tagged_loose]
    event['nJet_tagged'] = ak.num(event.tagJet)
    event['nJet_tagged_loose'] = ak.num(event.tagJet_loose)

    # For trigger emulation
    #   Calculate HT and other event variables
    ht_cfg  = jet_cfg.get('ht', {})
    ht_pt_min  = ht_cfg.get('pt_min', 30)
    ht_eta_max = ht_cfg.get('eta_max', 2.5)

    event['Jet', 'muon_cleaned'] = drClean(event.Jet, event.selMuon)[1]
    event['Jet', 'ht_selected'] = (event.Jet.pt >= ht_pt_min) & (np.abs(event.Jet.eta) < ht_eta_max) & event.Jet.muon_cleaned
    event['Jet', 'pfht_selected'] = (event.Jet.pt >= ht_pt_min) & (np.abs(event.Jet.eta) < ht_eta_max)
    compute_emulation_vars(event, useOnlyTop4=(not isRun3))

    return event


def apply_bRegCorr(
        jet: ak.Array,
        selected_label: str = 'selected',
        tagged_label: str = 'tagged',
        tagged_loose_label: str = 'tagged_loose'
    ) -> ak.Array:
    """
    Applies the bRegCorr correction factor to tagged jets and updates their properties.

    Parameters:
    -----------
    jet : ak.Array
        The jet collection containing fields such as `bRegCorr`, `tagged`, `selected`, and other jet properties.

    Returns:
    --------
    ak.Array
        A collection of jets with the bRegCorr correction applied to selected and tagged jets. The updated properties include:
        - `tagged`: Boolean mask for b-tagged jets.
        - `tagged_loose`: Boolean mask for loosely b-tagged jets.
        - `btagScore`: B-tagging score for jets.
        - `puId`: Pileup ID for jets.
        - `jetId`: Jet ID for jets.
        - `hadronFlavour` (if available): Flavour information for jets.
    """
    # Flatten the bRegCorr factors and tagged flags for processing
    bRegCorr_flat = copy(ak.flatten(jet.bRegCorr).to_numpy())
    # tagged_flags_flat = ak.flatten(jet.tagged)
    tagged_flags_flat = ak.flatten(jet[tagged_label])

    # Set bRegCorr to 1.0 for non-tagged jets
    bRegCorr_flat[~tagged_flags_flat] = 1.0

    # Unflatten the bRegCorr factors to match the original structure
    bRegCorr = ak.unflatten(bRegCorr_flat, ak.num(jet.bRegCorr))

    # Apply the bRegCorr factor to selected jets
    selected_jets = jet[jet[selected_label]] * bRegCorr[jet[selected_label]]

    # Update properties of the selected jets
    selected_jets[tagged_label] = jet[jet[selected_label]][tagged_label]
    selected_jets[tagged_loose_label] = jet[jet[selected_label]][tagged_loose_label]
    selected_jets["btagScore"] = jet[jet[selected_label]].btagScore
    selected_jets["puId"] = jet[jet[selected_label]].puId
    selected_jets["jetId"] = jet[jet[selected_label]].jetId

    # Include hadronFlavour if available
    if "hadronFlavour" in jet.fields:
        selected_jets["hadronFlavour"] = jet[jet[selected_label]].hadronFlavour

    return selected_jets

def lowpt_jet_selection(
    event: ak.Array,
    corrections_metadata: dict,
    isRun3: bool = False,
    isMC: bool = False,
    isSyntheticData: bool = False,
    isSyntheticMC: bool = False,
    dataset: str = '',
    doLeptonRemoval: bool = True,
    do_jet_veto_maps: bool = False,
    override_selected_with_flavor_bit: bool = False,
    sel_cfg: dict = None
) -> ak.Array:
    """
    Applies low-pT jet selection criteria to the event data.

    This function identifies and selects jets with low transverse momentum (pT) that meet specific criteria. It also
    applies b-tagging to the selected low-pT jets and calculates additional properties for analysis.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Jet` and `Lepton`.
    corrections_metadata : dict
        Metadata containing corrections and configuration information, such as b-tagging working points.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.
    isMC : bool, optional
        Whether the data is Monte Carlo simulation. Defaults to False.
    isSyntheticData : bool, optional
        Whether the data is synthetic. Defaults to False.
    isSyntheticMC : bool, optional
        Whether the Monte Carlo data is synthetic. Defaults to False.
    dataset : str, optional
        The dataset name. Defaults to an empty string.
    doLeptonRemoval : bool, optional
        Whether to perform lepton removal. Defaults to True.
    do_jet_veto_maps : bool, optional
        Whether to apply jet veto maps. Defaults to False.
    override_selected_with_flavor_bit : bool, optional
        Whether to override selected jets with flavor bit. Defaults to False.
    sel_cfg : dict, optional
        Dictionary of threshold overrides loaded from ``object_selection_thresholds.yml``.
        When ``None`` all thresholds fall back to their hardcoded defaults.

    Returns:
    --------
    ak.Array
        The input event data with additional fields for low-pT jet selection:
        - `Jet['selected_lowpt']`: Boolean mask for low-pT selected jets.
        - `selJets_lowpt`: Jets selected with low-pT criteria.
        - `nJet_selected_lowpt`: Number of low-pT selected jets.
        - `Jet['tagged_lowpt']`: Boolean mask for low-pT b-tagged jets.
        - `Jet['tagged_loose_lowpt']`: Boolean mask for loosely b-tagged low-pT jets.
        - `nJet_tagged_lowpt`: Number of low-pT b-tagged jets.
        - `nJet_tagged_loose_lowpt`: Number of loosely b-tagged low-pT jets.
        - `tagJet_lowpt`: Low-pT jets tagged as b-jets.

    Notes:
    ------
    - Low-pT jets are defined as jets with pT >= 15 GeV and |eta| <= 2.4 that are not already selected as high-pT jets.
    - The function uses b-tagging working points defined in the `corrections_metadata` to classify jets as tagged or
      loosely tagged.
    - The `apply_bRegCorr` function is used to apply regression corrections to the selected low-pT jets.
    """
    # Apply general jet selection first
    event = jet_selection(
        event,
        corrections_metadata=corrections_metadata,
        isRun3=isRun3,
        isMC=isMC,
        isSyntheticData=isSyntheticData,
        isSyntheticMC=isSyntheticMC,
        dataset=dataset,
        doLeptonRemoval=doLeptonRemoval,
        do_jet_veto_maps=do_jet_veto_maps,
        override_selected_with_flavor_bit=override_selected_with_flavor_bit,
        sel_cfg=sel_cfg
    )

    # Resolve low-pT thresholds
    lp_cfg    = (sel_cfg or {}).get('jet', {}).get('lowpt', {})
    lp_pt_min    = lp_cfg.get('pt_min', 15)
    lp_eta_max   = lp_cfg.get('eta_max', 2.4)
    lp_jetId_min = lp_cfg.get('jetId_min', 2)

    # Define low-pT jet selection criteria
    event['Jet', 'selected_lowpt'] = (
        (event.Jet.pt >= lp_pt_min) &
        (np.abs(event.Jet.eta) <= lp_eta_max) &
        ~event.Jet.pileup &
        (event.Jet.jetId >= lp_jetId_min) &
        event.Jet.lepton_cleaned &
        ~event.Jet.selected  # Exclude already selected jets
    )

    # Tagging jets
    event['Jet', 'tagged_lowpt'] = event.Jet.selected_lowpt & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
    event['Jet', 'tagged_loose_lowpt'] = event.Jet.selected_lowpt & (event.Jet.btagScore >= corrections_metadata['btagWP']['L'])

    event['nJet_selected_lowpt'] = ak.sum(event.Jet.selected_lowpt, axis=1)

    # Apply bRegCorr to low-pT selected jets
    event['selJet_no_bRegCorr_lowpt'] = event.Jet[event.Jet.selected_lowpt]
    event['selJet_lowpt'] = apply_bRegCorr(event.Jet, selected_label='selected_lowpt', tagged_label='tagged_lowpt', tagged_loose_label='tagged_loose_lowpt')

    event['tagJet_lowpt'] = event.selJet_lowpt[event.selJet_lowpt.tagged_lowpt]
    event['tagJet_loose_lowpt'] = event.selJet_lowpt[event.selJet_lowpt.tagged_loose_lowpt]
    event['nJet_tagged_lowpt'] = ak.num(event.tagJet_lowpt)
    event['nJet_tagged_loose_lowpt'] = ak.num(event.tagJet_loose_lowpt)

    event['allSelJet'] = ak.concatenate([event.selJet, event.selJet_lowpt], axis=1)
    event['allTagJet'] = ak.concatenate([event.tagJet, event.tagJet_lowpt], axis=1)

    return event
