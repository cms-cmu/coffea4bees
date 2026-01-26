import numpy as np
import awkward as ak
import logging
from coffea4bees.analysis.helpers.object_selection import (
    lepton_selection,
    jet_selection,
    lowpt_jet_selection
)
from src.physics.common import drClean


def apply_dilep_ttbar_selection(event: ak.Array, isRun3: bool = False) -> ak.Array:
    """
    Applies dilepton ttbar selection criteria to the event data.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Muon`, `Electron`, and `MET`.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.

    Returns:
    --------
    ak.Array
        A boolean mask indicating events passing the dilepton ttbar selection criteria.
    """
    dimuon_selection = (event.Muon.pt > 10) & (abs(event.Muon.eta) < 2.4) & (event.Muon.tightId) & (event.Muon.pfRelIso04_all < 0.05)
    nMuon_dimuon_selected = ak.sum(dimuon_selection, axis=1)
    dilepton_mask = nMuon_dimuon_selected >= 2

    jet_mask = event.nJet_tagged == 2

    # Require MET > 40 GeV
    met_mask = event.MET.pt > 30

    # Combine all selection criteria
    selection_mask = dilepton_mask & jet_mask & met_mask
    logging.debug(f"Selection mask: {selection_mask}\n\n")

    return selection_mask

def apply_boosted_4b_selection(event: ak.Array) -> ak.Array:
    """
    Applies boosted object selection criteria for 4b analysis.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `FatJet` and `bdt`.

    Returns:
    --------
    ak.Array
        The input event data with additional fields:
        - `passBoostedKin`: Boolean mask for events passing boosted kinematic selection.
        - `passBoostedSel`: Boolean mask for events passing boosted selection criteria.
    """
    # Sort FatJets by particleNetMD_Xbb in descending order
    event['FatJet'] = event.FatJet[ak.argsort(event.FatJet.particleNetMD_Xbb, axis=1, ascending=False)]

    # Apply kinematic selection to FatJets
    FatJet_selected = (event.FatJet.pt > 300) & (np.abs(event.FatJet.eta) < 2.4)
    nFatJet_selected = ak.sum(FatJet_selected, axis=1)

    # Check if events pass boosted kinematic selection
    event['passBoostedKin'] = nFatJet_selected >= 2

    # Apply additional selection criteria to candidate jets
    tmp_selev = event[event.passBoostedKin]
    candJet1 = (tmp_selev.FatJet[:, 0].msoftdrop > 50) & (tmp_selev.FatJet[:, 0].particleNetMD_Xbb > 0.8)
    candJet2 = (tmp_selev.FatJet[:, 1].particleNet_mass > 50)

    # Apply BDT-based selection if available
    if 'bdt' in tmp_selev.fields:
        passBDT = (tmp_selev.FatJet[:, 1].particleNetMD_Xbb > 0.950) & (tmp_selev.bdt['score'] > 0.03)  ### bdt_score only in picoAOD.chunk.withBDT.root files
        # passBDT = (tmp_selev.FatJet[:, 1].particleNetMD_Xbb > 0.980) & (tmp_selev.bdt['score'] > 0.43)  ### bdt_score only in picoAOD.chunk.withBDT.root files
    else:
        passBDT = np.full(len(tmp_selev), True)

    # Combine all selection criteria
    passBoostedSel = np.full(len(event), False)
    passBoostedSel[event.passBoostedKin] = candJet1 & candJet2 & passBDT
    event['passBoostedSel'] = passBoostedSel

    return event

def apply_4b_selection(
        event,
        corrections_metadata,
        *,
        config: dict = {"do_lepton_jet_cleaning": True, "override_selected_with_flavor_bit": False, "do_jet_veto_maps": False, "isRun3": False, "isMC": False, "isSyntheticData": False, "isSyntheticMC": False},
        dataset: str = '',
        loosePtForSkim: bool = False,
        apply_mixeddata_sel: bool = False
) -> ak.Array:
    """
    Applies object selection criteria for 4b analysis.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Jet` and `Lepton`.
    corrections_metadata : dict
        Metadata containing corrections and configuration information.
    dataset : str, optional
        The dataset name. Defaults to an empty string.
    loosePtForSkim : bool, optional
        Whether to use loose pT cuts for skimming. Defaults to False.
    apply_mixeddata_sel : bool, optional
        Whether to apply mixed data selection. Defaults to False.

    Returns:
    --------
    ak.Array
        The input event data with additional fields for object selection.
    """
    # Combined RunII and 3 selection
    event = lepton_selection(event, config["isRun3"])

    event = jet_selection(event, corrections_metadata, config["isRun3"], config["isMC"], config["isSyntheticData"], config["isSyntheticMC"], dataset, config["do_lepton_jet_cleaning"], config["do_jet_veto_maps"], apply_mixeddata_sel, config["override_selected_with_flavor_bit"])

    event['passJetMult'] = event['nJet_selected'] >= 4

    event['fourTag'] = (event['nJet_tagged'] >= 4)
    event['threeTag'] = (event['nJet_tagged_loose'] == 3) & (event['nJet_selected'] >= 4)
    event['twoTag'] = (event['nJet_tagged_loose'] == 2) & (event['nJet_selected'] >= 4)

    if config["isSyntheticData"] or config["isSyntheticMC"]:
        event['threeTag'] = False
        event['twoTag'] = False

    if config["isRun3"]:
        event['passPreSel'] = event.twoTag | event.threeTag | event.fourTag
    else:
        event['passPreSel'] = event.threeTag | event.fourTag

    event['tag'] = ak.zip({
        "twoTag": event.twoTag,
        "threeTag": event.threeTag,
        "fourTag": event.fourTag,
    })

    # For trigger emulation
    event['Jet', 'muon_cleaned'] = drClean(event.Jet, event.selMuon)[1]
    event['Jet', 'ht_selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) < 2.4) & event.Jet.muon_cleaned
    #  Calculate hT
    event["hT"] = ak.sum(event.Jet[event.Jet.selected_loose].pt, axis=1)
    event["hT_selected"] = ak.sum(event.Jet[event.Jet.selected].pt, axis=1)
    event["hT_trigger"] = ak.sum(event.Jet[event.Jet.ht_selected].pt, axis=1)

    # Only need 30 GeV jets for signal systematics
    if loosePtForSkim:
        mask_jet_lowpt_forskim = (event.Jet.pt >= 15) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId >= 2) & event.Jet.lepton_cleaned
        nJet_selected_lowpt_forskim = ak.sum(mask_jet_lowpt_forskim, axis=1)
        mask_tagjet_lowpt_forskim = mask_jet_lowpt_forskim & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
        event['passJetMult_lowpt_forskim'] = nJet_selected_lowpt_forskim >= 4
        nJet_tagged_lowpt_forskim = ak.num(event.Jet[mask_tagjet_lowpt_forskim])
        event["fourTag_lowpt_forskim"] = (nJet_tagged_lowpt_forskim >= 4)
        event['passPreSel_lowpt_forskim'] = event.threeTag | event.fourTag_lowpt_forskim

    return event

def apply_4b_lowpt_selection(
    event: ak.Array,
    corrections_metadata: dict,
    *,
    dataset: str = '',
    doLeptonRemoval: bool = True,
    loosePtForSkim: bool = False,
    override_selected_with_flavor_bit: bool = False,
    do_jet_veto_maps: bool = False,
    isRun3: bool = False,
    isMC: bool = False,  # Temporary for Run3
    isSyntheticData: bool = False,
    isSyntheticMC: bool = False
) -> ak.Array:
    """
    Applies low-pT jet selection and categorization for 4b analysis.

    This function selects low-pT jets, categorizes events based on tagging criteria, and updates event-level variables
    for further analysis.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Jet` and `Lepton`.
    corrections_metadata : dict
        Metadata containing corrections and configuration information, such as b-tagging working points.
    dataset : str, optional
        The dataset name. Defaults to an empty string.
    doLeptonRemoval : bool, optional
        Whether to perform lepton removal. Defaults to True.
    loosePtForSkim : bool, optional
        Whether to use loose pT cuts for skimming. Defaults to False.
    override_selected_with_flavor_bit : bool, optional
        Whether to override selected jets with flavor bit. Defaults to False.
    do_jet_veto_maps : bool, optional
        Whether to apply jet veto maps. Defaults to False.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.
    isMC : bool, optional
        Whether the data is Monte Carlo simulation. Defaults to False.
    isSyntheticData : bool, optional
        Whether the data is synthetic. Defaults to False.
    isSyntheticMC : bool, optional
        Whether the Monte Carlo data is synthetic. Defaults to False.

    Returns:
    --------
    ak.Array
        The input event data with additional fields for low-pT jet selection and categorization:
        - `passJetMult`: Boolean mask for events with at least 4 selected jets.
        - `fourTag`: Boolean mask for events with at least 4 b-tagged jets.
        - `threeTag`: Boolean mask for events with exactly 3 b-tagged jets.
        - `lowpt_fourTag`: Boolean mask for events with 3 b-tagged jets and at least 1 low-pT b-tagged jet.
        - `lowpt_threeTag`: Boolean mask for events with 3 loosely b-tagged jets and at least 1 low-pT jet.
        - `tag`: Zipped object containing tagging information.
        - `lowpt_categories`: Integer category for low-pT selection.
        - `passPreSel`: Boolean mask for events passing preselection.
        - `Jet['selected']`: Updated mask for selected jets, including low-pT jets.
    """
    # Apply lepton selection
    event = lepton_selection(event, isRun3)

    # Apply low-pT and nominal jet selection
    event = lowpt_jet_selection(
        event,
        corrections_metadata,
        isRun3=isRun3,
        isMC=isMC,
        isSyntheticData=isSyntheticData,
        isSyntheticMC=isSyntheticMC,
        dataset=dataset,
        doLeptonRemoval=doLeptonRemoval,
        do_jet_veto_maps=do_jet_veto_maps,
        override_selected_with_flavor_bit=override_selected_with_flavor_bit
    )

    # Define tagging and categorization
    event['passJetMult'] = event['nJet_selected'] >= 4
    event['fourTag'] = event['nJet_tagged'] >= 4
    event['threeTag'] = (event['nJet_tagged_loose'] == 3) & (
        event['nJet_selected'] >= 4)
    event['lowpt_fourTag'] = (
        (event['nJet_tagged'] == 3) &
        (event['nJet_selected'] >= 4) &
        (event['nJet_tagged_lowpt'] > 0) &
        ~event['fourTag']
    )
    event['lowpt_threeTag'] = (
        (event['nJet_tagged_loose'] == 3) &
        (event['nJet_selected'] >= 4) &
        (event['nJet_tagged_loose_lowpt'] == 0) &
        (event['nJet_selected_lowpt'] > 0)
    )

    # Create tagging object
    event['tag'] = ak.zip({
        "threeTag": event['threeTag'],
        "fourTag": event['fourTag'],
        "lowpt_fourTag": event['lowpt_fourTag'],
        "lowpt_threeTag": event['lowpt_threeTag']
    })

    # Define low-pT categories
    event['lowpt_categories'] = np.where(
        event['fourTag'], 0,
        np.where(
            event['lowpt_fourTag'], 5,
            np.where(
                event['lowpt_threeTag'], 7,
                np.where(event['threeTag'], 3, 9)
            )
        )
    )

    # Update preselection and jet selection
    event['passPreSel'] = event['lowpt_threeTag'] | event['lowpt_fourTag']

    return event
