import logging
import numpy as np
import awkward as ak
import yaml
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea.analysis_tools import PackedSelection, Weights
from coffea4bees.analysis.helpers.cutflow import cutflow_4b ## change thiss
from coffea4bees.analysis.helpers.object_selection import apply_bRegCorr ## (electron_selection, jet_selection, )
from src.physics.common import drClean
from src.physics.event_selection import apply_event_selection
from src.skimmer.picoaod import PicoAOD
from src.physics.objects.jet_corrections import apply_jet_veto_maps, apply_jerc_corrections
from src.skimmer.mc_weight_outliers import OutlierByMedian


def jet_selection_trg(
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
    override_selected_with_flavor_bit: bool = False
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

    # Run3-specific jet selection
    if isRun3:
        event['Jet', 'bRegCorr'] = 1.0
        event['Jet', 'btagScore'] = event.Jet.btagPNetB

        if not isSyntheticData:
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
                apply_jerc_corrections(
                    event,
                    corrections_metadata=corrections_metadata,
                    isMC=isMC,
                    run_systematics=False,
                    dataset=dataset,
                    jet_type="AK4PFPuppi.txt"
                )
            )

        event['Jet', 'puId'] = 10
        event['Jet', 'pileup'] = ((event.Jet.puId < 7) & (event.Jet.pt < 50)) | ((np.abs(event.Jet.eta) > 2.4) & (event.Jet.pt < 40))
        event['Jet', 'selected_loose'] = (event.Jet.pt >= 20) & (event.Jet.jetId >= 2) & event.Jet.lepton_cleaned & (np.abs(event.Jet.eta) <= 4.7)
        event['Jet', 'selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId >= 2) & event.Jet.lepton_cleaned


    # Tagging jets
    event['Jet', 'tagged'] = event.Jet.selected & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
    event['Jet', 'tagged_loose'] = event.Jet.selected & (event.Jet.btagScore >= corrections_metadata['btagWP']['L'])

    # Override selected jets with flavor bit if required
    if override_selected_with_flavor_bit and "jet_flavor_bit" in event.Jet.fields:
        event['Jet', 'selected'] = (event.Jet.selected) | (event.Jet.jet_flavor_bit == 1)
        event['Jet', 'selected_loose'] = True

    # Count selected jets
    event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)

    # Additional variables
    event['selJet_no_bRegCorr'] = event.Jet[event.Jet.selected]
    event['selJet'] = apply_bRegCorr(event.Jet)
    event['tagJet'] = event.selJet[event.selJet.tagged]
    event['tagJet_loose'] = event.selJet[event.selJet.tagged_loose]
    event['nJet_tagged'] = ak.num(event.tagJet)
    event['nJet_tagged_loose'] = ak.num(event.tagJet_loose)

    # For trigger emulation
    event['Jet', 'muon_cleaned'] = drClean(event.Jet, event.selMuon)[1]
    event['Jet', 'ht_selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) < 2.4) & event.Jet.muon_cleaned
    event['Jet', 'pfht_selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) < 2.4) 

    return event



def electron_selection_trg(electron: ak.Array, isRun3: bool = False) -> ak.Array:
    """
    Selects electrons based on kinematic, isolation, and identification criteria.

    Parameters:
    -----------
    electron : ak.Array
        The electron collection containing fields such as `pt`, `eta`, `pfRelIso03_all`, `mvaNoIso_WP90`, `mvaFall17V2Iso_WP90`, `dz`, and `dxy`.
    isRun3 : bool, optional
        Whether to apply Run 3 selection criteria. Defaults to False.

    Returns:
    --------
    ak.Array
        A boolean mask indicating selected electrons.
    """
    electron_kin = (electron.pt > 32) & (abs(electron.eta) < 2.5)
    electron_iso_ID = (electron.pfRelIso03_all < 0.15) & (
        getattr(electron, 'mvaIso_WP80') if isRun3 else getattr(electron, 'mvaFall17V2Iso_WP90')  ### Change this for Run 2
    )

    electron_IP = (
        ((abs(electron.eta) < 1.479) & (abs(electron.dz) < 0.1) & (abs(electron.dxy) < 0.05)) |
        ((abs(electron.eta) >= 1.479) & (abs(electron.dz) < 0.2) & (abs(electron.dxy) < 0.1))
    ) if isRun3 else True

    electron['selected'] = electron_kin & electron_iso_ID & electron_IP
    return electron[electron.selected]


def apply_1e4jet_selection(
        event, 
        corrections_metadata, 
        *,
        dataset: str = '',
        doLeptonRemoval: bool = True,
        loosePtForSkim: bool = False,
        override_selected_with_flavor_bit: bool = False,
        do_jet_veto_maps: bool = False,
        isRun3: bool = False,
        isMC: bool = False,  ### temporary for Run3
        isSyntheticData: bool = False,
        isSyntheticMC: bool = False,
        apply_mixeddata_sel: bool = False
) -> ak.Array:
    """
    Applies object selection criteria for trigger studies.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Jet` and `Lepton`.
    corrections_metadata : dict
        Metadata containing corrections and configuration information.
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
    apply_mixeddata_sel : bool, optional
        Whether to apply mixed data selection. Defaults to False.

    Returns:
    --------
    ak.Array
        The input event data with additional fields for object selection.
    """
    # Combined RunII and 3 selection
    event = electron_selection_trg(event, isRun3)
    
    event = jet_selection_trg(event, corrections_metadata, isRun3, isMC, isSyntheticData, isSyntheticMC, dataset, doLeptonRemoval, do_jet_veto_maps,apply_mixeddata_sel, override_selected_with_flavor_bit)

    event['passJetMult'] = event['nJet_selected'] >= 4

    event['fourTag'] = (event['nJet_tagged'] >= 4)
    event['threeTag'] = (event['nJet_tagged_loose'] == 3) & (event['nJet_selected'] >= 4)
    event['twoTag'] = (event['nJet_tagged_loose'] == 2) & (event['nJet_selected'] >= 4)

    if isSyntheticData or isSyntheticMC:
        event['threeTag'] = False
        event['twoTag'] = False

    if isRun3:
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


class Skimmer(PicoAOD):
    def __init__(
            self, 
            loosePtForSkim=False, 
            skim4b=False, 
            mc_outlier_threshold=200, 
            corrections_metadata=None,
            *args, **kwargs
        ):
        if skim4b:
            kwargs["pico_base_name"] = f'picoAOD_fourTag'
        super().__init__(*args, **kwargs)
        self.loosePtForSkim = loosePtForSkim
        self.skim4b = skim4b
        self.corrections_metadata = corrections_metadata if corrections_metadata is not None else {}
        self.mc_outlier_threshold = mc_outlier_threshold
        # Always use cutflow_4b unless explicitly overridden
        self._cutFlow = cutflow_4b()

    def select(self, events):
        year    = events.metadata['year']
        dataset = events.metadata['dataset']
        processName = events.metadata['processName']

        # Set process and datset dependent flags
        config = processor_config(processName, dataset, events)
        logging.debug(f'config={config}\n')

        events = apply_event_selection(
            events,
            self.corrections_metadata[year],
            cut_on_lumimask=config["cut_on_lumimask"]
        )

        if config["do_jet_calibration"]:
            jets = apply_jerc_corrections(
                events,
                corrections_metadata=self.corrections_metadata[year],
                isMC=config["isMC"],
                run_systematics=False,
                dataset=dataset
            )
            events["Jet"] = jets

        events = apply_1e4jet_selection(
            events,
            self.corrections_metadata[year],
            dataset=dataset,
            doLeptonRemoval=config["do_lepton_jet_cleaning"],
            loosePtForSkim=self.loosePtForSkim,
            isRun3=config["isRun3"],
            isMC=config["isMC"]
        )

        weights = Weights(len(events), storeIndividual=True)

        # general event weights
        if config["isMC"]:
            weights.add("genweight_", events.genWeight)

        selections = PackedSelection()
        selections.add("lumimask", events.lumimask)
        selections.add("passNoiseFilter", events.passNoiseFilter)
        selections.add("passHLT", (events.passHLT if config["cut_on_HLT_decision"] else np.full(len(events), True)))

        if self.loosePtForSkim:
            selections.add('passJetMult_lowpt_forskim', events.passJetMult_lowpt_forskim)
            selections.add("passPreSel_lowpt_forskim", events.passPreSel_lowpt_forskim)
            final_selection = selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult_lowpt_forskim=True, passPreSel_lowpt_forskim=True)
        elif self.skim4b:
            selections.add('passJetMult', events.passJetMult)
            selections.add("passPreSel", events.passPreSel)
            selections.add("passFourTag", events.fourTag)
            final_selection = selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult=True, passPreSel=True, passFourTag=True)
        else:
            selections.add('passJetMult', events.passJetMult)
            selections.add("passPreSel", events.passPreSel)
            final_selection = selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult=True, passPreSel=True)

        events["weight"] = weights.weight()

        self._cutFlow.fill("all", events, allTag=True)
        cumulative_cuts = []
        for cut in selections.names:
            cumulative_cuts.append(cut)
            self._cutFlow.fill(cut, events[selections.all(*cumulative_cuts)], allTag=True)

        processOutput = {}
        return final_selection, None, processOutput

    def preselect(self, events):
        dataset = events.metadata['dataset']
        processName = events.metadata['processName']
        config = processor_config(processName, dataset, events)
        if config["isMC"] and self.mc_outlier_threshold is not None and "genWeight" in events.fields:
            return OutlierByMedian(self.mc_outlier_threshold)(events.genWeight)
