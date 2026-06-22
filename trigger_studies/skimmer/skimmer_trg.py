import logging
import numpy as np
import awkward as ak
import yaml
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea.analysis_tools import PackedSelection, Weights
from coffea4bees.analysis.helpers.cutflow import cutflow_4b ## change thiss
from coffea4bees.analysis.helpers.object_selection import apply_bRegCorr, muon_selection
from src.physics.common import drClean
from src.physics.event_selection import apply_event_selection
from src.skimmer.picoaod import PicoAOD
from src.physics.objects.jet_corrections import apply_jet_veto_maps, apply_jerc_corrections
from src.skimmer.mc_weight_outliers import OutlierByMedian


def lepton_selection_trg(event: ak.Array, isRun3: bool = False) -> ak.Array:
    """
    Selects leptons (muons and electrons) and adds them to the event.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Muon` and `Electron`.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.

    Returns:
    --------
    ak.Array
        The input event data with additional fields:
        - `selMuon`: Selected muons.
        - `selElec`: Selected electrons (if present).
    """
    event['selMuon_HLT'] = muon_selection(event.Muon, isRun3)
    
    ## keep muon above pT > 10 GeV, mu.pfRelIso04_all < 0.15, |eta|<2.4 -- veto if this is satisfied
    if 'Electron' in event.fields:
        event['selElec_L1'] = electron_selection_trg(event.Electron, isRun3, pt = 32, working_point="")
        event['selElec_HLT'] = electron_selection_trg(event.Electron, isRun3, pt = 15, working_point="mvaIso_WP80")
        event['selDiLepton_HLT'] = ak.concatenate([event.selElec_HLT, event.selMuon_HLT], axis=1)
        event['elec_selected_L1'] = ak.sum(event.selElec_L1.selected == True, axis=1) == 1
        event['elec_selected_HLT'] = ak.sum(event.selElec_HLT.selected == True, axis=1) == 1
        event['muon_selected_HLT'] = ak.sum(event.selMuon_HLT.selected == True, axis=1) == 1
        # logging.info(f" SELMUONCHARGE {event[event.muon_selected_HLT].selMuon_HLT.charge} SELMUONCHARGE")
        # logging.info(f" SELMUONCHARGE {event[event.elec_selected_HLT].selElec_HLT.charge} SELMUONCHARGE")
        
    return event

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
    """
    # Initialize lepton-cleaned jets
    ## clean the jet vs selected electron with delR = 0.4
    event['Jet', 'lepton_cleaned'] = np.full(len(event), True) if not doLeptonRemoval else drClean(event.Jet, event['selMuon_HLT'], cone = 0.2)[1]

    # Apply jet veto maps if required
    if do_jet_veto_maps:
        event['Jet', 'jet_veto_maps'] = apply_jet_veto_maps(corrections_metadata['jet_veto_maps'], event.Jet)
        event['Jet'] = event['Jet'][event['Jet', 'jet_veto_maps']]

    # Run3-specific jet selection
    if isRun3:
        event['Jet', 'bRegCorr'] = 1.0
        event['Jet', 'btagScore'] = event.Jet.btagPNetB

        if not isSyntheticData:
            event['Jet'] = ak.where(   ### with neutrino if bjet, otherwise just puppi
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

        event['Jet', 'preselected']      = (event.Jet.pt > 30) & (np.abs(event.Jet.eta) < 2.5)
        event['nJet_preselected']        = ak.sum(event.Jet.preselected, axis=1)  ## Atleast 4 selected jets
        ### need atleast 4 jets but only 2 need to satisty MWP -- change this
        ### switch to particle net for run3
        # event['Jet', 'selected_L1']   = (event.nJet_preselected >= 4)
        event['Jet', 'tagged']        = (event.nJet_preselected >= 4) & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])  ### DeepJet medium working point
        event['nJet_tagged']          = ak.sum(event.Jet.tagged, axis=1)   ## Atleast 2 tagged jets for run3; 3 for run2
        event['selJet']      =  event.nJet_tagged >= 2  ## change this

        ### remove this
        ### do trigger emulation -- I don't need this -- revisit muon cleaned jets
        event['Jet', 'muon_cleaned'] = drClean(event.Jet, event.selMuon_HLT, cone = 0.2)[1] ### check again; keep nonisolated muons and don't want to clean with respect to those
        event['Jet', 'ht_selected'] = (event.Jet.muEF < 0.5) & (np.abs(event.Jet.eta) < 2.5) & event.Jet.muon_cleaned
        event['jet_ht_slected'] = ak.all(event.Jet.ht_selected == True, axis=1) == True
        # selev = event[(event.elec_selected_L1 & event.jet_ht_slected) | (event.selJet)]
    return event



def electron_selection_trg(electron: ak.Array, isRun3: bool = False, pt = 15, working_point = "") -> ak.Array:
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
    ### make selections above the trigger threshold
    ### Aplly this HLT_Ele30_WPTight_Gsf
    ### ot above 15 GeV and passes medium wp el.mvaIso_WP90 ## loose electron
    ## exactly 1e should pass tight selection, rest will have llose
    ### we need it to be from top 
    ## get me events with one electon
    electron_kin = (electron.pt > pt) & (abs(electron.eta) < 2.5)
    
    electron_IP = (
        ((abs(electron.eta) < 1.479) & (abs(electron.dz) < 0.1) & (abs(electron.dxy) < 0.05)) |
        ((abs(electron.eta) >= 1.479) & (abs(electron.dz) < 0.2) & (abs(electron.dxy) < 0.1))
    ) if isRun3 else True

    electron['selected'] = electron_kin & electron_IP
    if working_point:
        ### Offline electron has isolation ## 'mvaIso_WP80' for HLT
        electron_iso_ID = (electron.pfRelIso03_all < 0.15) & (
            getattr(electron, working_point) if isRun3 else getattr(electron, 'mvaFall17V2Iso_WP90')  ### Change this for Run 2
        )
        electron['selected'] = electron.selected & electron_iso_ID
    
    return electron[electron.selected]


def apply_dilep_jet_selection(
        event, 
        corrections_metadata, 
        *,
        dataset: str = '',
        doLeptonRemoval: bool = True,
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
    do_jet_veto_maps : bool, optional
        Whether to apply jet veto maps. Defaults to False.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.
    isMC : bool, optional
        Whether the data is Monte Carlo simulation. Defaults to False.
    Returns:
    --------
    ak.Array
        The input event data with additional fields for object selection.
    """
    ### First perform electron selection followed by jet selection
    event = lepton_selection_trg(event, isRun3)
    event = jet_selection_trg(event, corrections_metadata, isRun3, isMC, isSyntheticData, isSyntheticMC, dataset, doLeptonRemoval, do_jet_veto_maps,apply_mixeddata_sel, override_selected_with_flavor_bit)

    event['passJetMult'] = event['nJet_preselected'] >= 4   ### for L1, HLT
    event['passJetMult_tagged'] = (event['nJet_tagged'] >= 2) & (event['nJet_preselected'] >= 4)  ### for HLT
    
    ### selected events pass L1 skim requirement or HLT skim requirement
    selev = event[(event.elec_selected_L1 & event.passJetMult) | (event.elec_selected_HLT & event.muon_selected_HLT & event.selJet  )]

    # logging.info(f"selev {selev}")
    # logging.info(f"selev {selev}")
    # logging.info(f"selev {selev}")
    # logging.info(f"selev {selev}")
    selev["hT_trigger"] = ak.sum(selev.Jet.pt, axis=1)  ### calojet_ht; L1 is only from calorimeters without tracking/tagging
    
    ### for L1, only need ht; for HTL, we need jet4_pt
    selev["jet4_pt"] = selev.Jet[:,3].pt   ### this is pt sorted

    return event


class Skimmer(PicoAOD):
    def __init__(
            self, 
            mc_outlier_threshold=200, 
            corrections_metadata=None,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
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

        events = apply_dilep_jet_selection(
            events,
            self.corrections_metadata[year],
            dataset=dataset,
            doLeptonRemoval=config["do_lepton_jet_cleaning"],
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

        
        selections.add('passJetMult', events.passJetMult)
        selections.add("passJetMult_tagged", events.passJetMult_tagged)
        final_selection = selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult=True, passJetMult_tagged=True)

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
