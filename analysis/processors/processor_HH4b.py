from __future__ import annotations

import copy
import logging
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING

import awkward as ak
import numpy as np
import yaml
import gc
from src.physics.objects.jet_corrections import apply_jerc_corrections
from src.physics.common import update_events
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from coffea4bees.analysis.helpers.event_weights import (
    add_btagweights,
    add_pseudotagweights,
)
from src.physics.event_selection import apply_event_selection
from src.physics.event_weights import add_weights
from coffea4bees.analysis.helpers.event_selection import apply_dilep_ttbar_selection, apply_4b_selection
from coffea4bees.analysis.helpers.filling_histograms import (
    filling_nominal_histograms,
    filling_syst_histograms,
)
from src.friendtrees.FriendTreeSchema import FriendTreeSchema
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea4bees.analysis.helpers.candidates_selection import create_cand_jet_dijet_quadjet
from coffea4bees.analysis.helpers.SvB_helpers import setSvBVars, subtract_ttbar_with_SvB
from coffea4bees.analysis.helpers.topCandReconstruction import (
    adding_top_reco_to_event,
    buildTop,
    find_tops,
    find_tops_slow,
)
from src.hist_tools import Fill
from src.data_formats.root import Chunk, TreeReader
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import load
from memory_profiler import profile
import psutil
import os

from ..helpers.load_friend import (
    FriendTemplate,
    parse_friends,
    rename_FvT_friend,
    rename_SvB_friend,
)

if TYPE_CHECKING:
    from ..helpers.classifier.HCR import HCRModelMetadata
from coffea4bees.analysis.helpers.truth_tools import find_genpart

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


def _init_classfier(path: str | list[HCRModelMetadata]):
    if path is None:
        return None
    if isinstance(path, str):
        from ..helpers.classifier.HCR import Legacy_HCREnsemble
        return Legacy_HCREnsemble(path)
    else:
        from ..helpers.classifier.HCR import HCREnsemble
        return HCREnsemble(path)

def _init_classfier_FvT(path: str | list[HCRModelMetadata]):
    if path is None:
        return None
    from ..helpers.classifier.HCR import Legacy_HCREnsemble_FvT
    return Legacy_HCREnsemble_FvT(path)

class analysis(processor.ProcessorABC):
    """
    Coffea processor for HH→4b analysis workflows.

    This class orchestrates the event selection, object reconstruction, friend tree loading, weight calculation,
    systematic variations, and histogram filling for the HH→4b analysis. It supports both nominal and systematic
    processing, including jet energy corrections, b-tagging scale factors, trigger weights, and classifier outputs
    (FvT, SvB, etc.).

    Key Features:
        - Loads and applies friend trees (FvT, SvB, JCM, top reconstruction, etc.)
        - Handles MC and data workflows, including blinding and mixed data selection
        - Applies event selection, object selection, and cutflows
        - Calculates and applies event weights (MC, btag, pseudotag, trigger, resonance, etc.)
        - Supports systematic variations (JES, others)
        - Fills histograms and optionally dumps friend trees for classifier inputs and weights
        - Supports top candidate reconstruction (slow/fast algorithms or friend trees)
        - Memory usage logging (optional)

    Args:
        SvB (str | list[HCRModelMetadata], optional): Path or metadata for SvB classifier.
        SvB_MA (str | list[HCRModelMetadata], optional): Path or metadata for SvB_MA classifier.
        FvT (str | list[HCRModelMetadata], optional): Path or metadata for FvT classifier.
        blind (bool): Whether to blind data in signal region.
        apply_JCM (bool): Whether to apply Jet Combinatoric Model weights.
        JCM_file (str): Path to JCM weight file.
        corrections_metadata (dict): Metadata for corrections (JES, etc.).
        apply_trigWeight (bool): Whether to apply trigger weights.
        apply_btagSF (bool): Whether to apply b-tagging scale factors.
        apply_FvT (bool): Whether to apply FvT classifier/friend tree.
        apply_boosted_veto (bool): Whether to apply boosted event veto.
        run_dilep_ttbar_crosscheck (bool): Whether to run dilepton ttbar crosscheck selection.
        fill_histograms (bool): Whether to fill histograms.
        hist_cuts (list): List of cut names for histogram filling.
        run_SvB (bool): Whether to run SvB classifier/friend tree.
        top_reconstruction (str | None): Top candidate reconstruction mode ('slow', 'fast', or None).
        run_systematics (list): List of systematics to run (e.g., ['jes', 'others']).
        make_classifier_input (str): Path for dumping classifier input friend tree.
        make_top_reconstruction (str): Path for dumping top reconstruction friend tree.
        make_friend_JCM_weight (str): Path for dumping JCM weight friend tree.
        make_friend_FvT_weight (str): Path for dumping FvT weight friend tree.
        make_friend_SvB (str): Path for dumping SvB friend tree.
        subtract_ttbar_with_weights (bool): Whether to subtract ttbar using weights.
        apply_mixeddata_sel (bool): Whether to apply mixed data selection.
        friends (dict): Dictionary of friend tree templates or paths.

    Returns:
        dict: Output containing histograms, cutflow, and optionally dumped friend trees.
    """
    def __init__(
        self,
        *,
        SvB: str|list[HCRModelMetadata] = None,
        SvB_MA: str|list[HCRModelMetadata] = None,
        FvT: str|list[HCRModelMetadata] = None,
        blind: bool = False,
        apply_JCM: bool = True,
        JCM_file: str = "coffea4bees/analysis/weights/JCM/AN_24_089_v3/jetCombinatoricModel_SB_6771c35.yml",
        corrections_metadata: dict = None,
        apply_trigWeight: bool = True,
        apply_btagSF: bool = True,
        apply_FvT: bool = True,
        apply_boosted_veto: bool = False,
        run_dilep_ttbar_crosscheck: bool = False,
        fill_histograms: bool = True,
        hist_cuts = ['passPreSel'],
        run_SvB: bool = True,
        top_reconstruction: str | None = None,
        run_systematics: list = [],  #### Way of splitting systematics. It can be event_weights, jes, btag
        make_classifier_input: str = None,
        make_top_reconstruction: str = None,
        make_friend_JCM_weight: str = None,
        make_friend_FvT_weight: str = None,
        make_friend_SvB: str = None,
        subtract_ttbar_with_weights: bool = False,
        apply_mixeddata_sel: bool = False,  #### apply HIG-22-011 sel for mixeddata
        friends: dict[str, str|FriendTemplate] = None,
        return_events_for_display: bool = False,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.blind = blind
        self.apply_JCM = jetCombinatoricModel(JCM_file) if apply_JCM else None
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.apply_FvT = apply_FvT
        self.run_SvB = run_SvB
        self.fill_histograms = fill_histograms
        self.run_dilep_ttbar_crosscheck = run_dilep_ttbar_crosscheck
        self.apply_boosted_veto = apply_boosted_veto
        self.classifier_SvB = _init_classfier(SvB)
        self.classifier_SvB_MA = _init_classfier(SvB_MA)
        self.classifier_FvT = _init_classfier_FvT(FvT)
        self.corrections_metadata = corrections_metadata
        self.run_systematics = ['others', 'jes'] if 'all' in run_systematics else run_systematics
        self.make_top_reconstruction = make_top_reconstruction
        self.make_classifier_input = make_classifier_input
        self.make_friend_JCM_weight = make_friend_JCM_weight
        self.make_friend_FvT_weight = make_friend_FvT_weight
        self.make_friend_SvB = make_friend_SvB
        self.top_reconstruction = top_reconstruction
        if self.top_reconstruction is not None and self.top_reconstruction not in ["slow", "fast"]:
            raise ValueError(f"top_reconstruction must be None, 'slow', or 'fast', got: {self.top_reconstruction}")
        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights
        self.friends = parse_friends(friends)
        self.histCuts = hist_cuts
        self.apply_mixeddata_sel = apply_mixeddata_sel
        self.return_events_for_display = return_events_for_display
        
        # Track top 20 events with largest ps_hh across all chunks
        self.top_ps_hh_events = []
        
        # Memory monitoring
        self.debug_memory = False  # Set to False to disable memory monitoring
        
    def _log_memory(self, stage_name):
        """Log current memory usage"""
        if not self.debug_memory:
            return
            
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            rss_mb = memory_info.rss / 1024 / 1024  # MB
            vms_mb = memory_info.vms / 1024 / 1024  # MB
            logging.info(f"MEMORY: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB {stage_name}")
        except Exception as e:
            logging.warning(f"Memory monitoring failed at {stage_name}: {e}")

    # @profile
    def process(self, event):
        logging.debug(event.metadata)
        self._log_memory("process_start")
        
        fname   = event.metadata['filename']
        self.dataset = event.metadata['dataset']
        self.estart  = event.metadata['entrystart']
        self.estop   = event.metadata['entrystop']
        self.chunk   = f'{self.dataset}::{self.estart:6d}:{self.estop:6d} >>> '
        self.year    = event.metadata['year']
        self.year_label = self.corrections_metadata[self.year]['year_label']
        self.processName = event.metadata['processName']

        ### target is for new friend trees
        target = Chunk.from_coffea_events(event)
        self._log_memory("after_metadata_setup")

        #
        # Set process and datset dependent flags
        #
        self.config = processor_config(self.processName, self.dataset, event)
        logging.debug(f'{self.chunk} config={self.config}, for file {fname}\n')


        #
        #  If doing RW
        #
        # if self.config["isSyntheticData"] and not self.config["isPSData"]:
        #     with open(f"jet_clustering/jet-splitting-PDFs-00-08-00/hT-reweight-00-00-01/hT_weights_{self.year}.yml", "r") as f:
        #         self.hT_weights= yaml.safe_load(f)

        #
        #  If applying Gaussian Kernal to signal
        #
        self.gaussKernalMean = None
        if self.config["isSignal"] and (self.gaussKernalMean is not None) :
            bin_edges = np.linspace(0, 1200, 100)  # 100 bins from 0 to 1200
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
            sigma = 0.05 * self.gaussKernalMean  # Standard deviation of the Gaussian
            self.resonance_weights = np.exp(-0.5 * ((bin_centers - self.gaussKernalMean) / sigma) ** 2)  # Gaussian formula

        self.nEvent = len(event)

        #
        # Reading SvB friend trees
        #
        self._log_memory("before_friend_trees")
        path = fname.replace(fname.split("/")[-1], "")
        if self.apply_FvT and self.classifier_FvT is None:
            if "FvT" in self.friends:
                event["FvT"] = rename_FvT_friend(target, self.friends["FvT"])
                if self.config["isDataForMixed"] or self.config["isTTForMixed"]:
                    for _FvT_name in event.metadata["FvT_names"]:
                        event[_FvT_name] = rename_FvT_friend(target, self.friends[_FvT_name])
                        event[_FvT_name, _FvT_name] = event[_FvT_name].FvT

            else:
                # TODO: remove backward compatibility in the future
                if self.config["isMixedData"]:

                    FvT_name = event.metadata["FvT_name"]
                    event["FvT"] = getattr( 
                        NanoEventsFactory.from_root( 
                            f'{event.metadata["FvT_file"]}', 
                            entry_start=self.estart, 
                            entry_stop=self.estop,
                            schemaclass=FriendTreeSchema, 
                        ).events(),
                        FvT_name 
                    )

                    event["FvT", "FvT"] = getattr(event["FvT"], FvT_name)

                    #
                    # Dummies
                    #
                    event["FvT", "q_1234"] = np.full(len(event), -1, dtype=int)
                    event["FvT", "q_1324"] = np.full(len(event), -1, dtype=int)
                    event["FvT", "q_1423"] = np.full(len(event), -1, dtype=int)

                elif self.config["isDataForMixed"] or self.config["isTTForMixed"]:

                    #
                    # Use the first to define the FvT weights
                    #
                    event["FvT"] = getattr( 
                        NanoEventsFactory.from_root( 
                            f'{event.metadata["FvT_files"][0]}', 
                            entry_start=self.estart, 
                            entry_stop=self.estop, 
                            schemaclass=FriendTreeSchema, 
                        ).events(),
                        event.metadata["FvT_names"][0], 
                    )

                    event["FvT", "FvT"] = getattr( event["FvT"], event.metadata["FvT_names"][0] )

                    #
                    # Dummies
                    #
                    event["FvT", "q_1234"] = np.full(len(event), -1, dtype=int)
                    event["FvT", "q_1324"] = np.full(len(event), -1, dtype=int)
                    event["FvT", "q_1423"] = np.full(len(event), -1, dtype=int)

                    for _FvT_name, _FvT_file in zip( event.metadata["FvT_names"], event.metadata["FvT_files"] ):

                        event[_FvT_name] = getattr( 
                            NanoEventsFactory.from_root( 
                                f"{_FvT_file}", 
                                entry_start=self.estart, 
                                entry_stop=self.estop, 
                                schemaclass=FriendTreeSchema, 
                            ).events(),
                            _FvT_name, 
                        )

                        event[_FvT_name, _FvT_name] = getattr(event[_FvT_name], _FvT_name)

                else:
                    event["FvT"] = ( 
                        NanoEventsFactory.from_root( 
                            f'{fname.replace("picoAOD", "FvT")}', 
                            entry_start=self.estart, 
                            entry_stop=self.estop, 
                            schemaclass=FriendTreeSchema
                        ).events().FvT 
                    )

                if "std" not in event.FvT.fields:
                    event["FvT", "std"] = np.ones(len(event))
                    event["FvT", "pt4"] = np.ones(len(event))
                    event["FvT", "pt3"] = np.ones(len(event))
                    event["FvT", "pd4"] = np.ones(len(event))
                    event["FvT", "pd3"] = np.ones(len(event))

                event["FvT", "frac_err"] = event["FvT"].std / event["FvT"].FvT
                if not ak.all(event.FvT.event == event.event):
                    raise ValueError("ERROR: FvT events do not match events ttree")

        if self.run_SvB:
            for k in self.friends:
                if k.startswith("SvB"):
                    try:
                        event[k] = rename_SvB_friend(target, self.friends[k])
                        setSvBVars(k, event)
                    except Exception as e:
                        event[k] = self.friends[k].arrays(target)

            self._log_memory("after_friend_trees_loaded")

            if self.apply_mixeddata_sel: SvB_suffix = '_newSBDef'
            else: SvB_suffix = '_ULHH'

            if "SvB" not in self.friends and self.classifier_SvB is None:
                # SvB_file = f'{path}/SvB_newSBDef.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", "SvB")}'
                SvB_file = f'{path}/SvB{SvB_suffix}.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", f"SvB{SvB_suffix}")}'
                event["SvB"] = ( 
                    NanoEventsFactory.from_root( 
                        SvB_file,
                        entry_start=self.estart, 
                        entry_stop=self.estop, 
                        schemaclass=FriendTreeSchema
                    ).events().SvB 
                )

                if not ak.all(event.SvB.event == event.event):
                    raise ValueError("ERROR: SvB events do not match events ttree")
                # defining SvB for different SR
                setSvBVars("SvB", event)

            if "SvB_MA" not in self.friends and self.classifier_SvB_MA is None:
                # SvB_MA_file = f'{path}/SvB_MA_newSBDef.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", "SvB_MA")}'
                SvB_MA_file = f'{path}/SvB_MA{SvB_suffix}.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", f"SvB_MA{SvB_suffix}")}'
                event["SvB_MA"] = ( 
                    NanoEventsFactory.from_root( 
                        SvB_MA_file,
                        entry_start=self.estart, 
                        entry_stop=self.estop, 
                        schemaclass=FriendTreeSchema
                    ).events().SvB_MA
                )

                if not ak.all(event.SvB_MA.event == event.event):
                    raise ValueError("ERROR: SvB_MA events do not match events ttree")
                # defining SvB for different SR
                setSvBVars("SvB_MA", event)

        if self.config["isDataForMixed"]:

            #
            # Load the different JCMs
            #
            JCM_array = TreeReader( lambda x: [ s for s in x if s.startswith("pseudoTagWeight_3bDvTMix4bDvT_v") ] ).arrays(Chunk.from_coffea_events(event))

            for _JCM_load in event.metadata["JCM_loads"]:
                event[_JCM_load] = JCM_array[_JCM_load]

        #
        # Event selection
        #
        self._log_memory("before_event_selection")
        event = apply_event_selection( 
            event,
            self.corrections_metadata[self.year],
            cut_on_lumimask=self.config["cut_on_lumimask"]
        )
        self._log_memory("after_event_selection")


        ### adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights(
            event, 
            target=target,
            do_MC_weights=self.config["do_MC_weights"],
            dataset=self.dataset,
            year_label=self.year_label,
            friend_trigWeight=self.friends.get("trigWeight"),
            corrections_metadata=self.corrections_metadata[self.year],
            apply_trigWeight=self.apply_trigWeight,
            isTTForMixed=self.config["isTTForMixed"],
            run_systematics= 'others' in self.run_systematics,
        )


        #
        # Checking boosted selection (should change in the future)
        #
        event['notInBoostedSel'] = np.full(len(event), True)
        if self.apply_boosted_veto:

            if self.dataset.startswith("GluGluToHHTo4B_cHHH1"):
                boosted_file = load("coffea4bees/metadata/boosted_overlap_signal.coffea")['boosted']
                boosted_events = boosted_file.get(self.dataset, {}).get('event', event.event)
                boosted_events_set = set(boosted_events)
                event['notInBoostedSel'] = np.array([e not in boosted_events_set for e in event.event.to_numpy()])
            elif self.dataset.startswith("data"):
                boosted_file = load("coffea4bees/metadata/boosted_overlap_data.coffea")
                mask = np.array(boosted_file['BDTcat_index']) > 0  ### > 0 is all boosted categories, 1 is most sensitive
                filtered_runs = np.array(boosted_file['run'])[mask]
                filtered_lumis = np.array(boosted_file['luminosityBlock'])[mask]
                filtered_events = np.array(boosted_file['event'])[mask]
                boosted_events_set = set(zip(filtered_runs, filtered_lumis, filtered_events))
                event_tuples = zip(event.run.to_numpy(), event.luminosityBlock.to_numpy(), event.event.to_numpy())
                event['notInBoostedSel'] = np.array([t not in boosted_events_set for t in event_tuples])
            else:
                logging.info(f"Boosted veto not applied for dataset {self.dataset}")

        #
        # Calculate and apply Jet Energy Calibration
        #
        if self.config["do_jet_calibration"]:

            jets = apply_jerc_corrections(
                event,
                corrections_metadata=self.corrections_metadata[self.year],
                isMC=self.config["isMC"],
                run_systematics= 'jes' in self.run_systematics,
                dataset=self.dataset
            )
        else:
            jets = event.Jet


        # Determine which shifts to run
        if 'jes' in self.run_systematics:
            shifts = []
            shifts.extend([({"Jet": jets.JER.up}, f"CMS_res_j_{self.year_label}Up"), ({"Jet": jets.JER.down}, f"CMS_res_j_{self.year_label}Down")])

            for jesunc in self.corrections_metadata[self.year]["JES_uncertainties"]:
                shifts.extend( [ ({"Jet": jets[f"JES_{jesunc}"].up}, f"CMS_scale_j_{jesunc}Up"),
                                 ({"Jet": jets[f"JES_{jesunc}"].down}, f"CMS_scale_j_{jesunc}Down"), ] )
            logging.info(f"\nJet variations {[name for _, name in shifts]}")
        else:
            shifts = [({"Jet": jets}, None)]

        return processor.accumulate( self.process_shift(update_events(event, collections), name, weights, list_weight_names, target) for collections, name in shifts )

    # @profile
    def process_shift(self, event, shift_name, weights, list_weight_names, target):
        """For different jet variations. It computes event variations for the nominal case."""

        # Copy the weights to avoid modifying the original
        weights = copy.copy(weights)

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_4b_selection( 
            event, 
            self.corrections_metadata[self.year],
            dataset=self.dataset,
            doLeptonRemoval=self.config["do_lepton_jet_cleaning"],
            override_selected_with_flavor_bit=self.config["override_selected_with_flavor_bit"],
            do_jet_veto_maps=self.config["do_jet_veto_maps"],
            isRun3=self.config["isRun3"],
            isMC=self.config["isMC"], ### temporary
            isSyntheticData=self.config["isSyntheticData"],
            isSyntheticMC=self.config["isSyntheticMC"],
            apply_mixeddata_sel=self.apply_mixeddata_sel,
        )

        if self.run_dilep_ttbar_crosscheck:
            event['passDilepTtbar'] = apply_dilep_ttbar_selection(event, isRun3=self.config["isRun3"])
        #
        #  Test hT reweighting the synthetic data
        #
        # if self.config["isSyntheticData"] and not self.config["isPSData"]:
        #     hT_index = np.floor_divide(event.hT_selected,30).to_numpy()
        #     hT_index[hT_index > 48] = 48
        #
        #     vectorized_hT = np.vectorize(lambda i: self.hT_weights["weights"][int(i)])
        #     weights_hT = vectorized_hT(hT_index)
        #
        #     weights.add( "hT_reweight", weights_hT )
        #     list_weight_names.append(f"hT_reweight")



        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        #selections.add( "passHLT", ( np.full(len(event), True) if skip_HLT_cut else event.passHLT ) )
        selections.add( "passHLT", ( event.passHLT if self.config["cut_on_HLT_decision"] else np.full(len(event), True)  ) )
        selections.add( 'passJetMult', event.passJetMult )
        allcuts = [ 'lumimask', 'passNoiseFilter', 'passHLT', ]
        allcuts += [ 'passJetMult' ]
        event['weight'] = weights.weight()   ### this is for _cutflow

        #
        #  Cut Flows
        #
        processOutput = {}
        if not shift_name:
            processOutput['nEvent'] = {}
            processOutput['nEvent'][event.metadata['dataset']] = {
                'nEvent' : self.nEvent,
                'genWeights': np.sum(event.genWeight) if self.config["isMC"] else self.nEvent

            }

            #
            # Check outliers
            #
            # Checking for outliners in weights
            if self.config["isMC"]:
                tmp_weights = weights.weight()
                mean_weights = np.mean(tmp_weights)
                std_weights = np.std(tmp_weights)
                z_scores = np.abs((tmp_weights - mean_weights) / std_weights)
                pass_outliers = z_scores < 30
                event["passCleanGenWeight"] = pass_outliers
                if np.any(~pass_outliers) and std_weights > 0:
                    logging.warning(f"Outliers in weights:{tmp_weights[~pass_outliers]}, while mean is {mean_weights} and std is {std_weights} for event {event[~pass_outliers].event} in {self.dataset}\n")
                selections.add( "passCleanGenWeight", event.passCleanGenWeight)
                allcuts += ["passCleanGenWeight"]
            else:
                event['passCleanGenWeight'] = True
                selections.add( "passCleanGenWeight", event.passCleanGenWeight)

            #
            # Get Truth m4j
            #
            if self.config["isSignal"]:

                event['bfromHorZ_all']= find_genpart(event.GenPart, [5], [23, 25])

                if "status" in event.bfromHorZ_all.fields:
                    event['bfromHorZ'] = event.bfromHorZ_all[event.bfromHorZ_all.status == 23]
                else:
                    logging.warning(f"\nStatus Missing for GenParticles in dataset {self.dataset}\n")
                    event['bfromHorZ'] = event.bfromHorZ_all

                event['GenJet', 'selectedBs'] = (np.abs(event.GenJet.partonFlavour)==5)
                event['selGenBJet'] = event.GenJet[event.GenJet.selectedBs]
                event['matchedGenBJet'] = event.bfromHorZ.nearest( event.selGenBJet, threshold=10 )
                event["matchedGenBJet"] = event.matchedGenBJet[~ak.is_none(event.matchedGenBJet, axis=1)]

                event['pass4GenBJets'] = ak.num(event.matchedGenBJet) == 4
                event["truth_v4b"] = ak.where(  event.pass4GenBJets,
                                                event.matchedGenBJet.sum(axis=1),
                                                1e-10 * event.matchedGenBJet.sum(axis=1),
                                              )

                if self.gaussKernalMean is not None:
                    v4b_index = np.floor_divide(event.truth_v4b.mass, 12).to_numpy()
                    v4b_index[v4b_index > 98] = 98

                    vectorized_v4b = np.vectorize(lambda i: self.resonance_weights[int(i)])
                    weights_resonance = vectorized_v4b(v4b_index)
                    weights.add( "resonance_reweight", weights_resonance )
                    list_weight_names.append(f"resonance_reweight")

            else:
                event['pass4GenBJets'] = True


            selections.add( "pass4GenBJets", event.pass4GenBJets)

            #
            # Do the cutflow
            #
            sel_dict = OrderedDict({
                'all'               : selections.require(lumimask=True),
                'passCleanGenWeight': selections.require(lumimask=True, passCleanGenWeight=True),
                'pass4GenBJets'     : selections.require(lumimask=True, passCleanGenWeight=True, pass4GenBJets=True),
                'passNoiseFilter'   : selections.require(lumimask=True, passCleanGenWeight=True, passNoiseFilter=True),
                'passHLT'           : selections.require(lumimask=True, passCleanGenWeight=True, passNoiseFilter=True, passHLT=True),
            })
            sel_dict['passJetMult'] = selections.all(*allcuts)

            self._cutFlow = cutflow_4b(do_truth_hists=self.config["isSignal"])
            for cut, sel in sel_dict.items():
                self._cutFlow.fill( cut, event[sel], allTag=True )
                self._cutFlow.fill( f"{cut}_woTrig", event[sel], allTag=True,
                                    wOverride=weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[sel])


        #
        # Calculate and apply btag scale factors
        #
        if self.config["isMC"] and self.apply_btagSF:

            weights, list_weight_names = add_btagweights( 
                event, 
                weights,
                list_weight_names=list_weight_names,
                shift_name=shift_name,
                use_prestored_btag_SF=self.config["use_prestored_btag_SF"],
                run_systematics = 'others' in self.run_systematics,
                corrections_metadata=self.corrections_metadata[self.year]
            )
            logging.debug( f"Btag weight {weights.partial_weight(include=['CMS_btag'])[:10]}\n" )
            event["weight"] = weights.weight()
            if not shift_name:
                self._cutFlow.fill( "passJetMult_btagSF", event[selections.all(*allcuts)], allTag=True )
                self._cutFlow.fill( "passJetMult_btagSF_woTrig", event[selections.all(*allcuts)], allTag=True,
                               wOverride=weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] )

        #
        # Preselection: keep only three or four tag events
        #
        selections.add("passPreSel", event.passPreSel)
        allcuts.append("passPreSel")
        analysis_selections = selections.all(*allcuts)

        if not shift_name:
            self._cutFlow.fill( "passPreSel_allTag", event[selections.all(*allcuts)], allTag=True )
            self._cutFlow.fill( "passPreSel_allTag_woTrig", event[selections.all(*allcuts)], allTag=True,
                                wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))

        weights, list_weight_names = add_pseudotagweights( 
            event, 
            weights,
            JCM=self.apply_JCM,
            apply_FvT=self.apply_FvT,
            isDataForMixed=self.config["isDataForMixed"],
            list_weight_names=list_weight_names,
            event_metadata=event.metadata,
            year_label=self.year_label,
            len_event=len(event),
            )

        #
        # Example of how to write out event numbers
        #
        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_Run3_data_early
        # add_debug_Run3_data_early(event, processOutput)
        #from coffea4bees.analysis.helpers.write_debug_info import add_debug_Run3_data
        #add_debug_Run3_data(event, processOutput)

        selev = event[analysis_selections]
        #selev["passFvT50" ] = selev["FvT"].FvT > 50
        #selev["passFvT100"] = selev["FvT"].FvT > 100

        if self.subtract_ttbar_with_weights:

            pass_ttbar_filter_selev = subtract_ttbar_with_SvB(selev, self.dataset, self.year)

            pass_ttbar_filter = np.full( len(event), True)
            pass_ttbar_filter[ selections.all(*allcuts) ] = pass_ttbar_filter_selev
            selections.add( 'pass_ttbar_filter', pass_ttbar_filter )
            allcuts.append("pass_ttbar_filter")
            if not shift_name:
                self._cutFlow.fill( "pass_ttbar_filter", event[selections.all(*allcuts)], allTag=True )
                self._cutFlow.fill( "pass_ttbar_filter_woTrig", event[selections.all(*allcuts)], allTag=True,
                                    wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))


            analysis_selections = selections.all(*allcuts)
            selev = selev[pass_ttbar_filter_selev]

        #
        #  Build the top Candiates
        #
        if friend := self.friends.get("top_reconstruction"):
            top_cand = friend.arrays(target)[analysis_selections]
            adding_top_reco_to_event( selev, top_cand )

        else:
            if self.top_reconstruction in ["slow","fast"]:

                # sort the jets by btagging
                selev.selJet = selev.selJet[ ak.argsort(selev.selJet.btagScore, axis=1, ascending=False) ]

                if self.top_reconstruction == "slow":
                    top_cands = find_tops_slow(selev.selJet)
                else:
                    try:
                        top_cands = find_tops(selev.selJet)
                    except Exception as e:
                        logging.warning("WARNING: Fast top_reconstruction failed with exception: ")
                        logging.warning(f"{e}\n")
                        logging.warning("... Trying the slow top_reconstruction")
                        top_cands = find_tops_slow(selev.selJet)

                selev['top_cand'], _ = buildTop(selev.selJet, top_cands)
                ### with top friendtree we dont need the next two lines
                selev["xbW"] = selev.top_cand.xbW
                selev["xW"] = selev.top_cand.xW

        #
        #  Build di-jets and Quad-jets
        #
        selev = create_cand_jet_dijet_quadjet( 
            selev,
            apply_FvT=self.apply_FvT,
            classifier_FvT=self.classifier_FvT,
            run_SvB=self.run_SvB,
            run_systematics=self.run_systematics,
            classifier_SvB=self.classifier_SvB,
            classifier_SvB_MA=self.classifier_SvB_MA,
            processOutput = processOutput,
            isRun3=self.config["isRun3"],
            weights=weights,
            list_weight_names=list_weight_names,
            analysis_selections=analysis_selections,
            )



        #
        # Example of how to write out event numbers
        #
        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_info_to_output
        # add_debug_info_to_output(event, processOutput, weights, list_weight_names, analysis_selections)

        if self.return_events_for_display:
            #
            # Track top 20 events with largest SvB_MA.ps_hh across all chunks
            #
            logging.info(f"Tracking events enabled. Processing chunk with {len(selev)} events")
            if len(selev) > 0 and hasattr(selev, 'SvB_MA') and hasattr(selev.SvB_MA, 'ps_hh'):
                # Get SvB_MA.ps_hh values
                ps_hh_values = ak.to_numpy(ak.fill_none(selev.SvB_MA.ps_hh, -999))
                
                # Get run, lumi, event numbers
                run_numbers = ak.to_numpy(ak.fill_none(selev.run, -1))
                lumi_numbers = ak.to_numpy(ak.fill_none(selev.luminosityBlock, -1))
                event_numbers = ak.to_numpy(ak.fill_none(selev.event, -1))
                
                # Store events from this chunk in processOutput for accumulation
                chunk_events = []
                for i in range(len(ps_hh_values)):
                    chunk_events.append({
                        'ps_hh': float(ps_hh_values[i]),
                        'run': int(run_numbers[i]),
                        'lumi': int(lumi_numbers[i]),
                        'event': int(event_numbers[i])
                    })
                
                processOutput['top_ps_hh_events'] = chunk_events
                logging.info(f"Stored {len(chunk_events)} events from this chunk for accumulation")
            else:
                if len(selev) == 0:
                    logging.debug("No events in selev for this chunk")
                elif not hasattr(selev, 'SvB_MA'):
                    logging.warning("selev does not have SvB_MA attribute")
                elif not hasattr(selev.SvB_MA, 'ps_hh'):
                    logging.warning("selev.SvB_MA does not have ps_hh attribute")
                processOutput['top_ps_hh_events'] = []

        #
        # Blind data in fourTag SR
        #
        if not (self.config["isMC"] or "mix_v" in self.dataset) and self.blind:
            # blind_flag = ~(selev["quadJet_selected"].SR & selev.fourTag)
            blind_flag = ~( selev["quadJet_selected"].SR & (selev["SvB_MA"].ps_hh > 0.5) & selev.fourTag )
            blind_sel = np.full( len(event), True)
            blind_sel[ analysis_selections ] = blind_flag
            selections.add( 'blind', blind_sel )
            allcuts.append( 'blind' )

            if not shift_name:
                self._cutFlow.fill( "blind", event[selections.all(*allcuts)], allTag=True )
                self._cutFlow.fill( "blind_woTrig", event[selections.all(*allcuts)], allTag=True,
                                    wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))

            analysis_selections = selections.all(*allcuts)
            selev = selev[blind_flag]

        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[analysis_selections]
        selev["trigWeight"] = weights.partial_weight(include=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections]
        selev['weight_woTrig'] = weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections]
        selev["no_weight"] = np.ones(len(selev))
        if not shift_name:
            self._cutFlow.fill("passPreSel", selev)
            self._cutFlow.fill("passPreSel_woTrig", selev,
                               wOverride=selev['weight_woTrig'])
            self._cutFlow.fill("passDiJetMass", selev[selev.passDiJetMass])
            self._cutFlow.fill("passDiJetMass_woTrig", selev[selev.passDiJetMass],
                               wOverride=selev['weight_woTrig'][selev.passDiJetMass] )
            self._cutFlow.fill("boosted_veto_passPreSel", selev[selev.notInBoostedSel])
            self._cutFlow.fill("boosted_veto_SR", selev[selev.notInBoostedSel & selev["quadJet_selected"].SR])
            selev['passSR'] = selev.passDiJetMass & selev["quadJet_selected"].SR
            self._cutFlow.fill( "SR", selev[selev.passSR] )
            self._cutFlow.fill( "SR_woTrig", selev[selev.passSR],
                            wOverride=selev['weight_woTrig'][selev.passSR])
            selev['passSB'] = selev.passDiJetMass & selev["quadJet_selected"].SB
            self._cutFlow.fill( "SB", selev[(selev.passDiJetMass & selev["quadJet_selected"].SB)] )
            self._cutFlow.fill( "SB_woTrig", selev[(selev.passDiJetMass & selev["quadJet_selected"].SB)],
                            wOverride=selev['weight_woTrig'][selev.passSB] )
            self._cutFlow.fill("passVBFSel", selev[selev.passVBFSel])
            if self.run_SvB:
                self._cutFlow.fill("passSvB", selev[selev.passSvB])
                self._cutFlow.fill("passSvB_woTrig", selev[selev.passSvB],
                               wOverride=selev['weight_woTrig'][selev.passSvB] )
                self._cutFlow.fill("failSvB", selev[selev.failSvB])
                self._cutFlow.fill("failSvB_woTrig", selev[selev.failSvB],
                               wOverride=selev['weight_woTrig'][selev.failSvB] )
            if self.run_dilep_ttbar_crosscheck:
                self._cutFlow.fill("passDilepTtbar", selev[selev.passDilepTtbar], allTag=True,
                               wOverride=selev['weight_noJCM_noFvT'][selev.passDilepTtbar] )

            self._cutFlow.addOutput(processOutput, event.metadata["dataset"])



        #
        # Hists
        #
        if self.classifier_FvT: apply_FvT = True
        else: apply_FvT = self.apply_FvT
        hist = {}
        if self.fill_histograms:
            if not self.run_systematics:
                ## this can be simplified
                hist = filling_nominal_histograms(
                    selev, 
                    self.apply_JCM,
                    processName=self.processName,
                    year=self.year,
                    isMC=self.config["isMC"],
                    histCuts=self.histCuts,
                    apply_FvT=apply_FvT,
                    run_SvB=self.run_SvB,
                    run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                    top_reconstruction=self.top_reconstruction,
                    isDataForMixed=self.config['isDataForMixed'],
                    event_metadata=event.metadata
                    )

            #
            # Run systematics
            #
            else:
                hist = filling_syst_histograms(
                    selev, 
                    weights,
                    analysis_selections,
                    shift_name=shift_name,
                    processName=self.processName,
                    year=self.year,
                    histCuts=self.histCuts
                    )

        friends = { 'friends': {} }
        if self.make_top_reconstruction is not None:
            from ..helpers.dump_friendtrees import dump_top_reconstruction

            friends["friends"] = ( friends["friends"]
                | dump_top_reconstruction(
                    selev,
                    self.make_top_reconstruction,
                    f"top_reco{'_'+shift_name if shift_name else ''}",
                    analysis_selections,
                )
            )

        if self.make_classifier_input is not None:
            for k in ["ZZSR", "ZHSR", "HHSR", "SR", "SB"]:
                selev[k] = selev["quadJet_selected"][k]
            selev["nSelJets"] = ak.num(selev.selJet)

            from ..helpers.dump_friendtrees import dump_input_friend

            weight = "weight_noJCM_noFvT"
            if weight not in selev.fields:
                weight = "weight"
            friends["friends"] = ( friends["friends"]
                | dump_input_friend(
                    selev,
                    self.make_classifier_input,
                    "HCR_input",
                    analysis_selections,
                    weight=weight,
                    NotCanJet="notCanJet_coffea",
                )
            )
        if self.make_friend_JCM_weight is not None:
            from ..helpers.dump_friendtrees import dump_JCM_weight

            friends["friends"] = ( friends["friends"]
                | dump_JCM_weight(selev, self.make_friend_JCM_weight, "JCM_weight", analysis_selections)
            )

        if self.make_friend_FvT_weight is not None:
            from ..helpers.dump_friendtrees import dump_FvT_weight

            friends["friends"] = ( friends["friends"]
                | dump_FvT_weight(selev, self.make_friend_FvT_weight, "FvT_weight", analysis_selections)
            )

        if self.make_friend_SvB is not None:
            from ..helpers.dump_friendtrees import dump_SvB

            friends["friends"] = ( friends["friends"]
                | dump_SvB(selev, self.make_friend_SvB, "SvB", analysis_selections)
                | dump_SvB(selev, self.make_friend_SvB, "SvB_MA", analysis_selections)
            )

        # Log sizes of return objects
        if self.debug_memory:
            import sys
            hist_size = sys.getsizeof(hist) / 1024 / 1024  # MB
            output_size = sys.getsizeof(processOutput) / 1024 / 1024  # MB
            friends_size = sys.getsizeof(friends) / 1024 / 1024  # MB
            logging.info(f"Return object sizes - hist: {hist_size:.1f}MB, output: {output_size:.1f}MB, friends: {friends_size:.1f}MB")

        # Explicit cleanup before returning
        del selev, event, weights, analysis_selections
        gc.collect()
        
        return hist | processOutput | friends

    def postprocess(self, accumulator):
        # Write out top 20 events with largest ps_hh if tracking was enabled
        logging.info(f"Postprocess called. return_events_for_display={self.return_events_for_display}")
        
        if self.return_events_for_display and 'top_ps_hh_events' in accumulator:
            # Aggregate all events from all chunks
            all_events = accumulator['top_ps_hh_events']
            logging.info(f"Found {len(all_events)} total events from all chunks")
            
            if len(all_events) > 0:
                # Sort by ps_hh descending and take top 20
                all_events.sort(key=lambda x: x['ps_hh'], reverse=True)
                top_20_events = all_events[:20]
                
                output_filename = '/srv/top_20_ps_hh_events.txt'
                with open(output_filename, 'w') as f:
                    f.write("# Top 20 events with largest SvB_MA.ps_hh\n")
                    f.write("# Format: run luminosityBlock event ps_hh\n")
                    f.write("#" + "-"*60 + "\n")
                    for evt in top_20_events:
                        f.write(f"{evt['run']:d}:{evt['lumi']:d}:{evt['event']:d} {evt['ps_hh']:.6f}\n")
                
                logging.info(f"Wrote top 20 events with largest ps_hh to {output_filename}")
            else:
                logging.warning(f"No events were tracked across all chunks")
        
        return accumulator
