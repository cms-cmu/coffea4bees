from __future__ import annotations

import copy
import logging
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING

import awkward as ak
import numpy as np
import yaml
import gc
from src.physics.objects.jet_corrections import apply_jerc_corrections, apply_jerc_corrections_jsonpog
from src.physics.common import update_events
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from coffea4bees.analysis.helpers.event_weights import (
    add_btagweights,
    add_pseudotagweights,
)
from src.physics.event_selection import apply_event_selection
from coffea4bees.analysis.helpers.event_weights import add_weights
from coffea4bees.analysis.helpers.event_selection import (
    apply_boosted_4b_selection,
    apply_dilep_ttbar_selection,
    apply_4b_selection
)
from coffea4bees.analysis.helpers.object_selection import load_object_selection_config
from coffea4bees.analysis.helpers.filling_histograms import (
    filling_nominal_histograms,
    filling_syst_histograms,
)
from src.friendtrees.FriendTreeSchema import FriendTreeSchema
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea4bees.analysis.helpers.candidates_selection import create_cand_jet_dijet_quadjet, load_candidates_selection_config
from coffea4bees.analysis.helpers.SvB_helpers import setSvBVars, subtract_ttbar_with_FvT, setFvTVars
from coffea4bees.analysis.helpers.parking import (
    DEFAULT_LUMI_CFG_PATH as _PARKING_LUMI_CFG_DEFAULT,
    assign_is_parking,
    load_parking_lumi_cfg,
)
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
from coffea.nanoevents import NanoAODSchema
from src.compat import nano_from_root
from coffea.util import load
from memory_profiler import profile
import psutil
import os

from ..helpers.load_friend import (
    FriendTemplate,
    parse_friends,
    read_MvD_friend,
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


class _Unset:
    """Sentinel: SvB key absent from config — legacy ROOT fallback is allowed."""
_UNSET = _Unset()


def _init_classfier(path: str | list[HCRModelMetadata] | None | _Unset | bool):
    if path is None or isinstance(path, _Unset) or path is False:
        return None
    if isinstance(path, str):
        from ..helpers.classifier.HCR import Legacy_HCREnsemble
        return Legacy_HCREnsemble(path)
    else:
        from ..helpers.classifier.HCR import HCREnsemble
        return HCREnsemble(path)

def _init_classfier_FvT(path: str | list[HCRModelMetadata] | None | bool):
    if path is None or path is False:
        return None
    if isinstance(path, str):
        from ..helpers.classifier.HCR import Legacy_HCREnsemble_FvT
        return Legacy_HCREnsemble_FvT(path)
    else:
        from ..helpers.classifier.HCR import HCREnsemble
        return HCREnsemble(path)

def _init_feynnet(cfg):
    """Construct a FeynNet classifier from config.

    Two config schemas are supported:

    1. Flat list of model entries (back-compat) -> single ``FeynNetEnsemble``.

    2. Dict with ``parking`` and ``preparking`` keys, each a list of model
       entries -> ``FeynNetParkingDispatcher`` that picks the model per event
       based on ``event.is_parking``.

    Returns None if cfg is None.
    """
    if cfg is None:
        return None
    from coffea4bees.analysis.helpers.classifier.FeynNet import (
        FeynNetEnsemble,
        FeynNetParkingDispatcher,
    )
    if isinstance(cfg, dict) and "parking" in cfg and "preparking" in cfg:
        return FeynNetParkingDispatcher(cfg["parking"], cfg["preparking"])
    return FeynNetEnsemble(cfg)

class HH4bBaseProcessor(processor.ProcessorABC):
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
        make_friend_SvB_FeynNet (str): Path for dumping SvB_FeynNet friend tree.
        subtract_ttbar_with_weights (bool): Whether to subtract ttbar using weights.
        plot_ttbar_with_weights (bool): Whether to plot ttbar (3b and 4b) from 3b data using weights.
        apply_mixeddata_sel (bool): Whether to apply mixed data selection.
        friends (dict): Dictionary of friend tree templates or paths.

    Returns:
        dict: Output containing histograms, cutflow, and optionally dumped friend trees.

    Chunk-Scoped Instance Variables:
        These variables are set by process() for each chunk and used by helper methods.
        They are valid for the entire processing of one chunk and should not be accessed
        before process() has been called.

        Metadata variables (set in process()):
            - fname (str): Full path to the input file
            - dataset (str): Dataset name
            - estart (int): Starting entry number for this chunk
            - estop (int): Ending entry number for this chunk
            - chunk (str): Formatted string identifying this chunk
            - year (str): Data-taking year
            - year_label (str): Year label for corrections
            - processName (str): Process name (e.g., 'TTbar', 'HH4b')
            - target (Chunk): Chunk object for friend tree loading
            - path (str): Directory path containing the input file
            - nEvent (int): Number of events in this chunk
            - config (dict): Process/dataset dependent configuration flags
            - gaussKernalMean (float | None): Mean for Gaussian kernel reweighting
            - resonance_weights (np.ndarray | None): Resonance reweighting array
    """
    def __init__(
        self,
        *,
        SvB: str|list[HCRModelMetadata]|None|_Unset = _UNSET,
        SvB_MA: str|list[HCRModelMetadata]|None|_Unset = _UNSET,
        SvB_FeynNet: list[dict] | None = None,
        FvT: str|list[HCRModelMetadata] = None,
        blind: bool = False,
        apply_JCM: bool = True,
        JCM_file: str | None = None,
        weights: str | None = None,
        corrections_metadata: dict = None,
        apply_trigWeight: bool = True,
        apply_btagSF: bool = True,
        apply_FvT: bool = True,
        apply_boosted_veto: bool = False,
        run_dilep_ttbar_crosscheck: bool = False,
        fill_histograms: bool = True,
        hist_cuts = [],
        run_SvB: bool = True,
        run_SvB_FeynNet_comparison: bool = False,
        top_reconstruction: str | None = None,
        run_systematics: list = [],  #### Way of splitting systematics. It can be event_weights, jes, btag
        make_classifier_input: str = None,
        make_top_reconstruction: str = None,
        make_friend_JCM_weight: str = None,
        make_friend_FvT_weight: str = None,
        make_friend_SvB: str = None,
        make_friend_SvB_FeynNet: str = None,
        subtract_ttbar_with_weights: bool = False,
        plot_ttbar_with_weights: bool = False,
        plot_ttbar_with_MvD_weights: bool = False,
        apply_MvD: bool = False,
        apply_MvD_weight: bool = False,
        apply_mixeddata_sel: bool = False,  #### apply HIG-22-011 sel for mixeddata
        fourTag_use_tight: bool = False,  # Run3: redefine fourTag as 3 Tight + >=4 Medium b-tagged jets
        friends: dict[str, str|FriendTemplate] = None,
        return_events_for_display: bool = False,
        tracker = None,
        object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
        candidates_selection_cfg: str = "coffea4bees/analysis/metadata/candidates_selection_thresholds.yml",
        parking_lumi_cfg: str = _PARKING_LUMI_CFG_DEFAULT,
        year_override: bool = False,
        compute_hemi_mixing_diagnostics: bool = False,
        plot_extra_canjet_vars: bool = False,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.tracker = tracker
        self.sel_cfg = load_object_selection_config(object_selection_cfg) if object_selection_cfg else None
        self.cand_cfg = load_candidates_selection_config(candidates_selection_cfg) if candidates_selection_cfg else None
        self.blind = blind
        self.fourTag_use_tight = fourTag_use_tight

        import os
        self.weights_data = {}
        if weights and os.path.exists(weights):
            logging.info(f"Loading weights metadata in processor from: {weights}")
            self.weights_data = yaml.safe_load(open(weights, 'r')).get('weights', {})

        self.apply_JCM = {}
        if apply_JCM:
            if isinstance(JCM_file, str):
                self.apply_JCM = {"default": jetCombinatoricModel(JCM_file)}
            elif self.weights_data:
                for year, year_cfg in self.weights_data.items():
                    jcm_path = year_cfg.get("JCM_file") or year_cfg.get("JCM")
                    if jcm_path:
                        self.apply_JCM[year] = jetCombinatoricModel(jcm_path)
                if not self.apply_JCM:
                    raise ValueError("apply_JCM is True, but no JCM_file found in weights file.")
                logging.info(f"Loaded JCM models for {len(self.apply_JCM)} eras: {list(self.apply_JCM.keys())}")
            else:
                raise ValueError("apply_JCM is True, but JCM_file is not specified and no weights file is provided.")
        else:
            self.apply_JCM = None

        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.apply_FvT = apply_FvT
        self.apply_MvD = apply_MvD
        self.apply_MvD_weight = apply_MvD_weight
        self.run_SvB = run_SvB
        self.run_SvB_FeynNet_comparison = run_SvB_FeynNet_comparison
        self.fill_histograms = fill_histograms
        self.run_dilep_ttbar_crosscheck = run_dilep_ttbar_crosscheck
        self.apply_boosted_veto = apply_boosted_veto

        self.classifier_SvB = {}
        if SvB is True and self.weights_data:
            for year, year_cfg in self.weights_data.items():
                if "SvB" in year_cfg:
                    self.classifier_SvB[year] = _init_classfier(year_cfg["SvB"])
        else:
            self.classifier_SvB = _init_classfier(SvB)

        self.classifier_SvB_MA = {}
        if SvB_MA is True and self.weights_data:
            for year, year_cfg in self.weights_data.items():
                if "SvB_MA" in year_cfg:
                    self.classifier_SvB_MA[year] = _init_classfier(year_cfg["SvB_MA"])
        else:
            self.classifier_SvB_MA = _init_classfier(SvB_MA)

        self.classifier_SvB_FeynNet = _init_feynnet(SvB_FeynNet)
        # Skip legacy ROOT fallback only when the key was explicitly set to null
        # in the config (SvB=None). When the key is absent (_UNSET), the legacy
        # fallback is still allowed.
        self._skip_svb_legacy = set()
        if SvB is None:
            self._skip_svb_legacy.add("SvB")
        if SvB_MA is None:
            self._skip_svb_legacy.add("SvB_MA")

        self.classifier_FvT = {}
        if FvT is True and self.weights_data:
            for year, year_cfg in self.weights_data.items():
                if "FvT" in year_cfg:
                    self.classifier_FvT[year] = _init_classfier_FvT(year_cfg["FvT"])
        else:
            self.classifier_FvT = _init_classfier_FvT(FvT)

        self.corrections_metadata = corrections_metadata
        self.run_systematics = ['others', 'jes'] if 'all' in run_systematics else run_systematics
        self.make_top_reconstruction = make_top_reconstruction
        self.make_classifier_input = make_classifier_input
        self.make_friend_JCM_weight = make_friend_JCM_weight
        self.make_friend_FvT_weight = make_friend_FvT_weight
        self.make_friend_SvB = make_friend_SvB
        self.make_friend_SvB_FeynNet = make_friend_SvB_FeynNet
        self.top_reconstruction = top_reconstruction
        if self.top_reconstruction is not None and self.top_reconstruction not in ["slow", "fast"]:
            raise ValueError(f"top_reconstruction must be None, 'slow', or 'fast', got: {self.top_reconstruction}")
        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights

        if self.subtract_ttbar_with_weights and not self.apply_FvT:
            logging.info("subtract_ttbar_with_weights=True, but apply_FvT=False.  Setting apply_FvT=True\n")
            self.apply_FvT = True

        self.plot_ttbar_with_weights = plot_ttbar_with_weights
        self.plot_ttbar_with_MvD_weights = plot_ttbar_with_MvD_weights
        self.friends = parse_friends(friends)
        if self.friends:
            logging.info(f"Loaded friend tree definitions: {list(self.friends.keys())}")
        self.histCuts = hist_cuts
        self.apply_mixeddata_sel = apply_mixeddata_sel
        self.return_events_for_display = return_events_for_display
        self.year_override = year_override
        self.parking_lumi_cfg = load_parking_lumi_cfg(parking_lumi_cfg) if parking_lumi_cfg else None
        self.compute_hemi_mixing_diagnostics = compute_hemi_mixing_diagnostics
        self.plot_extra_canjet_vars = plot_extra_canjet_vars

        # Track top 20 events with largest ps_hh across all chunks
        self.top_ps_hh_events = []

        # Memory monitoring
        self.debug_memory = False  # Set to False to disable memory monitoring


    def _needs_is_parking(self) -> bool:
        """True if any active classifier consumes ``event.is_parking``.

        Currently only ``FeynNetParkingDispatcher`` reads it. Returning
        False short-circuits ``assign_is_parking`` so downstream analyses
        that load SvB_FeynNet from a friend tree don't require the
        is_parking friend tree to be present.
        """
        from coffea4bees.analysis.helpers.classifier.FeynNet import (
            FeynNetParkingDispatcher,
        )
        return isinstance(self.classifier_SvB_FeynNet, FeynNetParkingDispatcher)


    # @profile
    def process(self, event):
        """Process one chunk of events through the full analysis workflow.

        This is the main entry point for processing events. It initializes all chunk-scoped
        instance variables (see class docstring for list), loads friend trees, applies event
        selection, calculates weights, and processes all systematic variations.

        Chunk-Scoped Variables Initialized:
            This method sets all the chunk-scoped instance variables that helper methods depend on:
            - fname, dataset, estart, estop, chunk: File and chunk identification
            - year, year_label, processName: Analysis metadata
            - target: Chunk object for friend tree loading
            - path: Directory path for the input file
            - nEvent: Number of events in chunk
            - config: Process/dataset configuration flags
            - gaussKernalMean, resonance_weights: Resonance reweighting (for signal)

        Args:
            event: Coffea NanoEvents array for this chunk

        Returns:
            dict: Accumulated output from all systematic variations containing:
                - Histograms (if fill_histograms=True)
                - Cutflow tables
                - Friend tree outputs (if configured)
                - Event counts and metadata
        """
        logging.debug(event.metadata)
        self._log_memory("process_start")

        ### _stage is a context manager that logs the time and memory usage of each stage
        with self._stage("metadata_setup"):
            self.fname   = event.metadata['filename']
            self.dataset = event.metadata['dataset']
            self.estart  = event.metadata['entrystart']
            self.estop   = event.metadata['entrystop']
            self.chunk   = f'{self.dataset}::{self.estart:6d}:{self.estop:6d} >>> '
            self.year    = event.metadata['year']
            self.year_label = self.corrections_metadata[self.year]['year_label']
            weights_era = self.year
            if weights_era == "2018":
                weights_era = "UL18"
            elif weights_era == "2017":
                weights_era = "UL17"
            elif weights_era == "2016":
                weights_era = "UL16_preVFP"
            self.clf_SvB = self.classifier_SvB.get(weights_era) or self.classifier_SvB.get(self.year_label) or self.classifier_SvB.get(self.year) if isinstance(self.classifier_SvB, dict) else self.classifier_SvB
            self.clf_SvB_MA = self.classifier_SvB_MA.get(weights_era) or self.classifier_SvB_MA.get(self.year_label) or self.classifier_SvB_MA.get(self.year) if isinstance(self.classifier_SvB_MA, dict) else self.classifier_SvB_MA
            self.clf_FvT = self.classifier_FvT.get(weights_era) or self.classifier_FvT.get(self.year_label) or self.classifier_FvT.get(self.year) if isinstance(self.classifier_FvT, dict) else self.classifier_FvT
            self.jcm_model = self.apply_JCM.get(weights_era) or self.apply_JCM.get(self.year_label) or self.apply_JCM.get(self.year) or self.apply_JCM.get("default") if isinstance(self.apply_JCM, dict) else self.apply_JCM
            self.processName = event.metadata['processName']

            if (self.processName.find("mix") != -1 or self.dataset.find("syn") != -1) and not self.processName.startswith("mixeddata_all"):
                new_processName = self.dataset.replace(f'_{self.year}','')
                logging.debug(f"Overriding processName: {self.processName} to {new_processName} for dataset {self.dataset}")
                self.processName = new_processName

            ### target is for new friend trees
            self.target = Chunk.from_coffea_events(event)
            self._log_memory("after_metadata_setup")

        with self._stage("processor_config"):
            #
            # Set process and datset dependent flags
            #
            self.config = processor_config(self.processName, self.dataset, event)
            # print("HACK")
            if self.config["isRun3"]:
                self.config["isSyntheticData"] = bool(self.config["isMixedData"]) or self.config["isSyntheticData"]
                self.config["fourTag_use_tight"] = self.fourTag_use_tight
            logging.debug(f'{self.chunk} config={self.config}, for file {self.fname}\n')


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
        self.path = self.fname.replace(self.fname.split("/")[-1], "")
        with self._stage("load_friend_FvT"):
            if self.apply_FvT and self.clf_FvT is None:
                self.load_FvT(event)

        with self._stage("load_friend_MvD"):
            if self.apply_MvD and self.apply_MvD_weight:
                self.load_MvD(event)

        with self._stage("load_friend_SvB"):
            if self.run_SvB:
                self.load_SvB(event)
                # Keep run_SvB on if any SvB-like record was loaded (SvB, SvB_MA,
                # or any configurable SvB_* / SvB_FeynNet friend) or an on-the-fly
                # classifier is configured (its field is created later in
                # candidates_selection). Only disable when nothing is available.
                has_any_svb = any(f.startswith("SvB") for f in event.fields)
                if not has_any_svb and self.clf_SvB_MA is None and self.classifier_SvB_FeynNet is None:
                    logging.warning("No SvB-like friend or classifier available after load_SvB — disabling run_SvB for this chunk.")
                    self.run_SvB = False

        with self._stage("assign_is_parking"):
            if self.parking_lumi_cfg is not None and self._needs_is_parking():
                assign_is_parking(
                    event,
                    year=self.year,
                    is_mc=self.config["isMC"],
                    parking_lumi_cfg=self.parking_lumi_cfg,
                    friend=self.friends.get("is_parking"),
                    target=self.target,
                )

        with self._stage("load_JCM_friends"):
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
        with self._stage("event_selection"):
            event = apply_event_selection(
                event,
                self.corrections_metadata[self.year],
                cut_on_lumimask=self.config["cut_on_lumimask"]
            )
        self._log_memory("after_event_selection")

        ### adds all the event mc weights and 1 for data
        with self._stage("add_weights"):
            weights, list_weight_names = add_weights(
                event,
                config = self.config,
                target=self.target,
                dataset=self.dataset,
                year_label=self.year_label,
                friend_trigWeight=self.friends.get("trigWeight"),
                corrections_metadata=self.corrections_metadata[self.year],
                apply_trigWeight=self.apply_trigWeight,
                run_systematics= 'others' in self.run_systematics,
            )

        #
        # Checking boosted selection (should change in the future)
        #
        event['notInBoostedSel'] = np.full(len(event), True)
        with self._stage("boosted_veto"):
            if self.apply_boosted_veto:
                self.boosted_veto(event)

        #
        # Calculate and apply Jet Energy Calibration
        #
        with self._stage("jet_corrections"):
            if self.config["do_jet_calibration"]:

                jets = apply_jerc_corrections_jsonpog(
                    event,
                    corrections_metadata=self.corrections_metadata[self.year],
                    isMC=self.config["isMC"],
                    dataset=self.dataset,
                    run_systematics='jes' in self.run_systematics,
                )
            else:
                jets = event.Jet

        # Determine which shifts to run
        with self._stage("build_shifts"):
            if 'jes' in self.run_systematics:
                shifts = []
                shifts.extend([({"Jet": jets.JER.up}, f"CMS_res_j_{self.year_label}Up"), ({"Jet": jets.JER.down}, f"CMS_res_j_{self.year_label}Down")])

                for jesunc in self.corrections_metadata[self.year]["jes_unc"]:
                    shifts.extend( [ ({"Jet": jets[f"JES_{jesunc}"].up}, f"CMS_scale_j_{jesunc}Up"),
                                     ({"Jet": jets[f"JES_{jesunc}"].down}, f"CMS_scale_j_{jesunc}Down"), ] )
                logging.info(f"\nJet variations {[name for _, name in shifts]}")
            else:
                shifts = [({"Jet": jets}, None)]

        with self._stage("process_all_shifts"):
            return processor.accumulate( self.process_shift(update_events(event, collections), name, weights, list_weight_names) for collections, name in shifts )

    # @profile
    def process_shift(self, event, shift_name, weights, list_weight_names):
        """Process one systematic shift (or nominal).

        Args:
            event: Event array
            shift_name: Name of systematic shift (None for nominal)
            weights: Weights object
            list_weight_names: List of weight names

        Returns:
            dict: Combined output containing histograms, cutflow, and friend trees
        """
        label = shift_name or "nominal"

        # Copy weights to avoid modifying the original
        weights = copy.deepcopy(weights)

        with self._stage(f"{label}:apply_selection"):
            # Apply object selection
            event = self.apply_selection(event)

            if self.run_dilep_ttbar_crosscheck:
                apply_dilep_ttbar_selection(event, isRun3=self.config["isRun3"])

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

        with self._stage(f"{label}:build_selections"):
            # Build selections
            selections, allcuts = self.build_selections(event, weights)
            event['weight'] = weights.weight()  # For cutflow

        # Initialize output
        processOutput = {}
        with self._stage(f"{label}:cutflow"):
            if not shift_name:
                processOutput['nEvent'] = {
                    event.metadata['dataset']: {
                        'nEvent': self.nEvent,
                        'genWeights': np.sum(event.genWeight) if self.config["isMC"] else self.nEvent
                    }
                }
                # Build and fill cutflow
                self.build_cutflow(event, selections, allcuts, weights)

        with self._stage(f"{label}:btag_sf"):
            # Apply b-tagging scale factors
            weights, list_weight_names = self.apply_btag_sf(
                event, weights, list_weight_names,
                shift_name, selections, allcuts
            )

        with self._stage(f"{label}:presel_filter"):
            # Preselection: keep only three or four tag events
            selections.add("passPreSel", event.passPreSel)
            allcuts.append("passPreSel")
            analysis_selections = selections.all(*allcuts)

            if not shift_name:
                self.fill_cutflow_with_and_without_trig("passPreSel_allTag", event[analysis_selections], weights, analysis_selections)

        with self._stage(f"{label}:pseudotag_weights"):
            # Add pseudotag weights
            weights, list_weight_names = self.include_pseudotag_in_weight(event, weights, list_weight_names)

        # Select events passing all cuts
        selev = event[analysis_selections]

        # Subtract ttbar using SvB if requested
        if self.subtract_ttbar_with_weights:
            with self._stage(f"{label}:subtract_ttbar"):
                pass_ttbar_filter_4b = subtract_ttbar_with_FvT(selev, self.dataset, self.year, "d4_to_t4")
                pass_ttbar_filter_3b = subtract_ttbar_with_FvT(selev, self.dataset, self.year, "d3_to_t3")
                pass_ttbar_filter_selev = ak.where(selev.fourTag, pass_ttbar_filter_4b, pass_ttbar_filter_3b)

                pass_ttbar_filter = np.full(len(event), True)
                pass_ttbar_filter[selections.all(*allcuts)] = pass_ttbar_filter_selev
                selections.add('pass_ttbar_filter', pass_ttbar_filter)
                allcuts.append("pass_ttbar_filter")
                if not shift_name:
                    sel_mask = selections.all(*allcuts)
                    self.fill_cutflow_with_and_without_trig("pass_ttbar_filter", event[sel_mask], weights, sel_mask)
                analysis_selections = selections.all(*allcuts)
                selev = selev[pass_ttbar_filter_selev]

        if len(selev) == 0:
            return processOutput

        with self._stage(f"{label}:reconstruct_tops"):
            # Reconstruct top candidates
            self.reconstruct_tops(selev)

        with self._stage(f"{label}:build_candidates"):
            # Build jet/dijet/quadjet candidates
            selev = self.build_candidates(selev, weights, list_weight_names, analysis_selections, processOutput)


        # Custom Processing in the derived classes
        selev, analysis_selections = self.custom_processing(selev, self.config, selections, allcuts, len(event))

        # Track events for display if requested
        if self.return_events_for_display:
            self.events_for_display(selev, processOutput)

        # Blind data in fourTag SR
        if not (self.config["isMC"] or "mix_v" in self.dataset) and self.blind:
            with self._stage(f"{label}:blinding"):
                blind_flag = self._get_blind_flag(selev)
                if blind_flag is None:
                    pass  # warning already emitted inside _get_blind_flag
                else:
                    blind_sel = np.full(len(event), True)
                    blind_sel[analysis_selections] = blind_flag
                    selections.add('blind', blind_sel)
                    allcuts.append('blind')
                    if not shift_name:
                        sel_mask = selections.all(*allcuts)
                        self.fill_cutflow_with_and_without_trig("blind", event[sel_mask], weights, sel_mask)
                    analysis_selections = selections.all(*allcuts)
                    selev = selev[blind_flag]

        with self._stage(f"{label}:final_weights"):
            # Add weights to selected events
            logging.debug(f"final weight {weights.weight()[:10]}")
            selev["weight"] = weights.weight()[analysis_selections]
            selev["trigWeight"] = weights.partial_weight(include=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections]
            selev['weight_woTrig'] = weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections]
            selev["no_weight"] = np.ones(len(selev))

        with self._stage(f"{label}:detailed_cutflows"):
            # Fill detailed cutflows
            if not shift_name:
                self.fill_detailed_cutflows(selev)
                self._cutFlow.addOutput(processOutput, event.metadata["dataset"])
                if self.plot_ttbar_with_weights and hasattr(self, '_cutFlow_ttbar'):
                    era = event.metadata["dataset"].removeprefix("data_")
                    self._cutFlow_ttbar.addOutput(processOutput, f"TTbar_from_d3_{era}")
                if self.plot_ttbar_with_MvD_weights and hasattr(self, '_cutFlow_ttbar_MvD'):
                    era = event.metadata["dataset"].removeprefix("mixeddata_all_")
                    self._cutFlow_ttbar_MvD.addOutput(processOutput, f"TTbar4b_from_MvD_{era}")

        with self._stage(f"{label}:fill_histograms"):
            # Fill histograms
            hist = {}
            if self.fill_histograms:
                hist = self.histograms(event, selev, weights, analysis_selections, shift_name)

        with self._stage(f"{label}:dump_friends"):
            # Dump friend trees
            friends = self.dump_friend_trees(selev, analysis_selections, shift_name)

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

    def load_FvT(self, event):
        """Load FvT friend tree.

        Requires chunk-scoped variables: target, estart, estop, fname, config
        Must be called after process() has initialized these variables.

        Args:
            event: Event array
        """

        FvT_loaded = False
        if "FvT" in self.friends:
            FvT_arr = rename_FvT_friend(self.target, self.friends["FvT"])
            if FvT_arr is not None:
                event["FvT"] = FvT_arr
                FvT_loaded = True
                if self.config["isDataForMixed"] or self.config["isTTForMixed"]:
                    for _FvT_name in event.metadata["FvT_names"]:
                        event[_FvT_name] = rename_FvT_friend(self.target, self.friends[_FvT_name])
                        event[_FvT_name, _FvT_name] = event[_FvT_name].FvT

        if not FvT_loaded:
            # TODO: remove backward compatibility in the future
            if self.config["isMixedData"]:

                FvT_name = event.metadata["FvT_name"]
                event["FvT"] = getattr(
                    nano_from_root(
                        {f'{event.metadata["FvT_file"]}': "Events"},
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
                    nano_from_root(
                        {f'{event.metadata["FvT_files"][0]}': "Events"},
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
                        nano_from_root(
                            {f"{_FvT_file}": "Events"},
                            entry_start=self.estart,
                            entry_stop=self.estop,
                            schemaclass=FriendTreeSchema,
                        ).events(),
                        _FvT_name,
                    )

                    event[_FvT_name, _FvT_name] = getattr(event[_FvT_name], _FvT_name)

            else:
                event["FvT"] = (
                    nano_from_root(
                        {f'{self.fname.replace("picoAOD", "FvT")}': "Events"},
                        entry_start=self.estart,
                        entry_stop=self.estop,
                        schemaclass=FriendTreeSchema
                    ).events().FvT
                )


            if not ak.all(event.FvT.event == event.event):
                raise ValueError("ERROR: FvT events do not match events ttree")

        setFvTVars("FvT", event)


    def load_MvD(self, event):
        """Load MvD friend tree.

        Requires chunk-scoped variables: target
        Must be called after process() has initialized these variables.

        Args:
            event: Event array
        """
        if "MvD" in self.friends:
            event["MvD"] = read_MvD_friend(self.target, self.friends["MvD"])
        else:
            raise ValueError("apply_MvD=True but no 'MvD' entry found in friends dict")

    def load_SvB(self, event):
        """Load SvB and SvB_MA scores from one of three sources (in priority order):
        1. Friend tree arrays (modern method, via --friends config)
        2. Classifier model (on-the-fly inference, via SvB/SvB_MA config keys)
        3. Legacy ROOT files next to the picoAOD (fallback)

        Sources 1 and 2 are resolved before this method via self.friends and
        self.classifier_SvB/self.classifier_SvB_MA. The legacy ROOT fallback
        only runs if neither of those provided the field.

        Requires chunk-scoped variables: target, estart, estop, fname, path, dataset, config
        Must be called after process() has initialized these variables.

        Args:
            event: Event array
        """

        # Source 1: Load from friend tree arrays
        for k in self.friends:
            if k.startswith("SvB") and not k.startswith("SvB_FeynNet"):
                logging.debug(f"Loading SvB friend tree for {k}")
                try:
                    result = rename_SvB_friend(self.target, self.friends[k])
                    if result is None:
                        logging.warning(f"No SvB friend tree entries found for {k} (target not in friend mapping). Skipping.")
                        continue
                    event[k] = result
                    setSvBVars(k, event)
                except Exception as e:
                    logging.warning(f"rename_SvB_friend failed for {k}: {e}. Loading raw arrays.")
                    result = self.friends[k].arrays(self.target)
                    if result is None:
                        logging.warning(f"No raw SvB friend tree entries found for {k}. Skipping.")
                        continue
                    event[k] = result
                    setSvBVars(k, event)
            elif k.startswith("SvB_FeynNet"):
                # Load SvB_FeynNet directly from friend tree — fields are already in the correct
                # format (p_ggHH, p_ZZ, ps_hh, etc.) as stored by dump_SvB_FeynNet, no rename needed.
                try:
                    result = self.friends[k].arrays(self.target)
                    if result is None:
                        logging.warning(f"No SvB_FeynNet friend tree entries found (target not in friend mapping). Skipping.")
                        continue
                    event[k] = result
                    logging.debug(f"Loaded SvB_FeynNet friend tree for {k}")
                except Exception as e:
                    logging.warning(f"Failed to load SvB_FeynNet friend tree for {k}: {e}. Skipping.")

        self._log_memory("after_friend_trees_loaded")

        # Source 3: Legacy ROOT file fallback (only if no friend or classifier provides it)
        if self.apply_mixeddata_sel: SvB_suffix = '_newSBDef'
        else: SvB_suffix = '_ULHH'

        for svb_name, classifier in [("SvB", self.classifier_SvB), ("SvB_MA", self.classifier_SvB_MA)]:
            if svb_name in event.fields or classifier is not None:
                continue
            if svb_name in self._skip_svb_legacy:
                logging.debug(f"{svb_name} explicitly set to null in config — skipping legacy ROOT fallback.")
                continue
            # Legacy ROOT file fallback
            svb_file = f'{self.path}/{svb_name}{SvB_suffix}.root' if 'mix' in self.dataset else f'{self.fname.replace("picoAOD", f"{svb_name}{SvB_suffix}")}'
            try:
                event[svb_name] = (
                    nano_from_root(
                        {svb_file: "Events"},
                        entry_start=self.estart,
                        entry_stop=self.estop,
                        schemaclass=FriendTreeSchema
                    ).events()[svb_name]
                )

                if not ak.all(getattr(event, svb_name).event == event.event):
                    raise ValueError(f"ERROR: {svb_name} events do not match events ttree")
                setSvBVars(svb_name, event)
            except (FileNotFoundError, OSError):
                logging.debug(f"No {svb_name} source configured (no friend, classifier, or ROOT file at {svb_file}). Skipping.")

    def boosted_veto(self, event):
        """Apply veto for events selected in boosted analysis. This is for Run2 UL only.

        Requires chunk-scoped variables: dataset
        Must be called after process() has initialized these variables.

        Args:
            event: Event array
        """

        if self.dataset.startswith("GluGluToHHTo4B_cHHH1"):
            boosted_file = load("coffea4bees/metadata/boosted_overlap_signal.coffea")['boosted']
            boosted_events = boosted_file.get(self.dataset, {}).get('event', event.event)
            boosted_events_set = set(boosted_events)
            event['notInBoostedSel'] = np.array([e not in boosted_events_set for e in event.event.to_numpy()])
        elif self.dataset.startswith("VBFHHTo4B_kl_1p00_cv_1p00_c2v_1p00"):
            event = apply_boosted_4b_selection(event)
            event['notInBoostedSel'] = ~event.passBoostedSel
        elif self.dataset.startswith("VBFHHTo4B_kl_1p00_cv_1p00_c2v_1p00"):
            event = apply_boosted_4b_selection(event)
            event['notInBoostedSel'] = ~event.passBoostedSel
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
            logging.debug(f"Boosted veto not applied for dataset {self.dataset}")

    def build_selections(self, event, weights):
        """Build PackedSelection object with all cuts.

        Requires chunk-scoped variables: config
        Must be called after process() has initialized these variables.

        Args:
            event: Event array
            weights: Weights object

        Returns:
            tuple: (selections, allcuts) - PackedSelection object and list of cut names
        """
        selections = PackedSelection()
        selections.add("lumimask", event.lumimask)
        selections.add("passNoiseFilter", event.passNoiseFilter)
        selections.add("passHLT", event.passHLT if self.config["cut_on_HLT_decision"]
                      else np.full(len(event), True))
        selections.add('passJetMult', event.passJetMult)
        allcuts = ['lumimask', 'passNoiseFilter', 'passHLT', 'passJetMult']

        # Add MC-specific selections
        if self.config["isMC"]:
            self.add_genweight_check(event, selections, weights)
            allcuts.append("passCleanGenWeight")
        else:
            event['passCleanGenWeight'] = True
            selections.add("passCleanGenWeight", event.passCleanGenWeight)

        # Add signal-specific selections (truth matching)
        if self.config["isSignal"]:
            event['bfromHorZ_all'] = find_genpart(event.GenPart, [5], [23, 25])

            if "status" in event.bfromHorZ_all.fields:
                event['bfromHorZ'] = event.bfromHorZ_all[event.bfromHorZ_all.status == 23]
            else:
                logging.warning(f"\nStatus Missing for GenParticles in dataset {self.dataset}\n")
                event['bfromHorZ'] = event.bfromHorZ_all

            event['GenJet', 'selectedBs'] = (np.abs(event.GenJet.partonFlavour) == 5)
            event['selGenBJet'] = event.GenJet[event.GenJet.selectedBs]
            event['matchedGenBJet'] = event.bfromHorZ.nearest(event.selGenBJet, threshold=10)
            event["matchedGenBJet"] = event.matchedGenBJet[~ak.is_none(event.matchedGenBJet, axis=1)]

            event['pass4GenBJets'] = ak.num(event.matchedGenBJet) == 4
            event["truth_v4b"] = ak.where(event.pass4GenBJets,
                                          event.matchedGenBJet.sum(axis=1),
                                          1e-10 * event.matchedGenBJet.sum(axis=1))

            # Apply resonance reweighting if configured
            if self.gaussKernalMean is not None:
                v4b_index = np.floor_divide(event.truth_v4b.mass, 12).to_numpy()
                v4b_index[v4b_index > 98] = 98
                vectorized_v4b = np.vectorize(lambda i: self.resonance_weights[int(i)])
                weights_resonance = vectorized_v4b(v4b_index)
                weights.add("resonance_reweight", weights_resonance)
        else:
            event['pass4GenBJets'] = True
        selections.add("pass4GenBJets", event.pass4GenBJets)

        return selections, allcuts

    def add_genweight_check(self, event, selections, weights):
        """Check for outliers in event weights and add passCleanGenWeight selection.

        Requires chunk-scoped variables: dataset
        Must be called after process() has initialized these variables.

        Args:
            event: Event array
            selections: PackedSelection object to add cut to
            weights: Weights object for accessing genWeight
        """
        tmp_weights = weights.weight()
        mean_weights = np.mean(tmp_weights)
        std_weights = np.std(tmp_weights)
        z_scores = np.abs((tmp_weights - mean_weights) / std_weights)
        pass_outliers = z_scores < 30
        event["passCleanGenWeight"] = pass_outliers
        if np.any(~pass_outliers) and std_weights > 0:
            logging.warning(f"Outliers in weights:{tmp_weights[~pass_outliers]}, while mean is {mean_weights} and std is {std_weights} for event {event[~pass_outliers].event} in {self.dataset}\n")
        selections.add("passCleanGenWeight", event.passCleanGenWeight)

    def build_cutflow(self, event, selections, allcuts, weights):
        """Build and fill cutflow histograms.

        Requires chunk-scoped variables: dataset, config
        Must be called after process() has initialized these variables.

        Override this in subclasses to customize the cutflow selections.

        Args:
            event: Event array
            selections: PackedSelection object
            allcuts: List of all cut names
            weights: Weights object
        """
        # Build ordered dictionary of cutflow selections
        sel_dict = OrderedDict({
            'all': selections.require(lumimask=True),
            'passCleanGenWeight': selections.require(lumimask=True, passCleanGenWeight=True),
            'pass4GenBJets': selections.require(lumimask=True, passCleanGenWeight=True, pass4GenBJets=True),
            'passNoiseFilter': selections.require(lumimask=True, passCleanGenWeight=True, passNoiseFilter=True),
            'passHLT': selections.require(lumimask=True, passCleanGenWeight=True, passNoiseFilter=True, passHLT=True),
        })
        sel_dict['passJetMult'] = selections.all(*allcuts)

        # Fill cutflow histograms
        self._cutFlow = cutflow_4b(do_truth_hists=self.config["isSignal"])
        for cut, sel in sel_dict.items():
            self.fill_cutflow_with_and_without_trig(cut, event[sel], selection_mask=sel, weights=weights)

        if self.plot_ttbar_with_weights:
            self._cutFlow_ttbar = cutflow_4b(do_truth_hists=False)

        if self.plot_ttbar_with_MvD_weights:
            self._cutFlow_ttbar_MvD = cutflow_4b(do_truth_hists=False)

    def fill_cutflow_with_and_without_trig(self, cut_name, events, weights=None, selection_mask=None, allTag=None, weight_override=None):
        """Helper to fill cutflow with and without trigger weight.

        Supports two usage modes:
        1. Pre-selection mode: Pass full event array, weights object, and selection_mask
        - Used in build_cutflow, apply_btag_sf, process_shift for early cuts
        2. Post-selection mode: Pass filtered selev with embedded 'weight' and 'weight_woTrig'
        - Used in fill_detailed_cutflows for cuts after candidate building

        Subclasses can override this to customize trigger weight handling.

        Args:
            cut_name: Name of the cut for cutflow
            events: Event array (full or filtered selev)
            weights: Weights object (required for pre-selection mode, None for post-selection)
            selection_mask: Boolean mask for the selection (required for pre-selection mode)
            allTag: Whether to use allTag (default: True for pre-selection, None for post-selection)
            weight_override: Optional weight override for the main fill (not woTrig)
        """
        # Determine which mode we're in
        if weights is not None and selection_mask is not None:
            # Pre-selection mode: working with full event array
            use_allTag = True if allTag is None else allTag
            if weight_override is not None:
                self._cutFlow.fill(cut_name, events, allTag=use_allTag, wOverride=weight_override)
            else:
                self._cutFlow.fill(cut_name, events, allTag=use_allTag)

            trig_excluded = weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selection_mask]
            self._cutFlow.fill(f"{cut_name}_woTrig", events, allTag=use_allTag, wOverride=trig_excluded)

        elif 'weight' in events.fields and 'weight_woTrig' in events.fields:
            # Post-selection mode: selev already has embedded weights
            # Don't use allTag for post-selection fills unless explicitly specified
            if allTag is not None:
                fill_kwargs = {'allTag': allTag}
            else:
                fill_kwargs = {}

            if weight_override is not None:
                self._cutFlow.fill(cut_name, events, wOverride=weight_override, **fill_kwargs)
            else:
                self._cutFlow.fill(cut_name, events, **fill_kwargs)

            self._cutFlow.fill(f"{cut_name}_woTrig", events, wOverride=events['weight_woTrig'], **fill_kwargs)

        else:
            raise ValueError(
                f"Invalid arguments to fill_cutflow_with_and_without_trig for cut '{cut_name}'. "
                "Must provide either (weights + selection_mask) for pre-selection mode, "
                "or events with 'weight' and 'weight_woTrig' fields for post-selection mode."
            )

    def apply_btag_sf(self, event, weights, list_weight_names, shift_name, selections, allcuts):
        """Calculate and apply b-tagging scale factors.

        Args:
            event: Event array
            weights: Weights object
            list_weight_names: List of weight names
            shift_name: Name of systematic shift (None for nominal)
            selections: PackedSelection object
            allcuts: List of all cut names

        Returns:
            tuple: (weights, list_weight_names) - Updated weights and weight name list
        """
        if not (self.config["isMC"] and self.apply_btagSF):
            return weights, list_weight_names

        weights, list_weight_names = add_btagweights(
            event,
            weights,
            list_weight_names=list_weight_names,
            shift_name=shift_name,
            use_prestored_btag_SF=self.config["use_prestored_btag_SF"],
            run_systematics='others' in self.run_systematics,
            corrections_metadata=self.corrections_metadata[self.year],
            isRun3=self.config["isRun3"],
        )
        logging.debug(f"Btag weight {weights.partial_weight(include=['CMS_btag'])[:10]}\n")
        event["weight"] = weights.weight()

        if not shift_name:
            sel_mask = selections.all(*allcuts)
            self.fill_cutflow_with_and_without_trig("passJetMult_btagSF", event[sel_mask], weights, sel_mask)

        return weights, list_weight_names

    def reconstruct_tops(self, selev):
        """Build top candidate reconstruction.

        Requires chunk-scoped variables: target (if using friend trees)
        Must be called after process() has initialized these variables.

        Args:
            selev: Selected events array
        """
        if friend := self.friends.get("top_reconstruction"):
            top_cand = friend.arrays(self.target)[ak.to_numpy(selev.event)]
            adding_top_reco_to_event(selev, top_cand)
        elif self.top_reconstruction in ["slow", "fast"]:
            # Sort jets by b-tagging score
            selev["selJet"] = selev.selJet[ak.argsort(selev.selJet.btagScore, axis=1, ascending=False)]

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
            # With top friendtree we don't need the next two lines
            selev["xbW"] = selev.top_cand.xbW
            selev["xW"] = selev.top_cand.xW

    def build_candidates(self, selev, weights, list_weight_names, analysis_selections, processOutput):
        """Build di-jets and quad-jets candidates.

        Requires chunk-scoped variables: config (for isRun3)
        Must be called after process() has initialized these variables.

        Args:
            selev: Selected events array
            weights: Weights object
            list_weight_names: List of weight names
            analysis_selections: Boolean mask for analysis selection
            processOutput: Output dictionary

        Returns:
            Updated selev with candidate information
        """
        return create_cand_jet_dijet_quadjet(
            selev,
            apply_FvT=self.apply_FvT,
            classifier_FvT=self.clf_FvT,
            run_SvB=self.run_SvB,
            run_systematics=self.run_systematics,
            classifier_SvB=self.clf_SvB,
            classifier_SvB_MA=self.clf_SvB_MA,
            classifier_SvB_FeynNet=self.classifier_SvB_FeynNet,
            processOutput=processOutput,
            isRun3=self.config["isRun3"],
            weights=weights,
            list_weight_names=list_weight_names,
            analysis_selections=analysis_selections,
            cand_cfg=self.cand_cfg,
        )


    def custom_processing(self, selev, config, selections, allcuts, nEventTot):
        """
          Place holder for custom analysis implemeted in the derived classes

        """
        return selev, selections.all(*allcuts)


    def fill_detailed_cutflows(self, selev):
        """Fill detailed cutflow histograms after candidate building.

        Requires chunk-scoped variables: config (for isMC)
        Must be called after process() has initialized these variables.

        Args:
            selev: Selected events array with candidates (must have 'weight' and 'weight_woTrig')
        """
        # Now we can use the helper for all fills
        self.fill_cutflow_with_and_without_trig("passPreSel", selev)
        self.fill_cutflow_with_and_without_trig("passDiJetMass", selev[selev.passDiJetMass])
        self.fill_cutflow_with_and_without_trig("boosted_veto_passPreSel", selev[selev.notInBoostedSel])

        # These don't have _woTrig variants, keep as-is
        self._cutFlow.fill("boosted_veto_SR", selev[selev.notInBoostedSel & selev["quadJet_selected"].SR])

        selev['passSR'] = selev.passDiJetMass & selev["quadJet_selected"].SR
        self.fill_cutflow_with_and_without_trig("SR", selev[selev.passSR])

        selev['passSB'] = selev.passDiJetMass & selev["quadJet_selected"].SB
        self.fill_cutflow_with_and_without_trig("SB", selev[selev.passSB])

        self._cutFlow.fill("passVBFSel", selev[selev.passVBFSel])

        # passSvB/failSvB are only set when a signal-discriminant field
        # (SvB_MA or SvB_FeynNet) is present; skip these cutflow rows otherwise.
        if self.run_SvB and "passSvB" in selev.fields:
            self.fill_cutflow_with_and_without_trig("passSvB", selev[selev.passSvB])
            self.fill_cutflow_with_and_without_trig("failSvB", selev[selev.failSvB])

        if self.run_dilep_ttbar_crosscheck:
            # This one uses weight_noJCM_noFvT override
            self._cutFlow.fill("passMuMu", selev[selev.passMuMu], allTag=True,
                            wOverride=selev['weight_noJCM_noFvT'][selev.passMuMu])
            self._cutFlow.fill("passElMu", selev[selev.passElMu], allTag=True,
                            wOverride=selev['weight_noJCM_noFvT'][selev.passElMu])
            #self._cutFlow.fill("passElEl", selev[selev.passElEl], allTag=True,
            #                wOverride=selev['weight_noJCM_noFvT'][selev.passElEl])

        if self.plot_ttbar_with_weights:
            self._fill_ttbar_detailed_cutflows(selev)

        if self.plot_ttbar_with_MvD_weights:
            self._fill_ttbar_MvD_detailed_cutflows(selev)

    def _fill_ttbar_detailed_cutflows(self, selev):
        """Fill ttbar cutflow histograms using FvT d3_to_t4 and d3_to_t3 weights.

        Uses a single cutflow object where:
          - _cutFlowFourTag is filled with threeTag events weighted by weight_d3_to_t4
            (their contribution to the 4b signal region)
          - _cutFlowThreeTag is filled with threeTag events weighted by weight_d3_to_t3
            (their contribution to the 3b control region)

        Args:
            selev: Selected events array (must have passSR and passSB already set)
        """
        if 'weight_d3_to_t4' not in selev.fields:
            return

        def fill_ttbar_cut(cut, ev):
            ev3 = ev[ev.threeTag]
            n3 = len(ev3)
            self._cutFlow_ttbar._cutFlowFourTag [cut] = (float(np.sum(ev3['weight_d3_to_t4'])), n3)
            self._cutFlow_ttbar._cutFlowThreeTag[cut] = (float(np.sum(ev3['weight_d3_to_t3'])), n3)
            self._cutFlow_ttbar._cutFlowTwoTag  [cut] = (0, 0)

        fill_ttbar_cut("passPreSel", selev)
        fill_ttbar_cut("passDiJetMass", selev[selev.passDiJetMass])
        fill_ttbar_cut("boosted_veto_passPreSel", selev[selev.notInBoostedSel])
        fill_ttbar_cut("boosted_veto_SR", selev[selev.notInBoostedSel & selev["quadJet_selected"].SR])
        fill_ttbar_cut("SR", selev[selev.passSR])
        fill_ttbar_cut("SB", selev[selev.passSB])
        fill_ttbar_cut("passVBFSel", selev[selev.passVBFSel])

        if self.run_SvB and "passSvB" in selev.fields:
            fill_ttbar_cut("passSvB", selev[selev.passSvB])
            fill_ttbar_cut("failSvB", selev[selev.failSvB])

    def _fill_ttbar_MvD_detailed_cutflows(self, selev):
        """Fill TTbar cutflow histograms using MvD p_t4/p_mix4 weights.

        Only fills fourTag events (no threeTag component).
          - _cutFlowFourTag is filled with fourTag events weighted by weight_mix4_to_t4_MvD
          - _cutFlowThreeTag and _cutFlowTwoTag are set to zero

        Args:
            selev: Selected events array (must have passSR and passSB already set)
        """
        if 'weight_mix4_to_t4_MvD' not in selev.fields:
            return

        def fill_ttbar_MvD_cut(cut, ev):
            ev4 = ev[ev.fourTag]
            n4 = len(ev4)
            self._cutFlow_ttbar_MvD._cutFlowFourTag [cut] = (float(np.sum(ev4['weight_mix4_to_t4_MvD'])), n4)
            self._cutFlow_ttbar_MvD._cutFlowThreeTag[cut] = (0, 0)
            self._cutFlow_ttbar_MvD._cutFlowTwoTag  [cut] = (0, 0)

        fill_ttbar_MvD_cut("passPreSel", selev)
        fill_ttbar_MvD_cut("passDiJetMass", selev[selev.passDiJetMass])
        fill_ttbar_MvD_cut("boosted_veto_passPreSel", selev[selev.notInBoostedSel])
        fill_ttbar_MvD_cut("boosted_veto_SR", selev[selev.notInBoostedSel & selev["quadJet_selected"].SR])
        fill_ttbar_MvD_cut("SR", selev[selev.passSR])
        fill_ttbar_MvD_cut("SB", selev[selev.passSB])
        fill_ttbar_MvD_cut("passVBFSel", selev[selev.passVBFSel])

        if self.run_SvB and "passSvB" in selev.fields:
            fill_ttbar_MvD_cut("passSvB", selev[selev.passSvB])
            fill_ttbar_MvD_cut("failSvB", selev[selev.failSvB])

    def dump_friend_trees(self, selev, analysis_selections, shift_name):
        """Dump all requested friend trees.

        Requires chunk-scoped variables: config (for isMC, isSignal)
        Must be called after process() has initialized these variables.

        Args:
            selev: Selected events array
            analysis_selections: Boolean mask for analysis selection
            shift_name: Name of systematic shift (None for nominal)

        Returns:
            dict: Dictionary with 'friends' key containing friend tree data
        """
        friends = {'friends': {}}

        if self.make_top_reconstruction is not None:
            from ..helpers.dump_friendtrees import dump_top_reconstruction
            friends["friends"] |= dump_top_reconstruction(
                selev,
                self.make_top_reconstruction,
                f"top_reco{'_'+shift_name if shift_name else ''}",
                analysis_selections,
            )

        if self.make_classifier_input is not None:
            for k in ["ZZSR", "ZHSR", "HHSR", "SR", "SB"]:
                selev[k] = selev["quadJet_selected"][k]
            selev["nSelJets"] = ak.num(selev.selJet)

            from ..helpers.dump_friendtrees import dump_input_friend
            weight = "weight_noJCM_noFvT"
            if weight not in selev.fields:
                weight = "weight"
            friends["friends"] |= dump_input_friend(
                selev,
                self.make_classifier_input,
                "HCR_input",
                analysis_selections,
                weight=weight,
                NotCanJet="notCanJet_coffea",
            )

        if self.make_friend_JCM_weight is not None:
            from ..helpers.dump_friendtrees import dump_JCM_weight
            friends["friends"] |= dump_JCM_weight(selev, self.make_friend_JCM_weight, "JCM_weight", analysis_selections)

        if self.make_friend_FvT_weight is not None:
            from ..helpers.dump_friendtrees import dump_FvT_weight
            friends["friends"] |= dump_FvT_weight(selev, self.make_friend_FvT_weight, "FvT_weight", analysis_selections)

        if self.make_friend_SvB is not None:
            from ..helpers.dump_friendtrees import dump_SvB
            if "SvB" in selev.fields and self.clf_SvB is not None:
                friends["friends"] |= dump_SvB(selev, self.make_friend_SvB, "SvB", analysis_selections)
            if "SvB_MA" in selev.fields and self.clf_SvB_MA is not None:
                friends["friends"] |= dump_SvB(selev, self.make_friend_SvB, "SvB_MA", analysis_selections)

        if self.make_friend_SvB_FeynNet is not None:
            from ..helpers.dump_friendtrees import dump_SvB_FeynNet
            friends["friends"] |= dump_SvB_FeynNet(selev, self.make_friend_SvB_FeynNet, "SvB_FeynNet", analysis_selections)

        return friends

    def _fourtag_label(self):
        return "fourTag"

    def _get_blind_flag(self, selev):
        """Return a boolean mask (True = keep, False = blind/remove) for data blinding.

        Scans all SvB-like fields (any field starting with 'SvB') and removes SR events
        where *any* ps score (e.g. ps, ps_hh, ps_ttHbb, etc.) from *any* such classifier exceeds 0.8.
        Returns None if no usable SvB field with ps scores is found (blinding skipped).
        """
        svb_names = [f for f in selev.fields if f.startswith("SvB")]
        usable = [name for name in svb_names if any(f.startswith("ps") for f in selev[name].fields)]
        if not usable:
            logging.warning(
                f"Blinding requires at least one SvB field with ps scores, "
                f"but none found among {svb_names} — skipping blinding for this chunk."
            )
            return None
        in_sr = ak.to_numpy(selev["quadJet_selected"].SR)
        # Blind if the event is in SR and ANY ps score from ANY SvB classifier > 0.8
        is_signal = np.zeros(len(selev), dtype=bool)
        for name in usable:
            for field in selev[name].fields:
                if field.startswith("ps"):
                    score = ak.to_numpy(selev[name][field])
                    is_signal = is_signal | (score > 0.8)
        return ~(in_sr & is_signal)

    def apply_selection(self, event):
        """Apply selection to the events"""
        return apply_4b_selection(
            event,
            self.corrections_metadata[self.year],
            config=self.config,
            dataset=self.dataset,
            apply_mixeddata_sel=self.apply_mixeddata_sel,
            sel_cfg=self.sel_cfg,
        )


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

    def _stage(self, name: str):
        """Return a tracker stage context manager, or a no-op if no tracker is set."""
        if self.tracker is not None:
            return self.tracker.stage(name)
        return nullcontext()

    def include_pseudotag_in_weight(self, event, weights, list_weight_names):
        """Include pseudotag weight in the event weight.

        Requires chunk-scoped variables: year, dataset, config
        Must be called after process() has initialized these variables.

        Args:
            event: Event array
            weights: Weights object
            list_weight_names: List of weight names

        Returns:
            Updated weights object
        """
        # When FvT is evaluated on-the-fly via classifier, it hasn't run yet at
        # this point in the pipeline — weights are added later in _apply_ml_scores.
        apply_FvT = self.apply_FvT and self.clf_FvT is None
        return add_pseudotagweights(
            event,
            weights,
            JCM=self.jcm_model,
            apply_FvT=apply_FvT,
            isDataForMixed=self.config["isDataForMixed"],
            apply_MvD=self.apply_MvD,
            apply_MvD_weight=self.apply_MvD_weight,
            isMixedDataAll=self.config["isMixedDataAll"],
            list_weight_names=list_weight_names,
            event_metadata=event.metadata,
            year_label=self.year_label,
            len_event=len(event),
            )

    def events_for_display(self, selev, processOutput):
        """Track top 20 events with largest SvB_MA.ps_hh across all chunks.

        Requires chunk-scoped variables: dataset, year, config
        Must be called after process() has initialized these variables.

        Args:
            selev: Selected events array
            processOutput: Output dictionary to store tracked events
        """

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

    def _attach_hemi_mixing_diagnostics(self, selev):
        """Compute per-hemi 4-vector sums for closure diagnostics.

        Splits each event's jets into +/- hemispheres using the transverse-
        thrust axis and stores eta/pt/pz/mass of the per-hemi 4-vector sum
        for three jet collections:

          can    -- canJet            (4 HH-candidate jets per event)
          sel    -- selJet            (all selected jets, what matching pins)
          other  -- notCanJet_coffea  (the non-candidate selected jets)

        See ~/ClaudeBrain/physics/hemisphere-mixing-toy for the motivation.
        """
        from coffea4bees.hemisphere_mixing.mixing_helpers import (
            transverse_thrust_awkward_fast,
            split_hemispheres,
        )
        thrust = transverse_thrust_awkward_fast(
            selev.Jet, n_steps=720, refine_rounds=2)

        # "all"   = event.Jet  -- the collection the matching variables are
        #                          actually computed over (see compute_hemi_vars
        #                          in mixing_helpers.py).
        # "can"   = canJet     -- HH candidate jets (4 per event); HH observable.
        # "other" = notCanJet  -- selJet minus canJet; the slack carrier.
        collections = [
            ('can',   selev.canJet),
            ('all',   selev.Jet),
            ('other', selev.notCanJet_coffea),
        ]
        for prefix, coll in collections:
            pos, neg = split_hemispheres(coll, thrust)
            s_pos = pos.sum(axis=1)
            s_neg = neg.sum(axis=1)
            selev[f'hemi_{prefix}_pos_pt']   = s_pos.pt
            selev[f'hemi_{prefix}_pos_eta']  = s_pos.eta
            selev[f'hemi_{prefix}_pos_pz']   = s_pos.pz
            selev[f'hemi_{prefix}_pos_mass'] = s_pos.mass
            selev[f'hemi_{prefix}_neg_pt']   = s_neg.pt
            selev[f'hemi_{prefix}_neg_eta']  = s_neg.eta
            selev[f'hemi_{prefix}_neg_pz']   = s_neg.pz
            selev[f'hemi_{prefix}_neg_mass'] = s_neg.mass
        return selev

    def histograms(self, event, selev, weights, analysis_selections, shift_name):
        """Fill histograms for analysis.

        Requires chunk-scoped variables: processName, year, config
        Must be called after process() has initialized these variables.

        Args:
            event: Event array
            selev: Selected events array
            weights: Weights object
            analysis_selections: Boolean mask for analysis selection
            shift_name: Name of systematic shift (None for nominal)

        Returns:
            Dictionary with histogram outputs
        """

        if self.classifier_FvT: apply_FvT = True
        else: apply_FvT = self.apply_FvT

        if self.compute_hemi_mixing_diagnostics:
            selev = self._attach_hemi_mixing_diagnostics(selev)

        if not self.run_systematics:
            ## this can be simplified
            hist_nom = filling_nominal_histograms(
                selev,
                self.jcm_model,
                processName=self.processName,
                year=self.year,
                isMC=self.config["isMC"],
                histCuts=self.histCuts,
                apply_FvT=apply_FvT,
                apply_MvD=self.apply_MvD,
                apply_MvD_weight=self.apply_MvD_weight,
                run_SvB=self.run_SvB,
                run_SvB_FeynNet_comparison=self.run_SvB_FeynNet_comparison,
                run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                top_reconstruction=self.top_reconstruction,
                isDataForMixed=self.config['isDataForMixed'],
                event_metadata=event.metadata,
                year_override=self.year_override,
                compute_hemi_mixing_diagnostics=self.compute_hemi_mixing_diagnostics,
                plot_extra_canjet_vars=self.plot_extra_canjet_vars,
            )
            if not self.plot_ttbar_with_weights and not self.plot_ttbar_with_MvD_weights:
                return hist_nom


            hists = [hist_nom]

            if self.plot_ttbar_with_weights and self.processName == "data":
                hist_t4 = filling_nominal_histograms(
                    selev,
                    self.jcm_model,
                    processName="TTbar4b_from_d3",
                    year=self.year,
                    isMC=self.config["isMC"],
                    histCuts=self.histCuts,
                    apply_FvT=apply_FvT,
                    run_SvB=self.run_SvB,
                    run_SvB_FeynNet_comparison=self.run_SvB_FeynNet_comparison,
                    run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                    top_reconstruction=self.top_reconstruction,
                    isDataForMixed=self.config['isDataForMixed'],
                    event_metadata=event.metadata,
                    weight_name = "weight_d3_to_t4",
                    weight_noFvT_override="weight_d3_to_t4_noFvT",
                    year_override=self.year_override,
                    plot_extra_canjet_vars=self.plot_extra_canjet_vars,
                )

                hist_t3 = filling_nominal_histograms(
                    selev,
                    self.jcm_model,
                    processName="TTbar3b_from_d3",
                    year=self.year,
                    isMC=self.config["isMC"],
                    histCuts=self.histCuts,
                    apply_FvT=apply_FvT,
                    run_SvB=self.run_SvB,
                    run_SvB_FeynNet_comparison=self.run_SvB_FeynNet_comparison,
                    run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                    top_reconstruction=self.top_reconstruction,
                    isDataForMixed=self.config['isDataForMixed'],
                    event_metadata=event.metadata,
                    weight_name = "weight_d3_to_t3",
                    weight_noFvT_override="weight_d3_to_t3_noFvT",
                    year_override=self.year_override,
                    plot_extra_canjet_vars=self.plot_extra_canjet_vars,
                )
                hists += [hist_t4, hist_t3]

            if self.plot_ttbar_with_MvD_weights and self.apply_MvD_weight and self.config["isMixedDataAll"]:
                hist_mvd_t4 = filling_nominal_histograms(
                    selev,
                    self.jcm_model,
                    processName="TTbar4b_from_MvD",
                    year=self.year,
                    isMC=self.config["isMC"],
                    histCuts=self.histCuts,
                    apply_FvT=apply_FvT,
                    apply_MvD=self.apply_MvD,
                    apply_MvD_weight=self.apply_MvD_weight,
                    run_SvB=self.run_SvB,
                    run_SvB_FeynNet_comparison=self.run_SvB_FeynNet_comparison,
                    run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                    top_reconstruction=self.top_reconstruction,
                    isDataForMixed=self.config['isDataForMixed'],
                    event_metadata=event.metadata,
                    tag_list=["fourTag"],
                    weight_name="weight_mix4_to_t4_MvD",
                    weight_noMvD_override="weight_mix4_to_t4_MvD_noMvD",
                    year_override=self.year_override,
                    plot_extra_canjet_vars=self.plot_extra_canjet_vars,
                )
                hists.append(hist_mvd_t4)

            return processor.accumulate(hists)




        #
        # Run systematics
        #
        else:
            return filling_syst_histograms(
                selev,
                weights,
                analysis_selections,
                shift_name=shift_name,
                processName=self.processName,
                year=self.year,
                histCuts=self.histCuts,
                )




class analysis(HH4bBaseProcessor):
    """
    Backward compatibility alias for HH4bBaseProcessor.

    This class exists to maintain compatibility with existing code that uses
    the old 'analysis' class name. New code should use HH4bBaseProcessor directly.

    Deprecated: Use HH4bBaseProcessor instead.
    """
    pass
