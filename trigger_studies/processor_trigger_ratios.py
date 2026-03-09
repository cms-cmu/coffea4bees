import logging, warnings, copy, psutil, os
from src.hist_tools import Fill
from src.data_formats.root import Chunk, TreeReader
from coffea import processor
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import load
from memory_profiler import profile
from coffea.analysis_tools import PackedSelection
from coffea4bees.analysis.helpers.processor_config import processor_config
from src.physics.event_selection import apply_event_selection
from src.physics.objects.jet_corrections import apply_jerc_corrections
from coffea4bees.trigger_studies.skimmer.skimmer_trg import apply_dilep_jet_selection
from coffea4bees.trigger_studies.filling_histograms import filling_trigger_histograms
import awkward as ak
import numpy as np

class analysis(processor.ProcessorABC):
    """
    Coffea processor for trigger studies workflows.

    Returns:
        dict: Output containing histograms, cutflow, and optionally dumped friend trees.
    """
    def __init__(
        self,
        *,
        apply_btagSF: bool = True,
        apply_boosted_veto: bool = False,
        fill_histograms: bool = True,
        hist_cuts = ['passPreSel'],
        corrections_metadata: dict = {},
        run_systematics: list = [],  #### Way of splitting systematics. It can be event_weights, jes, btag
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.apply_btagSF = apply_btagSF
        self.fill_histograms = fill_histograms
        self.apply_boosted_veto = apply_boosted_veto
        self.run_systematics = ['others', 'jes'] if 'all' in run_systematics else run_systematics
        self.histCuts = hist_cuts        
        # Memory monitoring
        self.debug_memory = False  # Set to False to disable memory monitoring
        self.corrections_metadata = corrections_metadata
        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passJetMult",
            "passJetMult_btagSF",
        ]

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

        self.nEvent = len(event)

        #
        # Event selection
        #
        event = apply_event_selection(
            event,
            self.corrections_metadata[self.year],
            cut_on_lumimask=self.config["cut_on_lumimask"]
        )

        logging.info(f"chaeck flaf {event.L1.HTT280er}")
        logging.info(f"chaeck flaf {event.L1.fields}")
        
        #
        # Calculate and apply Jet Energy Calibration
        #
        jets = apply_jerc_corrections(
            event,
            corrections_metadata={}, #self.corrections_metadata[self.year],
            isMC=self.config["isMC"],
            run_systematics=False,
            dataset=self.dataset
        )
        event["Jet"] = jets
        
        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_dilep_jet_selection(
            event,
            self.corrections_metadata[self.year],
            # config=self.config,
            dataset=self.dataset,
            isRun3=self.config["isRun3"],
        )
        

        selev = event[(event.elec_selected_L1 & event.passJetMult) | (event.elec_selected_HLT & event.muon_selected_HLT & event.selJet  )]
        logging.info(f"selev {selev}")
        logging.info(f"selev {selev}")
        logging.info(f"selev {selev}")
        logging.info(f"selev {selev}")
        selev["passPreSel"] = ak.Array([True] * len(selev))
        selev["weight"] = ak.Array([1.0] * len(selev))
        selev["hT_trigger"] = ak.sum(selev.Jet.pt, axis=1)
        selev["jet4_pt"] = selev.Jet[:,3].pt
        selev['hT_trigger_vs_jet4_pt']  = ak.zip( { "hT_trigger": selev.hT_trigger,  "jet4_pt": selev.jet4_pt } )
        
        ## As defined by Marina
        L1_seed_dict = {
            "2022": ["L1_QuadJet60er2p5",
                     "L1_HTT280er",
                     "L1_HTT320er",
                     "L1_HTT360er",
                     "L1_HTT400er",
                     "L1_HTT450er",
                     "L1_HTT280er_QuadJet_70_55_40_35_er2p5",
                     "L1_HTT320er_QuadJet_70_55_40_40_er2p5",
                     "L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3",
                     "L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3",
                     "L1_Mu6_HTT240er"],
            "2023": ["L1_HTT280er",
                     "L1_Mu6_HTT240er"],}
        
        L1_seed_weights = L1_seed_dict[self.year_label]
        for weight_type in L1_seed_weights:
            weight_l1 = weight_type.split('L1_')[1]
            selev[f"{weight_type}"] = ak.values_astype(selev.L1[f"{weight_l1}"], "float64")

        hist = {}
        hist = filling_trigger_histograms(
            selev,
            processName=self.processName,
            year=self.year_label,
            isMC=self.config["isMC"],
            histCuts=self.histCuts,
            isDataForMixed=self.config['isDataForMixed'],
            event_metadata=event.metadata,
            L1_seed_dict=L1_seed_dict,
            )

        return hist 

    def postprocess(self, accumulator):
        return accumulator
