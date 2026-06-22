import logging, warnings, copy, psutil, os
from src.hist_tools import Fill
from src.data_formats.root import Chunk, TreeReader
from coffea import processor
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import load
from memory_profiler import profile
from coffea.analysis_tools import PackedSelection
from coffea4bees.analysis.helpers.processor_config import processor_config
from src.physics.common import drClean
from src.physics.event_selection import apply_event_selection
from src.physics.objects.jet_corrections import apply_jerc_corrections
from coffea4bees.trigger_studies.skimmer.skimmer_trg import apply_dilep_jet_selection
from coffea4bees.trigger_studies.filling_histograms import filling_trigger_histograms
import awkward as ak
import numpy as np

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
                "L1_Mu6_HTT240er"],
    "2024": ["L1_HTT280er",
                "L1_Mu6_HTT240er"],
                }

L1_all_seed_dict = {  ### perform OR of these seeds
    "2022": ["L1_HTT320er_QuadJet_70_55_40_40_er2p5",
                "L1_Mu6_HTT240er",
                "L1_HTT360er",],
    "2023": ["L1_Mu6_HTT240er", 
                "L1_HTT280er"],
    "2024": ["L1_Mu6_HTT240er", 
                "L1_HTT280er"],
            }

HLT_filters_dict = {
    "2022": {
        "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65": [
            "1:0:4:20:4PixelOnlyPFCentralJetTightIDPt20",
            "1:1:3:30:3PixelOnlyPFCentralJetTightIDPt30",
            "1:6:2:40:2PixelOnlyPFCentralJetTightIDPt40",
            "1:22:1:60:1PixelOnlyPFCentralJetTightIDPt60",
            "1:4:4:35:4PFCentralJetTightIDPt35",
            "1:8:3:40:3PFCentralJetTightIDPt40",
            "1:21:2:50:2PFCentralJetTightIDPt50",
            "1:23:1:70:1PFCentralJetTightIDPt70",
            "1:26:2:0.65:BTagCentralJetPt35PFParticleNet2BTagSum0p65"],
        "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5": [
            "3:2:0:280:L1sQuadJetC50to60IorHTT280to500IorHTT250to340QuadJet",
            "1:5:4:30:QuadCentralJet30",
            "3:3:0:320:CaloQuadJet30HT320",
            "1:11:2:0.17:BTagCaloDeepCSVp17Double",
            "1:12:4:30:PFCentralJetLooseIDQuad30",
            "1:13:1:75:1PFCentralJetLooseID75",
            "1:14:2:60:2PFCentralJetLooseID60",
            "1:15:3:45:3PFCentralJetLooseID45",
            "1:16:4:40:4PFCentralJetLooseID40",
            "3:4:0:330:PFCentralJetsLooseIDQuad30HT330",
            "1:25:3:0.24:BTagPFDeepJet4p5Triple"],
    },
    "2023": { 
        "HLT_QuadPFJet70_50_40_35_PNet2BTagMean0p65": [
            "1:0:4:20:4PixelOnlyPFCentralJetTightIDPt20",
            "1:1:3:30:3PixelOnlyPFCentralJetTightIDPt30",
            "1:6:2:40:2PixelOnlyPFCentralJetTightIDPt40",
            "1:22:1:60:1PixelOnlyPFCentralJetTightIDPt60",
            "1:4:4:35:4PFCentralJetTightIDPt35",
            "1:8:3:40:3PFCentralJetTightIDPt40",
            "1:21:2:50:2PFCentralJetTightIDPt50",
            "1:23:1:70:1PFCentralJetTightIDPt70",
            "1:26:2:0.65:BTagCentralJetPt35PFParticleNet2BTagSum0p65"],
        "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55": [
             "1:0:4:20:4PixelOnlyPFCentralJetTightIDPt20",
             "1:3:4:30:4PFCentralJetTightIDPt30",
             "3:5:0:280:PFHT280Jet30",
             "1:26:2:0.55:PFCentralJetPt30PNet2BTagMean0p55"],
        "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5" : [
              "3:2:0:240:L1sQuadJetC50to60IorHTT280to500IorHTT250to340QuadJet",
              "1:5:4:30:QuadCentralJet30",
              "3:3:0:320:CaloQuadJet30HT320",
              "1:11:2:0.17:BTagCaloDeepCSVp17Double",
              "1:12:4:30:PFCentralJetLooseIDQuad30",
              "1:13:1:75:1PFCentralJetLooseID75",
              "1:14:2:60:2PFCentralJetLooseID60",
              "1:15:3:45:3PFCentralJetLooseID45",
              "1:16:4:40:4PFCentralJetLooseID40",
              "3:4:0:330:PFCentralJetsLooseIDQuad30HT330",
              "1:25:3:0.24:BTagPFDeepJet4p5Triple"],
    },
    # note: the filter bits (second column) can be retrieved from here:
    #       https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv15/2024/
    #       (make sure to check the correct NanoAOD version!)
    # note: the filters to use for each trigger path can be retrieved from here:
    #       https://cmshltinfo.app.cern.ch/summary
    "2024": {
        "HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55": [
            "1:34:4:25:4PFCentralJetTightIDPt25",
            #"PFHT250Jet25", # this filter seems to be not stored in NanoAOD
            "1:35:2:0.55:PFCentralJetPt25PNet2BTagMean0p55"],
    },
    "2025": {
        "HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55": [
            "1:34:4:25:4PFCentralJetTightIDPt25",
            #"PFHT250Jet25", # this filter seems to be not stored in NanoAOD
            "1:35:2:0.55:PFCentralJetPt25PNet2BTagMean0p55"],
    }
}

def check_HLT_filter(selev, trgID, trgBit, trgMult):
    type_mask = selev.TrigObj.id == trgID
    bit_mask  = (( selev.TrigObj.filterBits >> trgBit) & 1) == 1
    passed_mask = type_mask & bit_mask

    if trgID == 3:   # for HT filter, need >=1 TrigObj passing
        return ak.num(selev.TrigObj.id[passed_mask]) >= 1

    passed_trgObjs = ak.zip({
        "eta": selev.TrigObj.eta[passed_mask],
        "phi": selev.TrigObj.phi[passed_mask],})

    # drClean returns jets with NO match within cone — want jets WITH a match
    _, no_match_mask = drClean(selev.Jet, passed_trgObjs, cone=0.5)
    match_mask = ~no_match_mask
    n_matched = ak.sum(match_mask, axis=-1)  
    return n_matched >= trgMult


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
            jet_type="AK4PFPuppi.txt", 
            corrections_metadata=self.corrections_metadata[self.year],
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
        selev["jet1_pt"] = selev.Jet[:,0].pt
        selev["jet2_pt"] = selev.Jet[:,1].pt
        selev["jet3_pt"] = selev.Jet[:,2].pt
        selev["jet4_pt"] = selev.Jet[:,3].pt
        selev['jetAll_pt']  = ak.zip( {  "hT_trigger": selev.hT_trigger,
                                    "jet1_pt": selev.jet1_pt,
                                    "jet2_pt": selev.jet2_pt,
                                    "jet3_pt": selev.jet3_pt,
                                    "jet4_pt": selev.jet4_pt,  } )
        selev['hT_trigger_vs_jet4_pt']  = ak.zip( { "hT_trigger": selev.hT_trigger,  "jet4_pt": selev.jet4_pt } )
        
        ############### L1 #############
        L1_seed_weights = L1_seed_dict[self.year_label]
        for weight_type in L1_seed_weights:
            weight_l1 = weight_type.split('L1_')[1]
            selev[f"{weight_type}"] = ak.values_astype(selev.L1[weight_l1], "float64")

        L1_all_seed_weights = L1_all_seed_dict[self.year_label]   
        weight_l1 = L1_all_seed_weights[0].split('L1_')[1]
        L1_all_bool = selev.L1[weight_l1]
        for weight_type in L1_all_seed_weights[1:]:
            weight_l1 = weight_type.split('L1_')[1]
            L1_all_bool = L1_all_bool | selev.L1[weight_l1]
        selev["L1_all"] = ak.values_astype(L1_all_bool, "float64")
        
        ############### HLT ###############
        
        # HLT_filters_paths = HLT_filters_dict[self.year_label]
        # for trigPath in HLT_filters_paths:
        #     selev[f"pass_{trigPath}"] = ak.ones_like(selev.L1_all, dtype=bool)
        #     for t in range(len(HLT_filters_paths[trigPath])):
        #         trigFilter = HLT_filters_paths[trigPath][t]
        #         trgID, trgBit, trgMult = [int(en) for en in trigFilter.split(":")[0:3]]
        #         if t == 0:
        #             selev[f"pass_{trigPath}_{trigFilter}"] = selev[f"pass_{trigPath}"] & check_HLT_filter(selev, trgID, trgBit, trgMult)
        #         else:
        #             trigFilter_prev = HLT_filters_paths[trigPath][t-1]
        #             selev[f"pass_{trigPath}_{trigFilter}"] = selev[f"pass_{trigPath}_{trigFilter_prev}"] & check_HLT_filter(selev, trgID, trgBit, trgMult)
                
        HLT_filters_paths = HLT_filters_dict[self.year_label]
        for trigPath in HLT_filters_paths:
            trig_filter_mask = ak.ones_like(selev.L1_all, dtype=bool)
            for trigFilter in HLT_filters_paths[trigPath]:
                trgID, trgBit, trgMult = [int(en) for en in trigFilter.split(":")[0:3]]
                trig_filter_mask = trig_filter_mask & check_HLT_filter(selev, trgID, trgBit, trgMult)
                selev[f"pass_{trigPath}_{trigFilter}"] = trig_filter_mask
                
                # logging.info(selev[f"pass_{trigPath}"])

        
        hist = {}
        hist = filling_trigger_histograms(
            selev,
            processName=self.processName,
            year=self.year_label,
            isMC=self.config["isMC"],
            histCuts=self.histCuts,
            isDataForMixed=self.config['isDataForMixed'],
            event_metadata=event.metadata,
            L1_seed_weights= L1_seed_weights + ["L1_all"],
            HLT_filters_paths=HLT_filters_paths,
            )

        return hist 

    def postprocess(self, accumulator):
        return accumulator
