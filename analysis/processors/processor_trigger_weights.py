import logging
import warnings
import awkward as ak
import yaml
import numpy as np
from src.physics.objects.jet_corrections import apply_jerc_corrections_jsonpog
from coffea4bees.analysis.trigger_emulator.TrigEmulatorTool   import TrigEmulatorTool
from coffea4bees.analysis.trigger_emulator.TriggerSFVectorized import TriggerSFVectorized
from src.physics.event_selection import apply_event_selection
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea4bees.analysis.helpers.candidates_selection import create_cand_jet_dijet_quadjet
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema
from coffea4bees.analysis.helpers.dump_friendtrees import dump_trigger_weight
from coffea4bees.analysis.helpers.processor_config import processor_config

#
#  Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")

class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
        make_classifier_input: str = None,
        corrections_metadata: str ="src/physics/corrections.yml",
        use_vectorized: bool = False,
        tagger: str = "DeepJet",
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = corrections_metadata
        self.make_classifier_input = make_classifier_input
        self.use_vectorized = use_vectorized
        self.tagger = tagger
        self.trig_sfs_vect = {}

        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passJetMult",
            "passJetMult_btagSF",
        ]


    def process(self, event):

        self.dataset = event.metadata['dataset']
        self.year    = event.metadata['year']
        self.processName = event.metadata['processName']

        #
        # Set process and datset dependent flags
        #
        self.config = processor_config(self.processName, self.dataset, event)
        logging.debug(f' config={self.config} \n')

        self.nEvent = len(event)

        #
        # Event selection
        #
        event = apply_event_selection(
            event,
            self.corrections_metadata[self.year],
            cut_on_lumimask=self.config["cut_on_lumimask"]
        )

        #
        # Calculate and apply Jet Energy Calibration
        #
        jets = apply_jerc_corrections_jsonpog(
            event,
            corrections_metadata=self.corrections_metadata[self.year],
            isMC=self.config["isMC"],
            run_systematics=False,
            dataset=self.dataset
        )
        event["Jet"] = jets

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_4b_selection(
            event,
            self.corrections_metadata[self.year],
            config=self.config,
            dataset=self.dataset,
        )

        event = create_cand_jet_dijet_quadjet(
            event,
            apply_FvT=False,
            run_SvB=False,
            run_systematics=False,
            classifier_SvB=None,
            classifier_SvB_MA=None,
        )

        year_label = self.corrections_metadata[self.year]['year_label'].replace("UL", "20").split("_")[0]
        event['trigWeight'] = {}

        if self.use_vectorized:
            year_int = int(year_label)
            if year_int not in self.trig_sfs_vect:
                self.trig_sfs_vect[year_int] = TriggerSFVectorized(year_int, map_path="coffea4bees/analysis/trigger_emulator/data/",tagger=self.tagger)

            trig_helper = self.trig_sfs_vect[year_int]
            data_eff, mc_eff, sf = trig_helper.calculate_event_sf(event)
            event['trigWeight', "Data"] = data_eff
            event['trigWeight', "MC"] = mc_eff

        else:
            emulator_data = TrigEmulatorTool("Test", year=year_label)
            emulator_mc   = TrigEmulatorTool("Test", year=year_label, useMCTurnOns=True)
            event['trigWeight', "Data"] = ak.Array([ emulator_data.GetWeightOR(selJet_pt, tagJet_pt, hT_trigger) for selJet_pt, tagJet_pt, hT_trigger in zip(event.selJet.pt, event.canJet.pt, event.hT_trigger) ])
            event['trigWeight', 'MC' ] = ak.Array([ emulator_mc.GetWeightOR(selJet_pt, tagJet_pt, hT_trigger) for selJet_pt, tagJet_pt, hT_trigger in zip(event.selJet.pt, event.canJet.pt, event.hT_trigger) ])

        logging.debug(f"trigger weight data: {event['trigWeight'].Data}")
        logging.debug(f"trigger weight mc: {event['trigWeight'].MC}")

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        allcuts = [ 'lumimask', 'passNoiseFilter' ]

        friends = {}

        friends["friends"] = dump_trigger_weight(
            event,
            self.make_classifier_input,
            "trigWeight",
            selections.all(*allcuts)
        )

        return friends

    def postprocess(self, accumulator):
        return accumulator
