import logging
import awkward as ak

import numpy as np
import yaml
from src.physics.objects.jet_corrections import apply_jerc_corrections_jsonpog
from src.skimmer.mc_weight_outliers import OutlierByMedian
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea4bees.analysis.helpers.object_selection import load_object_selection_config
from src.physics.event_selection import apply_event_selection
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel

from coffea.analysis_tools import PackedSelection, Weights
from src.skimmer.picoaod import PicoAOD
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from coffea4bees.hemisphere_mixing.mixing_helpers   import assign_mixed_subsamples, update_pseudoTagWeight_of_mixed_data

class MixedDataSplitter(PicoAOD):
    def __init__(
            self,
            skim4b=False,
            n_subsamples=16,
            mixed_subsample=0,
            corrections_metadata=None,
            apply_JCM: bool = True,
            JCM_file: str = "coffea4bees/skimmer/metadata/jetCombinatoricModel_for_mixed_splitting.txt",
            object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
            *args, **kwargs
        ):

        super().__init__(*args, **kwargs)
        logging.info(f"\nLoading JCM from file: {JCM_file} , apply_JCM = {apply_JCM}")
        self.apply_JCM = jetCombinatoricModel(JCM_file) if apply_JCM else None
        self.n_subsamples = n_subsamples
        self.mixed_subsample = mixed_subsample
        self.corrections_metadata = corrections_metadata if corrections_metadata is not None else {}
        self.sel_cfg = load_object_selection_config(object_selection_cfg) if object_selection_cfg else None
        # Always use cutflow_4b unless explicitly overridden
        self._cutFlow = cutflow_4b()
        self.histCuts = ["passPreSel"] #, "pass0OthJets", "pass1OthJets", "pass2OthJets"]

    def select(self, events):
        year    = events.metadata['year']
        dataset = events.metadata['dataset']
        processName = events.metadata['processName']
        estart  = events.metadata['entrystart']
        estop   = events.metadata['entrystop']
        chunk_str  = f'{dataset}::{estart:6d}:{estop:6d} >>> '

        logging.debug(f'Processing dataset: {dataset}, processName: {processName}, year: {year}\n')

        # Set process and datset dependent flags
        config = processor_config(processName, dataset, events)
        logging.debug(f'config={config}\n')

        events = apply_event_selection(
            events,
            self.corrections_metadata[year],
            cut_on_lumimask=config["cut_on_lumimask"]
        )

        if False and config["do_jet_calibration"]:
            jets = apply_jerc_corrections_jsonpog(events,
                                          corrections_metadata=self.corrections_metadata[year],
                                          isMC=config["isMC"],
                                          run_systematics=False,
                                          dataset=dataset
                                          )
            events["Jet"] = jets

        config["isSyntheticData"] = True if config["isRun3"] else False # HACK!!!
        events = apply_4b_selection(
            events,
            self.corrections_metadata[year],
            config=config,
            dataset=dataset,
            loosePtForSkim=False,
            sel_cfg=self.sel_cfg,
        )

        weights = Weights(len(events), storeIndividual=True)
        events["weight"] = weights.weight()

        #
        # Update pseudoTagWeight for mixed data
        #
        print(f"{chunk_str} event.pseudoTagWeight was {events.pseudoTagWeight[:10]} \n")
        update_pseudoTagWeight_of_mixed_data( events, self.apply_JCM )
        print(f"{chunk_str} event.pseudoTagWeight is now {events.pseudoTagWeight[:10]} \n")


        #
        #  Assign mixed data subsamples
        #
        assign_mixed_subsamples(events, n_subsamples=self.n_subsamples)
        events["passSubSample"] = events[f"pass_mixedSubSample_v{self.mixed_subsample}"]

        # general event weights
        if config["isMC"]:
            weights.add("genweight_", events.genWeight)

        selections = PackedSelection()
        selections.add("lumimask", events.lumimask)
        selections.add("passNoiseFilter", events.passNoiseFilter)
        selections.add("passHLT", (events.passHLT if config["cut_on_HLT_decision"] else np.full(len(events), True)))

        #if self.loosePtForSkim:
        #    selections.add('passJetMult_lowpt_forskim', events.passJetMult_lowpt_forskim)
        #    selections.add("passPreSel_lowpt_forskim", events.passPreSel_lowpt_forskim)
        #    final_selection = selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult_lowpt_forskim=True, passPreSel_lowpt_forskim=True)
        #elif self.skim4b:
        #    selections.add('passJetMult', events.passJetMult)
        #    selections.add("passPreSel", events.passPreSel)
        #    selections.add("passFourTag", events.fourTag)
        #    final_selection = selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult=True, passPreSel=True, passFourTag=True)
        #else:
        selections.add('passJetMult', events.passJetMult)
        selections.add("passPreSel", events.passPreSel)
        selections.add("passSubSample", events.passSubSample)

        final_selection = selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult=True, passPreSel=True, passSubSample=True)



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
