import logging

import numpy as np
import yaml
from src.physics.objects.jet_corrections import apply_jerc_corrections_jsonpog
from src.skimmer.mc_weight_outliers import OutlierByMedian
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea4bees.analysis.helpers.object_selection import load_object_selection_config
from src.physics.event_selection import apply_event_selection

from coffea.analysis_tools import PackedSelection, Weights
from src.skimmer.picoaod import PicoAOD
from coffea4bees.analysis.helpers.cutflow import cutflow_4b


class Skimmer(PicoAOD):
    def __init__(
            self,
            loosePtForSkim=False,
            skim4b=False,
            mc_outlier_threshold=200,
            corrections_metadata=None,
            object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
            *args, **kwargs
        ):
        if skim4b:
            kwargs["pico_base_name"] = f'picoAOD_fourTag'
        super().__init__(*args, **kwargs)
        self.loosePtForSkim = loosePtForSkim
        self.skim4b = skim4b
        self.corrections_metadata = corrections_metadata if corrections_metadata is not None else {}
        self.mc_outlier_threshold = mc_outlier_threshold
        self.sel_cfg = load_object_selection_config(object_selection_cfg) if object_selection_cfg else None
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
            jets = apply_jerc_corrections_jsonpog(
                events,
                corrections_metadata=self.corrections_metadata[year],
                isMC=config["isMC"],
                run_systematics=False,
                dataset=dataset
            )
            events["Jet"] = jets

        events = apply_4b_selection(
            events,
            self.corrections_metadata[year],
            config=config,
            dataset=dataset,
            loosePtForSkim=self.loosePtForSkim,
            sel_cfg=self.sel_cfg,
        )

        weights = Weights(len(events), storeIndividual=True)

        # general event weights
        if config["isMC"]:
            weights.add("genweight_", events.genWeight)

        selections = PackedSelection()
        selections.add("lumimask", np.full(len(events), True)) #events.lumimask)
        selections.add("passNoiseFilter", np.full(len(events), True)) #events.passNoiseFilter)
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
