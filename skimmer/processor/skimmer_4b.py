import logging

import numpy as np
from src.physics.objects.jet_corrections import apply_jerc_corrections_jsonpog
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from src.physics.event_selection import apply_event_selection

from coffea.analysis_tools import PackedSelection, Weights
from coffea4bees.skimmer.processor.skimmer_4b_base import Skimmer4b


class Skimmer(Skimmer4b):
    def __init__(
            self,
            loosePtForSkim=False,
            skim4b=False,
            split_tag_categories=False,
            mc_outlier_threshold=200,
            corrections_metadata=None,
            object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
            *args, **kwargs
        ):
        if skim4b:
            kwargs["pico_base_name"] = f'picoAOD_fourTag'
        super().__init__(
            mc_outlier_threshold=mc_outlier_threshold,
            corrections_metadata=corrections_metadata,
            object_selection_cfg=object_selection_cfg,
            *args, **kwargs,
        )
        self.loosePtForSkim = loosePtForSkim
        self.skim4b = skim4b
        self.split_tag_categories = split_tag_categories

    def select(self, events):
        m = self._parse_event_metadata(events)
        year, dataset, processName, config = m.year, m.dataset, m.processName, m.config
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

        events["weight"] = weights.weight()

        if self.split_tag_categories:
            selections.add('passJetMult', events.passJetMult)
            selections.add("passPreSel", events.passPreSel)
            selections.add("passFourTag", events.fourTag)
            selections.add("passThreeTag", events.threeTag)

            sel_4b = selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult=True, passPreSel=True, passFourTag=True)
            sel_3b = selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult=True, passPreSel=True, passThreeTag=True)

            self._cutFlow.fill("all", events, allTag=True)
            for cut in ["passNoiseFilter", "passHLT", "passJetMult", "passPreSel", "passFourTag"]:
                self._cutFlow.fill(cut, events[sel_4b], allTag=True)

            return {
                "fourTag": (sel_4b, None, {}),
                "threeTag": (sel_3b, None, {}),
            }

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

        self._cutFlow.fill("all", events, allTag=True)
        cumulative_cuts = []
        for cut in selections.names:
            cumulative_cuts.append(cut)
            self._cutFlow.fill(cut, events[selections.all(*cumulative_cuts)], allTag=True)

        processOutput = {}
        return final_selection, None, processOutput
