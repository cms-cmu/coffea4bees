import logging

import numpy as np
import yaml
from coffea.analysis_tools import PackedSelection, Weights

from src.physics.objects.jet_corrections import apply_jerc_corrections_jsonpog
from src.skimmer.mc_weight_outliers import OutlierByMedian
from src.skimmer.picoaod import PicoAOD
from src.physics.event_selection import apply_event_selection

from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea4bees.analysis.helpers.event_selection import apply_boosted_4b_selection
from coffea4bees.analysis.helpers.cutflow import cutflow_4b

class Skimmer(PicoAOD):
    def __init__(
            self, 
            mc_outlier_threshold:int|None=200, 
            corrections_metadata: dict = None, 
            *args, **kwargs
        ):

        super().__init__(*args, **kwargs)
        self.corrections_metadata = corrections_metadata
        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passBoostedSel",
        ]
        self.mc_outlier_threshold = mc_outlier_threshold
        self._cutFlow = cutflow_4b()
        logging.debug(f'Initialized processor with variables: {self.__dict__}')


    def select(self, event):

        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']

        #
        # Set process and datset dependent flags
        #
        config = processor_config(processName, dataset, event)
        logging.debug(f'config={config}\n')

        event = apply_event_selection( event, self.corrections_metadata[year], cut_on_lumimask=config["cut_on_lumimask"] )

        if config["do_jet_calibration"]:
            jets = apply_jerc_corrections_jsonpog(event,
                                      corrections_metadata=self.corrections_metadata[year],
                                      isMC=config["isMC"],
                                      run_systematics=False,
                                      dataset=dataset
                                      )
            event["Jet"] = jets

        event = apply_boosted_4b_selection(event)

        weights = Weights(len(event), storeIndividual=True)

        #
        # general event weights
        #
        if config["isMC"]:
            weights.add( "genweight_", event.genWeight )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( event.passHLT if config["cut_on_HLT_decision"] else np.full(len(event), True)  ) )
        selections.add( "passBoostedSel", event.passBoostedSel)

        event["weight"] = weights.weight()

        cumulative_cuts = ["lumimask"]
        self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )

        all_cuts = ["passNoiseFilter", "passHLT", "passBoostedSel" ]

        for cut in all_cuts:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        selection = event.lumimask & event.passNoiseFilter & event.passBoostedSel

        if not config["isMC"]: selection = selection & event.passHLT

        return selection

    def preselect(self, event):
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        config = processor_config(processName, dataset, event)
        if config["isMC"] and self.mc_outlier_threshold is not None and "genWeight" in event.fields:
            return OutlierByMedian(self.mc_outlier_threshold)(event.genWeight)