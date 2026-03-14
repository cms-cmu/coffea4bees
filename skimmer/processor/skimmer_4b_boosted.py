import logging

import numpy as np
from coffea.analysis_tools import PackedSelection, Weights

from src.physics.objects.jet_corrections import apply_jerc_corrections_jsonpog
from src.physics.event_selection import apply_event_selection

from coffea4bees.analysis.helpers.event_selection import apply_boosted_4b_selection
from coffea4bees.skimmer.processor.skimmer_4b_base import Skimmer4b


class Skimmer(Skimmer4b):
    def __init__(
            self,
            mc_outlier_threshold: int | None = 200,
            corrections_metadata: dict = None,
            *args, **kwargs
        ):
        super().__init__(
            mc_outlier_threshold=mc_outlier_threshold,
            corrections_metadata=corrections_metadata,
            object_selection_cfg=None,
            *args, **kwargs,
        )
        logging.debug(f'Initialized processor with variables: {self.__dict__}')


    def select(self, event):
        m = self._parse_event_metadata(event)
        year, dataset, processName, config = m.year, m.dataset, m.processName, m.config
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