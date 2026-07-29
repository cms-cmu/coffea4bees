import awkward as ak
import numpy as np
import warnings
import logging

from coffea.nanoevents import NanoAODSchema
from coffea import processor
from coffea.processor import column_accumulator

from coffea4bees.analysis.helpers.event_selection import (
    apply_boosted_4b_selection,
    apply_semiresolved_4b_selection,
    apply_4b_lowpt_selection,
)
from coffea4bees.analysis.helpers.object_selection import load_object_selection_config
from src.physics.event_selection import apply_event_selection

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


class analysis(processor.ProcessorABC):
    def __init__(
            self,
            *,
            corrections_metadata: dict = None,
            object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
            **kwargs
    ):
        self.corrections_metadata = corrections_metadata
        self.sel_cfg = load_object_selection_config(object_selection_cfg) if object_selection_cfg else None

    def process(self, event):
        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        isMC    = True if event.run[0] == 1 else False
        nEvent  = len(event)

        # Baseline noise filters, lumimask, etc.
        event = apply_event_selection(event, self.corrections_metadata[year], cut_on_lumimask=False)

        # Offline selections
        event = apply_4b_lowpt_selection(
            event,
            self.corrections_metadata[year],
            sel_cfg=self.sel_cfg,
            isMC=isMC,
        )
        event = apply_boosted_4b_selection(event)
        event = apply_semiresolved_4b_selection(event)

        out_dataset = {
            'numEvents': nEvent,
            'fourTag': column_accumulator(ak.to_numpy(event.fourTag)),
            'passBoostedSel': column_accumulator(ak.to_numpy(event.passBoostedSel)),
            'passSemiResolvedSel': column_accumulator(ak.to_numpy(event.passSemiResolvedSel)),
            'lowpt_fourTag': column_accumulator(ak.to_numpy(event.lowpt_fourTag)),
        }

        if 'HLT' in event.fields:
            for path in event.HLT.fields:
                out_dataset[f"HLT_{path}"] = column_accumulator(ak.to_numpy(event.HLT[path]))

        return {dataset: out_dataset}

    def postprocess(self, accumulator):
        return accumulator
