# coffea4bees/analysis/processors/processor_overlap_study.py
import awkward as ak
import numpy as np
import warnings

from coffea.nanoevents import NanoAODSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection

from coffea4bees.analysis.helpers.event_selection import (
    apply_4b_selection,
    apply_boosted_4b_selection,
    apply_semiresolved_4b_selection,
    apply_4b_lowpt_selection,
)
from coffea4bees.analysis.helpers.object_selection import load_object_selection_config
from src.physics.event_selection import apply_event_selection

import logging

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

        # Event-level selection (lumi, noise, HLT)
        event = apply_event_selection(event, self.corrections_metadata[year], cut_on_lumimask=False)

        # Object-level selections — each adds fields, does not filter events.
        # apply_4b_lowpt_selection re-runs jet_selection internally so must be called last.
        event = apply_4b_selection(event, self.corrections_metadata[year], sel_cfg=self.sel_cfg)
        event = apply_boosted_4b_selection(event)
        event = apply_semiresolved_4b_selection(event)
        event = apply_4b_lowpt_selection(
            event,
            self.corrections_metadata[year],
            sel_cfg=self.sel_cfg,
            isMC=isMC,
        )

        selections = PackedSelection()
        selections.add("lumimask",         event.lumimask)
        selections.add("passNoiseFilter",  event.passNoiseFilter)
        selections.add("passHLT",          np.full(nEvent, True) if isMC else event.passHLT)
        selections.add("fourTag",          event.fourTag)
        selections.add("passBoostedSel",   event.passBoostedSel)
        selections.add("passSemiResolved", event.passSemiResolvedSel)
        selections.add("passLowPt",        event.passPreSel)   # set by apply_4b_lowpt_selection

        base = ["lumimask", "passNoiseFilter", "passHLT"]

        def count(**kwargs):
            return int(ak.sum(selections.require(**{k: True for k in base}, **kwargs)))

        output = {
            dataset: {
                'numEvents': nEvent,
                # exclusive categories
                'onlyResolved':      count(fourTag=True,  passBoostedSel=False, passSemiResolved=False, passLowPt=False),
                'onlyBoosted':       count(fourTag=False, passBoostedSel=True,  passSemiResolved=False, passLowPt=False),
                'onlySemiResolved':  count(fourTag=False, passBoostedSel=False, passSemiResolved=True,  passLowPt=False),
                'onlyLowPt':         count(fourTag=False, passBoostedSel=False, passSemiResolved=False, passLowPt=True),
                # pairwise overlaps
                'resolved_and_boosted':       count(fourTag=True,  passBoostedSel=True,  passSemiResolved=False, passLowPt=False),
                'resolved_and_semiresolved':  count(fourTag=True,  passBoostedSel=False, passSemiResolved=True,  passLowPt=False),
                'resolved_and_lowpt':         count(fourTag=True,  passBoostedSel=False, passSemiResolved=False, passLowPt=True),
                'boosted_and_semiresolved':  count(fourTag=False, passBoostedSel=True,  passSemiResolved=True,  passLowPt=False),
                'boosted_and_lowpt':         count(fourTag=False, passBoostedSel=True,  passSemiResolved=False, passLowPt=True),
                'semiresolved_and_lowpt':    count(fourTag=False, passBoostedSel=False, passSemiResolved=True,  passLowPt=True),
                # anything passing at least one selection
                'anySelection': int(ak.sum(
                    selections.require(**{k: True for k in base}) &
                    (selections.all("fourTag") | selections.all("passBoostedSel") |
                     selections.all("passSemiResolved") | selections.all("passLowPt"))
                )),
                # none
                'none': count(fourTag=False, passBoostedSel=False, passSemiResolved=False, passLowPt=False),
            }
        }

        return output

    def postprocess(self, accumulator):
        return accumulator
