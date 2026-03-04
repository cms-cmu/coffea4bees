import time
import awkward as ak
import numpy as np
import yaml
import warnings

from coffea.nanoevents import NanoAODSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection

from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from coffea4bees.analysis.helpers.event_selection import apply_boosted_4b_selection, apply_4b_selection
from coffea4bees.analysis.helpers.object_selection import load_object_selection_config
from src.physics.event_selection import apply_event_selection

import logging

#
# Setup
#
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

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = corrections_metadata
        self.sel_cfg = load_object_selection_config(object_selection_cfg) if object_selection_cfg else None




    def process(self, event):

        tstart = time.time()
        fname   = event.metadata['filename']
        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        isMC    = True if event.run[0] == 1 else False
        nEvent = len(event)

        logging.debug(fname)
        logging.debug(f'Process {nEvent} Events')

        #
        # Event selection
        #
        event = apply_event_selection( event, self.corrections_metadata[year], cut_on_lumimask=False)

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_4b_selection( event, self.corrections_metadata[year], sel_cfg=self.sel_cfg )
        event = apply_boosted_4b_selection(event)

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        selections.add( 'passJetMult', event.passJetMult )
        selections.add( "passPreSel", event.passPreSel )
        selections.add( "passFourTag", ( event.passJetMult & event.passPreSel & event.fourTag) )
        selections.add( 'passBoostedSel', event.passBoostedSel )
        allcuts = [ 'passJetMult' ]

        #
        #  Cut Flows
        #
        processOutput = {}
        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = {
            'numEvents': nEvent,
            'boostedAndResolved': ak.sum(selections.require( passFourTag=True, passBoostedSel=True )),
            'onlyboosted': ak.sum(selections.require( passFourTag=False, passBoostedSel=True)),
            'onlyresolved': ak.sum(selections.require( passFourTag=True, passBoostedSel=False)),
            'none': ak.sum(selections.require( passFourTag=False, passBoostedSel=False)),
            #'boostedAndResolved': ak.sum(selections.require( passPreSel=True, passJetMult=True, passBoostedSel=True )),
            #'onlyboosted': ak.sum(selections.require( passPreSel=False, passJetMult=False, passBoostedSel=True)),
            #'onlyresolved': ak.sum(selections.require( passPreSel=True, passJetMult=True, passBoostedSel=False)),
            #'none': ak.sum(selections.require( passPreSel=False, passJetMult=False, passBoostedSel=False)),
        }

        boosted_events = event[ selections.require( passBoostedSel=True ) ]
        processOutput['boosted'] = {}
        processOutput['boosted'][event.metadata['dataset']] = {
            'run': boosted_events.run.to_list(),
            'event': boosted_events.event.to_list(),
            'lumi': boosted_events.luminosityBlock.to_list(),
        }

        event['weight'] = np.ones(len(event))
        self._cutFlow = cutflow_4b()
        self._cutFlow.fill( "all", event[selections.require(lumimask=True)], allTag=True )
        self._cutFlow.fill( "passNoiseFilter", event[selections.require(lumimask=True, passNoiseFilter=True)], allTag=True, )
        self._cutFlow.fill( "passHLT", event[ selections.require( lumimask=True, passNoiseFilter=True, passHLT=True ) ], allTag=True, )
        self._cutFlow.fill( "passJetMult", event[ selections.all(*allcuts)], allTag=True )
        allcuts.append("passPreSel")
        self._cutFlow.fill( "passPreSel", event[ selections.all(*allcuts)], allTag=True )


        #
        #  Adding infor for preliminary Boosted Synthetic data study
        #
        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_info_for_Boosted_Synthetic
        # add_debug_info_for_Boosted_Synthetic(boosted_events, processOutput)

        # print("\n")
        # print("Event fields", event.fields, "\n")
        # print("SubJet field", event.FatJet.subjets.fields, "\n")


        #
        # Preselection: keep only three or four tag events
        #

        self._cutFlow.addOutput(processOutput, event.metadata["dataset"])

        output = processOutput

        return output

    def postprocess(self, accumulator):
        return accumulator
