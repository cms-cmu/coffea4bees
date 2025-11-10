import time
import gc
import awkward as ak
import numpy as np
import correctionlib
import yaml
import warnings
import uproot


from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from coffea.util import load
from coffea.analysis_tools import Weights, PackedSelection
from coffea4bees.analysis.helpers.processor_config import processor_config

from src.hist_tools import Collection, Fill
from src.hist_tools.object import LorentzVector, Jet, Muon, Elec
#from coffea4bees.analysis.helpers.hist_templates import SvBHists, FvTHists, QuadJetHists

from coffea4bees.hemisphere_mixing.mixing_helpers   import transverse_thrust_awkward_fast

from coffea4bees.analysis.helpers.networks import HCREnsemble
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from src.friendtrees.FriendTreeSchema import FriendTreeSchema


from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from src.physics.objects.jet_corrections import apply_jerc_corrections
from src.physics.common import apply_btag_sf, update_events
from src.physics.event_weights import add_weights

from coffea4bees.analysis.helpers.SvB_helpers import setSvBVars, subtract_ttbar_with_SvB
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from src.physics.event_selection import apply_event_selection

import logging

from src.data_formats.root import TreeReader, Chunk

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")





class analysis(processor.ProcessorABC):
    def __init__(
            self,
            *,
            SvB=None,
            SvB_MA=None,
            threeTag=False,
            corrections_metadata: dict = None,
            run_SvB=True,
            subtract_ttbar_with_weights = False,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = corrections_metadata
        self.classifier_SvB = HCREnsemble(SvB) if SvB else None
        self.classifier_SvB_MA = HCREnsemble(SvB_MA) if SvB_MA else None
        self.run_SvB = run_SvB
        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights


        self.histCuts = ["passPreSel"] #, "pass0OthJets", "pass1OthJets", "pass2OthJets"]


    def process(self, event):

        tstart = time.time()
        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year    = event.metadata['year']
        year_label = self.corrections_metadata[year]['year_label']
        processName = event.metadata['processName']
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        nEvent = len(event)




        #
        # Set process and datset dependent flags
        #
        config = processor_config(processName, dataset, event)
        logging.debug(f'{chunk} config={config}, for file {fname}\n')

        #
        # Reading SvB friend trees (for TTbar subtraction)
        #
        path = fname.replace(fname.split("/")[-1], "")
        if self.run_SvB:
            if (self.classifier_SvB is None) | (self.classifier_SvB_MA is None):

                #SvB_file = f'{path}/SvB_newSBDef.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                SvB_file = f'{path}/SvB_ULHH.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                event["SvB"] = ( NanoEventsFactory.from_root( SvB_file,
                                                              entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().SvB )

                if not ak.all(event.SvB.event == event.event):
                    raise ValueError("ERROR: SvB events do not match events ttree")

                #SvB_MA_file = f'{path}/SvB_MA_newSBDef.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
                SvB_MA_file = f'{path}/SvB_MA_ULHH.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
                event["SvB_MA"] = ( NanoEventsFactory.from_root( SvB_MA_file,
                                                                 entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema ).events().SvB_MA )

                if not ak.all(event.SvB_MA.event == event.event):
                    raise ValueError("ERROR: SvB_MA events do not match events ttree")

                # defining SvB for different SR
                setSvBVars("SvB", event)
                setSvBVars("SvB_MA", event)


        #
        # Event selection
        #
        event = apply_event_selection( event, self.corrections_metadata[year], cut_on_lumimask=config["cut_on_lumimask"])


        ### target is for new friend trees
        target = Chunk.from_coffea_events(event)

        ### adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights( event, target=target,
                                                  do_MC_weights=config["do_MC_weights"],
                                                  dataset=dataset,
                                                  year_label=year_label,
                                                  friend_trigWeight=None,
                                                  corrections_metadata=self.corrections_metadata[year],
                                                  apply_trigWeight=True,
                                                  isTTForMixed=config["isTTForMixed"]
                                                 )


        logging.debug(f"weights event {weights.weight()[:10]}")
        logging.debug(f"Weight Statistics {weights.weightStatistics}")


        #
        # Calculate and apply Jet Energy Calibration
        #
        if config["do_jet_calibration"]:
            jets = apply_jerc_corrections(event,
                                          corrections_metadata=self.corrections_metadata[year],
                                          isMC=config["isMC"],
                                          run_systematics=False,
                                          dataset=dataset
                                          )
        else:
            jets = event.Jet

        event = update_events(event, {"Jet": jets})


        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_4b_selection( event,
                                    self.corrections_metadata[year],
                                    dataset=dataset,
                                    doLeptonRemoval=config["do_lepton_jet_cleaning"],
                                    override_selected_with_flavor_bit=config["override_selected_with_flavor_bit"],
                                    do_jet_veto_maps=config["do_jet_veto_maps"],
                                    isRun3=config["isRun3"],
                                    isMC=config["isMC"], ### temporary
                                    isSyntheticData=config["isSyntheticData"],
                                   )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( event.passHLT if config["cut_on_HLT_decision"] else np.full(len(event), True)  ) )
        selections.add( 'passJetMult', event.passJetMult )
        allcuts = [ 'lumimask', 'passNoiseFilter', 'passHLT', 'passJetMult' ]
        event['weight'] = weights.weight()   ### this is for _cutflow

        #
        #  Cut Flows
        #
        processOutput = {}

        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = {
            'nEvent' : nEvent,
            'genWeights': np.sum(event.genWeight) if config["isMC"] else nEvent

        }

        self._cutFlow = cutflow_4b()
        self._cutFlow.fill( "all", event[selections.require(lumimask=True)], allTag=True)
        self._cutFlow.fill( "passNoiseFilter", event[selections.require(lumimask=True, passNoiseFilter=True)], allTag=True)
        self._cutFlow.fill( "passHLT", event[ selections.require( lumimask=True, passNoiseFilter=True, passHLT=True ) ], allTag=True, )
        self._cutFlow.fill( "passJetMult", event[ selections.all(*allcuts)], allTag=True )


        #
        # Preselection: keep only four tag events
        #
        selections.add("passFourTag", event.fourTag)

        allcuts.append("passFourTag")
        selev = event[selections.all(*allcuts)]

        #
        # TTbar subtractions
        #
        if self.subtract_ttbar_with_weights:

            pass_ttbar_filter_selev = subtract_ttbar_with_SvB(selev, dataset, year)

            pass_ttbar_filter = np.full( len(event), True)
            pass_ttbar_filter[ selections.all(*allcuts) ] = pass_ttbar_filter_selev
            selections.add( 'pass_ttbar_filter', pass_ttbar_filter )
            allcuts.append("pass_ttbar_filter")
            selev = selev[pass_ttbar_filter_selev]


        # logging.info( f"\n {chunk} Event:  nSelJets {selev['nJet_selected']}\n")

        #
        #  To dump the testvectors
        #
        dumpTestVectors = False
        if dumpTestVectors:
            print(f'{chunk}\n\n')
            print(f'{chunk} self.input_jet_pt  = {[jets_for_clustering[iE].pt.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_eta  = {[jets_for_clustering[iE].eta.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_phi  = {[jets_for_clustering[iE].phi.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_mass  = {[jets_for_clustering[iE].mass.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_flavor  = {[jets_for_clustering[iE].jet_flavor.tolist() for iE in range(10)]}')
            print(f'{chunk}\n\n')


        #
        #  Get Thrust axis
        #
        res = transverse_thrust_awkward_fast(selev.Jet, n_steps=720, refine_rounds=2)

        selev["region"] = ak.zip({"SR": selev.fourTag})

        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[selections.all(*allcuts)]

        self._cutFlow.fill("passFourTag", selev )
        #self._cutFlow.fill("pass0OthJets",selev )
        #self._cutFlow.fill("pass1OthJets",selev )
        #self._cutFlow.fill("pass2OthJets",selev )

        self._cutFlow.addOutput(processOutput, event.metadata["dataset"])

        #
        # Hists
        #

        fill = Fill(process=processName, year=year, weight="weight")

        hist = Collection( process=[processName],
                           year=[year],
                           tag=["threeTag", "fourTag"],  # 3 / 4/ Other
                           region=['SR'],  # SR / SB / Other
                           **dict((s, ...) for s in self.histCuts)
                           )

        #
        # Jets
        #
        fill += Jet.plot(("selJets", "Selected Jets"),        "selJet",           skip=["deepjet_c"])
        # fill += Jet.plot(("canJets", "Higgs Candidate Jets"), "canJet",           skip=["deepjet_c"])
        # fill += Jet.plot(("othJets", "Other Jets"),           "notCanJet_coffea", skip=["deepjet_c"])
        # fill += Jet.plot(("tagJets", "Tag Jets"),             "tagJet",           skip=["deepjet_c"])

        # fill += Jet.plot(("notCanJet_sel", "Higgs Candidate Jets"), "notCanJet_sel",           skip=["deepjet_c"])
        # if self.do_declustering:
        #     fill += Jet.plot(("canJets_re", "Higgs Candidate Jets"), "canJet_re",           skip=["deepjet_c"])
        #     fill += Jet.plot(("notCanJet_sel_re", "Higgs Candidate Jets"), "notCanJet_sel_re",           skip=["deepjet_c"])

        #
        #  Make Jet Hists
        #
        skip_all_but_n = ["deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c", ]

        for iJ in range(4):
            fill += Jet.plot( (f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"], )


        #
        # fill histograms
        #
        # fill.cache(selev)
        fill(selev, hist)

        garbage = gc.collect()
        # print('Garbage:',garbage)


        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f"{chunk}{nEvent/elapsed:,.0f} events/s")

        output = hist.output | processOutput

        return output

    def postprocess(self, accumulator):
        return accumulator
