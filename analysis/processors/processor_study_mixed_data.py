import time
import gc
import awkward as ak
import numpy as np
import correctionlib
import yaml
import warnings
import uproot
import uuid

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from coffea.util import load
from coffea.analysis_tools import Weights, PackedSelection
from coffea4bees.analysis.helpers.processor_config import processor_config
from src.data_formats.awkward.zip import NanoAOD
from coffea4bees.analysis.helpers.candidates_selection import create_cand_jet_dijet_quadjet

from src.hist_tools import Collection, Fill
from src.hist_tools.object import LorentzVector, Jet, Muon, Elec
#from coffea4bees.analysis.helpers.hist_templates import SvBHists, FvTHists, QuadJetHists

from coffea4bees.hemisphere_mixing.mixing_helpers   import assign_mixed_subsamples, update_pseudoTagWeight_of_mixed_data
from coffea4bees.hemisphere_mixing.hemisphere_hist_templates import HemisphereHists

from coffea4bees.analysis.helpers.networks import HCREnsemble
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from src.friendtrees.FriendTreeSchema import FriendTreeSchema
from src.data_formats.root import TreeWriter, TreeReader
from src.storage.eos import EOS, PathLike
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from src.physics.objects.jet_corrections import apply_jerc_corrections
from src.physics.common import apply_btag_sf, update_events
from coffea4bees.analysis.helpers.event_weights import add_weights

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

_ROOT = ".root"




class analysis(processor.ProcessorABC):
    def __init__(
            self,
            *,
            SvB=None,
            SvB_MA=None,
            apply_JCM: bool = True,
            JCM_file: str = "output/mixeddata_cluster/jcm_for_subsampling/jetCombinatoricModel_SB_.txt",
            threeTag=False,
            corrections_metadata: dict = None,
            run_SvB=True,
            subtract_ttbar_with_weights = False,
    ):
        logging.debug("\nInitialize  processor_make_hemi_library\n")
        logging.info(f"\nLoading JCM from file: {JCM_file} , apply_JCM = {apply_JCM}")
        self.apply_JCM = jetCombinatoricModel(JCM_file) if apply_JCM else None
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
        chunk_str  = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year    = event.metadata['year']
        self.year_label = self.corrections_metadata[year]['year_label']
        processName = event.metadata['processName']
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        nEvent = len(event)

        chunk = Chunk.from_coffea_events(event)
        source_chunk = {str(chunk.path): [(chunk.entry_start, chunk.entry_stop)]}
        self._hemiLib_base_name = "hemisphereLib"


        #
        # Set process and datset dependent flags
        #
        self.config = processor_config(processName, dataset, event)
        logging.debug(f'{chunk_str} config={self.config}, for file {fname}\n')

        #
        # Reading SvB friend trees (for TTbar subtraction)
        #
        svb_path = fname.replace(fname.split("/")[-1], "")
        if self.run_SvB:
            if (self.classifier_SvB is None) | (self.classifier_SvB_MA is None):

                #SvB_file = f'{svb_path}/SvB_newSBDef.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                SvB_file = f'{svb_path}/SvB_ULHH.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                event["SvB"] = ( NanoEventsFactory.from_root( SvB_file,
                                                              entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().SvB )

                if not ak.all(event.SvB.event == event.event):
                    raise ValueError("ERROR: SvB events do not match events ttree")

                #SvB_MA_file = f'{svb_path}/SvB_MA_newSBDef.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
                SvB_MA_file = f'{svb_path}/SvB_MA_ULHH.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
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
        event = apply_event_selection( event, self.corrections_metadata[year], cut_on_lumimask=self.config["cut_on_lumimask"])


        ### target is for new friend trees
        target = Chunk.from_coffea_events(event)

        ### adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights( event, target=target,
                                                  dataset=dataset,
                                                  year_label=self.year_label,
                                                  friend_trigWeight=None,
                                                  corrections_metadata=self.corrections_metadata[year],
                                                  apply_trigWeight=True,
                                                  config=self.config,
                                                 )


        logging.debug(f"weights event {weights.weight()[:10]}")
        logging.debug(f"Weight Statistics {weights.weightStatistics}")


        #
        # Calculate and apply Jet Energy Calibration
        #
        if False and self.config["do_jet_calibration"]:
            jets = apply_jerc_corrections(event,
                                          corrections_metadata=self.corrections_metadata[year],
                                          isMC=self.config["isMC"],
                                          run_systematics=False,
                                          dataset=dataset
                                          )
        else:
            jets = event.Jet

        event = update_events(event, {"Jet": jets})


        # Apply object selection (function does not remove events, adds content to objects)
        self.config["isSyntheticData"] = 'True' # HACK!!!
        event = apply_4b_selection( event,
                                    self.corrections_metadata[year],
                                    config=self.config,
                                    dataset=dataset,
                                   )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( event.passHLT if self.config["cut_on_HLT_decision"] else np.full(len(event), True)  ) )
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
            'genWeights': np.sum(event.genWeight) if self.config["isMC"] else nEvent

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

        #
        # Add pseudotag weights for the mixed data
        #   These are fourTag events, but the weight should be calculed assuming with one less btagged jet
        #
        print(f"{chunk_str} event.pseudoTagWeight was {event.pseudoTagWeight[:10]} \n")
        update_pseudoTagWeight_of_mixed_data( event, self.apply_JCM )
        print(f"{chunk_str} event.pseudoTagWeight is now {event.pseudoTagWeight[:10]} \n")

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




        #
        #  Build di-jets and Quad-jets
        #
        selev = create_cand_jet_dijet_quadjet(
            selev,
            isRun3=self.config["isRun3"],
            )


        #
        #  Write out hemi library files
        #
        # selev["region"] = ak.zip({"SR": selev.fourTag})

        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[selections.all(*allcuts)]

        self._cutFlow.fill("passFourTag", selev )


        selev["thrustDeltaPhi"] = (selev.posHemiOld.thrust_phi - selev.newThrustPhi) % (2 * np.pi) - np.pi
        match_dist = ak.concatenate( [selev.posHemiNew.match_dist[:, np.newaxis], selev.negHemiNew.match_dist[:, np.newaxis]], axis=1 )
        hemis = ak.concatenate( [selev.posHemiNew[:, np.newaxis], selev.negHemiNew[:, np.newaxis]], axis=1 )

        selev["hemiMatchDist"] = match_dist

        selev["nSelJetOld"] = (selev.posHemiNew.nSelJet + selev.negHemiNew.nSelJet)

        #
        # Check how ofter then hemispheres from the same event are matched
        #
        same_event = (selev.posHemiNew.event == selev.negHemiNew.event)
        nSame = ak.sum(same_event)
        nSameSB = ak.sum(same_event & selev.region.SB)
        nSameSR = ak.sum(same_event & selev.region.SR)
        if ak.any(same_event):
            print(f"Found same event hemispheres! {nSame}, SB {nSameSB}, SR {nSameSR} out of {len(selev)}" )
            print(" SR hemi event  ",hemis.event[same_event].tolist(),"\n")
            print(" SR hemi run  ",  hemis.run[same_event].tolist(),"\n")
            print(" SR hemi luminosityBlock  ",  hemis.luminosityBlock[same_event].tolist(),"\n")
            print(" SR hemi hID  ",  hemis.hemisphereId[same_event].tolist(),"\n")
            #print(" SR hemi run   ", hemis.event[selev.region.SR])
            #print(" SR hemi run   ", hemis.run[selev.region.SR])
            #print(" SR hemi luminosityBlock   ", hemis.luminosityBlock[selev.region.SR])
            #print(" SR hemi hID   ", hemis.hemisphereId[selev.region.SR])


        #
        # Study the subsampling
        #
        n_subsamples = 16
        assign_mixed_subsamples(selev, n_subsamples=n_subsamples)

        sub_sample_counts = {}
        for i in range(n_subsamples):
            self._cutFlow.fill(f"pass_mixedSubSample_v{i}", selev[selev[f"pass_mixedSubSample_v{i}"]] )
            for j in range(i,n_subsamples):
                sub_sample_counts[f"{i}_{j}"] = ak.sum( selev[f"pass_mixedSubSample_v{i}"] & selev[f"pass_mixedSubSample_v{j}"] )

        self._cutFlow.addOutput(processOutput, event.metadata["dataset"])

        #
        # Hists
        #

        fill = Fill(process=processName, year=year, weight="weight")

        hist = Collection( process=[processName],
                           year=[year],
                           tag=["threeTag", "fourTag"],  # 3 / 4/ Other
                           region=['SR','SB'],  # SR / SB / Other
                           **dict((s, ...) for s in self.histCuts)
                           )

        #
        # Jets
        #
        fill += Jet.plot(("selJets", "Selected Jets"),        "selJet",           skip=["deepjet_c"])

        fill += Jet.plot(("selJets_v0", "Selected Jets"), "selJet", weight="pass_mixedSubSample_v0", skip=["deepjet_c"])
        fill += Jet.plot(("selJets_JCM","Selected Jets"), "selJet", weight="pseudoTagWeight",        skip=["deepjet_c"])


        #
        #  Make Jet Hists
        #
        skip_all_but_n = ["deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c", ]
        fill += Jet.plot(("selJets_noJCM", "Selected Jets"), "selJet", skip=skip_all_but_n)
        fill += Jet.plot(("tagJets_noJCM", "Tag Jets"),      "tagJet", skip=skip_all_but_n)

        #for iJ in range(4):
        #    fill += Jet.plot( (f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"], )


        fill += hist.add("thrustDeltaPhi", (50, -np.pi, np.pi, ("thrustDeltaPhi", 'Thrust Delta Phi')))
        fill += hist.add("matchDist",     (100, 0, 4, ("hemiMatchDist", 'Match distance')))
        fill += hist.add("selJets.nOld", (20, 0, 20, ("nSelJetOld", 'Number of selected jets before mixing')))

        #fill += HemisphereHists( (f"pos_hemis", f"Hemispheres"), f"pos_hemi" )
        #fill += HemisphereHists( (f"neg_hemis", f"Hemispheres"), f"neg_hemi" )

        #fill += HemisphereHists( (f"pos_hemis", f"Hemispheres"), f"pos_hemi" )

        # print("Event fields",   selev.fields,"\n")
        # print("Hemi old fields",selev.posHemiOld.fields,"\n")
        # print("Hemi new fields",selev.posHemiNew.fields,"\n")




        #
        # fill histograms
        #
        fill(selev, hist)


        # construct output
        metadata = (
            {
                "total_events": len(event),
                "saved_events": len(selev),
                "saved_events_SB": ak.sum(selev.region.SB),
                "saved_events_SR": ak.sum(selev.region.SR),
                "same_hemis":  nSame,
                "same_hemis_SB":  nSameSB,
                "same_hemis_SR":  nSameSR,
                "hemi_pair_SB_event":hemis.event[selev.region.SB].tolist(),
                "hemi_pair_SB_run":  hemis.run[selev.region.SB].tolist(),
                "hemi_pair_SB_hemisphereId":  hemis.hemisphereId[selev.region.SB].tolist(),
                "hemi_pair_SR_event":hemis.event[selev.region.SR].tolist(),
                "hemi_pair_SR_run":  hemis.run[selev.region.SR].tolist(),
                "hemi_pair_SR_hemisphereId":  hemis.hemisphereId[selev.region.SR].tolist(),
                "subsample_counts": sub_sample_counts,
                #"saved_hemis":  len(pos_hemi.thrust_phi) + len(neg_hemi.thrust_phi),
            }
        )

        result = {
            dataset: metadata
            | {
                "files": [],
                "source": source_chunk,
            }
        }


        garbage = gc.collect()


        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f"{chunk_str}{nEvent/elapsed:,.0f} events/s")

        output = hist.output | processOutput | result

        return output

    def postprocess(self, accumulator):
        return accumulator
