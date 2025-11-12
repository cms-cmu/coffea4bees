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

from src.hist_tools import Collection, Fill
from src.hist_tools.object import LorentzVector, Jet, Muon, Elec
#from coffea4bees.analysis.helpers.hist_templates import SvBHists, FvTHists, QuadJetHists

from coffea4bees.hemisphere_mixing.mixing_helpers   import transverse_thrust_awkward_fast, split_hemispheres, compute_hemi_vars
from coffea4bees.hemisphere_mixing.hemisphere_hist_templates import HemisphereHists

from coffea4bees.analysis.helpers.networks import HCREnsemble
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from src.friendtrees.FriendTreeSchema import FriendTreeSchema
from src.data_formats.root import TreeWriter, TreeReader
from src.storage.eos import EOS, PathLike
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

_ROOT = ".root"




class analysis(processor.ProcessorABC):
    def __init__(
            self,
            base_path: PathLike,
            *,
            SvB=None,
            SvB_MA=None,
            threeTag=False,
            corrections_metadata: dict = None,
            run_SvB=True,
            subtract_ttbar_with_weights = False,
            campaign: str = ...,
    ):
        logging.debug("\nInitialize  processor_make_hemi_library\n")

        self._base = EOS(base_path)

        if campaign is ...:
            campaign = f"mixing-{uuid.uuid4().hex[:8]}"
        if campaign is not None:
            logging.info(f"Using campaign name: {campaign}")
        self._campaign = campaign

        self._transform = NanoAOD(regular=False, jagged=True)


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
        year_label = self.corrections_metadata[year]['year_label']
        processName = event.metadata['processName']
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        nEvent = len(event)

        chunk = Chunk.from_coffea_events(event)
        source_chunk = {str(chunk.path): [(chunk.entry_start, chunk.entry_stop)]}
        self._hemiLib_base_name = "hemisphereLib"
        path = (
            self._base
            / f"{dataset}/{self._hemiLib_base_name}_{chunk.uuid}_{chunk.entry_start}_{chunk.entry_stop}{_ROOT}"
        )

        # check if chunks is already finished
        if self._campaign is not None:
            reader = TreeReader()
            try:
                cached = Chunk(path, fetch=True)
                metadata = reader.load_metadata(
                    self._campaign, cached, builtin_types=True
                )
                return {dataset: metadata | {"files": [cached], "source": source_chunk}}
            except Exception:
                pass


        #
        # Set process and datset dependent flags
        #
        config = processor_config(processName, dataset, event)
        logging.debug(f'{chunk_str} config={config}, for file {fname}\n')

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
            print(f'{chunk_str}\n\n')
            print(f'{chunk_str} self.input_jet_pt  = {[jets_for_clustering[iE].pt.tolist() for iE in range(10)]}')
            print(f'{chunk_str} self.input_jet_eta  = {[jets_for_clustering[iE].eta.tolist() for iE in range(10)]}')
            print(f'{chunk_str} self.input_jet_phi  = {[jets_for_clustering[iE].phi.tolist() for iE in range(10)]}')
            print(f'{chunk_str} self.input_jet_mass  = {[jets_for_clustering[iE].mass.tolist() for iE in range(10)]}')
            print(f'{chunk_str} self.input_jet_flavor  = {[jets_for_clustering[iE].jet_flavor.tolist() for iE in range(10)]}')
            print(f'{chunk_str}\n\n')


        #
        #  Get Thrust axis
        #
        thrust = transverse_thrust_awkward_fast(selev.Jet, n_steps=720, refine_rounds=2)

        #
        #  For outputs
        #
        jet_posHemi, jet_negHemi   = split_hemispheres(selev.Jet, thrust)
        muon_posHemi, muon_negHemi = split_hemispheres(selev.selMuon, thrust)
        elec_posHemi, elec_negHemi = split_hemispheres(selev.selElec, thrust)

        logging.debug("Jets pos",ak.num(jet_posHemi, axis=1))   # number of aligned jets per event
        logging.debug("Jets neg",ak.num(jet_negHemi, axis=1))      # number of anti-aligned jets per event


        #
        #  For mutltiplicity counting
        #
        tagJet_posHemi, tagJet_negHemi = split_hemispheres(selev.tagJet, thrust)
        selJet_posHemi, selJet_negHemi = split_hemispheres(selev.selJet, thrust)


        #
        #  Create hemispere objects
        #
        pos_hemi = ak.zip({"thrust_phi": thrust.phi,
                           "event": selev.event,
                           "run": selev.run,
                           "luminosityBlock" : selev.luminosityBlock,
                           "hemisphere_id": np.full(len(selev.run), +1),
                           "weight": selev.weight,
                           "nJet": ak.num(jet_posHemi, axis=1),
                           "nSelJet": ak.num(selJet_posHemi, axis=1),
                           "nTagJet": ak.num(tagJet_posHemi, axis=1),
                           "Jet": jet_posHemi,
                           "Muon": muon_posHemi,
                           "Elec": elec_posHemi
                           },
                          depth_limit=1
                          )
        pos_hemi = compute_hemi_vars(pos_hemi)

        neg_hemi = ak.zip({"thrust_phi": thrust.phi,
                           "event": selev.event,
                           "run" : selev.run,
                           "luminosityBlock" : selev.luminosityBlock,
                           "hemisphere_id": np.full(len(selev.run), -1),
                           "weight": selev.weight,
                           "nJet": ak.num(jet_negHemi, axis=1),
                           "nSelJet": ak.num(selJet_negHemi, axis=1),
                           "nTagJet": ak.num(tagJet_negHemi, axis=1),
                           "Jet": jet_negHemi,
                           "Muon": muon_negHemi,
                           "Elec": elec_negHemi
                           },
                          depth_limit=1
                          )
        neg_hemi = compute_hemi_vars(neg_hemi)

        hemis_all = ak.concatenate([pos_hemi, neg_hemi], axis=0)
        selev["pos_hemi"] = pos_hemi
        selev["neg_hemi"] = neg_hemi



        #
        #  Write out hemi library files
        #
        selev["region"] = ak.zip({"SR": selev.fourTag})

        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[selections.all(*allcuts)]

        self._cutFlow.fill("passFourTag", selev )
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


        #
        #  Make Jet Hists
        #
        skip_all_but_n = ["deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c", ]

        for iJ in range(4):
            fill += Jet.plot( (f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"], )

        fill += HemisphereHists( (f"pos_hemis", f"Hemispheres"), f"pos_hemi" )
        fill += HemisphereHists( (f"neg_hemis", f"Hemispheres"), f"neg_hemi" )


        #
        # fill histograms
        #
        fill(selev, hist)


        # construct output
        metadata = (
            {
                "total_events": len(event),
                "saved_events": len(selev),
                "saved_hemis":  len(hemis_all.thrust_phi),
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
        # print('Garbage:',garbage)

        with TreeWriter()(path) as writer:
            hemi_data = self._transform(hemis_all)
            writer.extend(hemi_data)
            if self._campaign is not None:
                writer.save_metadata(self._campaign, metadata)

            result[dataset]["files"].append(path)
        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f"{chunk_str}{nEvent/elapsed:,.0f} events/s")

        output = hist.output | processOutput | result

        return output

    def postprocess(self, accumulator):
        return accumulator
