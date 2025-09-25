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

from src.hist import Collection, Fill
from src.hist.object import LorentzVector, Jet, Muon, Elec
#from coffea4bees.analysis.helpers.hist_templates import SvBHists, FvTHists, QuadJetHists
from coffea4bees.jet_clustering.clustering_hist_templates import ClusterHists, ClusterHistsDetailed
from coffea4bees.jet_clustering.clustering   import cluster_bs, cluster_bs_fast
from coffea4bees.jet_clustering.declustering import compute_decluster_variables, make_synthetic_event, get_list_of_splitting_types, clean_ISR, get_list_of_ISR_splittings, get_list_of_combined_jet_types, get_list_of_all_sub_splittings, get_splitting_name, get_list_of_splitting_names

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
            clustering_pdfs_file = "jet_clustering/jet-splitting-PDFs-00-07-02/clustering_pdfs_vs_pT_XXX.yml",
            run_SvB=True,
            do_declustering=False,
            subtract_ttbar_with_weights = False,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = corrections_metadata
        self.clustering_pdfs_file = clustering_pdfs_file
        self.classifier_SvB = HCREnsemble(SvB) if SvB else None
        self.classifier_SvB_MA = HCREnsemble(SvB_MA) if SvB_MA else None
        self.run_SvB = run_SvB
        self.do_declustering = do_declustering
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

        clustering_pdfs_file = self.clustering_pdfs_file.replace("XXX", year)

        if not clustering_pdfs_file == "None":
            clustering_pdfs = yaml.safe_load(open(clustering_pdfs_file, "r"))
            logging.info(f"Loaded {len(clustering_pdfs.keys())} PDFs from {clustering_pdfs_file}")
        else:
            clustering_pdfs = None


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
        event = apply_4b_selection( event, self.corrections_metadata[year],
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
        # Preselection: keep only three or four tag events
        #
        #selections.add("passPreSel", event.passPreSel)
        selections.add("passFourTag", event.fourTag)

        #event['pass0OthJets'] = event.nJet_selected == 4
        #event['pass1OthJets'] = event.nJet_selected == 5
        #event['pass2OthJets'] = event.nJet_selected == 6
        #event['passMax1OthJets'] = event.nJet_selected < 6
        #event['passMax2OthJets'] = event.nJet_selected < 7
        #event['passMax4OthJets'] = event.nJet_selected < 9
        #selections.add("pass0OthJets",    event.pass0OthJets)
        #selections.add("pass1OthJets",    event.pass1OthJets)
        #selections.add("pass2OthJets",    event.pass2OthJets)
        #selections.add("passMax1OthJets", event.passMax1OthJets)
        #selections.add("passMax2OthJets", event.passMax2OthJets)
        #selections.add("passMax4OthJets", event.passMax4OthJets)
        allcuts.append("passFourTag")

        #allcuts.append("passMax1OthJets")
        #allcuts.append("passMax2OthJets")
        #allcuts.append("passMax4OthJets")
        #allcuts.append("pass2OthJets")

        selev = event[selections.all(*allcuts)]

        ## TTbar subtractions
        if self.subtract_ttbar_with_weights:

            pass_ttbar_filter_selev = subtract_ttbar_with_SvB(selev, dataset, year)

            pass_ttbar_filter = np.full( len(event), True)
            pass_ttbar_filter[ selections.all(*allcuts) ] = pass_ttbar_filter_selev
            selections.add( 'pass_ttbar_filter', pass_ttbar_filter )
            allcuts.append("pass_ttbar_filter")
            selev = selev[pass_ttbar_filter_selev]


        # logging.info( f"\n {chunk} Event:  nSelJets {selev['nJet_selected']}\n")

        #
        # Build and select boson candidate jets with bRegCorr applied
        #
        sorted_idx = ak.argsort( selev.Jet.btagScore * selev.Jet.selected, axis=1, ascending=False )
        canJet_idx = sorted_idx[:, 0:4]
        notCanJet_idx = sorted_idx[:, 4:]

        # apply bJES to canJets
        canJet = selev.Jet[canJet_idx] * selev.Jet[canJet_idx].bRegCorr
        canJet["bRegCorr"] = selev.Jet.bRegCorr[canJet_idx]
        canJet["btagScore"] = selev.Jet.btagScore[canJet_idx]
        canJet["puId"] = selev.Jet.puId[canJet_idx]
        canJet["jetId"] = selev.Jet.puId[canJet_idx]
        if config["isMC"]:
            canJet["hadronFlavour"] = selev.Jet.hadronFlavour[canJet_idx]

        #
        # pt sort canJets
        #
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
        selev["canJet"] = canJet
        for i in range(4):
            selev[f"canJet{i}"] = selev["canJet"][:, i]


        # print(selev.v4j.n)
        # selev['Jet', 'canJet'] = False
        notCanJet = selev.Jet[notCanJet_idx]
        notCanJet = notCanJet[notCanJet.selected_loose]
        notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

        notCanJet["isSelJet"] = 1 * ( (notCanJet.pt > 40) & (np.abs(notCanJet.eta) < 2.4) )  # should have been defined as notCanJet.pt>=40, too late to fix this now...
        selev["notCanJet_coffea"] = notCanJet

        #
        # Do the Clustering
        #
        canJet["jet_flavor"] = "b"
        notCanJet["jet_flavor"] = "j"

        jets_for_clustering = ak.concatenate([canJet, notCanJet], axis=1)
        jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]

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


        #clustered_jets, clustered_splittings = cluster_bs_fast(jets_for_clustering, debug=False)
        clustered_jets, clustered_splittings = cluster_bs(jets_for_clustering, debug=False)
        compute_decluster_variables(clustered_splittings)


        #
        #  add split name (can probably do this when making the splitting
        #
        split_name_flat = [get_splitting_name(i) for i in ak.flatten(clustered_splittings.jet_flavor)]
        split_name = ak.unflatten(split_name_flat, ak.num(clustered_splittings))
        clustered_splittings["splitting_name"] = split_name


        #
        #  get all splitting types that are used (ie: not pure ISR)
        #
        clustered_jets = clean_ISR(clustered_jets, clustered_splittings)

        cleaned_combined_jet_flavors = get_list_of_combined_jet_types(clustered_jets)
        cleaned_split_jet_flavors = []
        for _s in cleaned_combined_jet_flavors:
            cleaned_split_jet_flavors += get_list_of_all_sub_splittings(_s)

        #
        # Convert to list of cleaned splitting names
        #
        cleaned_splitting_name = [get_splitting_name(i) for i in cleaned_split_jet_flavors]
        cleaned_splitting_name = set(cleaned_splitting_name)

        #
        # Sort clusterings by type
        #
        for _s_type in cleaned_splitting_name:
            selev[f"splitting_{_s_type}"]   = clustered_splittings[clustered_splittings.splitting_name == _s_type]

        #print(f'{chunk} cleaned splitting types {cleaned_split_types}\n')

        # error_type = '(bj)((jj)b)'
        # found_error = error_type in cleaned_split_types
        #
        # if found_error:
        #     print(f"ERROR have splitting type {error_type}\n" )
        #
        #     error_mask = clustered_splittings.jet_flavor == error_type
        #     event_mask = ak.any(error_mask,axis=1 )
        #
        #     # print(f'{chunk} num splitting {ak.num(selev["splitting_b(bj)"])}')
        #     # print(f'{chunk} mask {ak.num(selev["splitting_b(bj)"]) > 0}')
        #     #bbj_mask = ak.num(selev["splitting_b(bj)"]) > 0
        #     jets_for_clustering_error = jets_for_clustering[event_mask]
        #     n_jets_error = len(jets_for_clustering_error)
        #     print(f'{chunk}\n\n')
        #     print(f'{chunk} self.input_jet_pt      = {[jets_for_clustering_error[iE].pt.tolist()         for iE in range(n_jets_error)]}')
        #     print(f'{chunk} self.input_jet_eta     = {[jets_for_clustering_error[iE].eta.tolist()        for iE in range(n_jets_error)]}')
        #     print(f'{chunk} self.input_jet_phi     = {[jets_for_clustering_error[iE].phi.tolist()        for iE in range(n_jets_error)]}')
        #     print(f'{chunk} self.input_jet_mass    = {[jets_for_clustering_error[iE].mass.tolist()       for iE in range(n_jets_error)]}')
        #     print(f'{chunk} self.input_jet_flavor  = {[jets_for_clustering_error[iE].jet_flavor.tolist() for iE in range(n_jets_error)]}')
        #     print(f'{chunk}\n\n')


        # from coffea4bees.jet_clustering.dumpTestVectors   import dumpTestVectors_bbj
        # dumpTestVectors_bbj(chunk, selev, jets_for_clustering)

        #
        # writing out bb splitting for Chris Berman
        #
        # out_data = {}
        # out_data["pt_comb"]  = ak.flatten(selev["splitting_bb"].pt)
        # out_data["eta_comb"] = ak.flatten(selev["splitting_bb"].eta)
        # out_data["zA"] = ak.flatten(selev["splitting_bb"].zA)
        # out_data["thetaA"] = ak.flatten(selev["splitting_bb"].thetaA)
        # out_data["mA"] = ak.flatten(selev["splitting_bb"].mA)
        # out_data["mB"] = ak.flatten(selev["splitting_bb"].mB)
        # out_data["decay_phi"] = ak.flatten(selev["splitting_bb"].decay_phi)
        #
        # for out_k, out_v in out_data.items():
        #     processOutput[out_k] = {}
        #     processOutput[out_k][event.metadata['dataset']] = list(out_v)


        #
        #  Declustering
        #
        if self.do_declustering:

            # clustered_jets = clean_ISR(clustered_jets, clustered_splittings)

            #
            # Declustering
            #

            #
            #  Read in the pdfs
            #

            declustered_jets = make_synthetic_event(clustered_jets, clustering_pdfs)

            declustered_jets = declustered_jets[ak.argsort(declustered_jets.pt, axis=1, ascending=False)]

            is_b_mask = declustered_jets.jet_flavor == "b"
            canJet_re = declustered_jets[is_b_mask]

            canJet_re["puId"] = 7
            canJet_re["jetId"] = 7 # selev.Jet.puId[canJet_idx]


            notCanJet_re = declustered_jets[~is_b_mask]
            notCanJet_re["puId"] = 7
            notCanJet_re["jetId"] = 7 # selev.Jet.puId[canJet_idx]

            selev["canJet_re"] = canJet_re
            selev["notCanJet_coffea_re"] = notCanJet_re

            #
            #  Recluster
            #
            jets_for_clustering = ak.concatenate([canJet_re, notCanJet_re], axis=1)
            jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]

            clustered_jets_reclustered, clustered_splittings_reclustered = cluster_bs(jets_for_clustering, debug=False)
            compute_decluster_variables(clustered_splittings_reclustered)

            # all_split_types_re = get_list_of_splitting_types(clustered_splittings_reclustered)
            # # ISR_splittings_re  = get_list_of_ISR_splittings(all_split_types_re)
            # ISR_splittings_re = [] # Hack Save all splitting for now
            # all_split_types_re = [item for item in all_split_types_re if item not in ISR_splittings_re]

            for _s_type in cleaned_splitting_name:
                selev[f"splitting_{_s_type}_re"]  = clustered_splittings_reclustered[clustered_splittings_reclustered.jet_flavor == _s_type]

            # print(f'{chunk} all splitting_re types {all_split_types_re}\n')

            debug_bbj = False
            if debug_bbj:
                bbj_mask = ak.num(selev["splitting_b(bj)_re"]) > 0
                #bbj_partA = selev["splitting_b(bj)_re"][bbj_mask].part_A

                selev_bbjj = selev[bbj_mask]

                bbj_partB_large_mass = selev_bbjj["splitting_b(bj)_re"].part_B.mass > 50
                print(f'{chunk} mass {selev_bbjj["splitting_b(bj)_re"].part_B.mass}')
                print(f'{chunk} have large {bbj_partB_large_mass}')
                print(f'{chunk} any {ak.any(bbj_partB_large_mass, axis=1)}')

                large_bbj_mb_event_mask = ak.any(bbj_partB_large_mass, axis=1)

                selev_large_bbj = selev_bbjj[large_bbj_mb_event_mask]

                print(f'{chunk} partB mass {selev_large_bbj["splitting_b(bj)_re"].part_B.mass}\n')
                print(f'{chunk} partB flav {selev_large_bbj["splitting_b(bj)_re"].part_B.jet_flavor}\n')
                print(f'{chunk} partB pt {selev_large_bbj["splitting_b(bj)_re"].part_B.pt}\n')
                print(f'{chunk} partB eta {selev_large_bbj["splitting_b(bj)_re"].part_B.eta}\n')


                print(f'{chunk} partA mass {selev_large_bbj["splitting_b(bj)_re"].part_A.mass}\n')
                print(f'{chunk} partA falv {selev_large_bbj["splitting_b(bj)_re"].part_A.jet_flavor}\n')
                print(f'{chunk} partA pt {selev_large_bbj["splitting_b(bj)_re"].part_A.pt}\n')
                print(f'{chunk} partA eta {selev_large_bbj["splitting_b(bj)_re"].part_A.eta}\n')

            dumpTestVectors = False
            if dumpTestVectors:
                print(f'{chunk}\n\n')
                print(f'{chunk} self.input_jet_pt  = {[jets_for_clustering[iE].pt.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_eta  = {[jets_for_clustering[iE].eta.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_phi  = {[jets_for_clustering[iE].phi.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_mass  = {[jets_for_clustering[iE].mass.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_flavor  = {[jets_for_clustering[iE].jet_flavor.tolist() for iE in range(10)]}')
                print(f'{chunk}\n\n')




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


        for _s_type in cleaned_splitting_name:
            fill += ClusterHists( (f"splitting_{_s_type}", f"{_s_type} Splitting"), f"splitting_{_s_type}" )

            # if _s_type in ["1b0j/1b0j", "1b0j/0b1j", "0b1j/0b1j", "1b1j/1b0j"]:
            #     fill += ClusterHistsDetailed( (f"detailed_splitting_{_s_type}",    f"{_s_type} Splitting"),    f"splitting_{_s_type}"    )


        if self.do_declustering:
            for _s_type in cleaned_splitting_name:
                fill += ClusterHists( (f"splitting_{_s_type}_re", f"${_s_type} Splitting"), f"splitting_{_s_type}_re" )


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
