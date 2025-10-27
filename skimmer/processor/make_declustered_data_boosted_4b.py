import yaml
from src.skimmer.picoaod import PicoAOD, fetch_metadata, resize
from coffea.nanoevents import NanoEventsFactory
from collections import OrderedDict
from coffea4bees.analysis.helpers.cutflow import cutFlow

from coffea4bees.jet_clustering.declustering import make_synthetic_event
from coffea4bees.analysis.helpers.cutflow import cutflow_4b

from src.math_tools.random import Squares
from src.physics.event_selection import apply_event_selection

from src.data_formats.root import Chunk, TreeReader
from coffea4bees.analysis.helpers.load_friend import (
    FriendTemplate,
    parse_friends
)

from coffea.analysis_tools import Weights, PackedSelection
import numpy as np
from src.physics.objects.jet_corrections import apply_jerc_corrections
from src.physics.common import update_events
from copy import copy
import logging
import awkward as ak
import uproot
from coffea.nanoevents.methods import vector

class DeClustererBoosted(PicoAOD):
    def __init__(self, clustering_pdfs_file = "None",
                declustering_rand_seed=5,
                friends: dict[str, str|FriendTemplate] = None,
                corrections_metadata: dict = None,
                *args, **kwargs):
        kwargs["pico_base_name"] = f'picoAOD_seed{declustering_rand_seed}'
        super().__init__(*args, **kwargs)

        logging.info(f"\nRunning Declusterer with these parameters: clustering_pdfs_file = {clustering_pdfs_file}, declustering_rand_seed = {declustering_rand_seed}, args = {args}, kwargs = {kwargs}")
        self.clustering_pdfs_file = clustering_pdfs_file

        self.friends = parse_friends(friends)
        self.declustering_rand_seed = declustering_rand_seed
        self.corrections_metadata = corrections_metadata

        self.skip_collections = kwargs["skip_collections"]
        self.skip_branches    = kwargs["skip_branches"]
        self.cutFlow = cutflow_4b()


    def select(self, event):

        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        fname   = event.metadata['filename']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        nEvent = len(event)
        year_label = self.corrections_metadata[year]['year_label']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        processName = event.metadata['processName']
        isMC    = True if event.run[0] == 1 else False

        clustering_pdfs_file = self.clustering_pdfs_file.replace("XXX", year)

        print(f"clustering_pdfs_file is {clustering_pdfs_file}\n")
        if not clustering_pdfs_file == "None":
            clustering_pdfs = yaml.safe_load(open(clustering_pdfs_file, "r"))
            logging.info(f"Loaded {len(clustering_pdfs.keys())} PDFs from {clustering_pdfs_file}\n")
        else:
            clustering_pdfs = None

        #
        # Event Selection
        #
        event = apply_event_selection( event, self.corrections_metadata[year],
                                       cut_on_lumimask = (not isMC),
                                      )

        selFatJet = event.FatJet
        selFatJet = selFatJet[selFatJet.particleNetMD_Xbb > 0.8]
        selFatJet = selFatJet[selFatJet.subJetIdx1 >= 0]
        selFatJet = selFatJet[selFatJet.subJetIdx2 >= 0]

        selFatJet = selFatJet[(selFatJet.subjets [:, :, 0] + selFatJet.subjets [:, :, 1]).pt > 300]
        selFatJet = selFatJet[(selFatJet.subjets [:, :, 0] + selFatJet.subjets [:, :, 1]).mass > 40]


        event["selFatJet"] = selFatJet


        #  Cehck How often do we have >=2 Fat Jets?
        event["passNFatJets"]  = (ak.num(event.selFatJet) == 2)



        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        selections.add( "passNFatJets",  event.passNFatJets )

        event["weight"] = 1.0



        #
        # Do the cutflow
        #
        sel_dict = OrderedDict({
            'all'               : selections.require(lumimask=True),
            'passNoiseFilter'   : selections.require(lumimask=True, passNoiseFilter=True),
            'passHLT'           : selections.require(lumimask=True, passNoiseFilter=True, passHLT=True),
            'passNFatJets'      : selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passNFatJets=True),
        })
        #sel_dict['passJetMult'] = selections.all(*allcuts)

        for cut, sel in sel_dict.items():
            self.cutFlow.fill( cut, event[sel], allTag=True )


        list_of_cuts = [ "lumimask", "passNoiseFilter", "passHLT", "passNFatJets" ]
        #list_of_cuts = [ "passNFatJets" ]
        analysis_selections = selections.all(*list_of_cuts)
        selev = event[analysis_selections]
        selection = selections.all(*list_of_cuts)



        #
        # Adding btag and jet flavor to fat jets
        #
        subjetIdx1_flat = ak.flatten(selev.selFatJet.subJetIdx1)
        subjetIdx2_flat = ak.flatten(selev.selFatJet.subJetIdx2)
        # print(f"subjetIdx1_flat: {subjetIdx1_flat}\n")
        # print(f"subjetIdx2_flat: {subjetIdx2_flat}\n")

        particleNet_HbbvsQCD_flat = ak.flatten(selev.selFatJet.particleNet_HbbvsQCD)
        # particleNet_HbbvsQCD_flat_str = [ f"({round(v,3)},{round(v,3)})" for v in particleNet_HbbvsQCD_flat ]
        # selev["selFatJet", "btag_string"] = ak.unflatten(particleNet_HbbvsQCD_flat_str, ak.num(selev.selFatJet))


        indices_str_flat = []
        for subJetIdxes in zip(subjetIdx1_flat, subjetIdx2_flat):
            indices_str_flat.append( f"({subJetIdxes[0]},{subJetIdxes[1]})" )


        indices_str = ak.unflatten(indices_str_flat, ak.num(selev.selFatJet))
        selev["selFatJet", "btag_string"] = indices_str

        fatjet_flavor_flat = np.array(["bb"] * len(particleNet_HbbvsQCD_flat))
        selev["selFatJet", "jet_flavor"] = ak.unflatten(fatjet_flavor_flat, ak.num(selev.selFatJet))






        # Create the PtEtaPhiMLorentzVectorArray
        clustered_jets = ak.zip(
            {
                "pt":   ak.values_astype((selev.selFatJet.subjets [:, :, 0] + selev.selFatJet.subjets [:, :, 1]).pt  , np.float64),
                "eta":  ak.values_astype((selev.selFatJet.subjets [:, :, 0] + selev.selFatJet.subjets [:, :, 1]).eta , np.float64),
                "phi":  ak.values_astype((selev.selFatJet.subjets [:, :, 0] + selev.selFatJet.subjets [:, :, 1]).phi , np.float64),
                "mass": ak.values_astype((selev.selFatJet.subjets [:, :, 0] + selev.selFatJet.subjets [:, :, 1]).mass, np.float64),
                #"pt":   ak.values_astype(selev.selFatJet.pt,   np.float64),
                #"eta":  ak.values_astype(selev.selFatJet.eta,  np.float64),
                #"phi":  ak.values_astype(selev.selFatJet.phi,  np.float64),
                #"mass": ak.values_astype(selev.selFatJet.mass, np.float64),
                "jet_flavor": selev.selFatJet.jet_flavor,
                "btag_string": selev.selFatJet.btag_string,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior
        )

        # Look at this function
        # fat_jet_splittings_events = compute_decluster_variables(fat_jet_splittings_events)
        #        print("new fields:", fat_jet_splittings_events.fields)

        clustered_jets["splitting_name"] = "1b0j/1b0j"


        #
        # Declustering
        #
        b_pt_threshold = 20 # Min pt of subjets ?
        declustered_jets = make_synthetic_event(clustered_jets, clustering_pdfs, declustering_rand_seed=self.declustering_rand_seed,
                                                b_pt_threshold=b_pt_threshold, dr_threshold=0, chunk=chunk, debug=False)

        declustered_jets = declustered_jets[ak.argsort(declustered_jets.btagScore, axis=1, ascending=True)]


        #
        # Assigng these jets to fat jets
        #


        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_info_to_output_declustering_outputs
        # add_debug_info_to_output_declustering_outputs(selev, declustered_jets, processOutput)


        n_jet = ak.num(declustered_jets)
        total_jet = int(ak.sum(n_jet))

        #print(f"Declustered jets: {declustered_jets.btagScore}\n")

        # These need to change
        out_branches = {
                # Update jets with new kinematics
                "SubJet_pt":              declustered_jets.pt, #ak.unflatten(np.full(total_jet, 7), n_jet),
                "SubJet_eta":             declustered_jets.eta,
                "SubJet_phi":             declustered_jets.phi,
                "SubJet_mass":            declustered_jets.mass,
                "SubJet_btagSCore":            declustered_jets.btagScore,
                # create new regular branch
                #"nClusteredJets":      selev.nClusteredJets,
            }

#        if config["isMC"]:
#            out_branches["trigWeight_Data"] = selev.trigWeight_Data
#            out_branches["trigWeight_MC"]   = selev.trigWeight_MC
#            out_branches["CMSbtag"]        = weights.partial_weight(include=["CMS_btag"])[selections.all(*cumulative_cuts)]

#        if '202' in dataset:
#            out_branches["Jet_PNetRegPtRawCorr"]         = ak.unflatten(np.full(total_jet, 1), n_jet)
#            out_branches["Jet_PNetRegPtRawCorrNeutrino"] = ak.unflatten(np.full(total_jet, 1), n_jet)
#            out_branches["Jet_btagPNetB"]                = declustered_jets.btagScore
#
#        else:
#            out_branches["Jet_bRegCorr"] = ak.unflatten(np.full(total_jet, 1), n_jet)
#            out_branches["Jet_btagDeepFlavB"] = declustered_jets.btagScore

        #
        #  Need to skip all the other jet branches to make sure they have the same number of jets
        #
        for f in selev.SubJet.fields:
            bname = f"SubJet_{f}"
            if bname not in out_branches:
                self.skip_branches.append(bname)

        self.update_branch_filter(self.skip_collections, self.skip_branches)
        branches = ak.Array(out_branches)

        processOutput = {}
        self.cutFlow.addOutput(processOutput, event.metadata["dataset"])
        processOutput["total_jet"] = total_jet

        return (selection,
                branches,
                processOutput,
                )
