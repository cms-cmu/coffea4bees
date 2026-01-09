import yaml
from src.skimmer.picoaod import PicoAOD #, fetch_metadata, resize
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea.nanoevents import NanoEventsFactory

from coffea4bees.jet_clustering.clustering   import cluster_bs
from coffea4bees.jet_clustering.declustering import make_synthetic_event, clean_ISR
from coffea4bees.analysis.helpers.SvB_helpers import setSvBVars, subtract_ttbar_with_SvB
from src.friendtrees.FriendTreeSchema import FriendTreeSchema
from src.math_tools.random import Squares
from coffea4bees.analysis.helpers.event_weights import add_btagweights
from coffea4bees.analysis.helpers.processor_config import processor_config
from src.physics.event_selection import apply_event_selection
from src.physics.event_weights import add_weights

from src.data_formats.root import Chunk, TreeReader
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
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

class DeClusterer(PicoAOD):
    def __init__(self, clustering_pdfs_file = "None",
                subtract_ttbar_with_weights = False,
                declustering_rand_seed=5,
                friends: dict[str, str|FriendTemplate] = None,
                corrections_metadata: dict = None,
                *args, **kwargs):
        kwargs["pico_base_name"] = f'picoAOD_seed{declustering_rand_seed}'
        super().__init__(*args, **kwargs)

        logging.info(f"\nRunning Declusterer with these parameters: clustering_pdfs_file = {clustering_pdfs_file}, subtract_ttbar_with_weights = {subtract_ttbar_with_weights}, declustering_rand_seed = {declustering_rand_seed}, args = {args}, kwargs = {kwargs}")
        self.clustering_pdfs_file = clustering_pdfs_file

        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights
        self.friends = parse_friends(friends)
        self.declustering_rand_seed = declustering_rand_seed
        self.corrections_metadata = corrections_metadata
        self._cutFlow = cutflow_4b()

        self.skip_collections = kwargs["skip_collections"]
        self.skip_branches    = kwargs["skip_branches"]


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

        ### target is for new friend trees
        target = Chunk.from_coffea_events(event)

        clustering_pdfs_file = self.clustering_pdfs_file.replace("XXX", year)

        print(f"clustering_pdfs_file is {clustering_pdfs_file}\n")
        if not clustering_pdfs_file == "None":
            clustering_pdfs = yaml.safe_load(open(clustering_pdfs_file, "r"))
            logging.info(f"Loaded {len(clustering_pdfs.keys())} PDFs from {clustering_pdfs_file}\n")
        else:
            clustering_pdfs = None

        #
        # Set process and datset dependent flags
        #
        config = processor_config(processName, dataset, event)
        logging.debug(f'{chunk} config={config}, for file {fname}\n')

        path = fname.replace(fname.split("/")[-1], "")

        if self.subtract_ttbar_with_weights:

            SvB_MA_file = f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
            event["SvB_MA"] = ( NanoEventsFactory.from_root( SvB_MA_file,
                                                             entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema ).events().SvB_MA )

            if not ak.all(event.SvB_MA.event == event.event):
                raise ValueError("ERROR: SvB_MA events do not match events ttree")

            # defining SvB_MA
            setSvBVars("SvB_MA", event)

        event = apply_event_selection( event, self.corrections_metadata[year], cut_on_lumimask=config["cut_on_lumimask"] )


        ## adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights( event, config["do_MC_weights"], dataset, year_label,
                                                  self.corrections_metadata[year],
                                                  isTTForMixed=False,
                                                  target=target,
                                                  friend_trigWeight=self.friends.get("trigWeight"),
                                                 )



        #
        # Calculate and apply Jet Energy Calibration
        #
        if config["do_jet_calibration"]:
            jets = apply_jerc_corrections(event,
                                          corrections_metadata=self.corrections_metadata[year],
                                          isMC=config["isMC"],
                                          dataset=dataset
                                          )
        else:
            jets = event.Jet


        event = update_events(event, {"Jet": jets})

        event = apply_4b_selection( event, self.corrections_metadata[year],
                                           dataset=dataset,
                                           doLeptonRemoval=config["do_lepton_jet_cleaning"],
                                           override_selected_with_flavor_bit=config["override_selected_with_flavor_bit"],
                                           do_jet_veto_maps = config["do_jet_veto_maps"],
                                           isRun3=config["isRun3"],
                                           isMC=config["isMC"],
                                           isSyntheticData=config["isSyntheticData"],
                                           isSyntheticMC=config["isSyntheticMC"],
                                           )


        #
        # Get the trigger weights
        #
        if config["isMC"]:
            if "GluGlu" in dataset:
                ### this is temporary until trigWeight is computed in new code
                # trigWeight_file = uproot.open(f'{event.metadata["filename"].replace("picoAOD", "trigWeight")}')['Events']
                # trigWeight = trigWeight_file.arrays(['event', 'trigWeight_Data', 'trigWeight_MC'], entry_start=estart,entry_stop=estop)
                # if not ak.all(trigWeight.event == event.event):
                #     raise ValueError('trigWeight events do not match events ttree')
                trigWeight = self.friends.get("trigWeight").arrays(target)

                event["trigWeight_Data"] = trigWeight.Data
                event["trigWeight_MC"]   = trigWeight.MC


        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( event.passHLT if config["cut_on_HLT_decision"] else np.full(len(event), True)  ) )
        selections.add( 'passJetMult',   event.passJetMult )
        selections.add( "passFourTag", event.fourTag)

        event["weight"] = weights.weight()

        cumulative_cuts = ["lumimask"]
        self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )

        other_cuts = ["passNoiseFilter", "passHLT", "passJetMult","passFourTag"]

        for cut in other_cuts:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        #
        # Add Btag SF
        #
        if config["isMC"]:

            weights, list_weight_names = add_btagweights( event, weights,
                                                          list_weight_names=list_weight_names,
                                                          corrections_metadata=self.corrections_metadata[year]
            )
            logging.debug( f"Btag weight {weights.partial_weight(include=['CMS_btag'])[:10]}\n" )
            event["weight"] = weights.weight()

            self._cutFlow.fill( "passFourTag_btagSF", event[selections.all(*cumulative_cuts)], allTag=True )

        selection = event.lumimask & event.passNoiseFilter & event.passJetMult & event.fourTag
        if not config["isMC"]: selection = selection & event.passHLT

        selev = event[selections.all(*cumulative_cuts)]

        #
        #  TTbar subtractions using weights
        #
        if self.subtract_ttbar_with_weights:

            pass_ttbar_filter_selev = subtract_ttbar_with_SvB(selev, dataset, year)

            pass_ttbar_filter = np.full( len(event), True)
            pass_ttbar_filter[ selections.all(*cumulative_cuts) ] = pass_ttbar_filter_selev
            selections.add( 'pass_ttbar_filter', pass_ttbar_filter )
            cumulative_cuts.append("pass_ttbar_filter")
            self._cutFlow.fill( "pass_ttbar_filter", event[selections.all(*cumulative_cuts)], allTag=True )

            selection = selection & pass_ttbar_filter
            selev = selev[pass_ttbar_filter_selev]

        #
        # Build and select boson candidate jets with bRegCorr applied
        #
        sorted_idx = ak.argsort( selev.Jet.btagScore * selev.Jet.selected, axis=1, ascending=False )
        canJet_idx = sorted_idx[:, 0:4]
        notCanJet_idx = sorted_idx[:, 4:]
        canJet = selev.Jet[canJet_idx]

        # apply bJES to canJets
        canJet = canJet * canJet.bRegCorr
        canJet["bRegCorr"] = selev.Jet.bRegCorr[canJet_idx]
        canJet["btagScore"] = selev.Jet.btagScore[canJet_idx]
        #if '202' in dataset:
        #    canJet["btagPNetB"] = selev.Jet.btagPNetB[canJet_idx]


        if config["isMC"]:
            canJet["hadronFlavour"] = selev.Jet.hadronFlavour[canJet_idx]

        #
        # pt sort canJets
        #
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]

        notCanJet = selev.Jet[notCanJet_idx]
        notCanJet = notCanJet[notCanJet.selected_loose]
        notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

        #
        # Do the Clustering
        #
        canJet["jet_flavor"] = "b"
        notCanJet["jet_flavor"] = "j"

        jets_for_clustering = ak.concatenate([canJet, notCanJet], axis=1)
        jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]

        processOutput = {}

        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_info_to_output_clustering_inputs
        # add_debug_info_to_output_clustering_inputs(selev, jets_for_clustering, processOutput)

        clustered_jets, _clustered_splittings = cluster_bs(jets_for_clustering, debug=False)
        clustered_jets = clean_ISR(clustered_jets, _clustered_splittings)

        mask_unclustered_jet = (clustered_jets.jet_flavor == "b") | (clustered_jets.jet_flavor == "j")
        selev["nClusteredJets"] = ak.num(clustered_jets[~mask_unclustered_jet])

        #
        # Declustering
        #
        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_info_to_output_clustering_outputs
        # add_debug_info_to_output_clustering_outputs(selev, clustered_jets, processOutput)

        b_pt_threshold = 30 if config["isRun3"] else 40
        declustered_jets = make_synthetic_event(clustered_jets, clustering_pdfs, declustering_rand_seed=self.declustering_rand_seed, b_pt_threshold=b_pt_threshold, chunk=chunk)

        declustered_jets = declustered_jets[ak.argsort(declustered_jets.pt, axis=1, ascending=False)]

        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_info_to_output_declustering_outputs
        # add_debug_info_to_output_declustering_outputs(selev, declustered_jets, processOutput)


        n_jet = ak.num(declustered_jets)
        total_jet = int(ak.sum(n_jet))


        out_branches = {
                # Update jets with new kinematics
                "Jet_pt":              declustered_jets.pt, #ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_eta":             declustered_jets.eta,
                "Jet_phi":             declustered_jets.phi,
                "Jet_mass":            declustered_jets.mass,
                "Jet_jet_flavor_bit":  declustered_jets.jet_flavor_bit,
                "Jet_jetId":           ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_puId":            ak.unflatten(np.full(total_jet, 7), n_jet),
                # create new regular branch
                "nClusteredJets":      selev.nClusteredJets,
            }

        if config["isMC"]:
            out_branches["trigWeight_Data"] = selev.trigWeight_Data
            out_branches["trigWeight_MC"]   = selev.trigWeight_MC
            out_branches["CMSbtag"]        = weights.partial_weight(include=["CMS_btag"])[selections.all(*cumulative_cuts)]

        if '202' in dataset:
            out_branches["Jet_PNetRegPtRawCorr"]         = ak.unflatten(np.full(total_jet, 1), n_jet)
            out_branches["Jet_PNetRegPtRawCorrNeutrino"] = ak.unflatten(np.full(total_jet, 1), n_jet)
            out_branches["Jet_btagPNetB"]                = declustered_jets.btagScore

        else:
            out_branches["Jet_bRegCorr"] = ak.unflatten(np.full(total_jet, 1), n_jet)
            out_branches["Jet_btagDeepFlavB"] = declustered_jets.btagScore

        #
        #  Need to skip all the other jet branches to make sure they have the same number of jets
        #
        for f in event.Jet.fields:
            bname = f"Jet_{f}"
            if bname not in out_branches:
                self.skip_branches.append(bname)

        self.update_branch_filter(self.skip_collections, self.skip_branches)
        branches = ak.Array(out_branches)

        processOutput["total_jet"] = total_jet

        return (selection,
                branches,
                processOutput,
                )
