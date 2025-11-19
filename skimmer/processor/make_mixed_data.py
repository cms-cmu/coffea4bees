import yaml
from src.skimmer.picoaod import PicoAOD #, fetch_metadata, resize
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea.nanoevents import NanoEventsFactory


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


from coffea4bees.hemisphere_mixing.mixing_helpers   import build_hemi_kdtrees
from coffea4bees.hemisphere_mixing.mixing_helpers   import split_events_into_hemispheres, get_filter

class HemiMixer(PicoAOD):
    def __init__(self,
                subtract_ttbar_with_weights = False,
                mixing_rand_seed=5,
                friends: dict[str, str|FriendTemplate] = None,
                corrections_metadata: dict = None,
                *args, **kwargs):
        kwargs["pico_base_name"] = f'picoAOD_seed{mixing_rand_seed}'
        super().__init__(*args, **kwargs)

        logging.info(f"\nRunning HemiMixer with these parameters: , subtract_ttbar_with_weights = {subtract_ttbar_with_weights}, mixing_rand_seed = {mixing_rand_seed}, args = {args}, kwargs = {kwargs}")

        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights
        self.friends = parse_friends(friends)
        self.mixing_rand_seed = mixing_rand_seed
        self.corrections_metadata = corrections_metadata
        self._cutFlow = cutflow_4b()

        self.skip_collections = kwargs["skip_collections"]
        self.skip_branches    = kwargs["skip_branches"]

        #
        #  Load the hemisphere libraries
        #
        yaml_file = 'coffea4bees/hemisphere_mixing/hemi_plots/hemi_statistics_UL18.yml'
        logging.info(f"\nLoading hemisphere libraries = {yaml_file}")


        jet_branches = ["Jet_phi", "Jet_pt", "Jet_eta", "Jet_mass", "Jet_btagDeepFlavB", "Jet_bRegCorr", "Jet_jetId"]
        #branch_list = ["nJet", "nSelJet", "nTagJet", "sumPt_T_minor", "sumPt_T", "combinedMass", "pz" ] + jet_branches
        self.hemi_summary_vars = ["sumPt_T_minor", "sumPt_T", "combinedMass", "pz" ]
        year_str = "UL18"

        self.hemi_kd_trees, self.hemi_points, self.hemi_jet_ranges, self.hemi_stats, self.hemi_data = build_hemi_kdtrees(hemi_metadata_yaml = yaml_file,
                                                                                                                         hemifiles = f"output/mixeddata_cluster/data_{year_str}*/*.root",
                                                                                                                         hemi_summary_vars = self.hemi_summary_vars,
                                                                                                                         jet_branches = jet_branches,
                                                                                                                         )





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
        selections.add( "passThreeTag", event.threeTag)

        event["weight"] = weights.weight()

        cumulative_cuts = ["lumimask"]
        self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )

        other_cuts = ["passNoiseFilter", "passHLT", "passJetMult", "passThreeTag"]

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

        selection = event.lumimask & event.passNoiseFilter & event.passJetMult & event.threeTag
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

        print("selMuon", type(selev.selMuon), selev.selMuon.tolist(),"\n")

        #
        #  Split event into hemispheres
        #
        pos_hemi, neg_hemi = split_events_into_hemispheres(selev)


        #
        #  Loop on hemisphere multiplcity bins
        #
        # Outer loop: tag multiplicity bins
        tag_keys = list(self.hemi_jet_ranges.keys())
        for itag, tag in enumerate(tag_keys):
            print(itag, tag, type(tag), "\n")
            # --- tag filter ----------------------------------------------------------
            tag_filter_pos = get_filter(pos_hemi, "nTagJet", tag, low_edge=(itag==0), high_edge=(itag==len(tag_keys)-1))
            tag_filter_neg = get_filter(neg_hemi, "nTagJet", tag, low_edge=(itag==0), high_edge=(itag==len(tag_keys)-1))

            # skip empty sub-ranges
            if not self.hemi_jet_ranges[tag]:
                print(f"ERROR: no sel jets for tag = {tag}")
                continue

            # -------------------------------------------------------------------------
            # Middle loop: selected-jet multiplicity bins
            sel_keys = list(self.hemi_jet_ranges[tag].keys())
            for isel, sel in enumerate(sel_keys):

                # --- sel filter ------------------------------------------------------
                sel_filter_pos = get_filter(pos_hemi, "nSelJet", sel, low_edge=(isel==0), high_edge=(isel==len(sel_keys)-1))
                sel_filter_neg = get_filter(neg_hemi, "nSelJet", sel, low_edge=(isel==0), high_edge=(isel==len(sel_keys)-1))

                # ---------------------------------------------------------------------
                # Inner loop: total-jet multiplicity bins
                jet_bins = self.hemi_jet_ranges[tag][sel]
                if not jet_bins:

                    jet_mult_key = (tag, sel, -1)
                    # special case: no jet bins defined

                    mask_pos = tag_filter_pos & sel_filter_pos
                    mask_neg = tag_filter_neg & sel_filter_neg

                else:

                    for ijet, jet in enumerate(jet_bins):

                        jet_filter_pos = get_filter(pos_hemi, "nJet", jet, low_edge=(ijet==0), high_edge=(ijet==len(jet_bins)-1))
                        jet_filter_neg = get_filter(neg_hemi, "nJet", jet, low_edge=(ijet==0), high_edge=(ijet==len(jet_bins)-1))
                        jet_mult_key = (tag, sel, jet)

                        # --- final selection ---------------------------------------------
                        mask_pos = tag_filter_pos & sel_filter_pos & jet_filter_pos
                        mask_neg = tag_filter_neg & sel_filter_neg & jet_filter_neg

                        # print(f"HemiStats for jet mult key {jet_mult_key}: {self.hemi_stats[jet_mult_key]}\n")

                        #
                        #  convert to zscores....
                        #
                        pos_hemi_points = np.column_stack([ (pos_hemi[mask_pos][name] - self.hemi_stats[jet_mult_key][name]["mean"]) / self.hemi_stats[jet_mult_key][name]["RMS"] for name in self.hemi_summary_vars])
                        neg_hemi_points = np.column_stack([ (neg_hemi[mask_pos][name] - self.hemi_stats[jet_mult_key][name]["mean"]) / self.hemi_stats[jet_mult_key][name]["RMS"] for name in self.hemi_summary_vars])
                        #neg_hemi_points = np.column_stack([ pos_hemi[mask_pos][name] for name in self.hemi_summary_vars])
                        #print(f"pos_hemi_points for jet mult key {jet_mult_key}:\n {pos_hemi_points}\n")

                        pos_match_dist, pos_match_idx = self.hemi_kd_trees[jet_mult_key].query(pos_hemi_points, k=1)
                        # print("pos_match",pos_match_dist,pos_match_idx,"\n")
                        # print(self.hemi_points[jet_mult_key][pos_match_idx])

                        #print("neg_match",self.hemi_kd_trees[jet_mult_key].query(neg_hemi_points, k=1),"\n")


        #
        #  Funciton to find the corect hemisphere libraries
        #
        print(f"pos_hemi.nJet = {pos_hemi.nTagJet, pos_hemi.nSelJet, ak.num(pos_hemi.Jet, axis=1)}\n")

        #
        #  Find nearest neighbor hemispheres
        #

        # Hack for Now
        #print("pos_match",self.hemi_kd_trees[(0, 1, 1)].query(self.hemi_points[(0, 1, 2)], k=1),"\n")
        #print("neg_match",self.hemi_kd_trees[(0, 1, 2)].query(self.hemi_points[(0, 1, 1)], k=1),"\n")

        processOutput = {}

        n_jet = ak.num(selev.Jet)
        total_jet = int(ak.sum(n_jet))
        out_branches = {}

        out_branches = {
                # Update jets with new kinematics
                "Jet_pt":              selev.Jet.pt, #ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_eta":             selev.Jet.eta,
                "Jet_phi":             selev.Jet.phi,
                "Jet_mass":            selev.Jet.mass,
                "Jet_jetId":           ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_puId":            ak.unflatten(np.full(total_jet, 7), n_jet),
                # create new regular branch
                #"nClusteredJets":      selev.nClusteredJets,
            }

        if config["isMC"]:
            out_branches["trigWeight_Data"] = selev.trigWeight_Data
            out_branches["trigWeight_MC"]   = selev.trigWeight_MC
            out_branches["CMSbtag"]        = weights.partial_weight(include=["CMS_btag"])[selections.all(*cumulative_cuts)]

        if '202' in dataset:
            out_branches["Jet_PNetRegPtRawCorr"]         = ak.unflatten(np.full(total_jet, 1), n_jet)
            out_branches["Jet_PNetRegPtRawCorrNeutrino"] = ak.unflatten(np.full(total_jet, 1), n_jet)
            out_branches["Jet_btagPNetB"]                = selev.Jet.btagPNetB

        else:
            out_branches["Jet_bRegCorr"] = ak.unflatten(np.full(total_jet, 1), n_jet)
            out_branches["Jet_btagDeepFlavB"] = selev.Jet.btagDeepFlavB

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
