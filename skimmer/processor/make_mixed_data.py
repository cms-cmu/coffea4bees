import yaml
from coffea4bees.skimmer.processor.skimmer_4b_base import Skimmer4b
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.methods import vector

from coffea4bees.analysis.helpers.SvB_helpers import setFvTVars, subtract_ttbar_with_FvT
from src.friendtrees.FriendTreeSchema import FriendTreeSchema
from src.math_tools.random import Squares
from coffea4bees.analysis.helpers.event_weights import add_btagweights
from src.physics.event_selection import apply_event_selection
from coffea4bees.analysis.helpers.event_weights import add_weights

from src.data_formats.root import Chunk, TreeReader
from coffea4bees.analysis.helpers.load_friend import (
    FriendTemplate,
    rename_FvT_friend,
)

from coffea.analysis_tools import Weights, PackedSelection
import numpy as np
from src.physics.objects.jet_corrections import apply_jerc_corrections_jsonpog
from src.physics.common import update_events
from copy import copy
import logging
import awkward as ak
import uproot


from coffea4bees.hemisphere_mixing.mixing_helpers   import build_hemi_kdtrees, compute_hemi_vars
from coffea4bees.hemisphere_mixing.mixing_helpers   import split_events_into_hemispheres, replace_hemis, replace_hemis_load_kdTrees, replace_hemis_topk_kdTrees, init_hemi_data, transverse_thrust_awkward_fast
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from coffea4bees.analysis.helpers.event_weights import add_pseudotagweights


class HemiMixer(Skimmer4b):
    def __init__(self,
                subtract_ttbar_with_weights = False,
                friends: dict[str, str|FriendTemplate] = None,
                apply_JCM: bool = True,
                JCM_file: str = "coffea4bees/analysis/weights/JCM/AN_24_089_v3/jetCombinatoricModel_SB_6771c35.yml",
                hemi_library_yaml: str = None,
                hemi_stats_path: str = None,
                corrections_metadata: dict = None,
                use_boost_corrected_matching: bool = False,
                use_topk_matching: bool = False,
                k_neighbors: int = 10,
                collision_mode: str = "retry",
                default_rank: int = 0,
                object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
                *args, **kwargs):
        super().__init__(
            corrections_metadata=corrections_metadata,
            object_selection_cfg=object_selection_cfg,
            friends=friends,
            *args, **kwargs,
        )

        logging.info(f"\nRunning HemiMixer with these parameters: , subtract_ttbar_with_weights = {subtract_ttbar_with_weights}, args = {args}, kwargs = {kwargs}")
        logging.info(f"\nLoading JCM from file: {JCM_file} , apply_JCM = {apply_JCM}\n")
        self.apply_JCM = jetCombinatoricModel(JCM_file) if apply_JCM else None

        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights

        self.skip_collections = kwargs["skip_collections"]
        self.skip_branches    = kwargs["skip_branches"]

        # Boost-corrected matching: match on 3 variables (no pz), then boost to correct pz
        self.use_boost_corrected_matching = use_boost_corrected_matching
        logging.info(f"use_boost_corrected_matching = {self.use_boost_corrected_matching}")

        # Top-K matching with rank selection: parallel implementation, opt-in via flag.
        # collision_mode: "ignore" | "drop" | "retry"; default_rank=0 reproduces nearest-neighbor.
        self.use_topk_matching = use_topk_matching
        self.k_neighbors       = k_neighbors
        self.collision_mode    = collision_mode
        self.default_rank      = default_rank
        logging.info(f"use_topk_matching = {self.use_topk_matching}, k_neighbors = {self.k_neighbors}, collision_mode = {self.collision_mode}, default_rank = {self.default_rank}")

        # Conditional matching variables based on boost correction mode
        if self.use_boost_corrected_matching:
            self.hemi_summary_vars = ["sumPt_T_minor", "sumPt_T", "combinedMass"]  # 3D matching
            self.hemi_load_vars    = self.hemi_summary_vars + ["pz"]               # also load pz for boost
        else:
            self.hemi_summary_vars = ["sumPt_T_minor", "sumPt_T", "combinedMass", "pz"]  # 4D matching
            self.hemi_load_vars    = self.hemi_summary_vars

        self.hemi_library_yaml = hemi_library_yaml
        self.hemi_stats_path = hemi_stats_path


    def select(self, event):
        m = self._parse_event_metadata(event)
        year, dataset, fname, estart, estop = m.year, m.dataset, m.fname, m.estart, m.estop
        nEvent, year_label, chunk, processName, config = m.nEvent, m.year_label, m.chunk, m.processName, m.config
        logging.debug(f'{chunk} config={config}, for file {fname}\n')

        self.jet_branches = ["Jet_phi", "Jet_pt", "Jet_eta", "Jet_mass", "Jet_jetId", "Jet_puId"]
        if '202' in dataset:
            self.jet_branches += ["Jet_btagPNetB", "Jet_PNetRegPtRawCorr", "Jet_PNetRegPtRawCorrNeutrino"]
        else:
            self.jet_branches += ["Jet_btagDeepFlavB", "Jet_bRegCorr"]

        ### target is for new friend trees
        target = Chunk.from_coffea_events(event)

        #
        #  Load the hemisphere libraries
        #
        year_str = year.replace("_preVFP", "").replace("_postVFP", "")

        yaml_file = f'{self.hemi_stats_path}/hemi_statistics_{year_str}.yml'
        logging.info(f"\nLoading hemisphere libraries = {yaml_file}\n")

        logging.info(f"\nLoading hemisphere library file: {self.hemi_library_yaml} for year {year_str}\b")

        test_load_hemi_kdTrees = True
        if test_load_hemi_kdTrees:
            hemi_data, hemi_jet_ranges, hemi_stats  = init_hemi_data(hemi_metadata_yaml = yaml_file,
                                                                     hemi_files_yaml = self.hemi_library_yaml,
                                                                     year = year_str,
                                                                     hemi_summary_vars = self.hemi_load_vars,
                                                                     jet_branches = self.jet_branches,
                                                                     )

        else:
            hemi_kd_trees, _, hemi_jet_ranges, hemi_stats, hemi_data = build_hemi_kdtrees(hemi_metadata_yaml = yaml_file,
                                                                                          hemi_files_yaml = self.hemi_library_yaml,
                                                                                          year = year_str,
                                                                                          hemi_summary_vars = self.hemi_summary_vars,
                                                                                          jet_branches = self.jet_branches,
                                                                                          )



        path = fname.replace(fname.split("/")[-1], "")

        if self.subtract_ttbar_with_weights:

            if "FvT" in self.friends:
                event["FvT"] = rename_FvT_friend(target, self.friends["FvT"])
            else:

                FvT_file = f'{fname.replace("picoAOD", "FvT")}'
                event["FvT"] = ( NanoEventsFactory.from_root( FvT_file,
                                                              entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().FvT )

                if not ak.all(event.FvT.event == event.event):
                    raise ValueError("ERROR: FvT events do not match events ttree")

            setFvTVars("FvT", event)


        event = apply_event_selection( event, self.corrections_metadata[year], cut_on_lumimask=config["cut_on_lumimask"] )


        ## adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights( event, dataset, year_label,
                                                  self.corrections_metadata[year],
                                                  target=target,
                                                  friend_trigWeight=self.friends.get("trigWeight"),
                                                  config=config,
                                                 )



        #
        # Calculate and apply Jet Energy Calibration
        #
        if config["do_jet_calibration"]:
            jets = apply_jerc_corrections_jsonpog(event,
                                          corrections_metadata=self.corrections_metadata[year],
                                          isMC=config["isMC"],
                                          dataset=dataset
                                          )
        else:
            jets = event.Jet


        event = update_events(event, {"Jet": jets})

        event = apply_4b_selection( event, self.corrections_metadata[year], config=config,
                                           dataset=dataset,
                                           sel_cfg=self.sel_cfg,
                                           )


        #
        # Apply JCM
        #
        event["weight"] = weights.weight()
        weights, list_weight_names = add_pseudotagweights(
            event,
            weights,
            JCM=self.apply_JCM,
            apply_FvT=False,
            isDataForMixed=False,
            list_weight_names=list_weight_names,
            event_metadata=event.metadata,
            year_label=year_label,
            len_event=len(event),
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
        selections.add( "passHLT", ( event.passHLT if config["cut_on_HLT_decision"] else npfull(len(event), True)  ) )
        selections.add( 'passJetMult',   event.passJetMult )
        selections.add( "passThreeTag", event.threeTag)

        cumulative_cuts = ["lumimask"]
        self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )

        other_cuts = ["passNoiseFilter", "passHLT", "passJetMult"]

        for cut in other_cuts:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        event["weight"] = weights.weight()

        cumulative_cuts.append( "passThreeTag")
        self._cutFlow.fill( "passThreeTag", event[selections.all(*cumulative_cuts)], allTag=False  )

        #
        # Add Btag SF
        #
        if config["isMC"]:

            weights, list_weight_names = add_btagweights( event, weights,
                                                          list_weight_names=list_weight_names,
                                                          corrections_metadata=self.corrections_metadata[year],
                                                          isRun3=config["isRun3"],
            )
            logging.debug( f"Btag weight {weights.partial_weight(include=['CMS_btag'])[:10]}\n" )


            self._cutFlow.fill( "passNTag_btagSF", event[selections.all(*cumulative_cuts)], allTag=True )

        selection = event.lumimask & event.passNoiseFilter & event.passJetMult & event.threeTag
        if not config["isMC"]: selection = selection & event.passHLT

        selev = event[selections.all(*cumulative_cuts)]

        #
        #  TTbar subtractions using weights
        #
        if self.subtract_ttbar_with_weights:

            pass_ttbar_filter_selev = subtract_ttbar_with_FvT(selev, dataset, year)

            pass_ttbar_filter = np.full( len(event), True)
            pass_ttbar_filter[ selections.all(*cumulative_cuts) ] = pass_ttbar_filter_selev
            selections.add( 'pass_ttbar_filter', pass_ttbar_filter )
            cumulative_cuts.append("pass_ttbar_filter")
            self._cutFlow.fill( "pass_ttbar_filter", event[selections.all(*cumulative_cuts)], allTag=True )

            selection = selection & pass_ttbar_filter
            selev = selev[pass_ttbar_filter_selev]


        #
        # Identify tagged and pstagged jets
        #
        selected_jets             = selev.Jet[selev.Jet.selected]
        sorted_selected_jets      = selected_jets[ak.argsort(selected_jets.btagScore, axis=1, ascending=False)]
        sorted_selected_local_idx = ak.local_index(sorted_selected_jets, axis=1)
        selected_jets["tagged_or_pstagged"] = (sorted_selected_local_idx < selev.nJet_ps_and_tag)
        selev["tag_or_psTag_Jet"] = selected_jets[selected_jets.tagged_or_pstagged]


        #
        #  Split event into hemispheres
        #
        pos_hemi, neg_hemi = split_events_into_hemispheres(selev, tagged_key="tag_or_psTag_Jet")


        #
        #  Do Hemisphere replacement
        #
        all_hemis = ak.concatenate([pos_hemi, neg_hemi], axis=0)
        all_hemis["replaced"] = 0
        all_hemis["match_dist"] = -1

        topk_kept = None
        if self.use_topk_matching:
            if not test_load_hemi_kdTrees:
                raise RuntimeError("use_topk_matching=True requires the load-kdTrees code path (test_load_hemi_kdTrees=True).")
            all_hemis, topk_kept = replace_hemis_topk_kdTrees(
                all_hemis=all_hemis, hemi_jet_ranges=hemi_jet_ranges,
                hemi_stats=hemi_stats, hemi_data=hemi_data,
                hemi_summary_vars=self.hemi_summary_vars, jet_branches=self.jet_branches,
                k_neighbors=self.k_neighbors,
                default_rank=self.default_rank,
                collision_mode=self.collision_mode,
                use_boost_corrected_matching=self.use_boost_corrected_matching,
            )
        elif test_load_hemi_kdTrees:
            all_hemis = replace_hemis_load_kdTrees(all_hemis=all_hemis, hemi_jet_ranges=hemi_jet_ranges,
                                                   hemi_stats=hemi_stats, hemi_data=hemi_data, hemi_summary_vars=self.hemi_summary_vars, jet_branches=self.jet_branches,
                                                   use_boost_corrected_matching=self.use_boost_corrected_matching
                                                   )

        else:
            all_hemis = replace_hemis(all_hemis=all_hemis, hemi_kd_trees=hemi_kd_trees, hemi_jet_ranges=hemi_jet_ranges,
                                      hemi_stats=hemi_stats, hemi_data=hemi_data, hemi_summary_vars=self.hemi_summary_vars, jet_branches=self.jet_branches)



        n_event = len(selev)
        pos_hemi_new = all_hemis[:n_event]
        neg_hemi_new = all_hemis[n_event:]

        #
        #  Drop events where the pos and neg replacement hemispheres came from the
        #  same library source event: that degenerately reconstructs a real 4-tag event
        #  and defeats the inter-hemisphere decorrelation mixing is meant to provide.
        #  Topk path: collisions are retried internally; topk_kept is False only for
        #  events whose collision could not be resolved within K. Legacy path: detect
        #  here and drop.
        #
        if self.use_topk_matching:
            not_same_event_selev = np.asarray(topk_kept, dtype=bool)
            n_same_event = int(np.sum(~not_same_event_selev))
            if n_same_event:
                logging.info(f"Dropping {n_same_event}/{n_event} events with unresolvable same-library-event hemisphere pairs (after K={self.k_neighbors})")
        else:
            same_event_selev = ak.to_numpy(
                (pos_hemi_new.event             == neg_hemi_new.event)
                & (pos_hemi_new.run             == neg_hemi_new.run)
                & (pos_hemi_new.luminosityBlock == neg_hemi_new.luminosityBlock)
            )
            not_same_event_selev = ~same_event_selev
            n_same_event = int(np.sum(same_event_selev))
            if n_same_event:
                logging.info(f"Dropping {n_same_event}/{n_event} events with same-library-event hemisphere pairs")

            # Legacy path doesn't emit match_rank; fill zeros so the picoAOD schema is consistent across A/B runs.
            pos_hemi_new = ak.with_field(pos_hemi_new, ak.zeros_like(pos_hemi_new.event), "match_rank")
            neg_hemi_new = ak.with_field(neg_hemi_new, ak.zeros_like(neg_hemi_new.event), "match_rank")

        not_same_event = np.full(len(event), True)
        not_same_event[selections.all(*cumulative_cuts)] = not_same_event_selev
        selections.add("pass_not_same_event_hemi", not_same_event)
        cumulative_cuts.append("pass_not_same_event_hemi")
        self._cutFlow.fill("pass_not_same_event_hemi", event[selections.all(*cumulative_cuts)], allTag=True)

        selection    = selection & not_same_event
        selev        = selev[not_same_event_selev]
        pos_hemi     = pos_hemi[not_same_event_selev]
        neg_hemi     = neg_hemi[not_same_event_selev]
        pos_hemi_new = pos_hemi_new[not_same_event_selev]
        neg_hemi_new = neg_hemi_new[not_same_event_selev]
        n_event      = len(selev)


        old_hemi_output_vars = ["thrust_phi",  "event", "run", "luminosityBlock", "weight", "hemisphereId"]
        new_hemi_output_vars = old_hemi_output_vars + ["match_dist", "match_rank", "nSelJet", "nTagJet", "nJet"]
        output_vars = []

        for var_name in old_hemi_output_vars:
            selev[f"posHemiOld_{var_name}"] = pos_hemi[var_name]
            output_vars.append(f"posHemiOld_{var_name}")

            selev[f"negHemiOld_{var_name}"] = neg_hemi[var_name]
            output_vars.append(f"negHemiOld_{var_name}")

        for var_name in new_hemi_output_vars:
            selev[f"posHemiNew_{var_name}"] = pos_hemi_new[var_name]
            output_vars.append(f"posHemiNew_{var_name}")

            selev[f"negHemiNew_{var_name}"] = neg_hemi_new[var_name]
            output_vars.append(f"negHemiNew_{var_name}")


        mixed_Jet = ak.concatenate([pos_hemi_new.Jet, neg_hemi_new.Jet], axis=1)
        selev["Jet"] = mixed_Jet


        #
        #  Sanity check: compute transverse thrust of new jets
        #
        new_thrust = transverse_thrust_awkward_fast(selev.Jet, n_steps=720, refine_rounds=2)
        selev["newThrustPhi"] = new_thrust.phi
        output_vars.append("newThrustPhi")

        #
        #  Add pseudoTagWeight
        #
        output_vars.append("pseudoTagWeight")

        processOutput = {}


        out_branches = {}


        #
        #  Add Jet branches
        #
        for var_name in self.jet_branches:
            var_key = var_name.replace("Jet_", "")
            out_branches[var_name] = selev.Jet[var_key]


        #
        #  Add hemi branches
        #
        for var_name in output_vars:
            out_branches[var_name] = selev[var_name]

        if config["isMC"]:
            out_branches["trigWeight_Data"] = selev.trigWeight_Data
            out_branches["trigWeight_MC"]   = selev.trigWeight_MC
            out_branches["CMSbtag"]        = weights.partial_weight(include=["CMS_btag"])[selections.all(*cumulative_cuts)]


        #
        #  Need to skip all the other jet branches to make sure they have the same number of jets
        #
        for f in event.Jet.fields:
            bname = f"Jet_{f}"
            if bname not in out_branches:
                self.skip_branches.append(bname)

        self.update_branch_filter(self.skip_collections, self.skip_branches)
        branches = ak.Array(out_branches)


        return (selection,
                branches,
                processOutput,
                )
