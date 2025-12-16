import yaml
from src.skimmer.picoaod import PicoAOD #, fetch_metadata, resize
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.methods import vector

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


from coffea4bees.hemisphere_mixing.mixing_helpers   import build_hemi_kdtrees, compute_hemi_vars
from coffea4bees.hemisphere_mixing.mixing_helpers   import split_events_into_hemispheres, replace_hemis, replace_hemis_load_kdTrees, init_hemi_data, transverse_thrust_awkward_fast
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from coffea4bees.analysis.helpers.event_weights import add_pseudotagweights


class HemiMixer(PicoAOD):
    def __init__(self,
                subtract_ttbar_with_weights = False,
                friends: dict[str, str|FriendTemplate] = None,
                apply_JCM: bool = True,
                JCM_file: str = "coffea4bees/analysis/weights/JCM/AN_24_089_v3/jetCombinatoricModel_SB_6771c35.yml",
                hemi_library_yaml: str = None,
                corrections_metadata: dict = None,
                *args, **kwargs):
        super().__init__(*args, **kwargs)

        logging.info(f"\nRunning HemiMixer with these parameters: , subtract_ttbar_with_weights = {subtract_ttbar_with_weights}, args = {args}, kwargs = {kwargs}")
        self.apply_JCM = jetCombinatoricModel(JCM_file) if apply_JCM else None

        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights
        self.friends = parse_friends(friends)
        self.corrections_metadata = corrections_metadata
        self._cutFlow = cutflow_4b()

        self.skip_collections = kwargs["skip_collections"]
        self.skip_branches    = kwargs["skip_branches"]

        #
        #  Load the hemisphere libraries
        #
        yaml_file = 'coffea4bees/hemisphere_mixing/hemi_plots/hemi_statistics_UL18.yml'
        logging.info(f"\nLoading hemisphere libraries = {yaml_file}")


        self.jet_branches = ["Jet_phi", "Jet_pt", "Jet_eta", "Jet_mass", "Jet_btagDeepFlavB", "Jet_bRegCorr", "Jet_jetId", "Jet_puId"]
        self.hemi_summary_vars = ["sumPt_T_minor", "sumPt_T", "combinedMass", "pz" ]
        year_str = "UL18"
        logging.info(f"\nLoading hemisphere library file: {hemi_library_yaml} for year {year_str}")
        with open(hemi_library_yaml, 'r') as f:
            hemi_library_data = yaml.safe_load(f)
            logging.debug("Keys",hemi_library_data.keys())
            hemi_files = hemi_library_data[year_str]
            logging.debug("Hemi files:", type(hemi_files), hemi_files)

        self.test_load_hemi_kdTrees = True
        if self.test_load_hemi_kdTrees:
            self.hemi_data, self.hemi_jet_ranges, self.hemi_stats  = init_hemi_data(hemi_metadata_yaml = yaml_file,
                                                                                    hemifiles = hemi_files,
                                                                                    hemi_summary_vars = self.hemi_summary_vars,
                                                                                    jet_branches = self.jet_branches,
                                                                                    )

        else:
            self.hemi_kd_trees, _, self.hemi_jet_ranges, self.hemi_stats, self.hemi_data = build_hemi_kdtrees(hemi_metadata_yaml = yaml_file,
                                                                                                              hemifiles = hemifiles,
                                                                                                              hemi_summary_vars = self.hemi_summary_vars,
                                                                                                              jet_branches = self.jet_branches,
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
        # Apply JCM
        #
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


        #
        # Identify tagged and pstagged jets
        #
        sorted_jets      = selev.Jet[ak.argsort(selev.Jet.btagScore, ascending=False)]
        sorted_local_idx = ak.local_index(sorted_jets)
        selev["Jet", "tagged_or_pstagged"] = (sorted_local_idx < selev.nJet_ps_and_tag)
        selev["tag_or_psTag_Jet"] = selev.Jet[selev.Jet.tagged_or_pstagged]

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

        if self.test_load_hemi_kdTrees:
            all_hemis = replace_hemis_load_kdTrees(all_hemis=all_hemis, hemi_jet_ranges=self.hemi_jet_ranges,
                                                   hemi_stats=self.hemi_stats, hemi_data=self.hemi_data, hemi_summary_vars=self.hemi_summary_vars, jet_branches=self.jet_branches
                                                   )

        else:
            all_hemis = replace_hemis(all_hemis=all_hemis, hemi_kd_trees=self.hemi_kd_trees, hemi_jet_ranges=self.hemi_jet_ranges,
                                      hemi_stats=self.hemi_stats, hemi_data=self.hemi_data, hemi_summary_vars=self.hemi_summary_vars, jet_branches=self.jet_branches)



        n_event = len(selev)
        pos_hemi_new = all_hemis[:n_event]
        neg_hemi_new = all_hemis[n_event:]


        old_hemi_output_vars = ["thrust_phi",  "event", "run", "luminosityBlock", "weight", "hemisphereId"]
        new_hemi_output_vars = old_hemi_output_vars + ["match_dist"]
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
