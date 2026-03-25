import logging
import awkward as ak

from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor
from ..helpers.load_friend import (
    FriendTemplate,
    parse_friends,
 )
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from coffea4bees.analysis.helpers.event_selection import apply_4b_lowpt_selection
from coffea4bees.analysis.helpers.event_weights import add_pseudotagweights
from coffea4bees.analysis.helpers.candidates_selection import create_cand_jet_dijet_quadjet
from coffea4bees.analysis.helpers.filling_histograms import (
    filling_nominal_histograms,
    filling_syst_histograms,
)
from coffea import processor

class analysis(HH4bBaseProcessor):
    def __init__(
        self,
        *,
        apply_JCM: bool = False,
        JCM_file: str = None,
        friends: dict[str, str|FriendTemplate] = None,
        **kwargs  # Accept additional arguments to pass to parent
    ):
        # Initialize parent without JCM (we'll handle it ourselves)
        super().__init__(apply_JCM=False, friends={}, **kwargs)
        
        # Set our own lowpt version of JCM
        self.apply_JCM = jetCombinatoricModel(JCM_file, lowpt_mode=True) if apply_JCM else None
        self.friends = parse_friends(friends)

    def apply_selection(self, event):
        return apply_4b_lowpt_selection(
            event,
            self.corrections_metadata[self.year],
            dataset=self.dataset,
            doLeptonRemoval=self.config["do_lepton_jet_cleaning"],
            override_selected_with_flavor_bit=self.config["override_selected_with_flavor_bit"],
            do_jet_veto_maps=self.config["do_jet_veto_maps"],
            isRun3=self.config["isRun3"],
            isMC=self.config["isMC"], ### temporary
            isSyntheticData=self.config["isSyntheticData"],
            isSyntheticMC=self.config["isSyntheticMC"],
            sel_cfg=self.sel_cfg,
        )

    def include_pseudotag_in_weight(self, event, weights, list_weight_names):
        return add_pseudotagweights(
            event,
            weights,
            JCM=self.apply_JCM,
            lowpt=True,
            apply_FvT=self.apply_FvT,
            isDataForMixed=self.config["isDataForMixed"],
            list_weight_names=list_weight_names,
            event_metadata=event.metadata,
            year_label=self.year_label,
            len_event=len(event),
            label3b="lowpt_threeTag",
        )

    def build_candidates(self, selev, weights, list_weight_names, analysis_selections, processOutput):

        return create_cand_jet_dijet_quadjet( 
            selev,
            apply_FvT=self.apply_FvT,
            run_SvB=self.run_SvB,
            run_systematics=self.run_systematics,
            classifier_SvB=self.classifier_SvB,
            classifier_SvB_MA=self.classifier_SvB_MA,
            processOutput=processOutput,
            isRun3=self.config["isRun3"],
            include_lowptjets=True,
            weights=weights,
            list_weight_names=list_weight_names,
            analysis_selections=analysis_selections,
            label3b="lowpt_threeTag",
        )
    

    def dump_friend_trees(self, selev, analysis_selections, shift_name):
        """Dump all requested friend trees.

        Requires chunk-scoped variables: config (for isMC, isSignal)
        Must be called after process() has initialized these variables.

        Args:
            selev: Selected events array
            analysis_selections: Boolean mask for analysis selection
            shift_name: Name of systematic shift (None for nominal)

        Returns:
            dict: Dictionary with 'friends' key containing friend tree data
        """
        friends = {'friends': {}}

        # if self.make_top_reconstruction is not None:
        #     from ..helpers.dump_friendtrees import dump_top_reconstruction
        #     friends["friends"] |= dump_top_reconstruction(
        #         selev,
        #         self.make_top_reconstruction,
        #         f"top_reco{'_'+shift_name if shift_name else ''}",
        #         analysis_selections,
        #     )

        if self.make_classifier_input is not None:
            for k in ["ZZSR", "ZHSR", "HHSR", "SR", "SB"]:
                selev[k] = selev["quadJet_selected"][k]
            selev["nSelJets_lowpt"] = ak.num(selev.selJet_lowpt)

            from ..helpers.dump_friendtrees import dump_input_friend
            weight = "weight_noJCM_noFvT"
            if weight not in selev.fields:
                weight = "weight"
            friends["friends"] |= dump_input_friend(
                selev,
                self.make_classifier_input,
                "HCR_input_lowpt",
                analysis_selections,
                weight=weight,
                NotCanJet="notCanJet_coffea",
                threeTag_label="lowpt_threeTag",
                fourTag_label="lowpt_fourTag",
                seljet_label="nSelJets_lowpt",
            )

        # if self.make_friend_JCM_weight is not None:
        #     from ..helpers.dump_friendtrees import dump_JCM_weight
        #     friends["friends"] |= dump_JCM_weight(selev, self.make_friend_JCM_weight, "JCM_weight", analysis_selections)

        # if self.make_friend_FvT_weight is not None:
        #     from ..helpers.dump_friendtrees import dump_FvT_weight
        #     friends["friends"] |= dump_FvT_weight(selev, self.make_friend_FvT_weight, "FvT_weight", analysis_selections)

        # if self.make_friend_SvB is not None:
        #     from ..helpers.dump_friendtrees import dump_SvB
        #     friends["friends"] |= dump_SvB(selev, self.make_friend_SvB, "SvB", analysis_selections)
        #     friends["friends"] |= dump_SvB(selev, self.make_friend_SvB, "SvB_MA", analysis_selections)

        return friends

    def histograms(self, event, selev, weights, analysis_selections, shift_name):
        """Fill histograms for analysis.
        
        Requires chunk-scoped variables: processName, year, config
        Must be called after process() has initialized these variables.
        
        Args:
            event: Event array
            selev: Selected events array
            weights: Weights object
            analysis_selections: Boolean mask for analysis selection
            shift_name: Name of systematic shift (None for nominal)
            
        Returns:
            Dictionary with histogram outputs
        """

        if self.classifier_FvT: apply_FvT = True
        else: apply_FvT = self.apply_FvT

        if not self.run_systematics:
            ## this can be simplified
            hist_nom = filling_nominal_histograms(
                selev,
                self.apply_JCM,
                processName=self.processName,
                year=self.year,
                isMC=self.config["isMC"],
                histCuts=self.histCuts,
                apply_FvT=apply_FvT,
                run_SvB=self.run_SvB,
                run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                top_reconstruction=self.top_reconstruction,
                isDataForMixed=self.config['isDataForMixed'],
                tag_list=["lowpt_fourTag", "lowpt_threeTag"],
                event_metadata=event.metadata
            )
            if not self.plot_ttbar_with_weights:
                return hist_nom


            hist_t4 = filling_nominal_histograms(
                selev,
                self.apply_JCM,
                processName="TTbar4b_from_d3",
                year=self.year,
                isMC=self.config["isMC"],
                histCuts=self.histCuts,
                apply_FvT=apply_FvT,
                run_SvB=self.run_SvB,
                run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                top_reconstruction=self.top_reconstruction,
                isDataForMixed=self.config['isDataForMixed'],
                event_metadata=event.metadata,
                tag_list=["lowpt_fourTag", "lowpt_threeTag"],
                weight_name = "weight_d3_to_t4"
            )

            hist_t3 = filling_nominal_histograms(
                selev,
                self.apply_JCM,
                processName="TTbar3b_from_d3",
                year=self.year,
                isMC=self.config["isMC"],
                histCuts=self.histCuts,
                apply_FvT=apply_FvT,
                run_SvB=self.run_SvB,
                run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                top_reconstruction=self.top_reconstruction,
                isDataForMixed=self.config['isDataForMixed'],
                event_metadata=event.metadata,
                tag_list=["lowpt_fourTag", "lowpt_threeTag"],
                weight_name = "weight_d3_to_t3"
            )

            return processor.accumulate([hist_nom, hist_t4, hist_t3])

        #
        # Run systematics
        #
        else:
            return filling_syst_histograms(
                selev, 
                weights,
                analysis_selections,
                shift_name=shift_name,
                processName=self.processName,
                year=self.year,
                histCuts=self.histCuts
                )


