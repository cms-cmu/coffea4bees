import logging
import awkward as ak

from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor, _Unset, _UNSET
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


def _init_classfier_lowpt(path):
    """Like _init_classfier but uses HCREnsemble_lowpt for modern (non-legacy) models."""
    if path is None or isinstance(path, _Unset):
        return None
    if isinstance(path, str):
        from ..helpers.classifier.HCR import Legacy_HCREnsemble
        return Legacy_HCREnsemble(path)
    from ..helpers.classifier.HCR import HCREnsemble_lowpt
    return HCREnsemble_lowpt(path)


class analysis(HH4bBaseProcessor):
    def __init__(
        self,
        *,
        apply_JCM: bool = False,
        JCM_file: str = None,
        friends: dict[str, str|FriendTemplate] = None,
        SvB_MA=_UNSET,
        weights: str | None = None,
        **kwargs  # Accept additional arguments to pass to parent
    ):
        # Initialize parent without JCM (we'll handle it ourselves).
        # Pass SvB_MA through so the parent's _skip_svb_legacy logic fires correctly,
        # then replace classifier_SvB_MA with the lowpt-aware variant below.
        super().__init__(apply_JCM=False, friends=friends, SvB_MA=SvB_MA, weights=weights, **kwargs)

        # Replace with lowpt-aware ensemble (reads nSelJets_lowpt instead of nJet_selected)
        self.classifier_SvB_MA = {}
        if SvB_MA is True and self.weights_data:
            for year, year_cfg in self.weights_data.items():
                if "SvB_MA" in year_cfg:
                    self.classifier_SvB_MA[year] = _init_classfier_lowpt(year_cfg["SvB_MA"])
        else:
            self.classifier_SvB_MA = _init_classfier_lowpt(SvB_MA)

        # Set our own lowpt version of JCM
        self.apply_JCM = {}
        if apply_JCM:
            if isinstance(JCM_file, str):
                self.apply_JCM = {"default": jetCombinatoricModel(JCM_file, lowpt_mode=True)}
            elif self.weights_data:
                for year, year_cfg in self.weights_data.items():
                    jcm_path = year_cfg.get("JCM_file") or year_cfg.get("JCM")
                    if jcm_path:
                        self.apply_JCM[year] = jetCombinatoricModel(jcm_path, lowpt_mode=True)
                if not self.apply_JCM:
                    raise ValueError("apply_JCM is True, but no JCM_file found in weights file.")
            else:
                raise ValueError("apply_JCM is True, but JCM_file is not specified and no weights file is provided.")
        else:
            self.apply_JCM = None

        self.friends = parse_friends(friends)

    def _fourtag_label(self):
        return "lowpt_fourTag"

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
            JCM=self.jcm_model,
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
            classifier_SvB=self.clf_SvB,
            classifier_SvB_MA=self.clf_SvB_MA,
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
            selev["nSelJets"] = ak.num(selev.selJet)

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
                seljet_label=["nSelJets_lowpt", "nSelJets"],
            )

        if self.make_friend_JCM_weight is not None:
            from ..helpers.dump_friendtrees import dump_JCM_weight
            friends["friends"] |= dump_JCM_weight(selev, self.make_friend_JCM_weight, "JCM_weight", analysis_selections)

        if self.make_friend_FvT_weight is not None:
            from ..helpers.dump_friendtrees import dump_FvT_weight
            friends["friends"] |= dump_FvT_weight(selev, self.make_friend_FvT_weight, "FvT_weight", analysis_selections)

        if self.make_friend_SvB is not None:
            from ..helpers.dump_friendtrees import dump_SvB
            if "SvB" in selev.fields and self.clf_SvB is not None:
                friends["friends"] |= dump_SvB(selev, self.make_friend_SvB, "SvB", analysis_selections)
            if "SvB_MA" in selev.fields and self.clf_SvB_MA is not None:
                friends["friends"] |= dump_SvB(selev, self.make_friend_SvB, "SvB_MA", analysis_selections)

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

        if self.run_SvB and "SvB_MA" in selev.fields and "passMinPs" not in selev.fields:
            selev["passMinPs"] = selev.SvB_MA.passMinPs

        if self.config["isMC"]:
            if self.config["isSignal"] and "bfromHorZ" in selev.fields and "tagJet_lowpt" in selev.fields:
                matched_genb = selev.tagJet_lowpt.nearest(selev.bfromHorZ, threshold=0.4)
                is_matched = ~ak.is_none(matched_genb, axis=1)
                local_idx = ak.local_index(selev.tagJet_lowpt, axis=1)
                matched_indices = local_idx[is_matched]
                has_any_match = ak.any(is_matched, axis=1)
                selev["matched_lowpt_jet_rank"] = ak.where(
                    has_any_match,
                    matched_indices,
                    ak.singletons(ak.full_like(has_any_match, -1, dtype=int))
                )
            else:
                selev["matched_lowpt_jet_rank"] = ak.singletons(ak.full_like(selev.run, -1, dtype=int))

        if not self.run_systematics:
            ## this can be simplified
            hist_nom = filling_nominal_histograms(
                selev,
                self.jcm_model,
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
                self.jcm_model,
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
                self.jcm_model,
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


