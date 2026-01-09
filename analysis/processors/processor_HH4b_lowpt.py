from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from coffea4bees.analysis.helpers.event_selection import apply_4b_lowpt_selection
from coffea4bees.analysis.helpers.event_weights import add_pseudotagweights
from coffea4bees.analysis.helpers.candidates_selection import create_cand_jet_dijet_quadjet
from coffea4bees.analysis.helpers.filling_histograms import (
    filling_nominal_histograms,
    filling_syst_histograms,
)

class analysis(HH4bBaseProcessor):
    def __init__(
        self,
        *,
        apply_JCM_lowpt: bool = False,
        JCM_lowpt_file: str = None,
        run_lowpt_selection: bool = False,
        **kwargs  # Accept additional arguments to pass to parent
    ):
        # Initialize parent first, passing all kwargs
        super().__init__(**kwargs)
        
        self.apply_JCM_lowpt = jetCombinatoricModel(JCM_lowpt_file) if apply_JCM_lowpt else None
        self.run_lowpt_selection = run_lowpt_selection

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
        )

    def include_pseudotag_in_weight(self, event, weights, list_weight_names):
        return add_pseudotagweights(
            event,
            weights,
            JCM_lowpt=self.apply_JCM_lowpt,
            JCM=self.apply_JCM,
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
        )

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
            return filling_nominal_histograms(
                    selev,
                    self.apply_JCM,
                    processName=self.processName,
                    year=self.year,
                    isMC=self.config["isMC"],
                    histCuts=self.histCuts,
                    apply_FvT=self.apply_FvT,
                    run_SvB=self.run_SvB,
                    run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                    top_reconstruction=self.top_reconstruction,
                    isDataForMixed=self.config['isDataForMixed'],
                    tag_list=["lowpt_fourTag", "lowpt_threeTag"],
                    event_metadata=event.metadata,
                )
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


