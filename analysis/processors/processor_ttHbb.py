from __future__ import annotations

import logging
import numpy as np
import awkward as ak
from coffea.analysis_tools import PackedSelection
from coffea import processor
from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor, _UNSET
from coffea4bees.analysis.helpers.filling_histograms import filling_ttHbb_histograms
from coffea4bees.analysis.helpers.SvB_helpers import set_ttHbb_SvB_vars

class ttHbbProcessor(HH4bBaseProcessor):
    """
    Fully decoupled Coffea processor for ttH(bb) analysis workflows.
    Inherits core event/object selection and corrections from HH4bBaseProcessor
    while isolating ttHbb SvB score derivation and histogram filling (skipping HH mass plots).
    """
    def __init__(
        self,
        *,
        friends=None,
        weights=None,
        SvB_MA=_UNSET,
        SvB=None,
        blind=False,
        apply_JCM=True,
        JCM_file=None,
        apply_FvT=True,
        apply_trigWeight=True,
        apply_btagSF=True,
        apply_boosted_veto=False,
        run_SvB=True,
        top_reconstruction="fast",
        plot_ttbar_with_weights=True,
        hist_cuts=[],
        **kwargs,
    ):
        logging.info("Initializing decoupled ttHbbProcessor")
        if weights is None or weights == "coffea4bees/metadata/weights/weights_HH4b.yml":
            weights = "coffea4bees/metadata/weights/weights_HH4b_2024_v2.yml"
        super().__init__(
            friends=friends,
            weights=weights,
            SvB_MA=SvB_MA,
            SvB=SvB,
            blind=blind,
            apply_JCM=apply_JCM,
            JCM_file=JCM_file,
            apply_FvT=apply_FvT,
            apply_trigWeight=apply_trigWeight,
            apply_btagSF=apply_btagSF,
            apply_boosted_veto=apply_boosted_veto,
            run_SvB=run_SvB,
            top_reconstruction=top_reconstruction,
            plot_ttbar_with_weights=plot_ttbar_with_weights,
            hist_cuts=hist_cuts,
            **kwargs,
        )

    def load_SvB(self, event):
        """Load SvB scores and derive native ttHbb fields without fake HH/ZH/ZZ assignments."""
        super().load_SvB(event)
        for k in self.friends:
            if k.startswith("SvB") and not k.startswith("SvB_FeynNet"):
                if getattr(event, k, None) is not None:
                    set_ttHbb_SvB_vars(k, event)

    def build_selections(self, event, weights):
        """Build PackedSelection object with all cuts and add selJets.n > 6 categorization."""
        selections, allcuts = super().build_selections(event, weights)

        # Define jet multiplicity pass/fail masks
        n_selJets = ak.num(event.selJets) if "selJets" in event.fields else ak.num(event.selJet)
        event["pass_nSelJets_gt6"] = n_selJets > 6
        event["fail_nSelJets_le6"] = n_selJets <= 6
        event["all_selJets"] = np.full(len(event), True)

        selections.add("pass_nSelJets_gt6", event.pass_nSelJets_gt6)
        selections.add("fail_nSelJets_le6", event.fail_nSelJets_le6)
        selections.add("all_selJets", event.all_selJets)

        return selections, allcuts

    def histograms(self, event, selev, weights, analysis_selections, shift_name):
        """Fill nominal ttHbb histograms as well as pass/fail selJets.n > 6 sub-categories."""
        n_selJets = ak.num(selev.selJets) if "selJets" in selev.fields else ak.num(selev.selJet)
        selev["pass_nSelJets_gt6"] = n_selJets > 6
        selev["fail_nSelJets_le6"] = n_selJets <= 6

        if self.classifier_FvT:
            apply_FvT = True
        else:
            apply_FvT = self.apply_FvT

        hist_dict = {}

        if not self.run_systematics:
            # 1. Fill default (all events) ttHbb histograms
            hist_nom = filling_ttHbb_histograms(
                selev,
                self.jcm_model,
                processName=self.processName,
                year=self.year,
                isMC=self.config["isMC"],
                histCuts=self.histCuts,
                apply_FvT=apply_FvT,
                run_SvB=self.run_SvB,
                top_reconstruction=self.top_reconstruction,
                isDataForMixed=self.config['isDataForMixed'],
                event_metadata=event.metadata,
                year_override=self.year_override,
            )

            if not self.plot_ttbar_with_weights or self.processName != "data":
                return hist_nom

            hists = [hist_nom]

            hist_t4 = filling_ttHbb_histograms(
                selev,
                self.jcm_model,
                processName="TTbar4b_from_d3",
                year=self.year,
                isMC=self.config["isMC"],
                histCuts=self.histCuts,
                apply_FvT=apply_FvT,
                run_SvB=self.run_SvB,
                top_reconstruction=self.top_reconstruction,
                isDataForMixed=self.config['isDataForMixed'],
                event_metadata=event.metadata,
                weight_name="weight_d3_to_t4",
                year_override=self.year_override,
            )

            hist_t3 = filling_ttHbb_histograms(
                selev,
                self.jcm_model,
                processName="TTbar3b_from_d3",
                year=self.year,
                isMC=self.config["isMC"],
                histCuts=self.histCuts,
                apply_FvT=apply_FvT,
                run_SvB=self.run_SvB,
                top_reconstruction=self.top_reconstruction,
                isDataForMixed=self.config['isDataForMixed'],
                event_metadata=event.metadata,
                weight_name="weight_d3_to_t3",
                year_override=self.year_override,
            )

            hists.append(hist_t4)
            hists.append(hist_t3)
            return processor.accumulate(hists)

        return hist_dict

# Alias for standard runner entry point
analysis = ttHbbProcessor
