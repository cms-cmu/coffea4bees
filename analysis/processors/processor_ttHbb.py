from __future__ import annotations

import logging
import numpy as np
import awkward as ak
from coffea.analysis_tools import PackedSelection
from coffea import processor
from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor, _UNSET
from coffea4bees.analysis.helpers.filling_histograms import filling_ttHbb_histograms
from coffea4bees.analysis.helpers.SvB_helpers_ttHbb import set_ttHbb_SvB_vars
from coffea4bees.analysis.helpers.candidates_selection_ttHbb import create_cand_jet_dijet_quadjet_ttHbb

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
        if weights is None:
            weights = "coffea4bees/metadata/weights/weights_ttHbb.yml"
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
        """Load SvB scores and derive native ttHbb fields without running HH4b setSvBVars."""
        for k in self.friends:
            if k.startswith("SvB") and not k.startswith("SvB_FeynNet"):
                logging.debug(f"Loading ttHbb SvB friend tree for {k}")
                try:
                    result = self.friends[k].arrays(self.target)
                    if result is not None:
                        event[k] = result
                        set_ttHbb_SvB_vars(k, event)
                except Exception as e:
                    logging.warning(f"Failed loading SvB friend tree {k} in ttHbbProcessor: {e}")

    def build_candidates(self, selev, weights, list_weight_names, analysis_selections, processOutput):
        """Build unconstrained di-jets and quad-jets candidates for ttHbb."""
        return create_cand_jet_dijet_quadjet_ttHbb(
            selev,
            apply_FvT=self.apply_FvT,
            classifier_FvT=self.clf_FvT,
            run_SvB=self.run_SvB,
            run_systematics=self.run_systematics,
            classifier_SvB=self.clf_SvB,
            classifier_SvB_MA=self.clf_SvB_MA,
            classifier_SvB_FeynNet=self.classifier_SvB_FeynNet,
            processOutput=processOutput,
            isRun3=self.config["isRun3"],
            weights=weights,
            list_weight_names=list_weight_names,
            analysis_selections=analysis_selections,
            cand_cfg=self.cand_cfg,
        )

    def fill_detailed_cutflows(self, selev):
        """Fill detailed cutflow histograms after ttHbb candidate building."""
        self.fill_cutflow_with_and_without_trig("passPreSel", selev)
        self.fill_cutflow_with_and_without_trig("passDiJetMass", selev[selev.passDiJetMass])
        self.fill_cutflow_with_and_without_trig("boosted_veto_passPreSel", selev[selev.notInBoostedSel])
        self._cutFlow.fill("boosted_veto_SR", selev[selev.notInBoostedSel & selev["quadJet_selected"].SR])

        selev['passSR'] = selev.passDiJetMass & selev["quadJet_selected"].SR
        self.fill_cutflow_with_and_without_trig("SR", selev[selev.passSR])

        selev['passSB'] = selev.passDiJetMass & selev["quadJet_selected"].SB
        self.fill_cutflow_with_and_without_trig("SB", selev[selev.passSB])

        self._cutFlow.fill("passVBFSel", selev[selev.passVBFSel])

        if self.run_SvB and "passSvB" in selev.fields:
            self.fill_cutflow_with_and_without_trig("passSvB", selev[selev.passSvB])
            self.fill_cutflow_with_and_without_trig("failSvB", selev[selev.failSvB])

    def build_selections(self, event, weights):
        """Build PackedSelection object with all cuts and add selJets.n > 6 categorization."""
        selections, allcuts = super().build_selections(event, weights)

        # Define jet multiplicity pass mask
        n_selJets = ak.num(event.selJets) if "selJets" in event.fields else ak.num(event.selJet)
        event["pass_nSelJets_gt6"] = n_selJets > 6
        event["all_selJets"] = np.full(len(event), True)

        selections.add("pass_nSelJets_gt6", event.pass_nSelJets_gt6)
        selections.add("all_selJets", event.all_selJets)

        return selections, allcuts

    def histograms(self, event, selev, weights, analysis_selections, shift_name):
        """Fill nominal ttHbb histograms as well as pass selJets.n > 6 sub-category."""
        n_selJets = ak.num(selev.selJets) if "selJets" in selev.fields else ak.num(selev.selJet)
        selev["pass_nSelJets_gt6"] = n_selJets > 6
        selev["SR"] = selev.passSR
        selev["SB"] = selev.passSB
        selev["region"] = ak.zip({"SR": selev.passSR, "SB": selev.passSB})
        selev["tag"] = ak.zip({"threeTag": selev.threeTag, "fourTag": selev.fourTag})

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
