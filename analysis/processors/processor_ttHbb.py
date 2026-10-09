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
from coffea4bees.analysis.helpers.truth_tools import label_jet_truth_parent

class ttHbbProcessor(HH4bBaseProcessor):
    """
    Fully decoupled Coffea processor for ttH(bb) analysis workflows.
    Inherits core event/object selection and corrections from HH4bBaseProcessor
    while isolating ttHbb SvB score derivation and histogram filling (skipping HH mass plots).

    pairing selects the candidate pairing: 'nominal' (default, 3 st/pt-sorted lead/subl pairings)
    or 'can_ttH' (6 ordered canH/canTT pairings, see create_cand_jet_dijet_quadjet_ttHbb).

    On ttHbb MC (processName starting with "ttHbb") jets are also truth-labeled (H/t/o),
    quadJet_truth (the truth H-H / t-t pairing) is stored and the truth histograms are filled.
    Truth labels never filter events, so quadJet/quadJet_selected and all other histograms are
    unaffected; quadJet_truth is None for events without a full H,H,t,t candidate-jet match.
    quadJet_truth and its histograms need the canH/canTT slots, so only exist for pairing='can_ttH'.
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
        classify_Z_decay=False,
        pairing="nominal",
        corrections_metadata: dict = None,
        **kwargs,
    ):
        logging.info("Initializing decoupled ttHbbProcessor")
        self.classify_Z_decay = classify_Z_decay
        self.pairing = pairing
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
            corrections_metadata=corrections_metadata,
            **kwargs,
        )

    def _is_ttHbb_MC(self):
        return self.config["isMC"] and self.processName.startswith("ttHbb")

    def apply_selection(self, event):
        """On ttHbb MC, attach truth parent labels to every jet before object selection so selJet/canJet inherit them."""
        if self._is_ttHbb_MC():
            if "GenPart" in event.fields:
                for name, values in label_jet_truth_parent(event.Jet, event.GenPart).items():
                    event["Jet", name] = values
            else:
                event["Jet", "truthParent"] = ak.unflatten(np.full(ak.sum(ak.num(event.Jet)), "o"), ak.num(event.Jet))
        return super().apply_selection(event)

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

    @staticmethod
    def _truth_pairing(diJet, quadJet):
        """Return the truth quadJet pairing per event: the pairing whose canH dijet (lead,
        slot 0) is the 2 truth-H canJets and whose canTT dijet (subl, slot 1) is the 2
        truth-t canJets. Events where canJet.truthParent is not exactly {H, H, t, t} get
        None. quadJet itself (incl. 'selected') is not modified, so quadJet_selected stays
        the algorithm's choice.

        Args:
            diJet: dijet array, shape [event][6 pairings][2 dijets], as built by
                _build_can_ttH_dijets (slot 0 = canH, slot 1 = canTT, unsorted).
                diJet.lead / diJet.subl are the two constituent jets of each dijet.
            quadJet: quadJet array, shape [event][6 pairings], as built by _build_can_ttH_quadjets.

        Returns:
            quadJet_truth: shape [event], option type (None for events with no truth pairing).
        """
        # A dijet is H-H (t-t) if both of its constituent jets are H (t); shape [event][6][2].
        dijet_is_HH = (diJet.lead.truthParent == "H") & (diJet.subl.truthParent == "H")
        dijet_is_tt = (diJet.lead.truthParent == "t") & (diJet.subl.truthParent == "t")

        # The 6 pairings are ordered, so exactly one has H-H in slot 0 and t-t in slot 1.
        pairing_is_truth = dijet_is_HH[:, :, 0] & dijet_is_tt[:, :, 1]
        event_has_truth_pairing = ak.any(pairing_is_truth, axis=1)

        quadJet_truth = quadJet[ak.argmax(pairing_is_truth, axis=1, keepdims=True)][:, 0]
        return ak.mask(quadJet_truth, event_has_truth_pairing)

    def build_candidates(self, selev, weights, list_weight_names, analysis_selections, processOutput):
        """Build canH/canTT di-jet and quad-jet candidates for ttHbb (same algorithm for data
        and MC). On ttHbb MC also store quadJet_truth: the truth H-H (lead) / t-t (subl) pairing,
        for events where canJet.truthParent is exactly {H, H, t, t} (None otherwise).
        """
        selev = create_cand_jet_dijet_quadjet_ttHbb(
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
            pairing=self.pairing,
        )

        if self._is_ttHbb_MC() and self.pairing == "can_ttH":
            ### truth H-H/t-t pairing; quadJet_selected stays the algorithm's choice
            selev["quadJet_truth"] = self._truth_pairing(selev.diJet, selev.quadJet)

        return selev

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

        if self.run_SvB and "pass_ps_min" in selev.fields:
            self.fill_cutflow_with_and_without_trig("pass_ps_min", selev[selev.pass_ps_min])

        if self.run_SvB and "passSvB" in selev.fields:
            self.fill_cutflow_with_and_without_trig("passSvB", selev[selev.passSvB])
            self.fill_cutflow_with_and_without_trig("failSvB", selev[selev.failSvB])

        # TTbar_from_d3 cutflow entries (the FvT-derived ttbar the closure tables subtract), as in
        # HH4bBaseProcessor.fill_detailed_cutflows; without them those entries are left empty
        if self.plot_ttbar_with_weights:
            self._fill_ttbar_detailed_cutflows(selev)

        if self.plot_ttbar_with_MvD_weights:
            self._fill_ttbar_MvD_detailed_cutflows(selev)

    def build_selections(self, event, weights):
        """Build PackedSelection object with all cuts and add selJets.n > 6 categorization."""
        selections, allcuts = super().build_selections(event, weights)

        # Define jet multiplicity pass mask
        n_selJets = ak.num(event.selJets) if "selJets" in event.fields else ak.num(event.selJet)
        event["pass_nSelJets_gt6"] = n_selJets > 6
        event["fail_nSelJets_le6"] = n_selJets <= 6
        event["all_selJets"] = np.full(len(event), True)
        event["passLeptonVeto"] = event.passLeptonVeto if "passLeptonVeto" in event.fields else np.full(len(event), True)

        selections.add("pass_nSelJets_gt6", event.pass_nSelJets_gt6)
        selections.add("fail_nSelJets_le6", event.fail_nSelJets_le6)
        selections.add("all_selJets", event.all_selJets)
        selections.add("passLeptonVeto", event.passLeptonVeto)

        return selections, allcuts

    def histograms(self, event, selev, weights, analysis_selections, shift_name):
        """Fill nominal ttHbb histograms as well as pass selJets.n > 6 sub-category."""
        n_selJets = ak.num(selev.selJets) if "selJets" in selev.fields else ak.num(selev.selJet)
        selev["pass_nSelJets_gt6"] = n_selJets > 6
        selev["fail_nSelJets_le6"] = n_selJets <= 6
        selev["passLeptonVeto"] = selev.passLeptonVeto if "passLeptonVeto" in selev.fields else np.full(len(selev), True)


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
                can_ttH=self.pairing == "can_ttH",
                classify_Z_decay=self.classify_Z_decay,
                truth_pairing_forced=self._is_ttHbb_MC() and self.pairing == "can_ttH",
                subsample_names=getattr(self, "subsample_names", None),
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
                can_ttH=self.pairing == "can_ttH",
                subsample_names=getattr(self, "subsample_names", None),
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
                can_ttH=self.pairing == "can_ttH",
                subsample_names=getattr(self, "subsample_names", None),
            )

            hists.append(hist_t4)
            hists.append(hist_t3)
            return processor.accumulate(hists)

        return hist_dict

# Alias for standard runner entry point
analysis = ttHbbProcessor
