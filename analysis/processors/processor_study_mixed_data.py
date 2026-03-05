import logging

import awkward as ak
import numpy as np

from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor
from src.hist_tools import Collection, Fill
from src.hist_tools.object import Jet

from coffea4bees.hemisphere_mixing.mixing_helpers import assign_mixed_subsamples, update_pseudoTagWeight_of_mixed_data
from coffea4bees.analysis.helpers.candidates_selection import create_cand_jet_dijet_quadjet
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from src.data_formats.root import Chunk


class analysis(HH4bBaseProcessor):
    def __init__(
            self,
            *,
            apply_JCM: bool = True,
            JCM_file: str = "output/mixeddata_cluster/jcm_for_subsampling/jetCombinatoricModel_SB_.txt",
            **kwargs,
    ):
        logging.info(f"\nLoading JCM from file: {JCM_file}, apply_JCM = {apply_JCM}")
        self._mixed_JCM = jetCombinatoricModel(JCM_file) if apply_JCM else None

        # Mixed data does not need standard pseudotag weight computation, SvB scoring,
        # b-tag SF, or FvT; pseudoTagWeight is updated in custom_processing instead.
        kwargs.setdefault("apply_JCM", False)
        kwargs.setdefault("run_SvB", False)
        kwargs.setdefault("apply_btagSF", False)
        kwargs.setdefault("apply_FvT", False)

        super().__init__(**kwargs)

    def process(self, event):
        """Override to capture source chunk metadata before processing."""
        chunk = Chunk.from_coffea_events(event)
        self._source_chunk = {str(chunk.path): [(chunk.entry_start, chunk.entry_stop)]}
        return super().process(event)

    def apply_selection(self, event):
        """Force isSyntheticData so mixed-data picoAODs skip JEC (already calibrated)."""
        self.config["isSyntheticData"] = True
        return super().apply_selection(event)

    def build_candidates(self, selev, weights, list_weight_names, analysis_selections, processOutput):
        """No-op: candidates are built after the fourTag cut in custom_processing."""
        return selev

    def fill_detailed_cutflows(self, selev):
        """No-op: detailed cutflows require dijet/quadjet candidates which are built later."""
        pass

    def custom_processing(self, selev, config, selections, allcuts, nEventTot):
        """Apply fourTag cut, update pseudoTagWeight, build candidates, run mixed-data studies."""

        # fourTag cut
        fourTag_sel = np.full(nEventTot, False)
        fourTag_sel[selections.all(*allcuts)] = selev.fourTag
        selections.add("passFourTag", fourTag_sel)
        allcuts.append("passFourTag")
        selev = selev[selev.fourTag]
        self._cutFlow.fill("passFourTag", selev)

        # Update pseudoTagWeight for mixed data (replaces standard add_pseudotagweights)
        logging.info(f"pseudoTagWeight before update: {selev.pseudoTagWeight[:10]}")
        update_pseudoTagWeight_of_mixed_data(selev, self._mixed_JCM)
        logging.info(f"pseudoTagWeight after update:  {selev.pseudoTagWeight[:10]}")

        # Build candidates (must follow fourTag cut)
        selev = create_cand_jet_dijet_quadjet(selev, isRun3=config["isRun3"])

        # Mixed-data-specific derived quantities
        selev["thrustDeltaPhi"] = (selev.posHemiOld.thrust_phi - selev.newThrustPhi) % (2 * np.pi) - np.pi
        match_dist = ak.concatenate([selev.posHemiNew.match_dist[:, np.newaxis], selev.negHemiNew.match_dist[:, np.newaxis]], axis=1)
        hemis = ak.concatenate([selev.posHemiNew[:, np.newaxis], selev.negHemiNew[:, np.newaxis]], axis=1)
        selev["hemiMatchDist"] = match_dist
        selev["nSelJetOld"] = selev.posHemiNew.nSelJet + selev.negHemiNew.nSelJet

        # Check how often hemispheres from the same event are matched
        same_event = selev.posHemiNew.event == selev.negHemiNew.event
        nSame   = ak.sum(same_event)
        nSameSB = ak.sum(same_event & selev.region.SB)
        nSameSR = ak.sum(same_event & selev.region.SR)
        if ak.any(same_event):
            logging.warning(f"Found same-event hemisphere pairs: {nSame} total, {nSameSB} SB, {nSameSR} SR out of {len(selev)}")
            logging.warning(f"  event: {hemis.event[same_event].tolist()}")
            logging.warning(f"  run:   {hemis.run[same_event].tolist()}")
            logging.warning(f"  lumiBlock: {hemis.luminosityBlock[same_event].tolist()}")
            logging.warning(f"  hemisphereId: {hemis.hemisphereId[same_event].tolist()}")

        # Subsample assignment study
        n_subsamples = 16
        assign_mixed_subsamples(selev, n_subsamples=n_subsamples)
        sub_sample_counts = {}
        for i in range(n_subsamples):
            self._cutFlow.fill(f"pass_mixedSubSample_v{i}", selev[selev[f"pass_mixedSubSample_v{i}"]])
            for j in range(i, n_subsamples):
                sub_sample_counts[f"{i}_{j}"] = ak.sum(selev[f"pass_mixedSubSample_v{i}"] & selev[f"pass_mixedSubSample_v{j}"])

        # Cache for dump_friend_trees
        self._study_metadata = {
            "saved_events_SB": ak.sum(selev.region.SB),
            "saved_events_SR": ak.sum(selev.region.SR),
            "same_hemis":      nSame,
            "same_hemis_SB":   nSameSB,
            "same_hemis_SR":   nSameSR,
            "hemi_pair_SB_event":       hemis.event[selev.region.SB].tolist(),
            "hemi_pair_SB_run":         hemis.run[selev.region.SB].tolist(),
            "hemi_pair_SB_hemisphereId":hemis.hemisphereId[selev.region.SB].tolist(),
            "hemi_pair_SR_event":       hemis.event[selev.region.SR].tolist(),
            "hemi_pair_SR_run":         hemis.run[selev.region.SR].tolist(),
            "hemi_pair_SR_hemisphereId":hemis.hemisphereId[selev.region.SR].tolist(),
            "subsample_counts": sub_sample_counts,
        }

        return selev, selections.all(*allcuts)

    def dump_friend_trees(self, selev, analysis_selections, shift_name):
        """Return mixed-data study metadata. Only runs on nominal shift."""
        if shift_name is not None:
            return {}

        metadata = {
            "total_events": self.nEvent,
            "saved_events": len(selev),
            **self._study_metadata,
        }

        return {
            self.dataset: metadata | {
                "files": [],
                "source": self._source_chunk,
            }
        }

    def histograms(self, event, selev, weights, analysis_selections, shift_name):

        fill = Fill(process=self.processName, year=self.year, weight="weight")

        hist = Collection(
            process=[self.processName],
            year=[self.year],
            tag=["threeTag", "fourTag"],
            region=['SR', 'SB'],
            **dict((s, ...) for s in self.histCuts),
        )

        fill += Jet.plot(("selJets",     "Selected Jets"), "selJet", skip=["deepjet_c"])
        fill += Jet.plot(("selJets_v0",  "Selected Jets"), "selJet", weight="pass_mixedSubSample_v0", skip=["deepjet_c"])
        fill += Jet.plot(("selJets_JCM", "Selected Jets"), "selJet", weight="pseudoTagWeight",        skip=["deepjet_c"])

        skip_all_but_n = ["deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c"]
        fill += Jet.plot(("selJets_noJCM", "Selected Jets"), "selJet", skip=skip_all_but_n)
        fill += Jet.plot(("tagJets_noJCM", "Tag Jets"),      "tagJet", skip=skip_all_but_n)

        fill += hist.add("thrustDeltaPhi", (50, -np.pi, np.pi, ("thrustDeltaPhi", "Thrust Delta Phi")))
        fill += hist.add("matchDist",      (100, 0, 4,          ("hemiMatchDist",  "Match distance")))
        fill += hist.add("selJets.nOld",   (20, 0, 20,          ("nSelJetOld",     "Number of selected jets before mixing")))

        fill(selev, hist)
        return hist.to_dict(nonempty=True)
