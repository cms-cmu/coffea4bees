import uuid
import logging
import numpy as np
import awkward as ak

from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor
from src.hist_tools import Collection, Fill
from src.hist_tools.object import Jet
from src.storage.eos import EOS, PathLike
from src.data_formats.root import TreeWriter, TreeReader, Chunk

from coffea4bees.analysis.helpers.candidates_selection import cand_jet_selection
from coffea4bees.hemisphere_mixing.mixing_helpers import split_events_into_hemispheres
from coffea4bees.hemisphere_mixing.hemisphere_hist_templates import HemisphereHists

from ..helpers.load_friend import FriendTemplate

_ROOT = ".root"


class analysis(HH4bBaseProcessor):
    def __init__(
            self,
            base_path: PathLike,
            *,
            campaign: str = ...,
            **kwargs,
    ):
        logging.debug("\nInitialize  processor_make_hemi_library\n")

        self._base = EOS(base_path)

        if campaign is ...:
            campaign = f"mixing-{uuid.uuid4().hex[:8]}"
        if campaign is not None:
            logging.info(f"Using campaign name: {campaign}")
        self._campaign = campaign

        # Defaults appropriate for hemisphere library building:
        # - No JCM weighting needed
        # - No SvB scoring needed
        # - No b-tag SF needed
        # - FvT loaded only if apply_FvT=True is passed (needed for subtract_ttbar_with_weights)
        kwargs.setdefault("apply_JCM", False)
        kwargs.setdefault("run_SvB", False)
        kwargs.setdefault("apply_FvT", False)
        kwargs.setdefault("apply_btagSF", False)

        super().__init__(**kwargs)

    def process(self, event):
        """Override to add chunk-level caching check before processing."""
        chunk = Chunk.from_coffea_events(event)
        dataset = event.metadata['dataset']
        source_chunk = {str(chunk.path): [(chunk.entry_start, chunk.entry_stop)]}

        path = (
            self._base
            / f"{dataset}/hemisphereLib_{chunk.uuid}_{chunk.entry_start}_{chunk.entry_stop}{_ROOT}"
        )

        # Return cached result if this chunk was already processed
        if self._campaign is not None:
            reader = TreeReader()
            try:
                cached = Chunk(path, fetch=True)
                metadata = reader.load_metadata(self._campaign, cached, builtin_types=True)
                return {dataset: metadata | {"files": [cached], "source": source_chunk}}
            except Exception:
                pass

        # Store for use in dump_friend_trees()
        self._hemi_path = path
        self._hemi_source_chunk = source_chunk

        return super().process(event)

    def build_candidates(self, selev, weights, list_weight_names, analysis_selections, processOutput):
        """No-op: hemisphere library does not need dijet/quadjet candidates."""
        return selev

    def custom_processing(self, selev, config, selections, allcuts, nEventTot):
        """Apply fourTag cut, build canJet fields, and split into hemispheres."""

        # fourTag cut (same pattern as processor_cluster_4b)
        fourTag_sel = np.full(nEventTot, False)
        fourTag_sel[selections.all(*allcuts)] = selev.fourTag
        selections.add("passFourTag", fourTag_sel)
        allcuts.append("passFourTag")
        selev = selev[selev.fourTag]
        self._cutFlow.fill("passFourTag", selev)

        selev = cand_jet_selection(selev, cand_cfg=self.cand_cfg)

        # Split events into hemispheres
        pos_hemi, neg_hemi = split_events_into_hemispheres(selev)
        selev["pos_hemi"] = pos_hemi
        selev["neg_hemi"] = neg_hemi

        selev["region"] = ak.zip({"SR": selev.fourTag})

        return selev, selections.all(*allcuts)

    def fill_detailed_cutflows(self, selev):
        """No-op: detailed cutflows require dijet/quadjet candidates which are not built here."""
        pass

    def dump_friend_trees(self, selev, analysis_selections, shift_name):
        """Write hemisphere library TreeWriter output. Only runs on nominal shift."""
        if shift_name is not None:
            return {}

        pos_hemi = selev.pos_hemi
        neg_hemi = selev.neg_hemi
        metadata = {
            "total_events": self.nEvent,
            "saved_events": len(selev),
            "saved_hemis": len(pos_hemi.thrust_phi) + len(neg_hemi.thrust_phi),
        }

        result = {
            self.dataset: metadata | {
                "files": [],
                "source": self._hemi_source_chunk,
            }
        }

        with TreeWriter()(self._hemi_path) as writer:
            writer.extend(selev.pos_hemi).extend(selev.neg_hemi)
            if self._campaign is not None:
                writer.save_metadata(self._campaign, metadata)
            result[self.dataset]["files"].append(self._hemi_path)

        return result

    def histograms(self, event, selev, weights, analysis_selections, shift_name):

        fill = Fill(process=self.processName, year=self.year, weight="weight")

        hist = Collection(
            process=[self.processName],
            year=[self.year],
            tag=["threeTag", "fourTag"],
            region=['SR'],
            **dict((s, ...) for s in self.histCuts),
        )

        fill += Jet.plot(("selJets", "Selected Jets"), "selJet", skip=["deepjet_c"])

        for iJ in range(4):
            fill += Jet.plot((f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"])

        fill += HemisphereHists(("pos_hemis", "Hemispheres"), "pos_hemi")
        fill += HemisphereHists(("neg_hemis", "Hemispheres"), "neg_hemi")

        fill(selev, hist)

        return hist.to_dict(nonempty=True)
