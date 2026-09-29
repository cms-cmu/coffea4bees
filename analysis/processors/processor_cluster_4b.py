import yaml
import fsspec
import logging
import numpy as np
import awkward as ak

from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor
from src.hist_tools import Collection, Fill
from src.hist_tools.object import Jet
from src.storage.eos import EOS
from src.data_formats.root import TreeWriter, Chunk

from coffea4bees.analysis.helpers.candidates_selection import cand_jet_selection
from coffea4bees.jet_clustering.clustering_hist_templates import ClusterHists
from coffea4bees.jet_clustering.clustering import cluster_bs
from coffea4bees.jet_clustering.declustering import (
    compute_decluster_variables,
    make_synthetic_event,
    clean_ISR,
    get_list_of_combined_jet_types,
    get_list_of_all_sub_splittings,
    get_splitting_name,
)
from coffea4bees.jet_clustering.splitting_library import build_splitting_library_rows

# Placeholder jet/pileup ID bit written to reclustered (synthetic) jets, which
# carry no detector-level ID. 7 = "passes tight".
_SYNTHETIC_JET_ID_BIT = 7


class analysis(HH4bBaseProcessor):
    # `friends` is an explicit parameter (not folded into **kwargs) because
    # runner.py only injects the per-year friend trees when the processor
    # __init__ signature names `friends` explicitly. Without it the FvT friend
    # is never supplied and the ttbar-subtraction path falls back to a
    # nonexistent sibling FvT.root next to each picoAOD.
    def __init__(self, friends: dict = None, **kwargs):
        self.clustering_pdfs_file = kwargs.pop("clustering_pdfs_file", "coffea4bees/jet_clustering/jet-splitting-PDFs-00-11-01/clustering_pdfs_vs_pT_XXX.yml")
        self.do_declustering      = kwargs.pop("do_declustering", False)

        # Library of real splittings for library-based declustering (declustering_method: library).
        # Written per chunk to <splitting_library_base_path>/<dataset>/ when a base path is given.
        splitting_library_base_path = kwargs.pop("splitting_library_base_path", None)
        self.splitting_library_base = EOS(splitting_library_base_path) if splitting_library_base_path not in (None, "None") else None
        self.splitting_library_carry_fields = list(kwargs.pop("splitting_library_carry_fields", ["btagScore"]))

        kwargs.setdefault("apply_JCM",    False)
        kwargs.setdefault("run_SvB",      False)
        kwargs.setdefault("apply_btagSF", False)

        super().__init__(friends=friends, **kwargs)
        logging.info("\nInitialize cluster 4b Processor")
        logging.info(f"subtract_ttbar_with_weights = {self.subtract_ttbar_with_weights}")
        logging.info(f"splitting_library_base = {self.splitting_library_base}, carry_fields = {self.splitting_library_carry_fields}")

    def process(self, event):
        """Record the chunk so dump_friend_trees can name its splitting-library file."""
        if self.splitting_library_base is not None:
            chunk = Chunk.from_coffea_events(event)
            dataset = event.metadata["dataset"]
            self._splitting_library_path = (
                self.splitting_library_base
                / f"{dataset}/splittingLib_{chunk.uuid}_{chunk.entry_start}_{chunk.entry_stop}.root"
            )
            self._splitting_library_source = {str(chunk.path): [(chunk.entry_start, chunk.entry_stop)]}
            self._splitting_library_rows = None
        return super().process(event)

    def build_candidates(self, selev, weights, list_weight_names, analysis_selections, processOutput):
        """No-op: candidate jets are built in custom_processing from btag-sorted jets."""
        return selev

    def fill_detailed_cutflows(self, selev):
        """No-op: detailed cutflows require dijet/quadjet candidates which are not built here."""
        pass

    def custom_processing(self, selev, config, selections, allcuts, nEventTot):
        logging.info("processor_cluster_4b.custom_processing")

        if self.clustering_pdfs_file != "None":
            clustering_pdfs_file = self.clustering_pdfs_file.replace("XXX", self.year)
            with fsspec.open(clustering_pdfs_file, "r") as f:     # local path or root:// URL
                clustering_pdfs = yaml.safe_load(f)
            logging.info(f"Loaded {len(clustering_pdfs.keys())} PDFs from {clustering_pdfs_file}")
        else:
            clustering_pdfs = None

        #
        #  Make four tag cut
        #
        fourTag_sel = np.full(nEventTot, False)
        fourTag_sel[selections.all(*allcuts)] = selev.fourTag
        selections.add("fourTag", fourTag_sel)
        allcuts.append("fourTag")
        selev = selev[selev.fourTag]

        self._cutFlow.fill("passFourTag", selev)

        selev = cand_jet_selection(selev, cand_cfg=self.cand_cfg)

        #
        # Do the Clustering
        #
        canJet    = selev.canJet
        notCanJet = selev.notCanJet_coffea
        canJet["jet_flavor"]    = "b"
        notCanJet["jet_flavor"] = "j"

        jets_for_clustering = ak.concatenate([canJet, notCanJet], axis=1)
        jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]

        #
        #  To dump the testvectors
        #
        dumpTestVectors = False
        if dumpTestVectors:
            print(f'{chunk}\n\n')
            print(f'{chunk} self.input_jet_pt  = {[jets_for_clustering[iE].pt.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_eta  = {[jets_for_clustering[iE].eta.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_phi  = {[jets_for_clustering[iE].phi.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_mass  = {[jets_for_clustering[iE].mass.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_flavor  = {[jets_for_clustering[iE].jet_flavor.tolist() for iE in range(10)]}')
            print(f'{chunk}\n\n')



        clustered_jets, clustered_splittings = cluster_bs(jets_for_clustering, debug=False)
        compute_decluster_variables(clustered_splittings)

        split_name_flat = [get_splitting_name(i) for i in ak.flatten(clustered_splittings.jet_flavor)]
        split_name = ak.unflatten(split_name_flat, ak.num(clustered_splittings))
        clustered_splittings["splitting_name"] = split_name

        clustered_jets = clean_ISR(clustered_jets, clustered_splittings)

        if self.splitting_library_base is not None:
            self._splitting_library_rows = build_splitting_library_rows(
                selev, jets_for_clustering, clustered_splittings, clustered_jets,
                carry_fields=self.splitting_library_carry_fields,
            )

        cleaned_combined_jet_flavors = get_list_of_combined_jet_types(clustered_jets)
        cleaned_split_jet_flavors = []
        for _s in cleaned_combined_jet_flavors:
            cleaned_split_jet_flavors += get_list_of_all_sub_splittings(_s)

        cleaned_splitting_name = [get_splitting_name(i) for i in cleaned_split_jet_flavors]
        self.cleaned_splitting_name = set(cleaned_splitting_name)

        for _s_type in cleaned_splitting_name:
            selev[f"splitting_{_s_type}"] = clustered_splittings[clustered_splittings.splitting_name == _s_type]

        #
        #  Declustering
        #
        if self.do_declustering:
            declustered_jets = make_synthetic_event(clustered_jets, clustering_pdfs)
            declustered_jets = declustered_jets[ak.argsort(declustered_jets.pt, axis=1, ascending=False)]

            is_b_mask = declustered_jets.jet_flavor == "b"
            canJet_re    = declustered_jets[is_b_mask]
            notCanJet_re = declustered_jets[~is_b_mask]

            canJet_re["puId"]    = _SYNTHETIC_JET_ID_BIT
            canJet_re["jetId"]   = _SYNTHETIC_JET_ID_BIT
            notCanJet_re["puId"]  = _SYNTHETIC_JET_ID_BIT
            notCanJet_re["jetId"] = _SYNTHETIC_JET_ID_BIT

            selev["canJet_re"]           = canJet_re
            selev["notCanJet_coffea_re"] = notCanJet_re

            #
            #  Recluster
            #
            jets_for_clustering = ak.concatenate([canJet_re, notCanJet_re], axis=1)
            jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]

            clustered_jets_reclustered, clustered_splittings_reclustered = cluster_bs(jets_for_clustering, debug=False)
            compute_decluster_variables(clustered_splittings_reclustered)

            for _s_type in cleaned_splitting_name:
                selev[f"splitting_{_s_type}_re"] = clustered_splittings_reclustered[clustered_splittings_reclustered.jet_flavor == _s_type]

        # Hack for plotting
        selev["region"] = ak.zip({"SR": selev.fourTag})

        return selev, selections.all(*allcuts)


    def dump_friend_trees(self, selev, analysis_selections, shift_name):
        """Write this chunk's splitting-library rows (nominal only). The returned
        {dataset: {files, source, ...}} entry has the same shape as the hemisphere library's, so
        workflows/scripts/regroup_hemi_library.py regroups it into a {year: [files]} registry."""
        result = super().dump_friend_trees(selev, analysis_selections, shift_name)
        if self.splitting_library_base is None or shift_name is not None or self._splitting_library_rows is None:
            return result

        rows = self._splitting_library_rows
        entry = {
            "total_events": self.nEvent,
            "saved_events": len(selev),
            "saved_splittings": len(rows),
            "files": [],
            "source": self._splitting_library_source,
        }
        if len(rows):
            with TreeWriter()(self._splitting_library_path) as writer:
                writer.extend(rows)
            entry["files"].append(self._splitting_library_path)

        return result | {self.dataset: entry}

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

        for _s_type in self.cleaned_splitting_name:
            fill += ClusterHists((f"splitting_{_s_type}", f"{_s_type} Splitting"), f"splitting_{_s_type}")

        if self.do_declustering:
            for _s_type in self.cleaned_splitting_name:
                fill += ClusterHists((f"splitting_{_s_type}_re", f"{_s_type} Splitting"), f"splitting_{_s_type}_re")

        fill(selev, hist)

        return hist.to_dict(nonempty=True)
