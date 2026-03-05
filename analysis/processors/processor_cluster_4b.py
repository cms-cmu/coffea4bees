import yaml
import logging
import numpy as np
import awkward as ak

from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor
from src.hist_tools import Collection, Fill
from src.hist_tools.object import Jet

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


class analysis(HH4bBaseProcessor):
    def __init__(self, **kwargs):
        self.clustering_pdfs_file = kwargs.pop("clustering_pdfs_file", "jet_clustering/jet-splitting-PDFs-00-07-02/clustering_pdfs_vs_pT_XXX.yml")
        self.do_declustering      = kwargs.pop("do_declustering", False)

        kwargs.setdefault("apply_JCM",    False)
        kwargs.setdefault("run_SvB",      False)
        kwargs.setdefault("apply_btagSF", False)

        super().__init__(**kwargs)
        logging.info("\nInitialize cluster 4b Processor")
        logging.info(f"subtract_ttbar_with_weights = {self.subtract_ttbar_with_weights}")

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
            clustering_pdfs = yaml.safe_load(open(clustering_pdfs_file, "r"))
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

            canJet_re["puId"]    = 7
            canJet_re["jetId"]   = 7
            notCanJet_re["puId"]  = 7
            notCanJet_re["jetId"] = 7

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
