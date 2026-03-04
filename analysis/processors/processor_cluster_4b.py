from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor
import numpy as np
import awkward as ak
from src.hist_tools import Collection, Fill
from src.hist_tools.object import LorentzVector, Jet, Muon, Elec
import logging

from coffea4bees.jet_clustering.clustering_hist_templates import ClusterHists, ClusterHistsDetailed
from coffea4bees.jet_clustering.clustering   import cluster_bs, cluster_bs_fast
from coffea4bees.jet_clustering.declustering import compute_decluster_variables, make_synthetic_event, get_list_of_splitting_types, clean_ISR, get_list_of_ISR_splittings, get_list_of_combined_jet_types, get_list_of_all_sub_splittings, get_splitting_name, get_list_of_splitting_names


class analysis(HH4bBaseProcessor):
    def __init__(
            self,
            **kwargs  # Accept additional arguments to pass to parent
    ):

        # Initialize parent without JCM (we'll handle it ourselves)
        print(f"kwargs was",kwargs.keys())
        self.clustering_pdfs_file = kwargs.pop("clustering_pdfs_file","jet_clustering/jet-splitting-PDFs-00-07-02/clustering_pdfs_vs_pT_XXX.yml")
        self.do_declustering      = kwargs.pop("do_declustering", False)
        print(f"kwargs is",kwargs.keys())
        super().__init__(**kwargs)
        logging.info("\nInitialize cluster 4b Processor")
        logging.info(f"subtract_ttbar_with_weights = {self.subtract_ttbar_with_weights}")


    def custom_processing(self, selev, config, selections, allcuts, nEventTot):
        logging.info(f"processor_cluster_4b.custom_processing")

        clustering_pdfs_file = self.clustering_pdfs_file.replace("XXX", self.year)

        if not self.clustering_pdfs_file == "None":
            clustering_pdfs = yaml.safe_load(open(self.clustering_pdfs_file, "r"))
            logging.info(f"Loaded {len(clustering_pdfs.keys())} PDFs from {self.clustering_pdfs_file}")
        else:
            clustering_pdfs = None


        #
        #  Make four tag cut
        #
        fourTag_sel = np.full(nEventTot, False)
        fourTag_sel[selections.all(*allcuts)] = selev.fourTag
        selections.add("fourTag", fourTag_sel)
        allcuts.append("fourTag")
        analysis_selections = selections.all(*allcuts)
        selev = selev[selev.fourTag]


        self._cutFlow.fill("passFourTag", selev )


        #
        # Build and select boson candidate jets with bRegCorr applied
        #
        sorted_idx = ak.argsort( selev.Jet.btagScore * selev.Jet.selected, axis=1, ascending=False )
        canJet_idx = sorted_idx[:, 0:4]
        notCanJet_idx = sorted_idx[:, 4:]

        # apply bJES to canJets
        canJet = selev.Jet[canJet_idx] * selev.Jet[canJet_idx].bRegCorr
        canJet["bRegCorr"] = selev.Jet.bRegCorr[canJet_idx]
        canJet["btagScore"] = selev.Jet.btagScore[canJet_idx]
        canJet["puId"] = selev.Jet.puId[canJet_idx]
        canJet["jetId"] = selev.Jet.puId[canJet_idx]
        if config["isMC"]:
            canJet["hadronFlavour"] = selev.Jet.hadronFlavour[canJet_idx]

        #
        # pt sort canJets
        #
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
        selev["canJet"] = canJet
        for i in range(4):
            selev[f"canJet{i}"] = selev["canJet"][:, i]


        # print(selev.v4j.n)
        # selev['Jet', 'canJet'] = False
        notCanJet = selev.Jet[notCanJet_idx]
        notCanJet = notCanJet[notCanJet.selected_loose]
        notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

        notCanJet["isSelJet"] = 1 * ( (notCanJet.pt > 40) & (np.abs(notCanJet.eta) < 2.4) )  # should have been defined as notCanJet.pt>=40, too late to fix this now...
        selev["notCanJet_coffea"] = notCanJet

        #
        # Do the Clustering
        #
        canJet["jet_flavor"] = "b"
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


        #clustered_jets, clustered_splittings = cluster_bs_fast(jets_for_clustering, debug=False)
        clustered_jets, clustered_splittings = cluster_bs(jets_for_clustering, debug=False)
        compute_decluster_variables(clustered_splittings)


        #
        #  add split name (can probably do this when making the splitting
        #
        split_name_flat = [get_splitting_name(i) for i in ak.flatten(clustered_splittings.jet_flavor)]
        split_name = ak.unflatten(split_name_flat, ak.num(clustered_splittings))
        clustered_splittings["splitting_name"] = split_name


        #
        #  get all splitting types that are used (ie: not pure ISR)
        #
        clustered_jets = clean_ISR(clustered_jets, clustered_splittings)

        cleaned_combined_jet_flavors = get_list_of_combined_jet_types(clustered_jets)
        cleaned_split_jet_flavors = []
        for _s in cleaned_combined_jet_flavors:
            cleaned_split_jet_flavors += get_list_of_all_sub_splittings(_s)

        #
        # Convert to list of cleaned splitting names
        #
        cleaned_splitting_name = [get_splitting_name(i) for i in cleaned_split_jet_flavors]
        self.cleaned_splitting_name = set(cleaned_splitting_name)

        #
        # Sort clusterings by type
        #
        for _s_type in cleaned_splitting_name:
            selev[f"splitting_{_s_type}"]   = clustered_splittings[clustered_splittings.splitting_name == _s_type]

        #print(f'{chunk} cleaned splitting types {cleaned_split_types}\n')

        # error_type = '(bj)((jj)b)'
        # found_error = error_type in cleaned_split_types
        #
        # if found_error:
        #     print(f"ERROR have splitting type {error_type}\n" )
        #
        #     error_mask = clustered_splittings.jet_flavor == error_type
        #     event_mask = ak.any(error_mask,axis=1 )
        #
        #     # print(f'{chunk} num splitting {ak.num(selev["splitting_b(bj)"])}')
        #     # print(f'{chunk} mask {ak.num(selev["splitting_b(bj)"]) > 0}')
        #     #bbj_mask = ak.num(selev["splitting_b(bj)"]) > 0
        #     jets_for_clustering_error = jets_for_clustering[event_mask]
        #     n_jets_error = len(jets_for_clustering_error)
        #     print(f'{chunk}\n\n')
        #     print(f'{chunk} self.input_jet_pt      = {[jets_for_clustering_error[iE].pt.tolist()         for iE in range(n_jets_error)]}')
        #     print(f'{chunk} self.input_jet_eta     = {[jets_for_clustering_error[iE].eta.tolist()        for iE in range(n_jets_error)]}')
        #     print(f'{chunk} self.input_jet_phi     = {[jets_for_clustering_error[iE].phi.tolist()        for iE in range(n_jets_error)]}')
        #     print(f'{chunk} self.input_jet_mass    = {[jets_for_clustering_error[iE].mass.tolist()       for iE in range(n_jets_error)]}')
        #     print(f'{chunk} self.input_jet_flavor  = {[jets_for_clustering_error[iE].jet_flavor.tolist() for iE in range(n_jets_error)]}')
        #     print(f'{chunk}\n\n')


        # from coffea4bees.jet_clustering.dumpTestVectors   import dumpTestVectors_bbj
        # dumpTestVectors_bbj(chunk, selev, jets_for_clustering)

        #
        # writing out bb splitting for Chris Berman
        #
        # out_data = {}
        # out_data["pt_comb"]  = ak.flatten(selev["splitting_bb"].pt)
        # out_data["eta_comb"] = ak.flatten(selev["splitting_bb"].eta)
        # out_data["zA"] = ak.flatten(selev["splitting_bb"].zA)
        # out_data["thetaA"] = ak.flatten(selev["splitting_bb"].thetaA)
        # out_data["mA"] = ak.flatten(selev["splitting_bb"].mA)
        # out_data["mB"] = ak.flatten(selev["splitting_bb"].mB)
        # out_data["decay_phi"] = ak.flatten(selev["splitting_bb"].decay_phi)
        #
        # for out_k, out_v in out_data.items():
        #     processOutput[out_k] = {}
        #     processOutput[out_k][event.metadata['dataset']] = list(out_v)


        #
        #  Declustering
        #
        if self.do_declustering:

            # clustered_jets = clean_ISR(clustered_jets, clustered_splittings)

            #
            # Declustering
            #

            #
            #  Read in the pdfs
            #

            declustered_jets = make_synthetic_event(clustered_jets, clustering_pdfs)

            declustered_jets = declustered_jets[ak.argsort(declustered_jets.pt, axis=1, ascending=False)]

            is_b_mask = declustered_jets.jet_flavor == "b"
            canJet_re = declustered_jets[is_b_mask]

            canJet_re["puId"] = 7
            canJet_re["jetId"] = 7 # selev.Jet.puId[canJet_idx]


            notCanJet_re = declustered_jets[~is_b_mask]
            notCanJet_re["puId"] = 7
            notCanJet_re["jetId"] = 7 # selev.Jet.puId[canJet_idx]

            selev["canJet_re"] = canJet_re
            selev["notCanJet_coffea_re"] = notCanJet_re

            #
            #  Recluster
            #
            jets_for_clustering = ak.concatenate([canJet_re, notCanJet_re], axis=1)
            jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]

            clustered_jets_reclustered, clustered_splittings_reclustered = cluster_bs(jets_for_clustering, debug=False)
            compute_decluster_variables(clustered_splittings_reclustered)

            # all_split_types_re = get_list_of_splitting_types(clustered_splittings_reclustered)
            # # ISR_splittings_re  = get_list_of_ISR_splittings(all_split_types_re)
            # ISR_splittings_re = [] # Hack Save all splitting for now
            # all_split_types_re = [item for item in all_split_types_re if item not in ISR_splittings_re]

            for _s_type in cleaned_splitting_name:
                selev[f"splitting_{_s_type}_re"]  = clustered_splittings_reclustered[clustered_splittings_reclustered.jet_flavor == _s_type]

            # print(f'{chunk} all splitting_re types {all_split_types_re}\n')

            debug_bbj = False
            if debug_bbj:
                bbj_mask = ak.num(selev["splitting_b(bj)_re"]) > 0
                #bbj_partA = selev["splitting_b(bj)_re"][bbj_mask].part_A

                selev_bbjj = selev[bbj_mask]

                bbj_partB_large_mass = selev_bbjj["splitting_b(bj)_re"].part_B.mass > 50
                print(f'{chunk} mass {selev_bbjj["splitting_b(bj)_re"].part_B.mass}')
                print(f'{chunk} have large {bbj_partB_large_mass}')
                print(f'{chunk} any {ak.any(bbj_partB_large_mass, axis=1)}')

                large_bbj_mb_event_mask = ak.any(bbj_partB_large_mass, axis=1)

                selev_large_bbj = selev_bbjj[large_bbj_mb_event_mask]

                print(f'{chunk} partB mass {selev_large_bbj["splitting_b(bj)_re"].part_B.mass}\n')
                print(f'{chunk} partB flav {selev_large_bbj["splitting_b(bj)_re"].part_B.jet_flavor}\n')
                print(f'{chunk} partB pt {selev_large_bbj["splitting_b(bj)_re"].part_B.pt}\n')
                print(f'{chunk} partB eta {selev_large_bbj["splitting_b(bj)_re"].part_B.eta}\n')


                print(f'{chunk} partA mass {selev_large_bbj["splitting_b(bj)_re"].part_A.mass}\n')
                print(f'{chunk} partA falv {selev_large_bbj["splitting_b(bj)_re"].part_A.jet_flavor}\n')
                print(f'{chunk} partA pt {selev_large_bbj["splitting_b(bj)_re"].part_A.pt}\n')
                print(f'{chunk} partA eta {selev_large_bbj["splitting_b(bj)_re"].part_A.eta}\n')

            dumpTestVectors = False
            if dumpTestVectors:
                print(f'{chunk}\n\n')
                print(f'{chunk} self.input_jet_pt  = {[jets_for_clustering[iE].pt.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_eta  = {[jets_for_clustering[iE].eta.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_phi  = {[jets_for_clustering[iE].phi.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_mass  = {[jets_for_clustering[iE].mass.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_flavor  = {[jets_for_clustering[iE].jet_flavor.tolist() for iE in range(10)]}')
                print(f'{chunk}\n\n')



        # Hack for plotting
        selev["region"] = ak.zip({"SR": selev.fourTag})

        #self._cutFlow.fill("pass0OthJets",selev )
        #self._cutFlow.fill("pass1OthJets",selev )
        #self._cutFlow.fill("pass2OthJets",selev )
        return selev, selections.all(*allcuts)

    #
    # Hists
    #
    def histograms(self, event, selev, weights, analysis_selections, shift_name):

        fill = Fill(process=self.processName, year=self.year, weight="weight")

        hist = Collection( process=[self.processName],
                           year=[self.year],
                           tag=["threeTag", "fourTag"],  # 3 / 4/ Other
                           region=['SR'],  # SR / SB / Other
                           **dict((s, ...) for s in self.histCuts)
                           )

        #
        # Jets
        #
        fill += Jet.plot(("selJets", "Selected Jets"),        "selJet",           skip=["deepjet_c"])
        # fill += Jet.plot(("canJets", "Higgs Candidate Jets"), "canJet",           skip=["deepjet_c"])
        # fill += Jet.plot(("othJets", "Other Jets"),           "notCanJet_coffea", skip=["deepjet_c"])
        # fill += Jet.plot(("tagJets", "Tag Jets"),             "tagJet",           skip=["deepjet_c"])

        # fill += Jet.plot(("notCanJet_sel", "Higgs Candidate Jets"), "notCanJet_sel",           skip=["deepjet_c"])
        # if self.do_declustering:
        #     fill += Jet.plot(("canJets_re", "Higgs Candidate Jets"), "canJet_re",           skip=["deepjet_c"])
        #     fill += Jet.plot(("notCanJet_sel_re", "Higgs Candidate Jets"), "notCanJet_sel_re",           skip=["deepjet_c"])

        #
        #  Make Jet Hists
        #
        skip_all_but_n = ["deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c", ]

        for iJ in range(4):
            fill += Jet.plot( (f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"], )


        for _s_type in self.cleaned_splitting_name:
            fill += ClusterHists( (f"splitting_{_s_type}", f"{_s_type} Splitting"), f"splitting_{_s_type}" )

            # if _s_type in ["1b0j/1b0j", "1b0j/0b1j", "0b1j/0b1j", "1b1j/1b0j"]:
            #     fill += ClusterHistsDetailed( (f"detailed_splitting_{_s_type}",    f"{_s_type} Splitting"),    f"splitting_{_s_type}"    )


        if self.do_declustering:
            for _s_type in self.cleaned_splitting_name:
                fill += ClusterHists( (f"splitting_{_s_type}_re", f"${_s_type} Splitting"), f"splitting_{_s_type}_re" )


        #
        # fill histograms
        #
        fill(selev, hist)

        return hist.to_dict(nonempty=True)


    def postprocess(self, accumulator):
        return accumulator
