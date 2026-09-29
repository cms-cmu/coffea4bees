import yaml
import fsspec
from coffea4bees.skimmer.processor.skimmer_4b_base import Skimmer4b
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea4bees.analysis.helpers.candidates_selection import cand_jet_selection
from src.compat import nano_from_root

from coffea4bees.jet_clustering.clustering   import cluster_bs
from coffea4bees.jet_clustering.declustering import make_synthetic_event, clean_ISR
from coffea4bees.jet_clustering.splitting_library import SplittingLibrary, carry_child_fields
from coffea4bees.analysis.helpers.SvB_helpers import setFvTVars, subtract_ttbar_with_FvT
from coffea4bees.analysis.helpers.object_selection import resolve_object_selection_config

# Placeholder values written to the synthetic picoAOD jet branches. The
# declustered jets have no detector-level ID/regression info, so we write a
# passing jet/pileup ID bit and unit regression/PNet correction factors.
_SYNTHETIC_JET_ID_BIT = 7   # Jet_jetId / Jet_puId "passes tight" bitmask
_UNIT_CORRECTION      = 1    # bRegCorr / PNetRegPtRawCorr(+Neutrino) = no correction

# Fallback declustering thresholds if neither the config nor the object-selection
# config supplies them. b-jet pT floor tracks the selected-jet pt_min (Run3 2022 /
# Run2); dr_threshold is the minimum angular separation of declustered jets.
_DEFAULT_B_PT_THRESHOLD_RUN3 = 30
_DEFAULT_B_PT_THRESHOLD_RUN2 = 40
_DEFAULT_DR_THRESHOLD        = 0.4
_DEFAULT_MAX_RETRY           = 8

from src.friendtrees.FriendTreeSchema import FriendTreeSchema
from src.math_tools.random import Squares
from coffea4bees.analysis.helpers.event_weights import add_btagweights
from src.physics.event_selection import apply_event_selection
from coffea4bees.analysis.helpers.event_weights import add_weights

from src.data_formats.root import Chunk, TreeReader
from coffea4bees.analysis.helpers.load_friend import (
    FriendTemplate,
    rename_FvT_friend,
)

from coffea.analysis_tools import Weights, PackedSelection
import numpy as np
from src.physics.objects.jet_corrections import apply_jerc_corrections_jsonpog
from src.physics.common import update_events
from copy import copy
import logging
import awkward as ak
import uproot

class DeClusterer(Skimmer4b):
    def __init__(self, clustering_pdfs_file = "None",
                subtract_ttbar_with_weights = False,
                declustering_rand_seed=5,
                b_pt_threshold=None,
                dr_threshold=_DEFAULT_DR_THRESHOLD,
                max_jet_retry=_DEFAULT_MAX_RETRY,
                max_event_retry=_DEFAULT_MAX_RETRY,
                friends: dict[str, str|FriendTemplate] = None,
                corrections_metadata: dict = None,
                object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
                declustering_method: str = "pdf",
                clustering_library_file: str = "None",
                library_carry_fields: list = ("btagScore",),
                library_min_entries: int = 10,
                library_scale_pt: bool = True,
                library_boost_z: bool = True,
                library_selection: str = "random",
                library_k_neighbors: int = 5,
                *args, **kwargs):
        # declustering_method "pdf" (default): sample the splittings from clustering_pdfs_file.
        # "library": replace them by real splittings from clustering_library_file (a {year: [files]}
        # registry written by processor_cluster_4b; XXX -> year). library_selection "random"
        # (default): each jet takes a reproducible random one of its library_k_neighbors nearest,
        # keyed on declustering_rand_seed + the jet kinematics, so seeds are equivalent replicas;
        # "rank": neighbour rank declustering_rand_seed. See jet_clustering/splitting_library.py.
        if declustering_method not in ("pdf", "library"):
            raise ValueError(f"declustering_method must be 'pdf' or 'library', got {declustering_method!r}")
        self.declustering_method = declustering_method
        self.clustering_library_file = clustering_library_file
        self.library_carry_fields = list(library_carry_fields)
        if "btagScore" not in self.library_carry_fields:
            self.library_carry_fields.insert(0, "btagScore")   # the output b-tag score comes from here
        self.library_min_entries = library_min_entries
        self.library_scale_pt = library_scale_pt
        self.library_boost_z = library_boost_z
        if library_selection not in ("random", "rank"):
            raise ValueError(f"library_selection must be 'random' or 'rank', got {library_selection!r}")
        self.library_selection = library_selection
        self.library_k_neighbors = int(library_k_neighbors)
        self._splitting_library_cache = {}

        pico_tag = "lib_" if declustering_method == "library" else ""
        kwargs["pico_base_name"] = f'picoAOD_{pico_tag}seed{declustering_rand_seed}'
        super().__init__(
            corrections_metadata=corrections_metadata,
            object_selection_cfg=object_selection_cfg,
            friends=friends,
            *args, **kwargs,
        )

        logging.info(f"\nRunning Declusterer with these parameters: declustering_method = {declustering_method}, clustering_library_file = {clustering_library_file}, library_carry_fields = {self.library_carry_fields}, library_min_entries = {library_min_entries}, library_scale_pt = {library_scale_pt}, library_boost_z = {library_boost_z}, library_selection = {library_selection}, library_k_neighbors = {library_k_neighbors}, clustering_pdfs_file = {clustering_pdfs_file}, subtract_ttbar_with_weights = {subtract_ttbar_with_weights}, declustering_rand_seed = {declustering_rand_seed}, b_pt_threshold = {b_pt_threshold}, dr_threshold = {dr_threshold}, max_jet_retry = {max_jet_retry}, max_event_retry = {max_event_retry}, args = {args}, kwargs = {kwargs}")
        self.clustering_pdfs_file = clustering_pdfs_file

        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights
        self.declustering_rand_seed = declustering_rand_seed
        # b_pt_threshold=None -> derive per-year from the selected-jet pt_min in
        # the object-selection config (era-aware; e.g. 25 GeV for 2023). An
        # explicit value here overrides that (escape hatch for studies).
        self.b_pt_threshold  = b_pt_threshold
        self.dr_threshold    = dr_threshold
        self.max_jet_retry   = max_jet_retry
        self.max_event_retry = max_event_retry

        self.skip_collections = kwargs["skip_collections"]
        self.skip_branches    = kwargs["skip_branches"]
        self._clustering_pdfs_cache = {}

    def _resolve_b_pt_threshold(self, year, isRun3):
        """The declustered b-jet pT floor. Defaults to the selected-jet pt_min
        from the object-selection config (era-resolved, so 2023 gets 25 GeV),
        keeping the declustering floor in sync with the analysis selection. An
        explicit ``b_pt_threshold`` in the config overrides."""
        if self.b_pt_threshold is not None:
            return self.b_pt_threshold
        if not self.sel_cfg:
            return _DEFAULT_B_PT_THRESHOLD_RUN3 if isRun3 else _DEFAULT_B_PT_THRESHOLD_RUN2
        jet_cfg = resolve_object_selection_config(self.sel_cfg, year).get('jet', {})
        if isRun3:
            return jet_cfg.get('run3', {}).get('selected', {}).get(
                'pt_min', _DEFAULT_B_PT_THRESHOLD_RUN3)
        return jet_cfg.get('run2', {}).get('default', {}).get('selected', {}).get(
            'pt_min', _DEFAULT_B_PT_THRESHOLD_RUN2)


    def select(self, event):
        m = self._parse_event_metadata(event)
        year, dataset, fname, estart, estop = m.year, m.dataset, m.fname, m.estart, m.estop
        nEvent, year_label, chunk, processName, config = m.nEvent, m.year_label, m.chunk, m.processName, m.config
        logging.debug(f'{chunk} config={config}, for file {fname}\n')

        ### target is for new friend trees
        target = Chunk.from_coffea_events(event)

        clustering_pdfs_file = self.clustering_pdfs_file.replace("XXX", year)

        if clustering_pdfs_file != "None":
            if clustering_pdfs_file not in self._clustering_pdfs_cache:
                # fsspec, not open(): this runs in the condor workers, and a roast publishes the
                # PDFs to EOS (root://...) rather than into the checkout the workers are shipped.
                with fsspec.open(clustering_pdfs_file, "r") as f:
                    self._clustering_pdfs_cache[clustering_pdfs_file] = yaml.safe_load(f)
                logging.info(f"Loaded {len(self._clustering_pdfs_cache[clustering_pdfs_file].keys())} PDFs from {clustering_pdfs_file}\n")
            clustering_pdfs = self._clustering_pdfs_cache[clustering_pdfs_file]
        else:
            clustering_pdfs = None

        splitting_library = None
        if self.declustering_method == "library":
            library_file = self.clustering_library_file.replace("XXX", year)
            if (library_file, year) not in self._splitting_library_cache:
                self._splitting_library_cache[(library_file, year)] = SplittingLibrary.from_files(
                    library_file, year, carry_fields=self.library_carry_fields, min_entries=self.library_min_entries)
                logging.info(f"Loaded {len(self._splitting_library_cache[(library_file, year)].flavor)} splittings for {year} from {library_file}\n")
            splitting_library = self._splitting_library_cache[(library_file, year)]

        path = fname.replace(fname.split("/")[-1], "")

        if self.subtract_ttbar_with_weights:

            if "FvT" in self.friends:
                event["FvT"] = rename_FvT_friend(target, self.friends["FvT"])
            else:

                FvT_file = f'{fname.replace("picoAOD", "FvT")}'
                event["FvT"] = ( nano_from_root( {FvT_file: "Events"},
                                                              entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().FvT )

                if not ak.all(event.FvT.event == event.event):
                    raise ValueError("ERROR: FvT events do not match events ttree")

            setFvTVars("FvT", event)

        event = apply_event_selection( event, self.corrections_metadata[year], cut_on_lumimask=config["cut_on_lumimask"] )


        ## adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights( event, dataset, year_label,
                                                  self.corrections_metadata[year],
                                                  target=target,
                                                  friend_trigWeight=self.friends.get("trigWeight"),
                                                  config=config,
                                                 )



        #
        # Calculate and apply Jet Energy Calibration
        #
        if config["do_jet_calibration"]:
            jets = apply_jerc_corrections_jsonpog(event,
                                          corrections_metadata=self.corrections_metadata[year],
                                          isMC=config["isMC"],
                                          dataset=dataset
                                          )
        else:
            jets = event.Jet


        event = update_events(event, {"Jet": jets})

        event = apply_4b_selection( event, self.corrections_metadata[year], config=config,
                                    dataset=dataset,
                                    sel_cfg=self.sel_cfg,
                                   )


        #
        # Get the trigger weights
        #
        if config["isMC"]:
            if "GluGlu" in dataset:
                ### this is temporary until trigWeight is computed in new code
                # trigWeight_file = uproot.open(f'{event.metadata["filename"].replace("picoAOD", "trigWeight")}')['Events']
                # trigWeight = trigWeight_file.arrays(['event', 'trigWeight_Data', 'trigWeight_MC'], entry_start=estart,entry_stop=estop)
                # if not ak.all(trigWeight.event == event.event):
                #     raise ValueError('trigWeight events do not match events ttree')
                trigWeight = self.friends.get("trigWeight").arrays(target)

                event["trigWeight_Data"] = trigWeight.Data
                event["trigWeight_MC"]   = trigWeight.MC


        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( event.passHLT if config["cut_on_HLT_decision"] else np.full(len(event), True)  ) )
        selections.add( 'passJetMult',   event.passJetMult )
        selections.add( "passFourTag", event.fourTag)

        event["weight"] = weights.weight()

        cumulative_cuts = ["lumimask"]
        self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )

        other_cuts = ["passNoiseFilter", "passHLT", "passJetMult","passFourTag"]

        for cut in other_cuts:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        #
        # Add Btag SF
        #
        if config["isMC"]:

            weights, list_weight_names = add_btagweights( event, weights,
                                                          list_weight_names=list_weight_names,
                                                          corrections_metadata=self.corrections_metadata[year],
                                                          isRun3=config["isRun3"],
            )
            logging.debug( f"Btag weight {weights.partial_weight(include=['CMS_btag'])[:10]}\n" )
            event["weight"] = weights.weight()

            self._cutFlow.fill( "passFourTag_btagSF", event[selections.all(*cumulative_cuts)], allTag=True )

        selection = event.lumimask & event.passNoiseFilter & event.passJetMult & event.fourTag
        if not config["isMC"]: selection = selection & event.passHLT

        selev = event[selections.all(*cumulative_cuts)]

        #
        #  TTbar subtractions using weights
        #
        if self.subtract_ttbar_with_weights:

            # These are FOUR-tag events: d4_to_t4, as processor_HH4b does for its 4b events. The
            # helper's default, d3_to_t3, is the 3b ttbar fraction (right for the mixer's 3b
            # events) and over-subtracted the 4b data by ~1.44x the ttbar MC (roast
            # declustered_run3_20260927_7e9990e-f0d7cdf).
            pass_ttbar_filter_selev = subtract_ttbar_with_FvT(selev, dataset, year, "d4_to_t4")

            pass_ttbar_filter = np.full( len(event), True)
            pass_ttbar_filter[ selections.all(*cumulative_cuts) ] = pass_ttbar_filter_selev
            selections.add( 'pass_ttbar_filter', pass_ttbar_filter )
            cumulative_cuts.append("pass_ttbar_filter")
            self._cutFlow.fill( "pass_ttbar_filter", event[selections.all(*cumulative_cuts)], allTag=True )

            selection = selection & pass_ttbar_filter
            selev = selev[pass_ttbar_filter_selev]

        selev = cand_jet_selection(selev)
        canJet    = selev.canJet
        notCanJet = selev.notCanJet_coffea

        #
        # Do the Clustering
        #
        canJet["jet_flavor"] = "b"
        notCanJet["jet_flavor"] = "j"

        jets_for_clustering = ak.concatenate([canJet, notCanJet], axis=1)
        jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]

        processOutput = {}

        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_info_to_output_clustering_inputs
        # add_debug_info_to_output_clustering_inputs(selev, jets_for_clustering, processOutput)

        clustered_jets, _clustered_splittings = cluster_bs(jets_for_clustering, debug=False)
        clustered_jets = clean_ISR(clustered_jets, _clustered_splittings)

        mask_unclustered_jet = (clustered_jets.jet_flavor == "b") | (clustered_jets.jet_flavor == "j")
        selev["nClusteredJets"] = ak.num(clustered_jets[~mask_unclustered_jet])

        #
        # Declustering
        #
        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_info_to_output_clustering_outputs
        # add_debug_info_to_output_clustering_outputs(selev, clustered_jets, processOutput)

        library_kwargs = {}
        if splitting_library is not None:
            # Jets never declustered keep their own carry fields (NaN on combined jets, whose
            # children bring the library's values)
            for field, values in carry_child_fields(clustered_jets, jets_for_clustering, self.library_carry_fields).items():
                clustered_jets[field] = values
            library_kwargs = dict(
                library=splitting_library,
                event_ids=np.column_stack([np.asarray(selev.run), np.asarray(selev.luminosityBlock), np.asarray(selev.event)]).astype(np.int64),
                library_scale_pt=self.library_scale_pt,
                library_boost_z=self.library_boost_z,
                library_selection=self.library_selection,
                library_k_neighbors=self.library_k_neighbors,
            )

        lookups_before = dict(splitting_library.lookup_counts) if splitting_library is not None else None

        b_pt_threshold = self._resolve_b_pt_threshold(year, config["isRun3"])
        declustered_jets = make_synthetic_event(clustered_jets, clustering_pdfs,
                                                declustering_rand_seed=self.declustering_rand_seed,
                                                b_pt_threshold=b_pt_threshold,
                                                dr_threshold=self.dr_threshold,
                                                max_jet_retry=self.max_jet_retry,
                                                max_event_retry=self.max_event_retry,
                                                chunk=chunk,
                                                **library_kwargs)

        declustered_jets = declustered_jets[ak.argsort(declustered_jets.pt, axis=1, ascending=False)]

        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_info_to_output_declustering_outputs
        # add_debug_info_to_output_declustering_outputs(selev, declustered_jets, processOutput)


        n_jet = ak.num(declustered_jets)
        total_jet = int(ak.sum(n_jet))


        out_branches = {
                # Update jets with new kinematics
                "Jet_pt":              declustered_jets.pt, #ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_eta":             declustered_jets.eta,
                "Jet_phi":             declustered_jets.phi,
                "Jet_mass":            declustered_jets.mass,
                "Jet_jet_flavor_bit":  declustered_jets.jet_flavor_bit,
                "Jet_jetId":           ak.unflatten(np.full(total_jet, _SYNTHETIC_JET_ID_BIT), n_jet),
                "Jet_puId":            ak.unflatten(np.full(total_jet, _SYNTHETIC_JET_ID_BIT), n_jet),
                # create new regular branch
                "nClusteredJets":      selev.nClusteredJets,
            }

        if config["isMC"]:
            out_branches["trigWeight_Data"] = selev.trigWeight_Data
            out_branches["trigWeight_MC"]   = selev.trigWeight_MC
            out_branches["CMSbtag"]        = weights.partial_weight(include=["CMS_btag"])[selections.all(*cumulative_cuts)]

        if config["isRun3"]:
            out_branches["Jet_PNetRegPtRawCorr"]         = ak.unflatten(np.full(total_jet, _UNIT_CORRECTION), n_jet)
            out_branches["Jet_PNetRegPtRawCorrNeutrino"] = ak.unflatten(np.full(total_jet, _UNIT_CORRECTION), n_jet)
            out_branches["Jet_btagPNetB"]                = declustered_jets.btagScore

        else:
            out_branches["Jet_bRegCorr"] = ak.unflatten(np.full(total_jet, _UNIT_CORRECTION), n_jet)
            out_branches["Jet_btagDeepFlavB"] = declustered_jets.btagScore

        # Other carried jet properties (btagScore is written above)
        if splitting_library is not None:
            for field in self.library_carry_fields:
                if field != "btagScore":
                    out_branches[f"Jet_{field}"] = declustered_jets[field]

        #
        #  Need to skip all the other jet branches to make sure they have the same number of jets
        #
        if not hasattr(self, "_branch_filter_initialized") or not self._branch_filter_initialized:
            skip_branches = set(self.skip_branches or [])
            for f in event.Jet.fields:
                bname = f"Jet_{f}"
                if bname not in out_branches:
                    skip_branches.add(bname)
            self.skip_branches = list(skip_branches)
            self.update_branch_filter(self.skip_collections, self.skip_branches)
            self._branch_filter_initialized = True

        branches = ak.Array(out_branches)

        processOutput["total_jet"] = total_jet
        if splitting_library is not None:
            # this chunk's library lookups by level (retries included) + last-resort self matches;
            # summed per dataset into the picoAOD registry (worker log warnings never reach the driver)
            processOutput["library_lookups"] = {k: v - lookups_before[k] for k, v in splitting_library.lookup_counts.items()}

        return (selection,
                branches,
                processOutput,
                )
