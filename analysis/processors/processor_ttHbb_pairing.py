"""Processor for ttHbb pairing study: analyzes dijet pairing metrics (min dR vs mass, pt, m4j)
across all 12 candidate selection cut stages without pass_dijet_mass.

How to run:
  1. Signal (ttHbb):
     ./run_container python runner.py \
         -p coffea4bees/analysis/processors/processor_ttHbb_pairing.py \
         -c coffea4bees/analysis/metadata/ttHbb_signals_pairing.yml \
         -m coffea4bees/metadata/datasets/ttHbb.yml \
         -d ttHbb -y UL18 \
         -o hists_ttHbb_pairing_UL18.coffea -op output/ttHbb_pairing/

  2. Background (TT_stitched):
     ./run_container python runner.py \
         -p coffea4bees/analysis/processors/processor_ttHbb_pairing.py \
         -c coffea4bees/analysis/metadata/ttHbb_TT_stitched_pairing.yml \
         -m coffea4bees/metadata/datasets/TT_stitched.yml \
         -d TTToHadronic_stitched TTToSemiLeptonic_stitched TTTo2L2Nu_stitched -y UL18 \
         -o hists_TT_stitched_pairing_UL18.coffea -op output/ttHbb_pairing/

  3. Control Data (data 3b):
     ./run_container python runner.py \
         -p coffea4bees/analysis/processors/processor_ttHbb_pairing.py \
         -c coffea4bees/analysis/metadata/ttHbb_data3b_pairing.yml \
         -m coffea4bees/metadata/datasets/data.yml \
         -d data -y UL18 \
         -o hists_data3b_pairing_UL18.coffea -op output/ttHbb_pairing/
"""

from __future__ import annotations

import copy
import logging
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING

import awkward as ak
import numpy as np
import yaml
from src.physics.objects.jet_corrections import apply_jerc_corrections_jsonpog
from src.physics.common import update_events
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea4bees.analysis.helpers.object_selection import load_object_selection_config
from src.physics.event_selection import apply_event_selection
from src.hist_tools import Fill
from src.data_formats.root import Chunk, TreeReader
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
import hist
from src.math_tools.random import Squares
from coffea4bees.analysis.helpers.event_weights import add_weights

from ..helpers.load_friend import (
    FriendTemplate,
    parse_friends,
)
from coffea4bees.analysis.helpers.candidates_selection import load_candidates_selection_config
from coffea4bees.analysis.helpers.candidates_selection_ttHbb import create_cand_jet_dijet_quadjet_ttHbb

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
        apply_trigWeight: bool = True,
        apply_btagSF: bool = True,
        fill_histograms: bool = True,
        corrections_metadata: dict = None,
        friends: dict[str, str|FriendTemplate] = None,
        object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
        candidates_selection_cfg: str = "coffea4bees/analysis/metadata/candidates_selection_thresholds_ttHbb.yml",
        tag_selection: str = "fourTag",
    ):
        logging.debug("\nInitialize ttHbb Pairing Processor")
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.fill_histograms = fill_histograms
        self.corrections_metadata = corrections_metadata
        self.friends = parse_friends(friends)
        self.sel_cfg = load_object_selection_config(object_selection_cfg) if object_selection_cfg else None
        self.cand_cfg = load_candidates_selection_config(candidates_selection_cfg) if candidates_selection_cfg else None
        self.tag_selection = tag_selection

        self.cutFlowCuts = [
            "all",
            "pass4GenBJets",
            "passHLT",
            "passNoiseFilter",
            "passJetMult",
            "passJetMult_btagSF",
            "passPreSel",
            "passMDR",
        ]

        self.histCuts = []

    def process(self, event):
        logging.info(event.metadata)
        fname   = event.metadata['filename']
        self.dataset = event.metadata['dataset']
        self.estart  = event.metadata['entrystart']
        self.estop   = event.metadata['entrystop']
        self.chunk   = f'{self.dataset}::{self.estart:6d}:{self.estop:6d} >>> '
        self.year    = event.metadata['year']
        self.year_label = self.corrections_metadata[self.year]['year_label']
        self.processName = event.metadata['processName']

        isMC = ("genWeight" in event.fields) and ("data" not in self.processName.lower())

        target = Chunk.from_coffea_events(event)

        event["passHLT"] = np.full(len(event), True)
        if isMC:
            weights, list_weight_names = add_weights(
                event, target=target,
                dataset=self.dataset,
                year_label=self.year_label,
                friend_trigWeight=self.friends.get("trigWeight"),
                corrections_metadata=self.corrections_metadata[self.year],
                apply_trigWeight=self.apply_trigWeight,
                config={"do_MC_weights": True, "isTTForMixed": False, "isRun3": False},
            )
            event = apply_event_selection(
                event,
                self.corrections_metadata[self.year],
                cut_on_lumimask=False,
            )
            jets = apply_jerc_corrections_jsonpog(
                event,
                corrections_metadata=self.corrections_metadata[self.year],
                isMC=True,
                run_systematics=False,
                dataset=self.dataset,
            )
        else:
            from coffea.analysis_tools import Weights
            weights = Weights(len(event), storeIndividual=False)
            weights.add('weight', np.ones(len(event), dtype=np.float32))
            list_weight_names = ['weight']
            event = apply_event_selection(
                event,
                self.corrections_metadata[self.year],
                cut_on_lumimask=True,
            )
            jets = apply_jerc_corrections_jsonpog(
                event,
                corrections_metadata=self.corrections_metadata[self.year],
                isMC=False,
                run_systematics=False,
                dataset=self.dataset,
            )

        config_opts = {
            "do_lepton_jet_cleaning": True,
            "override_selected_with_flavor_bit": False,
            "do_jet_veto_maps": False,
            "isRun3": False,
            "isMC": isMC,
            "isSyntheticData": False,
            "isSyntheticMC": False,
        }
        event = apply_4b_selection(
            event,
            self.corrections_metadata[self.year],
            dataset=self.dataset,
            config=config_opts,
            sel_cfg=self.sel_cfg,
        )

        selections = PackedSelection()
        selections.add("lumimask", event.lumimask)
        selections.add("passNoiseFilter", event.passNoiseFilter)
        selections.add("passHLT", np.full(len(event), True))
        selections.add("passJetMult", event.passJetMult)
        allcuts = ['lumimask', 'passNoiseFilter', 'passHLT', 'passJetMult']

        if self.tag_selection == "threeTag":
            tag_pass = event.threeTag
        elif self.tag_selection == "fourTag":
            tag_pass = event.fourTag
        else:
            tag_pass = event.passPreSel

        selections.add("tagCut", tag_pass)
        allcuts.append("tagCut")
        analysis_selections = selections.all(*allcuts)

        selev = event[analysis_selections]
        selev["weight"] = weights.weight()[analysis_selections]

        # Build candidates unconstrained for ttHbb
        selev = create_cand_jet_dijet_quadjet_ttHbb(selev, cand_cfg=self.cand_cfg)

        is_selected = selev["quadJet"].rank == np.max(selev["quadJet"].rank, axis=1)
        sr_sel = selev["quadJet_selected"].SR
        sbsr_sel = selev["quadJet_selected"].SR | selev["quadJet_selected"].SB

        # Pairing selections: strictly without passDiJetMass
        selection = {
            "none": selev["quadJet"].rank > 0,
            "none_SBSR": (selev["quadJet"].rank > 0) & sbsr_sel,
            "none_SR": (selev["quadJet"].rank > 0) & sr_sel,
            "passOneMDR": (selev["quadJet"].rank > 11) & (selev["quadJet"].rank < 12),
            "passOneMDR_SBSR": ((selev["quadJet"].rank > 11) & (selev["quadJet"].rank < 12)) & sbsr_sel,
            "passOneMDR_SR": ((selev["quadJet"].rank > 11) & (selev["quadJet"].rank < 12)) & sr_sel,
            "passMDR": selev["quadJet"].rank > 12,
            "passMDR_SBSR": (selev["quadJet"].rank > 12) & sbsr_sel,
            "passMDR_SR": (selev["quadJet"].rank > 12) & sr_sel,
            "selected": is_selected,
            "selected_SBSR": is_selected & sbsr_sel,
            "selected_SR": is_selected & sr_sel,
        }


        #
        # Histograms
        #
        process_axis = hist.axis.StrCategory([], name="process", label="Process", growth=True)
        sel_axis = hist.axis.StrCategory([], name="selection", label="Selection", growth=True)
        year_axis = hist.axis.StrCategory([], name="year", label="Year", growth=True)
        npar_axis = hist.axis.Integer(0, 4, name="n_pairings", label="Number of Pairings")
        mass_axis = hist.axis.Regular(100, 100, 1100, name="mass", label="$m_{4j}$ [GeV]")
        leadstmass_axis = hist.axis.Regular(100, 0., 1000., name="leadstmass", label="Leading $S_{T}$ Dijet Mass [GeV]")
        sublstmass_axis = hist.axis.Regular(100, 0., 1000., name="sublstmass", label="Subleading $S_{T}$ Dijet Mass [GeV]")
        leadstpt_axis = hist.axis.Regular(100, 0., 1000., name="leadstpt", label="Leading $S_{T}$ Dijet $p_{T}$ [GeV]")
        sublstpt_axis = hist.axis.Regular(100, 0., 1000., name="sublstpt", label="Subleading $S_{T}$ Dijet $p_{T}$ [GeV]")
        leadstdr_axis = hist.axis.Regular(100, 0., 5., name="leadstdr", label=r"Leading $S_{T}$ Boson Candidate $\Delta R(j,j)$")
        sublstdr_axis = hist.axis.Regular(100, 0., 5., name="sublstdr", label=r"Subleading $S_{T}$ Boson Candidate $\Delta R(j,j)$")

        hists = {
            'hists': {
                "npairs_vs_m4j": hist.Hist(process_axis, sel_axis, year_axis, mass_axis, npar_axis),
                "leadstmass_vs_sublstmass": hist.Hist(process_axis, sel_axis, year_axis, leadstmass_axis, sublstmass_axis),
                "leadstdr_vs_m4j": hist.Hist(process_axis, sel_axis, year_axis, mass_axis, leadstdr_axis),
                "sublstdr_vs_m4j": hist.Hist(process_axis, sel_axis, year_axis, mass_axis, sublstdr_axis),
                "leadstdr_vs_leadstmass": hist.Hist(process_axis, sel_axis, year_axis, leadstmass_axis, leadstdr_axis),
                "leadstdr_vs_leadstpt": hist.Hist(process_axis, sel_axis, year_axis, leadstpt_axis, leadstdr_axis),
                "sublstdr_vs_sublstmass": hist.Hist(process_axis, sel_axis, year_axis, sublstmass_axis, sublstdr_axis),
                "sublstdr_vs_sublstpt": hist.Hist(process_axis, sel_axis, year_axis, sublstpt_axis, sublstdr_axis),
            }
        }

        if "TTTo" in self.processName:
            hist_process = "TT"
        elif "data" in self.processName.lower():
            hist_process = "data_3b" if self.tag_selection == "threeTag" else "data"
        else:
            hist_process = self.processName

        for isel in selection.keys():
            num_pairs = ak.num(selev["quadJet"]["lead"][selection[isel]], axis=1)
            num_pairs_mass = ak.where(num_pairs > 0, ak.firsts(selev["quadJet"].v4jmass), -1)
            quadJet_v4jmass = ak.broadcast_arrays(selev["v4j"].mass[:, np.newaxis, np.newaxis], selev["quadJet"][selection[isel]].dr)[0]
            weight_quad = ak.broadcast_arrays(selev["weight"], selev["quadJet"]["lead"][selection[isel]].mass)[0]

            hists["hists"]["npairs_vs_m4j"].fill(
                process=hist_process,
                year=self.year,
                selection=isel,
                n_pairings=num_pairs,
                mass=num_pairs_mass,
                weight=selev["weight"],
            )

            hists["hists"]["leadstmass_vs_sublstmass"].fill(
                process=hist_process,
                year=self.year,
                selection=isel,
                leadstmass=ak.flatten(selev["quadJet"]["lead"][selection[isel]].mass, axis=1),
                sublstmass=ak.flatten(selev["quadJet"]["subl"][selection[isel]].mass, axis=1),
                weight=ak.flatten(weight_quad, axis=1),
            )

            hists["hists"]["leadstdr_vs_m4j"].fill(
                process=hist_process,
                year=self.year,
                selection=isel,
                leadstdr=ak.flatten(selev["quadJet"]["lead"][selection[isel]].dr, axis=1),
                mass=ak.flatten(quadJet_v4jmass[:, :, 0], axis=1),
                weight=ak.flatten(weight_quad, axis=1),
            )

            hists["hists"]["sublstdr_vs_m4j"].fill(
                process=hist_process,
                year=self.year,
                selection=isel,
                sublstdr=ak.flatten(selev["quadJet"]["subl"][selection[isel]].dr, axis=1),
                mass=ak.flatten(quadJet_v4jmass[:, :, 0], axis=1),
                weight=ak.flatten(weight_quad, axis=1),
            )

            hists["hists"]["leadstdr_vs_leadstmass"].fill(
                process=hist_process,
                year=self.year,
                selection=isel,
                leadstmass=ak.flatten(selev["quadJet"]["lead"][selection[isel]].mass, axis=1),
                leadstdr=ak.flatten(selev["quadJet"]["lead"][selection[isel]].dr, axis=1),
                weight=ak.flatten(weight_quad, axis=1),
            )

            hists["hists"]["leadstdr_vs_leadstpt"].fill(
                process=hist_process,
                year=self.year,
                selection=isel,
                leadstpt=ak.flatten(selev["quadJet"]["lead"][selection[isel]].pt, axis=1),
                leadstdr=ak.flatten(selev["quadJet"]["lead"][selection[isel]].dr, axis=1),
                weight=ak.flatten(weight_quad, axis=1),
            )

            hists["hists"]["sublstdr_vs_sublstmass"].fill(
                process=hist_process,
                year=self.year,
                selection=isel,
                sublstmass=ak.flatten(selev["quadJet"]["subl"][selection[isel]].mass, axis=1),
                sublstdr=ak.flatten(selev["quadJet"]["subl"][selection[isel]].dr, axis=1),
                weight=ak.flatten(weight_quad, axis=1),
            )

            hists["hists"]["sublstdr_vs_sublstpt"].fill(
                process=hist_process,
                year=self.year,
                selection=isel,
                sublstpt=ak.flatten(selev["quadJet"]["subl"][selection[isel]].pt, axis=1),
                sublstdr=ak.flatten(selev["quadJet"]["subl"][selection[isel]].dr, axis=1),
                weight=ak.flatten(weight_quad, axis=1),
            )

        return hists

    def postprocess(self, accumulator):
        return accumulator
