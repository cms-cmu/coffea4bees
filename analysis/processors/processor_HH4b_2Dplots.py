from __future__ import annotations

import copy
import logging
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING

import awkward as ak
import numpy as np
import yaml
from src.physics.objects.jet_corrections import apply_jerc_corrections
from src.physics.common import update_events
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from coffea4bees.analysis.helpers.filling_histograms import (
    filling_nominal_histograms,
    filling_syst_histograms,
)
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from src.physics.event_selection import apply_event_selection
from src.hist_tools import Fill
from src.data_formats.root import Chunk, TreeReader
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from memory_profiler import profile
import hist
from src.math_tools.random import Squares
from src.physics.event_weights import add_weights

from ..helpers.load_friend import (
    FriendTemplate,
    parse_friends,
)
from coffea4bees.analysis.helpers.candidates_selection import create_cand_jet_dijet_quadjet
#
# Setup
#

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


def _init_classfier(path: str | list[HCRModelMetadata]):
    if path is None:
        return None
    if isinstance(path, str):
        from ..helpers.classifier.HCR import Legacy_HCREnsemble
        return Legacy_HCREnsemble(path)
    else:
        from ..helpers.classifier.HCR import HCREnsemble
        return HCREnsemble(path)


class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
        apply_trigWeight: bool = True,
        apply_btagSF: bool = True,
        fill_histograms: bool = True,
        corrections_metadata: dict = None,
        friends: dict[str, str|FriendTemplate] = None,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.fill_histograms = fill_histograms
        self.corrections_metadata = corrections_metadata
        self.friends = parse_friends(friends)


        self.cutFlowCuts = [
            "all",
            "pass4GenBJets",
            "passHLT",
            "passNoiseFilter",
            "passJetMult",
            "passJetMult_btagSF",
            "passPreSel",
            "passDiJetMass",
        ]

        self.histCuts = ['passPreSel']

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

        ### target is for new friend trees
        target = Chunk.from_coffea_events(event)

        ### adds all the event mc weights and 1 for data 
        event["passHLT"] = np.full(len(event), True)
        weights, list_weight_names = add_weights(
            event, target=target,
            do_MC_weights=True,
            dataset=self.dataset,
            year_label=self.year_label,
            friend_trigWeight=self.friends.get("trigWeight"),
            corrections_metadata=self.corrections_metadata[self.year],
            apply_trigWeight=self.apply_trigWeight,
            isTTForMixed=False
        )

        #
        # Event selection
        #
        event = apply_event_selection( event,
                                        self.corrections_metadata[self.year],
                                        cut_on_lumimask=False
                                        )
        #
        # Calculate and apply Jet Energy Calibration
        #
        jets = apply_jerc_corrections(event,
                                        corrections_metadata=self.corrections_metadata[self.year],
                                        isMC=True,
                                        run_systematics=False,
                                        dataset=self.dataset
                                        )

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_4b_selection( event, self.corrections_metadata[self.year],
                                            dataset=self.dataset,
                                            doLeptonRemoval=True,
                                            override_selected_with_flavor_bit=False,
                                            run_lowpt_selection=False,
                                            do_jet_veto_maps=False,
                                            isRun3=False,
                                            isMC=True,
                                            isSyntheticData=False,
                                            isSyntheticMC=False,
                                            )


        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        #selections.add( "passHLT", ( np.full(len(event), True) if skip_HLT_cut else event.passHLT ) )
        selections.add( "passHLT", np.full(len(event), True)  )
        selections.add( 'passJetMult', event.passJetMult )
        allcuts = [ 'lumimask', 'passNoiseFilter', 'passHLT', ]
        allcuts += [ 'passJetMult' ]

        #
        # Preselection: keep only three or four tag events
        #
        selections.add("passPreSel", event.passPreSel)
        allcuts.append("passPreSel")
        analysis_selections = selections.all(*allcuts)

        selev = event[analysis_selections]
        selev["weight"] = weights.weight()[analysis_selections]
        selev = create_cand_jet_dijet_quadjet(selev)

        selection = {
            "none" : selev["quadJet"].rank > 0,
            "none_SBSR": (selev["quadJet"].rank > 0) & (selev["quadJet_selected"].SR | selev["quadJet_selected"].SB), 
            "none_SR": (selev["quadJet"].rank > 0) & selev["quadJet_selected"].SR,
            "passDiJetMass" : (selev["quadJet"].rank > 10),
            "passDiJetMass_SBSR": (selev["quadJet"].rank > 10) & (selev["quadJet_selected"].SR | selev["quadJet_selected"].SB), 
            "passDiJetMass_SR": (selev["quadJet"].rank > 10) & selev["quadJet_selected"].SR,
            "passDiJetMassOneMDR" : (selev["quadJet"].rank > 11) & (selev["quadJet"].rank < 12),
            "passDiJetMassOneMDR_SBSR": ((selev["quadJet"].rank > 11) & (selev["quadJet"].rank < 12)) & (selev["quadJet_selected"].SR | selev["quadJet_selected"].SB), 
            "passDiJetMassOneMDR_SR": ((selev["quadJet"].rank > 11) & (selev["quadJet"].rank < 12)) & selev["quadJet_selected"].SR,
            "passDiJetMassMDR" : selev["quadJet"].rank > 12,
            "passDiJetMassMDR_SBSR": (selev["quadJet"].rank > 12) & (selev["quadJet_selected"].SR | selev["quadJet_selected"].SB), 
            "passDiJetMassMDR_SR": (selev["quadJet"].rank > 12) & selev["quadJet_selected"].SR,
            "selected" : selev["quadJet"].rank == np.max(selev["quadJet"].rank, axis=1),
            "selected_SBSR": (selev["quadJet"].rank == np.max(selev["quadJet"].rank, axis=1)) & (selev["quadJet_selected"].SR | selev["quadJet_selected"].SB), 
            "selected_SR": (selev["quadJet"].rank == np.max(selev["quadJet"].rank, axis=1)) & selev["quadJet_selected"].SR 
        }


        #
        # Hists
        #
        process_axis = hist.axis.StrCategory([], name="process", label="Process", growth=True)
        sel_axis = hist.axis.StrCategory([], name="selection", label="Selection", growth=True)
        year_axis = hist.axis.StrCategory([], name="year", label="Year", growth=True)
        npar_axis = hist.axis.Integer(0, 4, name="n_pairings", label="Number of Pairings")
        mass_axis = hist.axis.Regular(100, 100, 1100, name="mass", label="$m_{4j}$ [GeV]")
        leadstmass_axis = hist.axis.Regular(50, 0., 250., name="leadstmass", label="Leading $S_{T}$ Dijet Mass [GeV]")
        sublstmass_axis = hist.axis.Regular(50, 0., 250., name="sublstmass", label="Subleading $S_{T}$ Dijet Mass [GeV]")
        leadstdr_axis = hist.axis.Regular(100, 0., 5., name="leadstdr", label="Leading $S_{T}$ Boson Candidate $\Delta R(j,j)$")
        sublstdr_axis = hist.axis.Regular(25, 0., 5., name="sublstdr", label="Subleading $S_{T}$ Boson Candidate $\Delta R(j,j)$")

        hists = {
            'hists' : {
                "npairs_vs_m4j": hist.Hist(process_axis, sel_axis, year_axis, mass_axis, npar_axis),
                "leadstmass_vs_sublstmass": hist.Hist(process_axis, sel_axis, year_axis, leadstmass_axis, sublstmass_axis),
                "leadstdr_vs_m4j": hist.Hist(process_axis, sel_axis, year_axis, mass_axis, leadstdr_axis),
                "sublstdr_vs_m4j": hist.Hist(process_axis, sel_axis, year_axis, mass_axis, sublstdr_axis),
            }
        }

        for isel in selection.keys():
            
            num_pairs = ak.num(selev["quadJet"]["lead"][selection[isel]], axis=1)
            num_pairs_mass = ak.where( num_pairs > 0, ak.firsts(selev["quadJet"].v4jmass), -1)
            quadJet_v4jmass = ak.broadcast_arrays(selev["v4j"].mass[:, np.newaxis, np.newaxis], selev["quadJet"][selection[isel]].dr)[0]
            weight_quad = ak.broadcast_arrays(selev["weight"], selev["quadJet"]["lead"][selection[isel]].mass)[0]

            hists["hists"]["npairs_vs_m4j"].fill(
                process=self.processName,
                year=self.year,
                selection=isel,
                n_pairings=num_pairs,
                mass=num_pairs_mass,
                weight=selev["weight"]
            )

            hists["hists"]["leadstmass_vs_sublstmass"].fill(
                process=self.processName,
                year=self.year,
                selection=isel,
                leadstmass=ak.flatten(selev["quadJet"]["lead"][selection[isel]].mass, axis=1),
                sublstmass=ak.flatten(selev["quadJet"]["subl"][selection[isel]].mass, axis=1),
                weight=ak.flatten(weight_quad, axis=1)
            )

            hists["hists"]["leadstdr_vs_m4j"].fill(
                process=self.processName,
                year=self.year,
                selection=isel,
                leadstdr=ak.flatten(selev["quadJet"]["lead"][selection[isel]].dr, axis=1),
                mass=ak.flatten(quadJet_v4jmass[:,:,0], axis=1),
                weight=ak.flatten(weight_quad, axis=1)
            )

            hists["hists"]["sublstdr_vs_m4j"].fill(
                process=self.processName,
                year=self.year,
                selection=isel,
                sublstdr=ak.flatten(selev["quadJet"]["subl"][selection[isel]].dr, axis=1),
                mass=ak.flatten(quadJet_v4jmass[:,:,0], axis=1),
                weight=ak.flatten(weight_quad, axis=1)
            )

        return hists

    def postprocess(self, accumulator):
        return accumulator
