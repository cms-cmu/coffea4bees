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
from coffea4bees.analysis.helpers.event_weights import (
    add_btagweights,
    add_pseudotagweights,
)
from coffea4bees.analysis.helpers.filling_histograms import (
    filling_nominal_histograms,
    filling_syst_histograms,
)
from src.friendtrees.FriendTreeSchema import FriendTreeSchema
from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea4bees.analysis.helpers.candidates_selection import create_cand_jet_dijet_quadjet
from coffea4bees.analysis.helpers.SvB_helpers import setSvBVars, subtract_ttbar_with_SvB
from coffea4bees.analysis.helpers.topCandReconstruction import (
    adding_top_reco_to_event,
    buildTop,
    find_tops,
    find_tops_slow,
)
from coffea4bees.analysis.helpers.event_selection import apply_4b_lowpt_selection
from src.physics.event_selection import apply_event_selection
from src.physics.event_weights import add_weights
from src.hist import Fill
from src.data_formats.root import Chunk, TreeReader
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import load
from memory_profiler import profile

from ..helpers.load_friend import (
    FriendTemplate,
    parse_friends,
    rename_FvT_friend,
    rename_SvB_friend,
)

if TYPE_CHECKING:
    from ..helpers.classifier.HCR import HCRModelMetadata
from coffea4bees.analysis.helpers.truth_tools import find_genpart

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
        SvB: str|list[HCRModelMetadata] = None,
        SvB_MA: str|list[HCRModelMetadata] = None,
        blind: bool = False,
        apply_JCM: bool = True,
        JCM_file: str = "coffea4bees/analysis/weights/JCM/AN_24_089_v3/jetCombinatoricModel_SB_6771c35.yml",
        corrections_metadata: dict = None,
        apply_JCM_lowpt: bool = False,
        JCM_lowpt_file: str = None,
        apply_trigWeight: bool = True,
        apply_btagSF: bool = True,
        apply_FvT: bool = True,
        apply_boosted_veto: bool = False,
        run_lowpt_selection: bool = False,
        run_dilep_ttbar_crosscheck: bool = False,
        fill_histograms: bool = True,
        hist_cuts = ['passPreSel'],
        run_SvB: bool = True,
        top_reconstruction: bool = False,
        run_systematics: list = [],
        make_classifier_input: str = None,
        make_top_reconstruction: str = None,
        make_friend_JCM_weight: str = None,
        make_friend_FvT_weight: str = None,
        make_friend_SvB: str = None,
        subtract_ttbar_with_weights: bool = False,
        friends: dict[str, str|FriendTemplate] = None,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.blind = blind
        self.apply_JCM = jetCombinatoricModel(JCM_file, zero_npt=True) if apply_JCM else None
        self.apply_JCM_lowpt = jetCombinatoricModel(JCM_lowpt_file) if apply_JCM_lowpt else None
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.apply_FvT = apply_FvT
        self.run_lowpt_selection = run_lowpt_selection
        self.run_SvB = run_SvB
        self.fill_histograms = fill_histograms
        self.run_dilep_ttbar_crosscheck = run_dilep_ttbar_crosscheck
        self.apply_boosted_veto = apply_boosted_veto
        self.classifier_SvB = _init_classfier(SvB)
        self.classifier_SvB_MA = _init_classfier(SvB_MA)
        self.corrections_metadata = corrections_metadata

        self.run_systematics = run_systematics
        self.make_top_reconstruction = make_top_reconstruction
        self.make_classifier_input = make_classifier_input
        self.make_friend_JCM_weight = make_friend_JCM_weight
        self.make_friend_FvT_weight = make_friend_FvT_weight
        self.make_friend_SvB = make_friend_SvB
        self.top_reconstruction = top_reconstruction
        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights
        self.friends = parse_friends(friends)
        self.histCuts = hist_cuts

    def process(self, event):
        logging.debug(event.metadata)
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

        if self.top_reconstruction:
            self.top_reconstruction = self.top_reconstruction
            logging.info(f"top_reconstruction overridden to {self.top_reconstruction}\n")
        else:
            self.top_reconstruction = event.metadata.get("top_reconstruction", None)

        #
        # Set process and datset dependent flags
        #
        self.config = processor_config(self.processName, self.dataset, event)
        logging.debug(f'{self.chunk} config={self.config}, for file {fname}\n')

        #
        #  If applying Gaussian Kernal to signal
        #
        self.gaussKernalMean = None
        if self.config["isSignal"] and (self.gaussKernalMean is not None) :
            bin_edges = np.linspace(0, 1200, 100)  # 100 bins from 0 to 1200
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
            sigma = 0.05 * self.gaussKernalMean  # Standard deviation of the Gaussian
            self.resonance_weights = np.exp(-0.5 * ((bin_centers - self.gaussKernalMean) / sigma) ** 2)  # Gaussian formula

        self.nEvent = len(event)

        #
        # Reading SvB friend trees
        #
        path = fname.replace(fname.split("/")[-1], "")
        if self.apply_FvT:
                event["FvT"] = rename_FvT_friend(target, self.friends["FvT"])
                if self.config["isDataForMixed"] or self.config["isTTForMixed"]:
                    for _FvT_name in event.metadata["FvT_names"]:
                        event[_FvT_name] = rename_FvT_friend(target, self.friends[_FvT_name])
                        event[_FvT_name, _FvT_name] = event[_FvT_name].FvT
            
        if self.run_SvB:
            for k in self.friends:
                if k.startswith("SvB"):
                    try:
                        event[k] = rename_SvB_friend(target, self.friends[k])
                        setSvBVars(k, event)
                    except Exception as e:
                        event[k] = self.friends[k].arrays(target)

        #
        # Event selection
        #
        event = apply_event_selection( event,
                                        self.corrections_metadata[self.year],
                                        cut_on_lumimask=self.config["cut_on_lumimask"]
                                        )


        ### adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights(
            event, target=target,
            do_MC_weights=self.config["do_MC_weights"],
            dataset=self.dataset,
            year_label=self.year_label,
            friend_trigWeight=self.friends.get("trigWeight"),
            corrections_metadata=self.corrections_metadata[self.year],
            apply_trigWeight=self.apply_trigWeight,
            isTTForMixed=self.config["isTTForMixed"]
        )

        #
        # Calculate and apply Jet Energy Calibration
        #
        if self.config["do_jet_calibration"]:
            jets = apply_jerc_corrections(event,
                                          corrections_metadata=self.corrections_metadata[self.year],
                                          isMC=self.config["isMC"],
                                          run_systematics=self.run_systematics,
                                          dataset=self.dataset
                                          )
        else:
            jets = event.Jet


        shifts = [({"Jet": jets}, None)]
        if self.run_systematics:
            for jesunc in self.corrections_metadata[self.year]["JES_uncertainties"]:
                shifts.extend( [ ({"Jet": jets[f"JES_{jesunc}"].up}, f"CMS_scale_j_{jesunc}Up"),
                                 ({"Jet": jets[f"JES_{jesunc}"].down}, f"CMS_scale_j_{jesunc}Down"), ] )

            shifts.extend( [({"Jet": jets.JER.up}, f"CMS_res_j_{self.year_label}Up"), ({"Jet": jets.JER.down}, f"CMS_res_j_{self.year_label}Down")] )

            logging.info(f"\nJet variations {[name for _, name in shifts]}")

        return processor.accumulate( self.process_shift(update_events(event, collections), name, weights, list_weight_names, target) for collections, name in shifts )

    def process_shift(self, event, shift_name, weights, list_weight_names, target):
        """For different jet variations. It computes event variations for the nominal case."""

        # Copy the weights to avoid modifying the original
        weights = copy.copy(weights)

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_4b_lowpt_selection( event, self.corrections_metadata[self.year],
                                    dataset=self.dataset,
                                    doLeptonRemoval=self.config["do_lepton_jet_cleaning"],
                                    override_selected_with_flavor_bit=self.config["override_selected_with_flavor_bit"],
                                    do_jet_veto_maps=self.config["do_jet_veto_maps"],
                                    isRun3=self.config["isRun3"],
                                    isMC=self.config["isMC"], ### temporary
                                    isSyntheticData=self.config["isSyntheticData"],
                                    isSyntheticMC=self.config["isSyntheticMC"],
                                    )


        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        #selections.add( "passHLT", ( np.full(len(event), True) if skip_HLT_cut else event.passHLT ) )
        selections.add( "passHLT", ( event.passHLT if self.config["cut_on_HLT_decision"] else np.full(len(event), True)  ) )
        selections.add( 'passJetMult', event.passJetMult )
        allcuts = [ 'lumimask', 'passNoiseFilter', 'passHLT', ]
        allcuts += [ 'passJetMult' ]
        event['weight'] = weights.weight()   ### this is for _cutflow

        #
        #  Cut Flows
        #
        processOutput = {}
        if not shift_name:
            processOutput['nEvent'] = {}
            processOutput['nEvent'][event.metadata['dataset']] = {
                'nEvent' : self.nEvent,
                'genWeights': np.sum(event.genWeight) if self.config["isMC"] else self.nEvent

            }

            #
            # Get Truth m4j
            #
            if self.config["isSignal"]:

                event['bfromHorZ_all']= find_genpart(event.GenPart, [5], [23, 25])

                if "status" in event.bfromHorZ_all.fields:
                    event['bfromHorZ'] = event.bfromHorZ_all[event.bfromHorZ_all.status == 23]
                else:
                    logging.warning(f"\nStatus Missing for GenParticles in dataset {self.dataset}\n")
                    event['bfromHorZ'] = event.bfromHorZ_all

                event['GenJet', 'selectedBs'] = (np.abs(event.GenJet.partonFlavour)==5)
                event['selGenBJet'] = event.GenJet[event.GenJet.selectedBs]
                event['matchedGenBJet'] = event.bfromHorZ.nearest( event.selGenBJet, threshold=10 )
                event["matchedGenBJet"] = event.matchedGenBJet[~ak.is_none(event.matchedGenBJet, axis=1)]

                event['pass4GenBJets'] = ak.num(event.matchedGenBJet) == 4
                event["truth_v4b"] = ak.where(  event.pass4GenBJets,
                                                event.matchedGenBJet.sum(axis=1),
                                                1e-10 * event.matchedGenBJet.sum(axis=1),
                                              )

                if self.gaussKernalMean is not None:
                    v4b_index = np.floor_divide(event.truth_v4b.mass, 12).to_numpy()
                    v4b_index[v4b_index > 98] = 98

                    vectorized_v4b = np.vectorize(lambda i: self.resonance_weights[int(i)])
                    weights_resonance = vectorized_v4b(v4b_index)
                    weights.add( "resonance_reweight", weights_resonance )
                    list_weight_names.append(f"resonance_reweight")

            else:
                event['pass4GenBJets'] = True

            selections.add( "pass4GenBJets", event.pass4GenBJets)

            #
            # Do the cutflow
            #
            sel_dict = OrderedDict({
                'all'               : selections.require(lumimask=True),
                'pass4GenBJets'     : selections.require(lumimask=True, pass4GenBJets=True),
                'passNoiseFilter'   : selections.require(lumimask=True, passNoiseFilter=True),
                'passHLT'           : selections.require(lumimask=True, passNoiseFilter=True, passHLT=True),
            })
            sel_dict['passJetMult'] = selections.all(*allcuts)

            self._cutFlow = cutFlow(do_truth_hists=self.config["isSignal"])
            for cut, sel in sel_dict.items():
                self._cutFlow.fill( cut, event[sel], allTag=True )
                self._cutFlow.fill( f"{cut}_woTrig", event[sel], allTag=True,
                                    wOverride=weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[sel])


        #
        # Calculate and apply btag scale factors
        #
        if self.config["isMC"] and self.apply_btagSF:

            weights, list_weight_names = add_btagweights(
                event,
                weights,
                list_weight_names=list_weight_names,
                shift_name=shift_name,
                use_prestored_btag_SF=self.config["use_prestored_btag_SF"],
                run_systematics=self.run_systematics,
                corrections_metadata=self.corrections_metadata[self.year],
            )
            logging.debug( f"Btag weight {weights.partial_weight(include=['CMS_btag'])[:10]}\n" )
            event["weight"] = weights.weight()
            if not shift_name:
                self._cutFlow.fill( "passJetMult_btagSF", event[selections.all(*allcuts)], allTag=True )
                self._cutFlow.fill( "passJetMult_btagSF_woTrig", event[selections.all(*allcuts)], allTag=True,
                               wOverride=weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] )

        #
        # Preselection: keep only three or four tag events
        #
        selections.add("passPreSel", event.passPreSel)
        allcuts.append("passPreSel")
        analysis_selections = selections.all(*allcuts)

        if not shift_name:
            self._cutFlow.fill( "passPreSel_allTag", event[selections.all(*allcuts)], allTag=True )
            self._cutFlow.fill( "passPreSel_allTag_woTrig", event[selections.all(*allcuts)], allTag=True,
                                wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))

        weights, list_weight_names = add_pseudotagweights(
            event,
            weights,
            JCM_lowpt=self.apply_JCM_lowpt,
            JCM=self.apply_JCM,
            apply_FvT=self.apply_FvT,
            isDataForMixed=self.config["isDataForMixed"],
            list_weight_names=list_weight_names,
            event_metadata=event.metadata,
            year_label=self.year_label,
            len_event=len(event),
            label3b="lowpt_threeTag",
        )

        #
        # Example of how to write out event numbers
        #
        #from coffea4bees.analysis.helpers.write_debug_info import add_debug_Run3_data
        #add_debug_Run3_data(event, processOutput)

        selev = event[analysis_selections]
        #selev["passFvT50" ] = selev["FvT"].FvT > 50
        #selev["passFvT100"] = selev["FvT"].FvT > 100

        #
        #  Build the top Candiates
        #
        if friend := self.friends.get("top_reconstruction"):
            top_cand = friend.arrays(target)[analysis_selections]
            adding_top_reco_to_event( selev, top_cand )

        else:
            if self.top_reconstruction in ["slow","fast"]:

                # sort the jets by btagging
                selev.selJet = selev.selJet[ ak.argsort(selev.selJet.btagScore, axis=1, ascending=False) ]

                if self.top_reconstruction == "slow":
                    top_cands = find_tops_slow(selev.selJet)
                else:
                    try:
                        top_cands = find_tops(selev.selJet)
                    except Exception as e:
                        logging.warning("WARNING: Fast top_reconstruction failed with exception: ")
                        logging.warning(f"{e}\n")
                        logging.warning("... Trying the slow top_reconstruction")
                        top_cands = find_tops_slow(selev.selJet)

                selev['top_cand'], _ = buildTop(selev.selJet, top_cands)
                ### with top friendtree we dont need the next two lines
                selev["xbW"] = selev.top_cand.xbW
                selev["xW"] = selev.top_cand.xW

        #
        #  Build di-jets and Quad-jets
        #
        selev = create_cand_jet_dijet_quadjet( selev,
            apply_FvT=self.apply_FvT,
            run_SvB=self.run_SvB,
            run_systematics=self.run_systematics,
            classifier_SvB=self.classifier_SvB,
            classifier_SvB_MA=self.classifier_SvB_MA,
            processOutput=processOutput,
            isRun3=self.config["isRun3"],
            include_lowptjets=True,
            )




        #
        # Example of how to write out event numbers
        #
        # from coffea4bees.analysis.helpers.write_debug_info import add_debug_info_to_output
        # add_debug_info_to_output(event, processOutput, weights, list_weight_names, analysis_selections)


        #
        # Blind data in fourTag SR
        #
        # if not (self.config["isMC"] or "mix_v" in self.dataset) and self.blind:
        #     # blind_flag = ~(selev["quadJet_selected"].SR & selev.fourTag)
        #     blind_flag = ~( selev["quadJet_selected"].SR & (selev["SvB_MA"].ps_hh > 0.5) & selev.fourTag )
        #     blind_sel = np.full( len(event), True)
        #     blind_sel[ analysis_selections ] = blind_flag
        #     selections.add( 'blind', blind_sel )
        #     allcuts.append( 'blind' )

        #     if not shift_name:
        #         self._cutFlow.fill( "blind", event[selections.all(*allcuts)], allTag=True )
        #         self._cutFlow.fill( "blind_woTrig", event[selections.all(*allcuts)], allTag=True,
        #                             wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))

        #     analysis_selections = selections.all(*allcuts)
        #     selev = selev[blind_flag]

        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[analysis_selections]
        selev["trigWeight"] = weights.partial_weight(include=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections]
        selev['weight_woTrig'] = weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections]
        selev["no_weight"] = np.ones(len(selev))
        if not shift_name:
            self._cutFlow.fill("passPreSel", selev)
            self._cutFlow.fill("passPreSel_woTrig", selev,
                               wOverride=selev['weight_woTrig'])
            self._cutFlow.fill("passDiJetMass", selev[selev.passDiJetMass])
            self._cutFlow.fill("passDiJetMass_woTrig", selev[selev.passDiJetMass],
                               wOverride=selev['weight_woTrig'][selev.passDiJetMass] )
            selev['passSR'] = selev.passDiJetMass & selev["quadJet_selected"].SR
            self._cutFlow.fill( "SR", selev[selev.passSR] )
            self._cutFlow.fill( "SR_woTrig", selev[selev.passSR],
                            wOverride=selev['weight_woTrig'][selev.passSR])
            selev['passSB'] = selev.passDiJetMass & selev["quadJet_selected"].SB
            self._cutFlow.fill( "SB", selev[(selev.passDiJetMass & selev["quadJet_selected"].SB)] )
            self._cutFlow.fill( "SB_woTrig", selev[(selev.passDiJetMass & selev["quadJet_selected"].SB)],
                            wOverride=selev['weight_woTrig'][selev.passSB] )
            if self.run_SvB:
                self._cutFlow.fill("passSvB", selev[selev.passSvB])
                self._cutFlow.fill("passSvB_woTrig", selev[selev.passSvB],
                               wOverride=selev['weight_woTrig'][selev.passSvB] )
                self._cutFlow.fill("failSvB", selev[selev.failSvB])
                self._cutFlow.fill("failSvB_woTrig", selev[selev.failSvB],
                               wOverride=selev['weight_woTrig'][selev.failSvB] )
            if self.run_dilep_ttbar_crosscheck:
                self._cutFlow.fill("passDilepTtbar", selev[selev.passDilepTtbar], allTag=True,
                               wOverride=selev['weight_noJCM_noFvT'][selev.passDilepTtbar] )

            self._cutFlow.addOutput(processOutput, event.metadata["dataset"])

        #
        # Hists
        #
        hist = {}
        if self.fill_histograms:
            if not self.run_systematics:
                ## this can be simplified
                hist = filling_nominal_histograms(
                    selev,
                    self.apply_JCM,
                    JCM_lowpt=self.apply_JCM_lowpt,
                    processName=self.processName,
                    year=self.year,
                    isMC=self.config["isMC"],
                    histCuts=self.histCuts,
                    apply_FvT=self.apply_FvT,
                    run_SvB=self.run_SvB,
                    run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                    top_reconstruction=self.top_reconstruction,
                    isDataForMixed=self.config['isDataForMixed'],
                    tag_list=["lowpt_fourTag", "lowpt_threeTag"],
                    event_metadata=event.metadata,
                )

            #
            # Run systematics
            #
            else:
                hist = filling_syst_histograms(selev, 
                    weights,
                    analysis_selections,
                    shift_name=shift_name,
                    processName=self.processName,
                    year=self.year,
                    histCuts=self.histCuts)

        friends = { 'friends': {} }
        if self.make_top_reconstruction is not None:
            from ..helpers.dump_friendtrees import dump_top_reconstruction

            friends["friends"] = ( friends["friends"]
                | dump_top_reconstruction(
                    selev,
                    self.make_top_reconstruction,
                    f"top_reco{'_'+shift_name if shift_name else ''}",
                    analysis_selections,
                )
            )

        if self.make_classifier_input is not None:
            for k in ["ZZSR", "ZHSR", "HHSR", "SR", "SB"]:
                selev[k] = selev["quadJet_selected"][k]
            selev["nSelJets"] = ak.num(selev.selJet)

            from ..helpers.dump_friendtrees import dump_input_friend

            weight = "weight_noJCM_noFvT"
            if weight not in selev.fields:
                weight = "weight"
            friends["friends"] = ( friends["friends"]
                | dump_input_friend(
                    selev,
                    self.make_classifier_input,
                    "HCR_input",
                    analysis_selections,
                    weight=weight,
                    NotCanJet="notCanJet_coffea",
                )
            )
        if self.make_friend_JCM_weight is not None:
            from ..helpers.dump_friendtrees import dump_JCM_weight

            friends["friends"] = ( friends["friends"]
                | dump_JCM_weight(selev, self.make_friend_JCM_weight, "JCM_weight", analysis_selections)
            )

        if self.make_friend_FvT_weight is not None:
            from ..helpers.dump_friendtrees import dump_FvT_weight

            friends["friends"] = ( friends["friends"]
                | dump_FvT_weight(selev, self.make_friend_FvT_weight, "FvT_weight", analysis_selections)
            )

        if self.make_friend_SvB is not None:
            from ..helpers.dump_friendtrees import dump_SvB

            friends["friends"] = ( friends["friends"]
                | dump_SvB(selev, self.make_friend_SvB, "SvB", analysis_selections)
                | dump_SvB(selev, self.make_friend_SvB, "SvB_MA", analysis_selections)
            )

        return hist | processOutput | friends

    def postprocess(self, accumulator):
        return accumulator
