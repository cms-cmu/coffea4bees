import time
import gc
import awkward as ak
import numpy as np
import correctionlib
import yaml
import warnings
import uproot
import copy
import hist as hist2


from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection

from src.hist_tools import Collection, Fill
from src.hist_tools.object import LorentzVector, Jet

from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from src.friendtrees.FriendTreeSchema import FriendTreeSchema

from src.physics.common import apply_btag_sf, update_events
from coffea4bees.analysis.helpers.truth_tools import find_genpart

from src.physics.event_selection import apply_event_selection
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection

import logging

from src.data_formats.root import TreeReader, Chunk

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")



class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
        corrections_metadata: dict = None,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = corrections_metadata

        self.histCuts = ["pass4GenBJets00",    "pass4GenBJets20",    "pass4GenBJets40",
                         "pass4GenBJets00_1j", "pass4GenBJets20_1j", "pass4GenBJets40_1j",
                         "pass4GenBJetsb203b40_1j_i", "pass4GenBJetsb203b40_1j_e",
                         "pass4GenBJets2b202b40_2j_i", "pass4GenBJets2b202b40_2j_e",
                         ]



    def process(self, event):

        tstart = time.time()
        fname   = event.metadata['filename']
        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        isMC    = True if event.run[0] == 1 else False
        nEvent = len(event)
        weights = Weights(len(event), storeIndividual=True)
        logging.debug(fname)
        logging.debug(f'Process {nEvent} Events')

        #
        # Event selection
        #
        event = apply_event_selection( event, self.corrections_metadata[year], cut_on_lumimask=False)

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_4b_selection( event, self.corrections_metadata[year] )

        # selections.add( 'passJetMult', event.passJetMult )
        # selections.add( "passPreSel", event.passPreSel )
        # selections.add( "passFourTag", ( event.passJetMult & event.passPreSel & event.fourTag) )
        # selections.add( 'passBoostedSel', event.passBoostedSel )
        # allcuts = [ 'passJetMult' ]

        #
        #  Cut Flows
        #
        processOutput = {}
        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = {
            'numEvents': nEvent,
        }


        #
        #  genJet -> b-quark matching
        #
        event['bfrom_Z_or_H']= find_genpart(event.GenPart, [5], [23,25])

        event['GenJet', 'selectedBs00'] = (event.GenJet.pt >= 0) & (np.abs(event.GenJet.eta) <= 2.4) & (np.abs(event.GenJet.partonFlavour)==5)
        event['selGenBJet00'] = event.GenJet[event.GenJet.selectedBs00]
        event['matchedGenBJet00'] = event.bfrom_Z_or_H.nearest( event.selGenBJet00, threshold=0.2 )
        event['matchedGenBJet00'] = event.matchedGenBJet00[ ak.argsort(event.matchedGenBJet00.pt, axis=1, ascending=False) ]
        event['matchedGenBJet00'] = event.matchedGenBJet00[~ak.is_none(event.matchedGenBJet00, axis=1)]

        event['matchedGenBJet15'] = event.matchedGenBJet00[event.matchedGenBJet00.pt > 15]
        event['matchedGenBJet20'] = event.matchedGenBJet00[event.matchedGenBJet00.pt > 20]
        event['matchedGenBJet30'] = event.matchedGenBJet00[event.matchedGenBJet00.pt > 30]
        event['matchedGenBJet40'] = event.matchedGenBJet00[event.matchedGenBJet00.pt > 40]


        event['pass4GenBJets00'] = ak.num(event.matchedGenBJet00) >= 4
        event['pass4GenBJets15'] = ak.num(event.matchedGenBJet15) >= 4
        event['pass4GenBJets20'] = ak.num(event.matchedGenBJet20) >= 4
        event['pass4GenBJets30'] = ak.num(event.matchedGenBJet30) >= 4
        event['pass4GenBJets40'] = ak.num(event.matchedGenBJet40) >= 4
        event['pass3GenBJets40'] = ak.num(event.matchedGenBJet40) >= 3
        event['pass2GenBJets40'] = ak.num(event.matchedGenBJet40) >= 2
        event['weight'] = weights.weight()   ### this is for _cutflow

        #
        #  Only look at event where the 4 b-jets are in the tracker
        #
        selev = event[event.pass4GenBJets00]

        for iJ in range(4):
            selev[f"matchedGenBJet00_{iJ}"] = selev.matchedGenBJet00[:, iJ]


        selev["v4j00"] = ak.where(  selev.pass4GenBJets00,
                                    selev.matchedGenBJet00.sum(axis=1),
                                    1e-10 * selev.matchedGenBJet00.sum(axis=1),
                                  )

        selev["v4j15"] = ak.where(  selev.pass4GenBJets15,
                                    selev.matchedGenBJet15.sum(axis=1),
                                    1e-10 * selev.matchedGenBJet15.sum(axis=1),
                                  )


        selev["v4j20"] = ak.where(  selev.pass4GenBJets20,
                                    selev.matchedGenBJet20.sum(axis=1),
                                    1e-10 * selev.matchedGenBJet20.sum(axis=1),
                                  )


        selev["v4j30"] = ak.where(  selev.pass4GenBJets30,
                                    selev.matchedGenBJet30.sum(axis=1),
                                    1e-10 * selev.matchedGenBJet30.sum(axis=1),
                                  )


        selev["v4j40"] = ak.where(  selev.pass4GenBJets40,
                                    selev.matchedGenBJet40.sum(axis=1),
                                    1e-10 * selev.matchedGenBJet40.sum(axis=1),

                                  )


        #
        #  Other Jets
        #
        selev['GenJet', 'other00'] = (np.abs(selev.GenJet.eta) <= 2.4) & (~selev.GenJet.selectedBs00)
        selev['otherGenJets00'] = selev.GenJet[selev.GenJet.other00]
        selev['otherGenJets40'] = selev.otherGenJets00[selev.otherGenJets00.pt > 40]

        selev['pass1OtherJet40'] = ak.num(selev.otherGenJets40) >= 1
        selev['pass2OtherJet40'] = ak.num(selev.otherGenJets40) >= 2

        selev['pass4GenBJets00_1j'] = selev.pass4GenBJets00 & selev.pass1OtherJet40
        selev['pass4GenBJets20_1j'] = selev.pass4GenBJets20 & selev.pass1OtherJet40
        selev['pass4GenBJets40_1j'] = selev.pass4GenBJets40 & selev.pass1OtherJet40

        selev['pass4GenBJetsb203b40_1j_i'] = selev.pass4GenBJets20 & selev.pass3GenBJets40 & selev.pass1OtherJet40
        selev['pass4GenBJetsb203b40_1j_e'] = selev.pass4GenBJetsb203b40_1j_i & ~selev.pass4GenBJets40

        selev['pass4GenBJets2b202b40_2j_i'] = selev.pass4GenBJets20 & selev.pass2GenBJets40 & selev.pass2OtherJet40
        selev['pass4GenBJets2b202b40_2j_e'] = selev.pass4GenBJets2b202b40_2j_i & ~selev.pass4GenBJets40 & ~selev.pass4GenBJetsb203b40_1j_i


        # selev['Jet', 'selected'] = (selev.Jet.pt >= 40) & (np.abs(selev.Jet.eta) <= 2.4)
        # selev['selJet'] = selev.Jet[ selev.Jet.selected ]
        # selev['matchedRecoJet'] = selev.bfromH.nearest( selev.selJet, threshold=0.2 )
        # selev['matchedRecoJet'] = selev.matchedRecoJet[ ak.argsort(selev.matchedRecoJet.pt, axis=1, ascending=False) ]

        #
        #  Hacks for plotting (all events count as SR and fourTag
        #
        selev["region"] = np.full(len(selev), 0b10)
        selev["tag"]    = np.full(len(selev), 4)

        selections = PackedSelection()
        selections.add( "passHLT",            event.passHLT )
        selections.add( "pass4GenBJets00",    event.pass4GenBJets00)
        selections.add( "pass4GenBJets20",    event.pass4GenBJets20)
        selections.add( "pass4GenBJets40",    event.pass4GenBJets40)
        selections.add( "pass3GenBJets40",    event.pass3GenBJets40)
        selections.add( "pass2GenBJets40",    event.pass2GenBJets40)

        event_pass1OtherJet40 = np.full(len(event), False)
        event_pass1OtherJet40[event.pass4GenBJets00] = selev.pass1OtherJet40
        event["pass1OtherJet40"] = event_pass1OtherJet40

        event_pass2OtherJet40 = np.full(len(event), False)
        event_pass2OtherJet40[event.pass4GenBJets00] = selev.pass2OtherJet40
        event["pass2OtherJet40"] = event_pass2OtherJet40


        selections.add( "pass1OtherJet40",    event.pass1OtherJet40)
        selections.add( "pass2OtherJet40",    event.pass2OtherJet40)

        self._cutFlow = cutflow_4b()
        self._cutFlow.fill( "all", event, allTag=True)
        self._cutFlow.fill( "pass4GenBJets00",    event[selections.require(pass4GenBJets00=True)], allTag=True)
        self._cutFlow.fill( "pass4GenBJets20",    event[selections.require(pass4GenBJets20=True)], allTag=True)
        self._cutFlow.fill( "pass4GenBJets40",    event[selections.require(pass4GenBJets40=True)], allTag=True)
        self._cutFlow.fill( "pass4GenBJets00_1j", event[selections.require(pass4GenBJets00=True, pass1OtherJet40=True)], allTag=True)
        self._cutFlow.fill( "pass4GenBJets20_1j", event[selections.require(pass4GenBJets20=True, pass1OtherJet40=True)], allTag=True)
        self._cutFlow.fill( "pass4GenBJets40_1j", event[selections.require(pass4GenBJets40=True, pass1OtherJet40=True)], allTag=True)
        self._cutFlow.fill( "pass4GenBJetsb203b40_1j_i", event[selections.require(pass4GenBJets20=True, pass3GenBJets40=True, pass1OtherJet40=True)], allTag=True)
        self._cutFlow.fill( "pass4GenBJetsb203b40_1j_e", event[selections.require(pass4GenBJets20=True, pass3GenBJets40=True, pass1OtherJet40=True, pass4GenBJets40=False)], allTag=True)
        self._cutFlow.fill( "pass4GenBJets2b202b40_2j_i", event[selections.require(pass4GenBJets20=True, pass2GenBJets40=True, pass2OtherJet40=True)], allTag=True)
        self._cutFlow.fill( "pass4GenBJets2b202b40_2j_e", event[selections.require(pass4GenBJets20=True, pass2GenBJets40=True, pass2OtherJet40=True, pass4GenBJets40=False, pass3GenBJets40=False)], allTag=True)

        fill = Fill(process=processName, year=year, weight="weight")

        hist = Collection( process=[processName],
                           year=[year],
                           tag=[3, 4, 0],  # 3 / 4/ Other
                           region=[2, 1, 0],  # SR / SB / Other
                           **dict((s, ...) for s in self.histCuts)
                          )

        #
        #  Jets
        #
        fill += LorentzVector.plot(('matchedGenBJet20', 'Selected Gen Candidate'), 'matchedGenBJet20')
        fill += LorentzVector.plot(('matchedGenBJet40', 'Selected Gen Candidate'), 'matchedGenBJet40')

        fill += LorentzVector.plot(('otherGenJet00', 'Non matched Gen Candidate'), 'otherGenJets00')
        fill += LorentzVector.plot(('otherGenJet40', 'Non matched Gen Candidate'), 'otherGenJets40')


        for iJ in range(4):
            fill += LorentzVector.plot((f'genBJet{iJ}', f'Matched Gen {iJ}'), f'matchedGenBJet00_{iJ}', skip=["n"])


        #
        #  m4js
        #
        fill += LorentzVector.plot_pair( ("v4j00", R"$HH_{4b}$"), "v4j00", skip=["n", "dr", "dphi", "st"], bins={"mass": (120, 0, 1200)}, )
        fill += LorentzVector.plot_pair( ("v4j15", R"$HH_{4b}$"), "v4j15", skip=["n", "dr", "dphi", "st"], bins={"mass": (120, 0, 1200)}, )
        fill += LorentzVector.plot_pair( ("v4j20", R"$HH_{4b}$"), "v4j20", skip=["n", "dr", "dphi", "st"], bins={"mass": (120, 0, 1200)}, )
        fill += LorentzVector.plot_pair( ("v4j30", R"$HH_{4b}$"), "v4j30", skip=["n", "dr", "dphi", "st"], bins={"mass": (120, 0, 1200)}, )
        fill += LorentzVector.plot_pair( ("v4j40", R"$HH_{4b}$"), "v4j40", skip=["n", "dr", "dphi", "st"], bins={"mass": (120, 0, 1200)}, )


        fill(selev, hist)

        self._cutFlow.addOutput(processOutput, event.metadata["dataset"])

        output = processOutput | hist.output

        return output

        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f"{chunk}{nEvent/elapsed:,.0f} events/s")


    def postprocess(self, accumulator):
        return accumulator
