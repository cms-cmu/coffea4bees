import logging

import numpy as np
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea.analysis_tools import PackedSelection, Weights
import awkward as ak
from coffea4bees.analysis.helpers.truth_tools import find_genpart
from coffea4bees.skimmer.processor.skimmer_4b_base import Skimmer4b


class Skimmer(Skimmer4b):
    def __init__(self, loosePtForSkim=False, mc_outlier_threshold: int | None = 200, corrections_metadata: dict = None, *args, **kwargs):
        super().__init__(
            mc_outlier_threshold=mc_outlier_threshold,
            corrections_metadata=corrections_metadata,
            object_selection_cfg=None,
            *args, **kwargs,
        )
        self.skip_collections = kwargs["skip_collections"]
        self.skip_branches    = kwargs["skip_branches"]


    def select(self, event):

        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']

        #
        # Set process and datset dependent flags
        #
        config = processor_config(processName, dataset, event)
        logging.debug(f'config={config}\n')

        event['bfromHorZ_all']= find_genpart(event.GenPart, [5], [23, 25])

        if "status" in event.bfromHorZ_all.fields:
            event['bfromHorZ'] = event.bfromHorZ_all[event.bfromHorZ_all.status == 23]
        else:
            logging.warning(f"\nStatus Missing for GenParticles in dataset {self.dataset}\n")
            event['bfromHorZ'] = event.bfromHorZ_all

        event['GenJet', 'selectedBs'] = (np.abs(event.GenJet.partonFlavour)==5) & (np.abs(event.GenJet.eta) < 2.5) & (event.GenJet.pt >= 40)
        event['selGenBJet'] = event.GenJet[event.GenJet.selectedBs]
        event['matchedGenBJet'] = event.bfromHorZ.nearest( event.selGenBJet, threshold=10 )
        event["matchedGenBJet"] = event.matchedGenBJet[~ak.is_none(event.matchedGenBJet, axis=1)]

        print("nselGenBJet",ak.num(event.selGenBJet),"\n")
        print("selGenBJet",event.selGenBJet.pt[0:2].tolist(),"\n")
        print("nbsfromHorZ",ak.num(event.bfromHorZ),"\n")
        print("bsfromHorZ",event.bfromHorZ.pt[0:2].tolist(),"\n")
        print("nmatchGenBJet",ak.num(event.matchedGenBJet),"\n")
        print("matchedGenBJet",event.matchedGenBJet.pt[0:2].tolist(),"\n")

        event['pass4GenBJets'] = (ak.num(event.matchedGenBJet) == 4)

        #event['pass4GenBJets'] = event.event % 11 == 0

        ### weights = Weights(len(event), storeIndividual=True)

        #
        # general event weights
        #
        ### if config["isMC"]:
        ###     weights.add( "genweight_", event.genWeight )

        ### selections = PackedSelection()
        ### selections.add( 'pass4GenBJets',   event.pass4GenBJets )

        ### event["weight"] = weights.weight()

        ### cumulative_cuts = []
        ### self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )
        ###
        ### all_cuts = ["pass4GenBJets"]
        ###
        ### for cut in all_cuts:
        ###     cumulative_cuts.append(cut)
        ###     self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        selection = event.pass4GenBJets & self.preselect(event) # & event.passPreSel
        event = event[selection]

        n_genJets = ak.num(event.GenJet)
        total_genJets = int(ak.sum(n_genJets))

        n_selGenJets = ak.num(event.matchedGenBJet)

        out_branches = {
            "GenJet_eta":             ak.unflatten(ak.flatten(event.matchedGenBJet.eta          ).tolist(), n_selGenJets),
            "GenJet_pt":              ak.unflatten(ak.flatten(event.matchedGenBJet.pt           ).tolist(), n_selGenJets),
            "GenJet_mass":            ak.unflatten(ak.flatten(event.matchedGenBJet.mass         ).tolist(), n_selGenJets),
            "GenJet_phi":             ak.unflatten(ak.flatten(event.matchedGenBJet.phi          ).tolist(), n_selGenJets),
            "GenJet_hadronFlavour":   ak.unflatten(ak.flatten(event.matchedGenBJet.hadronFlavour).tolist(), n_selGenJets),
            "GenJet_partonFlavour":   ak.unflatten(ak.flatten(event.matchedGenBJet.partonFlavour).tolist(), n_selGenJets),
        }

        #
        #  Need to skip all the other jet branches to make sure they have the same number of jets
        #
        for f in event.GenJet.fields:
            bname = f"GenJet_{f}"
            if bname not in out_branches:
                self.skip_branches.append(bname)

        self.update_branch_filter(self.skip_collections, self.skip_branches)
        branches = ak.Array(out_branches)


        processOutput = {}
        processOutput["total_event"] = len(event)
        #processOutput["sel_event"] = len(selev)
        return (selection,
                branches,
                processOutput)
