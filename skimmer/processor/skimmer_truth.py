import logging

import numpy as np
from coffea.analysis_tools import PackedSelection, Weights
import awkward as ak
from coffea4bees.analysis.helpers.truth_tools import find_genpart
from coffea4bees.skimmer.processor.skimmer_4b_base import Skimmer4b


class Skimmer(Skimmer4b):
    def __init__(self, loosePtForSkim=False, mc_outlier_threshold: int | None = 200, corrections_metadata: dict = None, *args, **kwargs):
        self.skip_collections = kwargs.get("skip_collections", [])
        self.skip_branches    = kwargs.get("skip_branches", [])
        super().__init__(
            mc_outlier_threshold=mc_outlier_threshold,
            corrections_metadata=corrections_metadata,
            object_selection_cfg=None,
            *args, **kwargs,
        )


    def select(self, event):
        m = self._parse_event_metadata(event)
        year, dataset, processName, config = m.year, m.dataset, m.processName, m.config
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

        event['pass4GenBJets'] = (ak.num(event.matchedGenBJet) == 4)

        selection = event.pass4GenBJets
        event = event[selection]

        out_branches = {
            "GenJet_eta":             event.matchedGenBJet.eta,
            "GenJet_pt":              event.matchedGenBJet.pt,
            "GenJet_mass":            event.matchedGenBJet.mass,
            "GenJet_phi":             event.matchedGenBJet.phi,
            "GenJet_hadronFlavour":   event.matchedGenBJet.hadronFlavour,
            "GenJet_partonFlavour":   event.matchedGenBJet.partonFlavour,
        }

        #
        #  Need to skip all the other jet branches to make sure they have the same number of jets
        #
        if not hasattr(self, "_branch_filter_initialized") or not self._branch_filter_initialized:
            skip_branches = set(self.skip_branches or [])
            for f in event.GenJet.fields:
                bname = f"GenJet_{f}"
                if bname not in out_branches:
                    skip_branches.add(bname)
            self.skip_branches = list(skip_branches)
            self.update_branch_filter(self.skip_collections, self.skip_branches)
            self._branch_filter_initialized = True

        branches = ak.Array(out_branches)

        processOutput = {}
        processOutput["total_event"] = len(event)
        return (selection,
                branches,
                processOutput)
