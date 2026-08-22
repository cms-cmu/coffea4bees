import unittest
import sys
import os
import awkward as ak
from coffea.nanoevents.methods import vector

sys.path.insert(0, os.getcwd())
from coffea4bees.analysis.helpers.candidates_selection_ttHbb import create_cand_jet_dijet_quadjet_ttHbb


class quadJetTTHbbTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jets = {
            'pt': [[128.5, 137.2, 102.9, 77.4, 71.3, 62.1], [216.8, 111.1, 56.3, 60.3, 59.5, 12.9]],
            'eta': [[-0.47, 0.41, 2.29, -0.44, -3.07, 1.74], [1.09, 0.61, 0.49, 1.81, -0.16, 3.43]],
            'phi': [[-1.02, 1.61, 0.86, 2.96, -2.46, -1.00], [1.56, -1.58, -2.14, -1.45, -1.16, 2.35]],
            'mass': [[10.8, 13.9, 11.9, 7.6, 12.1, 11.7], [34.2, 13.6, 6.3, 9.2, 11.4, 0.0]],
            'btagScore': [[0.99, 0.99, 0.004, 0.99, -1.0, 0.019], [0.99, 0.004, 0.99, 0.81, 0.99, -1.0]],
            'bRegCorr': [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            'puId': [[10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10]],
            'jetId': [[6, 6, 6, 6, 6, 6], [6, 6, 6, 6, 6, 4]],
            'selected': [[True, True, True, True, False, True], [True, True, True, True, True, False]],
            'selected_loose': [[True, True, True, True, True, True], [True, True, True, True, True, False]],
        }

        cls.input_jets = ak.zip(
            {
                "pt": jets["pt"],
                "eta": jets["eta"],
                "phi": jets["phi"],
                "mass": jets["mass"],
                "btagScore": jets["btagScore"],
                "bRegCorr": jets["bRegCorr"],
                "puId": jets["puId"],
                "jetId": jets["jetId"],
                "selected": jets["selected"],
                "selected_loose": jets["selected_loose"],
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        class DictToClass:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
            def __setitem__(self, key, value):
                setattr(self, key, value)
            def __getitem__(self, key):
                return getattr(self, key)
            @property
            def fields(self):
                return list(self.__dict__.keys())

        event_dict = {
            "Jet": cls.input_jets,
            "event": ak.Array([1, 2]),
        }
        cls.event = DictToClass(event_dict)

    def test_quadJets_ttHbb(self):
        selev = create_cand_jet_dijet_quadjet_ttHbb(self.event)
        self.assertIn("quadJet_selected", selev.fields)
        self.assertIn("leadStM_selected", selev.fields)
        self.assertIn("sublStM_selected", selev.fields)
        self.assertIn("region", selev.fields)
        self.assertTrue(ak.all(selev.passDiJetMass))
        self.assertTrue(len(selev.quadJet_selected.lead.mass) == 2)


if __name__ == '__main__':
    unittest.main()
