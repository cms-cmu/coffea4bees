import unittest
import awkward as ak
import numpy as np
import os
import sys

# Ensure workspace root is in path
sys.path.append(os.getcwd())

from coffea4bees.analysis.trigger_emulator.TriggerSFVectorized import TriggerSFVectorized

class TestTriggerSFVectorized(unittest.TestCase):
    def test_calculate_sf_2018(self):
        # Create a dummy event array
        # 2 events.
        # Event 1: 4 jets, should pass basics
        # Event 2: 3 jets (should be padded)
        
        jets_pt = [[100.0, 80.0, 60.0, 40.0], [50.0, 40.0, 30.0]]
        jets_eta = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        jets_btag = [[0.9, 0.8, 0.1, 0.1], [0.1, 0.1, 0.1]]
        
        events = ak.Array({
            "Jet": {
                "pt": jets_pt,
                "eta": jets_eta,
                "btagDeepFlavB": jets_btag
            }
        })
        
        # Check if file exists, otherwise we might just get defaults (which is fine for unit test of structure)
        map_path = "coffea4bees/analysis/trigger_emulator/data/"
        
        tsf = TriggerSFVectorized(2018, map_path=map_path)
        
        data_eff, mc_eff, sf = tsf.calculate_event_sf(events)
        
        # Check shapes
        self.assertEqual(len(data_eff), 2)
        self.assertEqual(len(mc_eff), 2)
        self.assertEqual(len(sf), 2)
        
        # Check values are valid floats
        import math
        self.assertFalse(math.isnan(data_eff[0]))
        self.assertFalse(math.isnan(mc_eff[0]))
        
        print(f"Data eff: {data_eff}")
        print(f"MC eff: {mc_eff}")
        
    def test_calculate_sf_padding(self):
        # Test extreme case with 0 jets
        jets_pt = [[]]
        jets_eta = [[]]
        jets_btag = [[]]
        events = ak.Array({
            "Jet": {
                "pt": jets_pt, 
                "eta": jets_eta, 
                "btagDeepFlavB": jets_btag
            }
        })
        
        tsf = TriggerSFVectorized(2018, map_path="coffea4bees/analysis/trigger_emulator/data/")
        data_eff, mc_eff, sf = tsf.calculate_event_sf(events)
        
        # Should handle it gracefully (likely returning 0 efficiency or 1 depending on logic)
        self.assertEqual(len(data_eff), 1)

if __name__ == '__main__':
    unittest.main()
