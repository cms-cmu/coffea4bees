import unittest
from unittest.mock import MagicMock, patch
import awkward as ak
import numpy as np
import os
import sys

# Ensure workspace root is in path
sys.path.append(os.getcwd())

from coffea4bees.analysis.trigger_emulator.TriggerSFVectorized import TriggerSFVectorized

class TestTriggerSFVectorized(unittest.TestCase):
    
    def setUp(self):
        # Create dummy data for lookups that resembles a turn-on curve
        self.dummy_x = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 500.0])
        self.dummy_y = np.array([0.05, 0.1,  0.5,  0.8,  0.9,   0.95, 1.0])
        self.dummy_err = np.array([0.01]*7)
        
        self.dummy_lookup_data = {
            "x": self.dummy_x,
            "y": self.dummy_y,
            "y_err_up": self.dummy_err,
            "y_err_down": self.dummy_err,
            "type": "graph"
        }

    def _mock_load_root_file(self, inst, path, is_l1=False):
        """
        Mock for _load_root_file that populates the lookup dictionaries
        with keys expected by the calculation methods.
        """
        store = inst.data_lookups
        mc_store = inst.mc_lookups
        
        # Comprehensive list of keys used in TriggerSFVectorized for all years
        keys = [
            # 2018 / 2017 keys
            "L1filterHT", "QuadCentralJet30", "CaloQuadJet30HT320", "CaloQuadJet30HT300",
            "PFCentralJetLooseIDQuad30", "1PFCentralJetLooseID75", 
            "2PFCentralJetLooseID60", "3PFCentralJetLooseID45", 
            "4PFCentralJetLooseID40", "PFCentralJetsLooseIDQuad30HT330", "PFCentralJetsLooseIDQuad30HT300",
            "BTagCaloDeepCSVp17Double_Efficiency", "BTagPFDeepCSV4p5Triple_Efficiency",
            "BTagCaloCSVp05Double_Efficiency", "BTagPFCSVp070Triple_Efficiency",
            
            # L1
            "L1calojetsPFHT", "L1All_postEE", "L1All_preEE", "L1_HTT280er_postBPix", "L1_HTT280er_preBPix",
            
            # 2022 PNet keys
            "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_4PixelOnlyPFCentralJetTightIDPt20",
            "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_3PixelOnlyPFCentralJetTightIDPt30",
            "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_2PixelOnlyPFCentralJetTightIDPt40",
            "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_1PixelOnlyPFCentralJetTightIDPt60",
            "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_4PFCentralJetTightIDPt35",
            "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_3PFCentralJetTightIDPt40",
            "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_2PFCentralJetTightIDPt50",
            "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_1PFCentralJetTightIDPt70",
            "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_BTagCentralJetPt35PFParticleNet2BTagSum0p65",
            
            # 2023 PostBPix
            "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55_4PixelOnlyPFCentralJetTightIDPt20",
            "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55_4PFCentralJetTightIDPt30",
            "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55_PFHT280Jet30",
            "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55_PFCentralJetPt30PNet2BTagMean0p55",
            
            # 2D Placeholders
            "JetLeg", "InclusiveBTagLeg"
        ]
        
        for k in keys:
             # Add both bare key and key with "Efficiency" suffix just in case
             store[k] = self.dummy_lookup_data
             mc_store[k] = self.dummy_lookup_data
             store[k + "_Efficiency"] = self.dummy_lookup_data
             mc_store[k + "_Efficiency"] = self.dummy_lookup_data

    def test_2018_DeepJet(self):
        """Test 2018 logic with DeepJet"""
        with patch.object(TriggerSFVectorized, '_load_root_file', autospec=True) as mock_load:
            mock_load.side_effect = self._mock_load_root_file
            
            tsf = TriggerSFVectorized(2018, map_path="dummy", tagger="DeepJet")
            
            # Create dummy events
            # Event 1: Passes everything (high pt, high btag)
            # Event 2: Fails pt (low pt)
            jets_pt = [
                [200.0, 150.0, 100.0, 80.0], 
                [30.0, 30.0, 30.0, 20.0]
            ]
            jets_btag = [
                [0.9, 0.9, 0.9, 0.9],
                [0.1, 0.1, 0.1, 0.1]
            ]
            
            events = ak.Array({
                "Jet": {
                    "pt": jets_pt,
                    "eta": [[0.0]*4, [0.0]*4],
                    "btagDeepFlavB": jets_btag,
                    "pfht_selected": [[True]*4, [True]*4],
                    "ht_selected": [[True]*4, [True]*4],
                # }?
            # })
                    "btagDeepFlavB": jets_btag,
                    "pfht_selected": [[True]*4, [True]*4],
                    "ht_selected": [[True]*4, [True]*4]
                }
            })
            
            data, mc, sf = tsf.calculate_event_sf(events)
            
            self.assertEqual(len(data), 2)
            # High pt event should have high efficiency (close to 1.0 given dummy data max is 1.0)
            self.assertTrue(data[0] > 0.0)
            
            # Low pt event should have lower efficiency
            self.assertTrue(data[1] < data[0])

    def test_2022_PNet(self):
        """Test 2022 logic with PNet (checks for pn_b column)"""
        with patch.object(TriggerSFVectorized, '_load_root_file', autospec=True) as mock_load:
            mock_load.side_effect = self._mock_load_root_file
            
            tsf = TriggerSFVectorized(2022, map_path="dummy", tagger="PNet")
            
            # 2022 uses PNet (pn_b)
            jets_pt = [[100.0, 80.0, 60.0, 40.0]]
            jets_pn_b = [[0.9, 0.8, 0.5, 0.1]]
            
            events = ak.Array({
                "Jet": {
                    "pt": jets_pt,
                    "pn_b": jets_pn_b,
                    "pfht_selected": [[True]*4],
                    "ht_selected": [[True]*4]
                }
            })
            
            data, mc, sf = tsf.calculate_event_sf(events)
            self.assertEqual(len(sf), 1)
            self.assertTrue(data[0] > 0.0)

    def test_2017_ParT(self):
        """Test 2017 logic"""
        with patch.object(TriggerSFVectorized, '_load_root_file', autospec=True) as mock_load:
            mock_load.side_effect = self._mock_load_root_file
            
            # 2017 uses different keys
            tsf = TriggerSFVectorized(2017, map_path="dummy", tagger="ParT") # Tagger affects B-score column selection
            
            jets_pt = [[100.0, 80.0, 60.0, 40.0]]
            jets_b = [[0.9, 0.8, 0.5, 0.1]]
            
            events = ak.Array({
                "Jet": {
                    "pt": jets_pt,
                    "btagDeepFlavB": jets_b, # 2017 fallback to DeepFlavB
                    "pfht_selected": [[True]*4],
                    "ht_selected": [[True]*4]
                }
            })
            
            data, mc, sf = tsf.calculate_event_sf(events)
            self.assertEqual(len(sf), 1)

    def test_missing_column_fallback(self):
        """Test that missing pn_b falls back to DeepFlavB for sorting"""
        with patch.object(TriggerSFVectorized, '_load_root_file', autospec=True) as mock_load:
            mock_load.side_effect = self._mock_load_root_file
            
            tsf = TriggerSFVectorized(2022, map_path="dummy", tagger="PNet")
            
            events = ak.Array({
                "Jet": {
                    "pt": [[100.0, 80.0, 60.0, 40.0]],
                    "btagDeepFlavB": [[0.9, 0.8, 0.5, 0.1]],
                    # No pn_b, should fallback
                    "pfht_selected": [[True]*4],
                    "ht_selected": [[True]*4]
                }
            })
            
            # Should not crash, should use DeepFlavB
            data, mc, sf = tsf.calculate_event_sf(events)
            self.assertEqual(len(sf), 1)

if __name__ == '__main__':
    unittest.main()
