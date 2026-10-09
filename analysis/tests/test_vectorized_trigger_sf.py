import unittest
from unittest.mock import MagicMock, patch
import awkward as ak
import numpy as np
import os
import sys
from coffea4bees.analysis.trigger_emulator.helpers import compute_emulation_vars

# Ensure workspace root is in path
sys.path.append(os.getcwd())

from coffea4bees.analysis.trigger_emulator.TriggerSFVectorized import TriggerSFVectorized, trigger_era_code

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

            events['Jet', 'btagScore'] = events.Jet.btagDeepFlavB
            compute_emulation_vars(events, useOnlyTop4=True)


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

            events['Jet', 'btagScore'] = events.Jet.pn_b
            compute_emulation_vars(events)


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

            events['Jet', 'btagScore'] = events.Jet.btagDeepFlavB
            compute_emulation_vars(events, useOnlyTop4=True)

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

            events['Jet', 'btagScore'] = events.Jet.btagDeepFlavB

            compute_emulation_vars(events)
            # Should not crash, should use DeepFlavB
            data, mc, sf = tsf.calculate_event_sf(events)
            self.assertEqual(len(sf), 1)

    def test_2023_jet_leg_2d(self):
        """2023 jet leg is a 2D (PF HT x 4th-jet pT) map; data and MC enter the event efficiency"""
        jet_leg = {
            "type": "hist2d",
            "x_edges": np.array([200.0, 300.0, 1000.0]),   # PF HT
            "y_edges": np.array([30.0, 50.0, 150.0]),      # 4th-jet pT
            "eff": np.array([[0.5, 0.9],
                             [0.8, 1.0]]),
        }

        def load(inst, path, is_l1=False):
            self._mock_load_root_file(inst, path, is_l1)
            inst.data_lookups["Data__Efficiency2D_Inclusive-PerLeg-ForthJetPt-vs-alljets_PFHT"] = jet_leg
            mc = dict(jet_leg, eff=jet_leg["eff"] * 0.5)
            inst.mc_lookups["Simulation__Efficiency2D_Inclusive-PerLeg-ForthJetPt-vs-alljets_PFHT"] = mc

        with patch.object(TriggerSFVectorized, '_load_root_file', autospec=True) as mock_load:
            mock_load.side_effect = load
            tsf = TriggerSFVectorized(2023, map_path="dummy", tagger="PNet")

            name = "Efficiency2D_Inclusive-PerLeg-ForthJetPt-vs-alljets_PFHT"
            # in range, below range (clipped to the first bin), above range (clipped to the last)
            ht = ak.Array([250.0, 100.0, 5000.0])
            pt4 = ak.Array([60.0, 10.0, 500.0])
            np.testing.assert_allclose(tsf.lookup_efficiency_2d(name, ht, pt4), [0.9, 0.5, 1.0])
            np.testing.assert_allclose(tsf.lookup_efficiency_2d(name, ht, pt4, is_data=False), [0.45, 0.25, 0.5])

            # the jet leg is now in the product: MC eff is half the data eff for every other leg equal
            d, m, sf = tsf._calculate_2023(pt4, ht, ht, ak.Array([1.0, 1.0, 1.0]))
            np.testing.assert_allclose(ak.to_numpy(sf), 2.0)

    def test_2d_map_missing_raises(self):
        """A missing 2D map is an error, not a silent efficiency of 1"""
        with patch.object(TriggerSFVectorized, '_load_root_file', autospec=True) as mock_load:
            mock_load.side_effect = self._mock_load_root_file
            tsf = TriggerSFVectorized(2023, map_path="dummy", tagger="PNet")
            with self.assertRaises(KeyError):
                tsf.lookup_efficiency_2d("Efficiency2D_Inclusive", ak.Array([300.0]), ak.Array([40.0]))

    def test_tefficiency_2d_fill(self):
        """Empty bins: forward-filled along y within a row, empty rows copy the nearest filled row"""
        class H:
            def __init__(self, vals, cls="TH2D"):
                self.classname, self._v = cls, np.array(vals, dtype=float)
            def values(self):
                return self._v
            def member(self, name):
                ax = MagicMock()
                ax.edges.return_value = np.arange(self._v.shape[0 if name == "fXaxis" else 1] + 1, dtype=float)
                return ax

        total = H([[0, 0, 0], [10, 0, 0], [10, 10, 10]])
        passed = H([[0, 0, 0], [5, 0, 0], [8, 9, 10]])
        teff = MagicMock()
        teff.member.side_effect = lambda n: {"fTotalHistogram": total, "fPassedHistogram": passed}[n]
        table = TriggerSFVectorized._tefficiency_2d(teff)
        np.testing.assert_allclose(table["eff"], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.8, 0.9, 1.0]])

    def test_era_codes(self):
        """Each Run 3 era gets its own code; other years keep the calendar year"""
        self.assertEqual(trigger_era_code("2022_preEE", "2022"), 2021)
        self.assertEqual(trigger_era_code("2022_EE", "2022"), 2022)
        self.assertEqual(trigger_era_code("2023_preBPix", "2023"), 2023)
        self.assertEqual(trigger_era_code("2023_BPix", "2023"), 2020)
        self.assertEqual(trigger_era_code("2024", "2024"), 2024)
        self.assertEqual(trigger_era_code("UL18", "2018"), 2018)

    def test_l1_curve_per_era(self):
        """preEE/postEE and preBPix/postBPix read different L1 curves"""
        names = []
        orig = TriggerSFVectorized.lookup_efficiency

        def spy(inst, name, values, is_data=True):
            names.append(name)
            return orig(inst, name, values, is_data)

        events = ak.Array({"Jet": {"pt": [[100.0, 80.0, 60.0, 40.0]], "pn_b": [[0.9, 0.8, 0.5, 0.1]],
                                   "pfht_selected": [[True] * 4], "ht_selected": [[True] * 4]}})
        events['Jet', 'btagScore'] = events.Jet.pn_b
        compute_emulation_vars(events)
        jet_leg = {"type": "hist2d", "x_edges": np.array([0.0, 2000.0]), "y_edges": np.array([0.0, 500.0]),
                   "eff": np.array([[1.0]])}

        def load(inst, path, is_l1=False):
            self._mock_load_root_file(inst, path, is_l1)
            inst.data_lookups["Data__Efficiency2D_Inclusive-PerLeg-ForthJetPt-vs-alljets_PFHT"] = jet_leg
            inst.mc_lookups["Simulation__Efficiency2D_Inclusive-PerLeg-ForthJetPt-vs-alljets_PFHT"] = jet_leg

        expected = {2021: "L1All_preEE", 2022: "L1All_postEE",
                    2023: "L1_HTT280er_preBPix", 2020: "L1_HTT280er_postBPix"}
        with patch.object(TriggerSFVectorized, '_load_root_file', autospec=True) as mock_load, \
             patch.object(TriggerSFVectorized, 'lookup_efficiency', autospec=True, side_effect=spy):
            mock_load.side_effect = load
            for code, l1 in expected.items():
                names.clear()
                TriggerSFVectorized(code, map_path="dummy", tagger="PNet").calculate_event_sf(events)
                self.assertIn(l1, names, code)
                other = set(expected.values()) - {l1}
                self.assertFalse(other & set(names), (code, other & set(names)))

    def test_missing_1d_curve_warns(self):
        """A missing 1D curve still gives 1.0 but is logged (once)"""
        with patch.object(TriggerSFVectorized, '_load_root_file', autospec=True) as mock_load:
            mock_load.side_effect = self._mock_load_root_file
            tsf = TriggerSFVectorized(2022, map_path="dummy", tagger="PNet")
            with self.assertLogs(level="WARNING") as logs:
                eff, _, _ = tsf.lookup_efficiency("NoSuchLeg", ak.Array([50.0, 60.0]))
                tsf.lookup_efficiency("NoSuchLeg", ak.Array([50.0]))
            np.testing.assert_allclose(ak.to_numpy(eff), 1.0)
            self.assertEqual(sum("NoSuchLeg" in m for m in logs.output), 1)

    def test_real_maps_run3_eras(self):
        """The shipped Run 3 maps have every curve each era asks for (no fallback warning)"""
        map_path = "coffea4bees/analysis/trigger_emulator/data/"
        if not os.path.exists(os.path.join(map_path, "TriggerEfficiency_Fit_2023_18April2025.root")):
            self.skipTest("trigger maps not available (run from the barista root)")
        events = ak.Array({"Jet": {"pt": [[150.0, 100.0, 70.0, 50.0], [60.0, 50.0, 40.0, 35.0]],
                                   "pn_b": [[0.99, 0.95, 0.5, 0.1], [0.9, 0.8, 0.2, 0.1]],
                                   "pfht_selected": [[True] * 4] * 2, "ht_selected": [[True] * 4] * 2}})
        events['Jet', 'btagScore'] = events.Jet.pn_b
        compute_emulation_vars(events)
        sfs = {}
        for era, code in [("2022_preEE", 2021), ("2022_EE", 2022), ("2023_preBPix", 2023), ("2023_BPix", 2020),
                          ("2024", 2024)]:
            tsf = TriggerSFVectorized(code, map_path=map_path, tagger="PNet")
            d, m, sf = tsf.calculate_event_sf(events)
            self.assertEqual(tsf._warned_missing, set(), era)
            self.assertTrue(np.all((ak.to_numpy(d) > 0) & (ak.to_numpy(d) <= 1)), era)
            sfs[era] = ak.to_numpy(sf)
        # the per-era L1 curves differ, so the eras of a year must not give identical SFs
        self.assertFalse(np.allclose(sfs["2022_preEE"], sfs["2022_EE"]))
        self.assertFalse(np.allclose(sfs["2023_preBPix"], sfs["2023_BPix"]))

if __name__ == '__main__':
    unittest.main()
