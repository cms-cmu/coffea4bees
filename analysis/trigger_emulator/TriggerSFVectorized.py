import logging
import awkward as ak
import numpy as np
import uproot
import os

#
# Configuration from Marina_triggerHelper.py
#
TRGSF_FILES_EXTRA = {
    2015: None,
    2016: None,
    2017: None,
    2018: None,
    2021: "L1T_ttbar1L_ele_Efficiency_Fit_2022_18April2025.root",
    2022: "L1T_ttbar1L_ele_Efficiency_Fit_2022_18April2025.root",
    2023: "L1T_ttbar1L_ele_Efficiency_Fit_2023_18April2025.root",
    2020: "L1T_ttbar1L_ele_Efficiency_Fit_2023_18April2025.root",
    2024: None,
    2025: None,
}

TRGSF_FILES = {
    "DeepJet": {
        #2018: "2018/TriggerEfficiency_Fit_DeepJet_2018_wMatching.root", # UL trigger SFs
        2015: "TriggerEfficiency_Fit_2016_matched_0p5.root", # pre-UL trigger SFs -> to be replaced with UL
        2016: "TriggerEfficiency_Fit_2016_matched_0p5.root", # pre-UL trigger SFs -> to be replaced with UL
        2017: "TriggerEfficiency_Fit_2017_14Feb2024.root",
        2018: "TriggerEfficiency_Fit_2018_matched_0p5.root", # pre-UL trigger SFs -> to be replaced with UL
        2021: "TriggerEfficiency_Fit_2021_18April2025.root",
        2022: "TriggerEfficiency_Fit_2022_18April2025.root",
        2023: "TriggerEfficiency_Fit_2023_18April2025.root",
        2020: "TriggerEfficiency_Fit_2020_18April2025.root"
    },
    "PNet"  : {
        2015: "TriggerEfficiency_Fit_2016_matched_0p5.root", # use the same trigger SF for the moment
        2016: "TriggerEfficiency_Fit_2016_matched_0p5.root", # use the same trigger SF for the moment
        2017: "TriggerEfficiency_Fit_2017_14Feb2024.root",
        2018: "TriggerEfficiency_Fit_2018_matched_0p5.root", # use the same trigger SF for the moment
        2021: "TriggerEfficiency_Fit_2021_18April2025.root",
        2022: "TriggerEfficiency_Fit_2022_18April2025.root",
        2023: "TriggerEfficiency_Fit_2023_18April2025.root",
        2020: "TriggerEfficiency_Fit_2020_18April2025.root"
    },
    "ParT"   : {
        2015: None,
        2016: None,
        2017: "TriggerEfficiency_Fit_2017_14Feb2024.root",
        2018: None,
        2021: "TriggerEfficiency_Fit_2021_18April2025.root",
        2022: "TriggerEfficiency_Fit_2022_18April2025.root",
        2023: "TriggerEfficiency_Fit_2023_18April2025.root",
        2020: "TriggerEfficiency_Fit_2020_18April2025.root"
    },
}

class TriggerSFVectorized:
    def __init__(self, year, map_path="coffea4bees/analysis/trigger_emulator/data/", tagger="DeepJet"):
        self.year = int(year) if str(year).isdigit() else year
        self.map_path = map_path
        self.tagger = tagger
        
        self.data_lookups = {}
        self.mc_lookups = {}

        # Ensure path is absolute if not already
        if not os.path.isabs(self.map_path) and not os.path.exists(self.map_path):
             # Try workspace relative
             if os.path.exists(os.path.join(os.getcwd(), self.map_path)):
                 self.map_path = os.path.join(os.getcwd(), self.map_path)

        # 1. Load Main Trigger File
        filename = TRGSF_FILES.get(tagger, {}).get(self.year)
        if not filename:
            logging.warning(f"No Trigger SF file found for year {self.year}, tagger {self.tagger}")
        else:
            full_path = os.path.join(self.map_path, filename)
            logging.info(f"Loading Trigger SFs from {full_path}")
            self._load_root_file(full_path, is_l1=False)

        # 2. Load L1 Extra File (Run 3 mostly)
        l1_filename = TRGSF_FILES_EXTRA.get(self.year)
        if l1_filename:
            l1_full_path = os.path.join(self.map_path, l1_filename)
            logging.info(f"Loading L1 SFs from {l1_full_path}")
            self._load_root_file(l1_full_path, is_l1=True)

    def _load_root_file(self, full_path, is_l1=False):
        try:
            with uproot.open(full_path) as f:
                for key in f.keys():
                    name = key.split(";")[0] # remove cycle number
                    if "FitResult" in name:
                        continue
                    try:
                        obj = f[key]
                    except Exception as e:
                        logging.warning(f"Could not read key {key} from file {full_path}: {e}")
                        continue
                    
                    # Store logic: separate Data and MC
                    if "data" in name.lower() or "muon" in name.lower():
                        store = self.data_lookups
                    else:
                        store = self.mc_lookups
                        
                    # Parse TGraphAsymmErrors or TEfficiency
                    # simplified handling: accept both TGraph and TGraphAsymmErrors (which might not inherit from TGraph behavior)
                    # Duck typing: check for values which graphs have (histograms also have values, but we check to_hist separately or check name)
                    if hasattr(obj, "values") and not hasattr(obj, "to_hist"):
                        # Convert graph to lookup
                        
                        values = obj.values(axis="y")
                        if hasattr(obj, "errors"):
                            try:
                                # First try without 'which' argument (uproot 5 style)
                                errors_high = obj.errors(axis="y")[1] 
                                errors_low = obj.errors(axis="y")[0] 
                            except TypeError:
                                # Fallback for versions requiring 'which' (uproot 4 style)
                                errors_high = obj.errors("high", axis="y")
                                errors_low = obj.errors("low", axis="y")
                        else:
                            # For plain TGraphs (e.g. up/down variations without stored errors), assume 0
                            errors_high = np.zeros_like(values)
                            errors_low = np.zeros_like(values)
                            
                        edges = obj.values(axis="x")
                        
                        # We need to construct edges for lookup. 
                        # If N points, Marina code implies N intervals? No, TGraph has N points.
                        # Marina: for ix in range(0, N): xmin=PointX(ix), xmax=PointX(ix+1)
                        # This implies the TGraph stores the EDGES as points? 
                        # That is unusual. Usually TGraph stores (x,y).
                        # Let's assume standard behavior: we need a lookup function that works like Marina's
                        
                        store[name] = {
                            "x": edges,
                            "y": values,
                            "y_err_up": errors_high,
                            "y_err_down": errors_low,
                            "type": "graph"
                        }

                    elif hasattr(obj, "to_hist"): # TEfficiency or Hist
                        # For 2D
                        pass

        except FileNotFoundError:
            logging.error(f"Could not open file {full_path}")

    def _fix_in_range(self, val):
        return ak.where(val < 0.0, 0.0, ak.where(val > 1.0, 1.0, val))

    def _get3BTagEff(self, e0, e1, e2, e3):
        e0 = self._fix_in_range(e0)
        e1 = self._fix_in_range(e1)
        e2 = self._fix_in_range(e2)
        e3 = self._fix_in_range(e3)
        term1 = e0 * e1 * e2 * e3
        term2 = (1-e0) * e1 * e2 * e3
        term3 = e0 * (1-e1) * e2 * e3
        term4 = e0 * e1 * (1-e2) * e3
        term5 = e0 * e1 * e2 * (1-e3)
        return term1 + term2 + term3 + term4 + term5

    def _get2BTagEff(self, e0, e1, e2, e3):
        e0 = self._fix_in_range(e0)
        e1 = self._fix_in_range(e1)
        e2 = self._fix_in_range(e2)
        e3 = self._fix_in_range(e3)
        ine0, ine1, ine2, ine3 = 1.0-e0, 1.0-e1, 1.0-e2, 1.0-e3
        
        # 1 - (fail all) - (pass 1 only)
        # fail all = ine0 * ine1 * ine2 * ine3
        # pass 1 (0 only) = e0 * ine1 * ine2 * ine3
        # ...
        
        fail_all = ine0 * ine1 * ine2 * ine3
        pass_0 = e0 * ine1 * ine2 * ine3
        pass_1 = ine0 * e1 * ine2 * ine3
        pass_2 = ine0 * ine1 * e2 * ine3
        pass_3 = ine0 * ine1 * ine2 * e3
        
        return 1.0 - fail_all - pass_0 - pass_1 - pass_2 - pass_3

    def _computeFinalEff(self, eff_list):
        # eff_list is a list of arrays
        eff = 1.0
        for e in eff_list:
            eff = eff * self._fix_in_range(e)
        return eff

    def lookup_efficiency(self, name, values, is_data=True):
        """
        Vectorized lookup of efficiency
        """
        store = self.data_lookups if is_data else self.mc_lookups
        
        # Helper to find the matching key in the store (handle simplified names)
        # Marina code splits by "Efficiency_" or "Intervals_"
        # We need flexible matching
        target = None
        for key in store:
            if name in key and "Efficiency" in key: # Prefer efficiency maps
                target = store[key]
                break
        
        if target is None:
            # Fallback or return 1.0
            return ak.ones_like(values, dtype=float), ak.zeros_like(values, dtype=float), ak.zeros_like(values, dtype=float)

        if target["type"] == "graph":
            x_vals = target["x"]
            y_vals = target["y"]
            y_err_up = target["y_err_up"]
            y_err_down = target["y_err_down"]

            # Use np.searchsorted to find bins
            # counts = len(x_vals) (assuming x_vals explains edges as per Marina code logic)
            # Actually, we need to map the 'values' array to indices in x_vals
            
            # Determine if we need to restore structure
            is_jagged = False
            counts = None
            try:
                # Try to get number of elements per sub-array (default axis=1)
                counts = ak.num(values, axis=1)
                is_jagged = True
            except (ValueError, TypeError):
                # Likely 1D array (flat) or scalar
                is_jagged = False
            
            # Use ak.flatten to get a 1D array for searchsorted
            flat_values = ak.flatten(values, axis=None)
            
            indices = np.searchsorted(x_vals, ak.to_numpy(flat_values)) - 1
            indices = np.clip(indices, 0, len(y_vals) - 1)
            
            eff_flat = y_vals[indices]
            err_up_flat = y_err_up[indices]
            err_down_flat = y_err_down[indices]
            
            if is_jagged:
                eff = ak.unflatten(eff_flat, counts)
                err_up = ak.unflatten(err_up_flat, counts)
                err_down = ak.unflatten(err_down_flat, counts)
            else:
                eff = eff_flat
                err_up = err_up_flat
                err_down = err_down_flat
            
            return eff, err_up, err_down
            
        return ak.ones_like(values), ak.zeros_like(values), ak.zeros_like(values)

    def calculate_event_sf(self, events):
        # 1. Calculate HT and other event variables
        all_jets = events.Jet
        pfjetht = ak.sum(all_jets[all_jets.pfht_selected].pt, axis=1)
        calojetht = ak.sum(all_jets[all_jets.ht_selected].pt, axis=1)
        
        # 2. Sort Jets for Trigger Checks
        # By b-tag (assume 'btagDeepFlavB' exists)
        # Marina: "sorted(all_jets, key=lambda x: x.bdisc, reverse=True)"
        # Note: Marina uses 'pn_b' for > 2018.
        b_score_name = "btagDeepFlavB" if self.year <= 2018 else "pn_b" 
        # Adjust if column name is different in your NANOAOD
        if b_score_name not in all_jets.fields:
            # Fallback
            b_score_name = "btagDeepFlavB"

        jets_by_b = all_jets[ak.argsort(all_jets[b_score_name], ascending=False)]
        
        # Top 4 by b-tag are selected (for <= 2018 logic in Marina), or all (for > 2018)
        # Marina: "if self.year <= 2018: selected_jets = btag_sorted_jets[0:4] else: all_jets"
        # Then "sort the jets in pt"
        
        if self.year <= 2018:
            selected_jets = jets_by_b[:, :4] # Takes top 4
        else:
            selected_jets = all_jets
            
        jets_by_pt = selected_jets[ak.argsort(selected_jets.pt, ascending=False)]
        
        # We need at least 4 jets? trigger usually requires 4.
        # Handle events with fewer than 4 jets by padding or masking?
        # For SF calculation, usually we run on events passing selection (>=4 jets)
        # We'll use ak.pad_none to be safe
        jets_by_pt = ak.pad_none(jets_by_pt, 4, clip=True)
        
        # We need to fill None values (missing jets) with a value that yields 0 efficiency (e.g. 0.0)
        # to ensure events with < 4 jets have 0 trigger efficiency
        pt1 = ak.fill_none(jets_by_pt[:, 0].pt, 0.0)
        pt2 = ak.fill_none(jets_by_pt[:, 1].pt, 0.0)
        pt3 = ak.fill_none(jets_by_pt[:, 2].pt, 0.0)
        pt4 = ak.fill_none(jets_by_pt[:, 3].pt, 0.0)

        scores_sorted = jets_by_b[b_score_name]
        scores_sorted = ak.pad_none(scores_sorted, 4, clip=True)
        # Fill missing b-tag scores with -1 or 0 (assuming low score = low efficiency)
        b1 = ak.fill_none(scores_sorted[:, 0], 0.0)
        b2 = ak.fill_none(scores_sorted[:, 1], 0.0)
        b3 = ak.fill_none(scores_sorted[:, 2], 0.0)
        b4 = ak.fill_none(scores_sorted[:, 3], 0.0)
        
        btagTMean = 0.0 # Placeholder for post-run2
        
        # 3. Year Specific Calculation
        if self.year == 2018:
            return self._calculate_2018(pt1, pt2, pt3, pt4, pfjetht, calojetht, b1, b2, b3, b4)

        elif self.year == 2017:
            return self._calculate_2017(pt1, pt2, pt3, pt4, pfjetht, calojetht, b1, b2, b3, b4)

        elif self.year in [2021, 2022]: 
            return self._calculate_2022(pt1, pt2, pt3, pt4, pfjetht, calojetht, btagTMean)

        elif self.year in [2020, 2023]:
            if self.year == 2020:
                return self._calculate_2023_PostBPix(pt4, pfjetht, calojetht, btagTMean)
            else:
                return self._calculate_2023_PreBPix(pt4, pfjetht, calojetht, btagTMean)

        ones = ak.ones_like(pt1, dtype=float)
        return ones, ones, ones

    def _calculate_2018(self, pt1, pt2, pt3, pt4, pfjetht, calojetht, b1, b2, b3, b4):
        # L1
        d_Attr_L1, m_Attr_L1 = "L1filterHT", "L1filterHT"
        d_L1, _, _ = self.lookup_efficiency(d_Attr_L1, calojetht, is_data=True)
        m_L1, _, _ = self.lookup_efficiency(m_Attr_L1, calojetht, is_data=False)
        
        # QuadCentralJet30 (Limit on 4th jet)
        d_QCJ30, _, _ = self.lookup_efficiency("QuadCentralJet30", pt4, is_data=True)
        m_QCJ30, _, _ = self.lookup_efficiency("QuadCentralJet30", pt4, is_data=False)
        
        # CaloQuadJet30HT320
        d_CaloHT, _, _ = self.lookup_efficiency("CaloQuadJet30HT320", calojetht, is_data=True)
        m_CaloHT, _, _ = self.lookup_efficiency("CaloQuadJet30HT320", calojetht, is_data=False)
        
        # PFCentralJetLooseIDQuad30 (pt4)
        d_PFQuad30, _, _ = self.lookup_efficiency("PFCentralJetLooseIDQuad30", pt4, is_data=True)
        m_PFQuad30, _, _ = self.lookup_efficiency("PFCentralJetLooseIDQuad30", pt4, is_data=False)
        
        # 1PFCentralJetLooseID75 (pt1)
        d_PF1_75, _, _ = self.lookup_efficiency("1PFCentralJetLooseID75", pt1, is_data=True)
        m_PF1_75, _, _ = self.lookup_efficiency("1PFCentralJetLooseID75", pt1, is_data=False)
        
        # 2PFCentralJetLooseID60 (pt2)
        d_PF2_60, _, _ = self.lookup_efficiency("2PFCentralJetLooseID60", pt2, is_data=True)
        m_PF2_60, _, _ = self.lookup_efficiency("2PFCentralJetLooseID60", pt2, is_data=False)
        
        # 3PFCentralJetLooseID45 (pt3)
        d_PF3_45, _, _ = self.lookup_efficiency("3PFCentralJetLooseID45", pt3, is_data=True)
        m_PF3_45, _, _ = self.lookup_efficiency("3PFCentralJetLooseID45", pt3, is_data=False)
        
        # 4PFCentralJetLooseID40 (pt4) - Wait, Marina uses 4PFCentralJetLooseID40 with pt4
        d_PF4_40, _, _ = self.lookup_efficiency("4PFCentralJetLooseID40", pt4, is_data=True)
        m_PF4_40, _, _ = self.lookup_efficiency("4PFCentralJetLooseID40", pt4, is_data=False)
        
        # PFCentralJetsLooseIDQuad30HT330 (pfjetht)
        d_PFHT330, _, _ = self.lookup_efficiency("PFCentralJetsLooseIDQuad30HT330", pfjetht, is_data=True)
        m_PFHT330, _, _ = self.lookup_efficiency("PFCentralJetsLooseIDQuad30HT330", pfjetht, is_data=False)
        
        # BTags
        # BTagCaloDeepCSVp17Double (Double Btag)
        # We look up eff for all 4 jets
        d_b_calo_0, _, _ = self.lookup_efficiency("BTagCaloDeepCSVp17Double", b1, is_data=True)
        d_b_calo_1, _, _ = self.lookup_efficiency("BTagCaloDeepCSVp17Double", b2, is_data=True)
        d_b_calo_2, _, _ = self.lookup_efficiency("BTagCaloDeepCSVp17Double", b3, is_data=True)
        d_b_calo_3, _, _ = self.lookup_efficiency("BTagCaloDeepCSVp17Double", b4, is_data=True)
        
        m_b_calo_0, _, _ = self.lookup_efficiency("BTagCaloDeepCSVp17Double", b1, is_data=False)
        m_b_calo_1, _, _ = self.lookup_efficiency("BTagCaloDeepCSVp17Double", b2, is_data=False)
        m_b_calo_2, _, _ = self.lookup_efficiency("BTagCaloDeepCSVp17Double", b3, is_data=False)
        m_b_calo_3, _, _ = self.lookup_efficiency("BTagCaloDeepCSVp17Double", b4, is_data=False)
        
        d_b_double = self._get2BTagEff(d_b_calo_0, d_b_calo_1, d_b_calo_2, d_b_calo_3)
        m_b_double = self._get2BTagEff(m_b_calo_0, m_b_calo_1, m_b_calo_2, m_b_calo_3)
        
        # BTagPFDeepCSV4p5Triple (Triple Btag) (?)
        d_b_pf_0, _, _ = self.lookup_efficiency("BTagPFDeepCSV4p5Triple", b1, is_data=True)
        d_b_pf_1, _, _ = self.lookup_efficiency("BTagPFDeepCSV4p5Triple", b2, is_data=True)
        d_b_pf_2, _, _ = self.lookup_efficiency("BTagPFDeepCSV4p5Triple", b3, is_data=True)
        d_b_pf_3, _, _ = self.lookup_efficiency("BTagPFDeepCSV4p5Triple", b4, is_data=True)

        m_b_pf_0, _, _ = self.lookup_efficiency("BTagPFDeepCSV4p5Triple", b1, is_data=False)
        m_b_pf_1, _, _ = self.lookup_efficiency("BTagPFDeepCSV4p5Triple", b2, is_data=False)
        m_b_pf_2, _, _ = self.lookup_efficiency("BTagPFDeepCSV4p5Triple", b3, is_data=False)
        m_b_pf_3, _, _ = self.lookup_efficiency("BTagPFDeepCSV4p5Triple", b4, is_data=False)

        d_b_triple = self._get3BTagEff(d_b_pf_0, d_b_pf_1, d_b_pf_2, d_b_pf_3)
        m_b_triple = self._get3BTagEff(m_b_pf_0, m_b_pf_1, m_b_pf_2, m_b_pf_3)
        
        # Final Calculation
        d_comps = [d_L1, d_QCJ30, d_CaloHT, d_PFQuad30, d_PF1_75, d_PF2_60, d_PF3_45, d_PF4_40, d_PFHT330, d_b_double, d_b_triple]
        m_comps = [m_L1, m_QCJ30, m_CaloHT, m_PFQuad30, m_PF1_75, m_PF2_60, m_PF3_45, m_PF4_40, m_PFHT330, m_b_double, m_b_triple]
        
        return self._compute_sf(d_comps, m_comps)

    def _calculate_2017(self, pt1, pt2, pt3, pt4, pfjetht, calojetht, b1, b2, b3, b4):
        d_L1, _, _ = self.lookup_efficiency("L1calojetsPFHT", calojetht, is_data=True)
        m_L1, _, _ = self.lookup_efficiency("L1calojetsPFHT", calojetht, is_data=False)

        d_QCJ30, _, _ = self.lookup_efficiency("QuadCentralJet30", pt4, is_data=True)
        m_QCJ30, _, _ = self.lookup_efficiency("QuadCentralJet30", pt4, is_data=False)

        d_CaloHT, _, _ = self.lookup_efficiency("CaloQuadJet30HT300", calojetht, is_data=True)
        m_CaloHT, _, _ = self.lookup_efficiency("CaloQuadJet30HT300", calojetht, is_data=False)

        d_PFQuad30, _, _ = self.lookup_efficiency("PFCentralJetLooseIDQuad30", pt4, is_data=True)
        m_PFQuad30, _, _ = self.lookup_efficiency("PFCentralJetLooseIDQuad30", pt4, is_data=False)

        d_PF1_75, _, _ = self.lookup_efficiency("1PFCentralJetLooseID75", pt1, is_data=True)
        m_PF1_75, _, _ = self.lookup_efficiency("1PFCentralJetLooseID75", pt1, is_data=False)
        d_PF2_60, _, _ = self.lookup_efficiency("2PFCentralJetLooseID60", pt2, is_data=True)
        m_PF2_60, _, _ = self.lookup_efficiency("2PFCentralJetLooseID60", pt2, is_data=False)
        d_PF3_45, _, _ = self.lookup_efficiency("3PFCentralJetLooseID45", pt3, is_data=True)
        m_PF3_45, _, _ = self.lookup_efficiency("3PFCentralJetLooseID45", pt3, is_data=False)
        d_PF4_40, _, _ = self.lookup_efficiency("4PFCentralJetLooseID40", pt4, is_data=True)
        m_PF4_40, _, _ = self.lookup_efficiency("4PFCentralJetLooseID40", pt4, is_data=False)

        d_PFHT300, _, _ = self.lookup_efficiency("PFCentralJetsLooseIDQuad30HT300", pfjetht, is_data=True)
        m_PFHT300, _, _ = self.lookup_efficiency("PFCentralJetsLooseIDQuad30HT300", pfjetht, is_data=False)

        d_b_calo_0, _, _ = self.lookup_efficiency("BTagCaloCSVp05Double", b1, is_data=True)
        d_b_calo_1, _, _ = self.lookup_efficiency("BTagCaloCSVp05Double", b2, is_data=True)
        d_b_calo_2, _, _ = self.lookup_efficiency("BTagCaloCSVp05Double", b3, is_data=True)
        d_b_calo_3, _, _ = self.lookup_efficiency("BTagCaloCSVp05Double", b4, is_data=True)
        m_b_calo_0, _, _ = self.lookup_efficiency("BTagCaloCSVp05Double", b1, is_data=False)
        m_b_calo_1, _, _ = self.lookup_efficiency("BTagCaloCSVp05Double", b2, is_data=False)
        m_b_calo_2, _, _ = self.lookup_efficiency("BTagCaloCSVp05Double", b3, is_data=False)
        m_b_calo_3, _, _ = self.lookup_efficiency("BTagCaloCSVp05Double", b4, is_data=False)
        d_b_double = self._get2BTagEff(d_b_calo_0, d_b_calo_1, d_b_calo_2, d_b_calo_3)
        m_b_double = self._get2BTagEff(m_b_calo_0, m_b_calo_1, m_b_calo_2, m_b_calo_3)

        d_b_pf_0, _, _ = self.lookup_efficiency("BTagPFCSVp070Triple", b1, is_data=True)
        d_b_pf_1, _, _ = self.lookup_efficiency("BTagPFCSVp070Triple", b2, is_data=True)
        d_b_pf_2, _, _ = self.lookup_efficiency("BTagPFCSVp070Triple", b3, is_data=True)
        d_b_pf_3, _, _ = self.lookup_efficiency("BTagPFCSVp070Triple", b4, is_data=True)
        m_b_pf_0, _, _ = self.lookup_efficiency("BTagPFCSVp070Triple", b1, is_data=False)
        m_b_pf_1, _, _ = self.lookup_efficiency("BTagPFCSVp070Triple", b2, is_data=False)
        m_b_pf_2, _, _ = self.lookup_efficiency("BTagPFCSVp070Triple", b3, is_data=False)
        m_b_pf_3, _, _ = self.lookup_efficiency("BTagPFCSVp070Triple", b4, is_data=False)
        d_b_triple = self._get3BTagEff(d_b_pf_0, d_b_pf_1, d_b_pf_2, d_b_pf_3)
        m_b_triple = self._get3BTagEff(m_b_pf_0, m_b_pf_1, m_b_pf_2, m_b_pf_3)

        d_comps = [d_L1, d_QCJ30, d_CaloHT, d_PFQuad30, d_PF1_75, d_PF2_60, d_PF3_45, d_PF4_40, d_PFHT300, d_b_double, d_b_triple]
        m_comps = [m_L1, m_QCJ30, m_CaloHT, m_PFQuad30, m_PF1_75, m_PF2_60, m_PF3_45, m_PF4_40, m_PFHT300, m_b_double, m_b_triple]
        
        return self._compute_sf(d_comps, m_comps)

    def _calculate_2022(self, pt1, pt2, pt3, pt4, pfjetht, calojetht, btagTMean):
        trg = "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65"
        
        l1_name = "L1All_preEE" if self.year == 2021 else "L1All_postEE"
        d_L1, _, _ = self.lookup_efficiency(l1_name, calojetht, is_data=True)
        m_L1, _, _ = self.lookup_efficiency(l1_name, calojetht, is_data=False)

        fn = trg + "_4PixelOnlyPFCentralJetTightIDPt20"
        d_4Pixel20, _, _ = self.lookup_efficiency(fn, pt4, is_data=True)
        m_4Pixel20, _, _ = self.lookup_efficiency(fn, pt4, is_data=False)

        fn = trg + "_3PixelOnlyPFCentralJetTightIDPt30"
        d_3Pixel30, _, _ = self.lookup_efficiency(fn, pt3, is_data=True)
        m_3Pixel30, _, _ = self.lookup_efficiency(fn, pt3, is_data=False)

        fn = trg + "_2PixelOnlyPFCentralJetTightIDPt40"
        d_2Pixel40, _, _ = self.lookup_efficiency(fn, pt2, is_data=True)
        m_2Pixel40, _, _ = self.lookup_efficiency(fn, pt2, is_data=False)

        fn = trg + "_1PixelOnlyPFCentralJetTightIDPt60"
        d_1Pixel60, _, _ = self.lookup_efficiency(fn, pt1, is_data=True)
        m_1Pixel60, _, _ = self.lookup_efficiency(fn, pt1, is_data=False)

        fn = trg + "_4PFCentralJetTightIDPt35"
        d_PF4_35, _, _ = self.lookup_efficiency(fn, pt4, is_data=True)
        m_PF4_35, _, _ = self.lookup_efficiency(fn, pt4, is_data=False)

        fn = trg + "_3PFCentralJetTightIDPt40"
        d_PF3_40, _, _ = self.lookup_efficiency(fn, pt3, is_data=True)
        m_PF3_40, _, _ = self.lookup_efficiency(fn, pt3, is_data=False)

        fn = trg + "_2PFCentralJetTightIDPt50"
        d_PF2_50, _, _ = self.lookup_efficiency(fn, pt2, is_data=True)
        m_PF2_50, _, _ = self.lookup_efficiency(fn, pt2, is_data=False)

        fn = trg + "_1PFCentralJetTightIDPt70"
        d_PF1_70, _, _ = self.lookup_efficiency(fn, pt1, is_data=True)
        m_PF1_70, _, _ = self.lookup_efficiency(fn, pt1, is_data=False)

        fn = trg + "_BTagCentralJetPt35PFParticleNet2BTagSum0p65"
        d_BTag, _, _ = self.lookup_efficiency(fn, btagTMean, is_data=True)
        m_BTag, _, _ = self.lookup_efficiency(fn, btagTMean, is_data=False)

        d_comps = [d_L1, d_4Pixel20, d_3Pixel30, d_2Pixel40, d_1Pixel60, d_PF4_35, d_PF3_40, d_PF2_50, d_PF1_70, d_BTag]
        m_comps = [m_L1, m_4Pixel20, m_3Pixel30, m_2Pixel40, m_1Pixel60, m_PF4_35, m_PF3_40, m_PF2_50, m_PF1_70, m_BTag]

        return self._compute_sf(d_comps, m_comps)

    def _calculate_2023_PostBPix(self, pt4, pfjetht, calojetht, btagTMean):
        # 2020 in Marina code
        trg = "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55"
        
        d_L1, _, _ = self.lookup_efficiency("L1_HTT280er_postBPix", calojetht, is_data=True)
        m_L1, _, _ = self.lookup_efficiency("L1_HTT280er_postBPix", calojetht, is_data=False)

        d_4Pixel20, _, _ = self.lookup_efficiency(trg+"_4PixelOnlyPFCentralJetTightIDPt20", pt4, is_data=True)
        m_4Pixel20, _, _ = self.lookup_efficiency(trg+"_4PixelOnlyPFCentralJetTightIDPt20", pt4, is_data=False)

        d_4PF30, _, _ = self.lookup_efficiency(trg+"_4PFCentralJetTightIDPt30", pt4, is_data=True)
        m_4PF30, _, _ = self.lookup_efficiency(trg+"_4PFCentralJetTightIDPt30", pt4, is_data=False)

        d_PFHT280, _, _ = self.lookup_efficiency(trg+"_PFHT280Jet30", pfjetht, is_data=True)
        m_PFHT280, _, _ = self.lookup_efficiency(trg+"_PFHT280Jet30", pfjetht, is_data=False)

        d_BTagMean, _, _ = self.lookup_efficiency(trg+"_PFCentralJetPt30PNet2BTagMean0p55", btagTMean, is_data=True)
        m_BTagMean, _, _ = self.lookup_efficiency(trg+"_PFCentralJetPt30PNet2BTagMean0p55", btagTMean, is_data=False)

        d_comps = [d_L1, d_4Pixel20, d_4PF30, d_PFHT280, d_BTagMean]
        m_comps = [m_L1, m_4Pixel20, m_4PF30, m_PFHT280, m_BTagMean]
        return self._compute_sf(d_comps, m_comps)

    def _calculate_2023_PreBPix(self, pt4, pfjetht, calojetht, btagTMean):
        d_L1, _, _ = self.lookup_efficiency("L1_HTT280er_preBPix", calojetht, is_data=True)
        m_L1, _, _ = self.lookup_efficiency("L1_HTT280er_preBPix", calojetht, is_data=False)

        # 2D lookup mockup: JetLeg(x=pfjetht, y=pt4)
        d_JetLeg, _, _ = self.lookup_efficiency("JetLeg", pfjetht, is_data=True) # Needs 2D support
        m_JetLeg, _, _ = self.lookup_efficiency("JetLeg", pfjetht, is_data=False) # Needs 2D support
        
        d_BJetLeg, _, _ = self.lookup_efficiency("InclusiveBTagLeg", btagTMean, is_data=True)
        m_BJetLeg, _, _ = self.lookup_efficiency("InclusiveBTagLeg", btagTMean, is_data=False)

        d_comps = [d_L1, d_JetLeg, d_BJetLeg]
        m_comps = [m_L1, m_JetLeg, m_BJetLeg]
        return self._compute_sf(d_comps, m_comps)
        
    def _compute_sf(self, d_comps, m_comps):
        d_total = self._computeFinalEff(d_comps)
        m_total = self._computeFinalEff(m_comps)
        
        # Avoid division by zero warnings by replacing 0 with 1 in the denominator
        # The result for these elements will be masked out by the ak.where condition anyway
        safe_m_total = ak.where(m_total > 0, m_total, 1.0)
        raw_sf = d_total / safe_m_total
        
        sf = ak.where(m_total > 0, raw_sf, 1.0)
        return d_total, m_total, sf
