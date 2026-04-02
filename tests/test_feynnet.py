# coffea4bees/tests/test_feynnet.py
import os
import unittest

import numpy as np
import awkward as ak
from coffea4bees.analysis.helpers.classifier.FeynNet import FeynNetEnsemble, _select_forward_jets, _higgs_cand_flags
from unittest.mock import MagicMock
import json


def _make_mock_event(n=10):
    """Minimal mock selev with canJet, quadJet_selected, Jet fields."""
    rng = np.random.default_rng(42)
    pts = np.sort(rng.uniform(40, 300, (n, 4)), axis=1)[:, ::-1].astype("float32")
    etas = rng.uniform(-2.4, 2.4, (n, 4)).astype("float32")
    phis = rng.uniform(-np.pi, np.pi, (n, 4)).astype("float32")
    masses = rng.uniform(0, 20, (n, 4)).astype("float32")

    canJet = ak.zip({"pt": pts, "eta": etas, "phi": phis, "mass": masses})

    h1b1 = ak.zip({"pt": pts[:, 0], "eta": etas[:, 0], "phi": phis[:, 0], "mass": masses[:, 0]})
    h1b2 = ak.zip({"pt": pts[:, 1], "eta": etas[:, 1], "phi": phis[:, 1], "mass": masses[:, 1]})
    h2b1 = ak.zip({"pt": pts[:, 2], "eta": etas[:, 2], "phi": phis[:, 2], "mass": masses[:, 2]})
    h2b2 = ak.zip({"pt": pts[:, 3], "eta": etas[:, 3], "phi": phis[:, 3], "mass": masses[:, 3]})
    lead_dijet = ak.zip({"lead": h1b1, "subl": h1b2})
    subl_dijet = ak.zip({"lead": h2b1, "subl": h2b2})
    quadJet_selected = ak.zip({"lead": lead_dijet, "subl": subl_dijet})

    jet_pts   = np.concatenate([pts, rng.uniform(30, 200, (n, 4)).astype("float32")], axis=1)
    jet_etas  = np.concatenate([etas, rng.uniform(-4.5, 4.5, (n, 4)).astype("float32")], axis=1)
    jet_phis  = np.concatenate([phis, rng.uniform(-np.pi, np.pi, (n, 4)).astype("float32")], axis=1)
    jet_masses = np.concatenate([masses, rng.uniform(0, 20, (n, 4)).astype("float32")], axis=1)
    jet_tagged = np.zeros((n, 8), dtype=bool)
    jet_tagged[:, :4] = True  # canJets are b-tagged
    jet_selected = np.ones((n, 8), dtype=bool)
    Jet = ak.zip({
        "pt": jet_pts, "eta": jet_etas, "phi": jet_phis, "mass": jet_masses,
        "tagged": jet_tagged, "selected": jet_selected,
    })

    event_ids = rng.integers(0, 1000000, n).astype(np.uint64)

    return ak.zip({
        "canJet": canJet,
        "quadJet_selected": quadJet_selected,
        "Jet": Jet,
        "event": event_ids,
    }, depth_limit=1)


def _make_mock_preprocess():
    def _var(name):
        return {name: {"median": 0, "norm_factor": 1, "replace_inf_value": 0,
                        "lower_bound": -1e32, "upper_bound": 1e32, "pad": 0}}
    j_vars = ["j_log_pt","j_log_mass","j_eta","j_sinphi","j_cosphi",
              "j_h1b1_cand","j_h1b2_cand","j_h2b1_cand","j_h2b2_cand"]
    f_vars = ["f_log_pt","f_log_mass","f_eta","f_sinphi","f_cosphi","f_min_b_dr"]
    def _group(var_names, length):
        infos = {}
        for v in var_names:
            infos.update(_var(v))
        return {"var_names": var_names, "var_infos": infos, "var_length": length}
    return {
        "output_names": ["event_probs","event_logits","event_reweight"],
        "input_names": ["j","j_p4","f","f_p4"],
        "j": _group(j_vars, 4),
        "j_p4": _group(["j_pt","j_mass","j_eta","j_phi"], 4),
        "f": _group(f_vars, 2),
        "f_p4": _group(["f_pt","f_mass","f_eta","f_phi"], 2),
    }


class FeynNetTestCase(unittest.TestCase):

    def test_higgs_cand_flags_shape(self):
        event = _make_mock_event(10)
        flags = _higgs_cand_flags(event)
        self.assertEqual(flags.shape, (10, 4, 4))

    def test_higgs_cand_flags_sum_to_one_per_role(self):
        event = _make_mock_event(10)
        flags = _higgs_cand_flags(event)
        # Each role column sums to 1 across the 4 canJets
        self.assertTrue(np.all(flags.sum(axis=1) == 1))
        # Each canJet is assigned to at most one role
        self.assertTrue(np.all(flags.sum(axis=2) <= 1))

    def test_select_forward_jets_shape(self):
        event = _make_mock_event(10)
        f_pt, f_eta, f_phi, f_mass = _select_forward_jets(event)
        self.assertEqual(f_pt.shape, (10, 2))
        self.assertEqual(f_eta.shape, (10, 2))

    def test_select_forward_jets_not_btagged(self):
        event = _make_mock_event(10)
        f_pt, f_eta, f_phi, f_mass = _select_forward_jets(event)
        # canJets (slots 0-3) are all b-tagged, so forward jets come from slots 4-7
        self.assertTrue(np.any(f_pt > 0))

    def test_feynnet_ensemble_output_shape(self):
        n = 12
        event = _make_mock_event(n)
        prep = _make_mock_preprocess()

        mock_session = MagicMock()
        def fake_run(output_names, inputs):
            batch = list(inputs.values())[0].shape[0]
            return [
                np.ones((batch, 5), dtype=np.float32) / 5,
                np.zeros((batch, 5), dtype=np.float32),
                np.ones((batch, 3), dtype=np.float32),
            ]
        mock_session.run = fake_run

        ensemble = FeynNetEnsemble.__new__(FeynNetEnsemble)
        ensemble.sessions = [mock_session]
        ensemble.preprocessors = [prep]
        ensemble.n_folds = 1
        ensemble.classes = ["ggHH", "qqHH", "ZZ", "ZH", "Background"]
        ensemble._batched = True

        c_score, q_score = ensemble(event)

        self.assertEqual(c_score.shape, (n, 5), f"Expected ({n},5), got {c_score.shape}")
        self.assertEqual(q_score.shape, (n, 3), f"Expected ({n},3), got {q_score.shape}")
        self.assertFalse(np.any(np.isnan(c_score)))
        self.assertFalse(np.any(np.isnan(q_score)))


from coffea4bees.analysis.helpers.SvB_helpers import compute_SvB_FeynNet


class TestComputeSvBFeynNet(unittest.TestCase):
    def test_compute_SvB_FeynNet_fields(self):
        """compute_SvB_FeynNet adds SvB_FeynNet field with expected sub-fields."""
        n = 10
        event = _make_mock_event(n)
        mask = np.ones(n, dtype=bool)

        mock_ensemble = MagicMock()
        c_score = np.zeros((n, 5), dtype=np.float32)
        c_score[:, 0] = 0.6   # ggHH dominant
        c_score[:, 1] = 0.05  # qqHH
        c_score[:, 2] = 0.1   # ZZ
        c_score[:, 3] = 0.1   # ZH
        c_score[:, 4] = 0.15  # Background
        q_score = np.ones((n, 3), dtype=np.float32)
        mock_ensemble.return_value = (c_score, q_score)
        mock_ensemble.classes = ["ggHH", "qqHH", "ZZ", "ZH", "Background"]

        compute_SvB_FeynNet(event, mask, SvB_FeynNet=mock_ensemble)

        self.assertIn("SvB_FeynNet", event.fields)
        for field in ("p_ggHH", "p_qqHH", "p_ZZ", "p_ZH", "p_bkg",
                      "ps", "passMinPs", "hh", "zz", "zh",
                      "ps_hh", "ps_zz", "ps_zh", "reweight", "tt_vs_mj"):
            self.assertIn(field, event["SvB_FeynNet"].fields, f"Missing field: {field}")

        # hh should be True when combined ggHH+qqHH (0.65) > ZZ (0.1) and ZH (0.1)
        self.assertTrue(np.all(ak.to_numpy(event["SvB_FeynNet"].hh)))


class TestProcessorWiring(unittest.TestCase):
    def test_processor_SvB_FeynNet_param_exists(self):
        """processor_HH4b accepts SvB_FeynNet=None without error."""
        import importlib
        proc_module = importlib.import_module("coffea4bees.analysis.processors.processor_HH4b")
        HH4b = proc_module.analysis
        p = HH4b(SvB_FeynNet=None, SvB=None, SvB_MA=None)
        self.assertTrue(hasattr(p, "classifier_SvB_FeynNet"))
        self.assertIsNone(p.classifier_SvB_FeynNet)


if __name__ == "__main__":
    unittest.main()
