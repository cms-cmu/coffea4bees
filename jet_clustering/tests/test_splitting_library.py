import os
import sys
import tempfile
import unittest

import numpy as np
import awkward as ak
import uproot
from coffea.nanoevents.methods import vector

sys.path.insert(0, os.getcwd())
from coffea4bees.jet_clustering.clustering import cluster_bs
from coffea4bees.jet_clustering.declustering import clean_ISR, get_list_of_all_sub_splittings
from coffea4bees.jet_clustering.splitting_library import (
    build_splitting_library_rows,
    decode_flavor,
)
from src.data_formats.root import TreeWriter


def make_toy_events(n_events=50, seed=7):
    """Events with 4 b jets and 0-4 extra j jets, each jet with a distinct btagScore."""
    rng = np.random.default_rng(seed)
    pt, eta, phi, mass, flavor, btag = [], [], [], [], [], []
    for _ in range(n_events):
        n_j = rng.integers(0, 5)
        n = 4 + n_j
        _pt = np.sort(rng.uniform(30, 250, n))[::-1]
        pt.append(_pt.tolist())
        eta.append(rng.uniform(-2.4, 2.4, n).tolist())
        phi.append(rng.uniform(-np.pi, np.pi, n).tolist())
        mass.append((0.1 * _pt).tolist())
        _flav = ["b"] * 4 + ["j"] * n_j
        rng.shuffle(_flav)
        flavor.append(_flav)
        btag.append(rng.uniform(0, 1, n).tolist())

    jets = ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "mass": mass, "jet_flavor": flavor, "btagScore": btag},
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )
    selev = ak.zip({
        "run":             np.full(n_events, 1, dtype=np.uint32),
        "luminosityBlock": np.full(n_events, 2, dtype=np.uint32),
        "event":           np.arange(n_events, dtype=np.uint64) + 100,
    })
    return selev, jets


class splittingLibraryTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.selev, cls.jets = make_toy_events()
        cls.clustered_jets, cls.splittings = cluster_bs(cls.jets, debug=False)
        cls.clean_jets = clean_ISR(cls.clustered_jets, cls.splittings)
        cls.rows = build_splitting_library_rows(cls.selev, cls.jets, cls.splittings, cls.clean_jets,
                                                carry_fields=["btagScore"])

    def test_one_row_per_splitting(self):
        self.assertEqual(len(self.rows), int(np.sum(ak.num(self.splittings))))
        expected_event = ak.flatten(ak.broadcast_arrays(self.selev.event, self.splittings.pt)[0])
        np.testing.assert_array_equal(np.asarray(self.rows.event), np.asarray(expected_event))

    def test_flavor_roundtrip(self):
        decoded = decode_flavor(self.rows.jet_flavor)
        self.assertEqual(decoded.tolist(), ak.to_list(ak.flatten(self.splittings.jet_flavor)))

    def test_carried_btag(self):
        """Single-jet children carry their input jet's btagScore; combined children get NaN."""
        flat = {k: ak.to_list(ak.flatten(v)) for k, v in
                {"A_f": self.splittings.part_A.jet_flavor, "A_s": self.splittings.part_A.btag_string,
                 "B_f": self.splittings.part_B.jet_flavor, "B_s": self.splittings.part_B.btag_string}.items()}
        for tag in ("A", "B"):
            carried = np.asarray(self.rows[f"{tag}_btagScore"])
            for i, (f, s) in enumerate(zip(flat[f"{tag}_f"], flat[f"{tag}_s"])):
                if len(f) == 1:
                    # btag_string is the input score rounded to 3 decimals
                    self.assertAlmostEqual(carried[i], float(s), places=3)
                else:
                    self.assertTrue(np.isnan(carried[i]))

    def test_in_clean_tree(self):
        """Flagged splittings of an event are exactly the nodes of its cleaned clustered jets."""
        flags = ak.unflatten(np.asarray(self.rows.in_clean_tree), ak.num(self.splittings))
        for iE in range(len(self.splittings)):
            expected = []
            for f in ak.to_list(self.clean_jets.jet_flavor[iE]):
                expected += get_list_of_all_sub_splittings(f)
            flagged = [f for f, keep in zip(ak.to_list(self.splittings.jet_flavor[iE]), ak.to_list(flags[iE])) if keep]
            self.assertEqual(sorted(flagged), sorted(expected))
        # toy events have ISR splittings, so some rows must be unflagged
        self.assertLess(int(np.sum(self.rows.in_clean_tree)), len(self.rows))

    def test_children_sum_to_parent(self):
        px = self.rows.A_pt * np.cos(self.rows.A_phi) + self.rows.B_pt * np.cos(self.rows.B_phi)
        py = self.rows.A_pt * np.sin(self.rows.A_phi) + self.rows.B_pt * np.sin(self.rows.B_phi)
        np.testing.assert_allclose(np.hypot(px, py), np.asarray(self.rows.pt), rtol=1e-4)

    def test_write_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "splittingLib.root")
            with TreeWriter()(path) as writer:
                writer.extend(self.rows)
            with uproot.open(path) as f:
                back = f["Events"].arrays(library="ak")
        self.assertEqual(len(back), len(self.rows))
        self.assertEqual(decode_flavor(back.jet_flavor).tolist(), decode_flavor(self.rows.jet_flavor).tolist())
        np.testing.assert_allclose(np.asarray(back.A_btagScore), np.asarray(self.rows.A_btagScore), equal_nan=True)


if __name__ == "__main__":
    unittest.main()
