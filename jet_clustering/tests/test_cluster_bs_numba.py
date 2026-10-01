"""cluster_bs (numba) against cluster_bs_reference (the original per-event Python clustering).

Same topology required: per event identical jet_flavor / btag_string of the clustered jets, of
every splitting and of its part_A / part_B, in the same order; kinematics within TOL. The
reference does its first merges in float32 scalar math (cluster_bs in float64), so its masses
carry float32 cancellation in sqrt(t^2 - p^2): up to ~1e-4 relative on boosted pairs.

    python coffea4bees/jet_clustering/tests/test_cluster_bs_numba.py
"""
import os
import sys
import time
import unittest

import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector

sys.path.insert(0, os.getcwd())
from coffea4bees.jet_clustering.clustering import cluster_bs, cluster_bs_reference
from coffea4bees.jet_clustering.tests import test_clustering    # module import: its TestCase isn't re-run here

# (rtol, atol) per field: measured max differences are pt 9e-7 rel, eta/phi 4e-6 abs, mass 8e-5 rel
TOL = {"pt": (1e-5, 1e-6), "eta": (0, 1e-5), "phi": (0, 1e-5), "mass": (2e-4, 1e-3)}
P4 = ("pt", "eta", "phi", "mass")
STRINGS = ("jet_flavor", "btag_string")


def make_jets(pt, eta, phi, mass, flavor, btag):
    return ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass, "jet_flavor": flavor, "btagScore": btag},
                  with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)


def random_events(n_events, seed, n_jets=(4, 10), n_b=(3, 6), degenerate=False):
    """float32 jets like NanoAOD; at least 3 b's per event (the reference clusters until a merge
    would combine three b's and fails on events that never get there)."""
    rng = np.random.default_rng(seed)
    nj = rng.integers(n_jets[0], n_jets[1] + 1, n_events)
    nb = np.minimum(rng.integers(n_b[0], n_b[1] + 1, n_events), nj)
    tot = int(nj.sum())
    pt   = rng.uniform(30, 300, tot).astype(np.float32)
    eta  = rng.uniform(-2.5, 2.5, tot).astype(np.float32)
    phi  = rng.uniform(-np.pi, np.pi, tot).astype(np.float32)
    mass = rng.uniform(4, 30, tot).astype(np.float32)
    btag = rng.uniform(0, 1, tot).astype(np.float32)
    flavor = []
    for n, b in zip(nj, nb):
        f = ["b"] * b + ["j"] * (n - b)
        rng.shuffle(f)
        flavor += f
    if degenerate:
        # exact copies of the previous jet -> dR = 0 / equal pt ties for the argmin and child order
        offsets = np.concatenate([[0], np.cumsum(nj)])
        for iE in range(n_events):
            if rng.uniform() < 0.5:
                k = offsets[iE] + 1
                pt[k], eta[k], phi[k], mass[k] = pt[k - 1], eta[k - 1], phi[k - 1], mass[k - 1]
            if rng.uniform() < 0.5:
                k = offsets[iE] + 2
                pt[k] = pt[k - 1]
    counts = nj.astype(np.int64)
    u = lambda a: ak.unflatten(a, counts)
    jets = make_jets(u(pt), u(eta), u(phi), u(mass), u(ak.Array(flavor)), u(btag))
    return jets[ak.argsort(jets.pt, axis=1, ascending=False)]


def mass_from_children(splittings):
    """float64 invariant mass of part_A + part_B for every splitting (flat)."""
    p = {}
    for tag in ("part_A", "part_B"):
        c = splittings[tag]
        pt, eta, phi, m = (ak.to_numpy(ak.flatten(c[f])).astype(np.float64) for f in P4)
        p[tag] = (pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt(m ** 2 + (pt * np.cosh(eta)) ** 2))
    x, y, z, t = (a + b for a, b in zip(p["part_A"], p["part_B"]))
    m2 = t ** 2 - (x ** 2 + y ** 2 + z ** 2)
    return np.copysign(np.sqrt(np.abs(m2)), m2)


def mismatches(ref, new, tol=TOL):
    """Indices of events whose clustering differs (strings exact, kinematics to tol)."""
    (rj, rs), (nj, ns) = ref, new
    bad = set()
    for name, r, n in (("jets", rj, nj), ("split", rs, ns), ("A", rs.part_A, ns.part_A), ("B", rs.part_B, ns.part_B)):
        same_n = ak.to_numpy(ak.num(r) == ak.num(n))
        bad.update(np.flatnonzero(~same_n).tolist())
        ok = same_n
        for f in STRINGS:
            eq = ak.to_list(r[f]), ak.to_list(n[f])
            ok = ok & np.array([a == b for a, b in zip(*eq)])
        for f in P4:
            rf = ak.to_list(r[f])
            nf = ak.to_list(n[f])
            rtol, atol = tol[f]
            ok = ok & np.array([len(a) == len(b) and np.allclose(a, b, rtol=rtol, atol=atol) for a, b in zip(rf, nf)])
        bad.update(np.flatnonzero(~ok).tolist())
    return sorted(bad)


class ClusterBsNumbaTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_clustering.clusteringTestCase.setUpClass()
        cls.vectors = {name: getattr(test_clustering.clusteringTestCase, name) for name in
                       ("input_jets_4", "input_jets_5", "input_jets_bbj", "input_jets_6",
                        "input_jets_5b", "input_jets_HH_3b", "input_jets_bad_split", "input_jets_all")}

    def _compare(self, jets, label, allow=0, tol=TOL):
        ref = cluster_bs_reference(ak.copy(jets))
        new = cluster_bs(ak.copy(jets))
        bad = mismatches(ref, new, tol)
        if bad:
            iE = bad[0]
            print(f"\n[{label}] {len(bad)}/{len(jets)} events differ; first {iE}:")
            print("  ref split", ak.to_list(ref[1].jet_flavor[iE]), ak.to_list(ref[1].pt[iE]))
            print("  new split", ak.to_list(new[1].jet_flavor[iE]), ak.to_list(new[1].pt[iE]))
        self.assertLessEqual(len(bad), allow, f"{label}: {len(bad)} events differ")
        return ref, new

    def test_vectors(self):
        for name, jets in self.vectors.items():
            with self.subTest(name):
                self._compare(jets, name)

    def test_output_layout(self):
        ref, new = self._compare(self.vectors["input_jets_all"], "layout")
        for r, n in zip(ref, new):
            self.assertEqual(r.fields, n.fields)
        self.assertEqual(ref[1].part_A.fields, new[1].part_A.fields)
        self.assertEqual(ak.parameters(ak.flatten(new[0])).get("__record__"), "PtEtaPhiMLorentzVector")
        self.assertEqual(ak.parameters(ak.flatten(new[1].part_A)).get("__record__"), "PtEtaPhiMLorentzVector")
        self.assertEqual(str(ak.type(new[1].jet_flavor)).split("*")[-1].strip(), "string")

    def test_btag_string_side_effect(self):
        jets_ref, jets_new = ak.copy(self.vectors["input_jets_all"]), ak.copy(self.vectors["input_jets_all"])
        cluster_bs_reference(jets_ref)
        cluster_bs(jets_new)
        self.assertEqual(ak.to_list(jets_ref.btag_string), ak.to_list(jets_new.btag_string))

    def test_random_float32(self):
        self._compare(random_events(300, seed=1), "random")

    def test_random_degenerate(self):
        # exact ties (dR = 0 copies, equal pt): same (dij, i, j) resolution and child order. The
        # collinear merges are where the reference's float32 mass cancels worst: 2m comes out up
        # to ~0.4% off (the error grows like (pt/m)^2), so its masses only get 1% here and
        # cluster_bs is checked against a float64 recomputation from its own children instead.
        ref, new = self._compare(random_events(300, seed=2, degenerate=True), "degenerate",
                                 tol={**TOL, "mass": (1e-2, 1e-1)})
        self.assertTrue(np.allclose(ak.to_numpy(ak.flatten(new[1].mass)), mass_from_children(new[1]), rtol=1e-9, atol=1e-9))

    def test_speed(self):
        jets = random_events(300, seed=3)
        cluster_bs(ak.copy(jets[:10]))            # jit compile
        t0 = time.perf_counter(); cluster_bs_reference(ak.copy(jets)); t_ref = time.perf_counter() - t0
        t0 = time.perf_counter(); cluster_bs(ak.copy(jets));           t_new = time.perf_counter() - t0
        big = random_events(100_000, seed=4)
        t0 = time.perf_counter(); cluster_bs(big);                     t_big = time.perf_counter() - t0
        print(f"\ncluster_bs_reference {1e3 * t_ref / len(jets):.3f} ms/event, "
              f"cluster_bs {1e3 * t_new / len(jets):.4f} ms/event "
              f"(x{t_ref / t_new:.0f}); 1e5 events in {t_big:.2f}s")
        self.assertLess(t_new, t_ref)


if __name__ == "__main__":
    unittest.main()
