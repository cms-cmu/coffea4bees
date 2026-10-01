"""Toy-library tests of replace_hemis_topk_kdTrees: the source-event veto and the seeded random
rank selection used by the 4b mixing (MakeMixedData, mixing.source: fourTag).

    python -m unittest coffea4bees.hemisphere_mixing.tests.test_topk_matching
"""
import unittest

import awkward as ak
import numpy as np

from coffea4bees.hemisphere_mixing.mixing_helpers import replace_hemis_topk_kdTrees

SUMMARY = ["sumPt_T_minor", "sumPt_T", "combinedMass", "pz"]
JETS = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass"]
# two tag bins so every bin mask is an array; every toy hemisphere has nTagJet = 2 (first bin)
RANGES = {2: {2: []}, 3: {2: []}}
KEY = (2, 2, -1)


def _library(n_events, rng):
    """n_events library events, two hemispheres each (pos then neg block)."""
    n = 2 * n_events
    event = np.tile(np.arange(1000, 1000 + n_events), 2)
    lib = {
        "event": event, "run": np.full(n, 360000), "luminosityBlock": np.full(n, 7),
        "hemisphereId": np.repeat([1, -1], n_events), "thrust_phi": rng.uniform(-np.pi, np.pi, n),
        "weight": np.ones(n), "nJet": np.full(n, 2), "nSelJet": np.full(n, 2), "nTagJet": np.full(n, 2),
    }
    for v in SUMMARY:
        lib[v] = rng.normal(size=n)
    for v in JETS:
        lib[v] = ak.Array([[1.0 + i, 2.0 + i] for i in range(n)])
    stats = {KEY: {v: {"mean": 0.0, "RMS": 1.0} for v in SUMMARY}}
    return lib, stats


def _query(lib, rows):
    """The library hemispheres `rows` (pos block then neg block) as the hemispheres to be mixed."""
    rec = {k: np.asarray(lib[k])[rows] for k in ("event", "run", "luminosityBlock", "hemisphereId",
                                                 "thrust_phi", "weight", "nSelJet", "nTagJet", *SUMMARY)}
    return ak.zip(rec)


def _mix(lib, stats, hemis, **kw):
    hemi_data = dict(lib)      # the matcher caches kd-trees in it: fresh copy per call
    return replace_hemis_topk_kdTrees(all_hemis=hemis, hemi_stats=stats, hemi_data=hemi_data,
                                      hemi_jet_ranges=RANGES, hemi_summary_vars=SUMMARY,
                                      jet_branches=JETS, k_neighbors=10, **kw)


class TopKMatchingTest(unittest.TestCase):

    def setUp(self):
        self.n_events = 400
        self.lib, self.stats = _library(self.n_events, np.random.default_rng(1))
        # mix the library's own events: pos hemis 0..n-1, neg hemis n..2n-1
        rows = np.concatenate([np.arange(self.n_events), self.n_events + np.arange(self.n_events)])
        self.hemis = _query(self.lib, rows)

    def _ids(self, out):
        n = self.n_events
        return np.asarray(out.event[:n]), np.asarray(out.event[n:])

    def test_without_veto_each_hemisphere_matches_itself(self):
        # the hazard the veto exists for: a library event mixed at rank 0 comes back unchanged
        out, kept = _mix(self.lib, self.stats, self.hemis, collision_mode="ignore")
        pos, neg = self._ids(out)
        src = np.asarray(self.hemis.event[:self.n_events])
        self.assertTrue(np.all(pos == src) and np.all(neg == src))
        self.assertTrue(np.all(np.asarray(out.match_dist) == 0))

    def test_veto_excludes_both_own_hemispheres(self):
        for mode in ("ignore", "drop", "retry"):
            for sel in ("fixed", "random"):
                out, kept = _mix(self.lib, self.stats, self.hemis, collision_mode=mode,
                                 exclude_source_event=True, rank_selection=sel, mixing_seed=3)
                pos, neg = self._ids(out)
                src = np.asarray(self.hemis.event[:self.n_events])
                with self.subTest(mode=mode, sel=sel):
                    self.assertFalse(np.any(pos == src))
                    self.assertFalse(np.any(neg == src))
                    self.assertTrue(np.all(np.asarray(out.match_dist) > 0))
                    if mode == "retry":
                        self.assertTrue(np.all(kept))
                        self.assertFalse(np.any(pos == neg))        # no pos/neg collision left

    def test_veto_rank0_is_nearest_other_event(self):
        out, _ = _mix(self.lib, self.stats, self.hemis, collision_mode="ignore", exclude_source_event=True)
        pts = np.column_stack([self.lib[v] for v in SUMMARY])
        for i in range(0, self.n_events, 37):
            d = np.linalg.norm(pts - pts[i], axis=1)
            d[self.lib["event"] == self.lib["event"][i]] = np.inf
            self.assertEqual(int(out.event[i]), int(self.lib["event"][np.argmin(d)]))

    def test_random_rank_is_seeded_and_uniform(self):
        kw = dict(collision_mode="ignore", exclude_source_event=True, rank_selection="random", k_random=10)
        a, _ = _mix(self.lib, self.stats, self.hemis, mixing_seed=5, **kw)
        b, _ = _mix(self.lib, self.stats, self.hemis, mixing_seed=5, **kw)
        c, _ = _mix(self.lib, self.stats, self.hemis, mixing_seed=6, **kw)
        ra, rc = np.asarray(a.match_rank), np.asarray(c.match_rank)
        self.assertTrue(np.array_equal(ra, np.asarray(b.match_rank)))
        self.assertTrue(np.array_equal(np.asarray(a.event), np.asarray(b.event)))
        self.assertFalse(np.array_equal(ra, rc))
        self.assertEqual(ra.min(), 0)
        self.assertEqual(ra.max(), 9)
        counts = np.bincount(ra, minlength=10)
        # 800 draws over 10 ranks: every rank within 5 sigma of 80
        self.assertTrue(np.all(np.abs(counts - 80) < 5 * np.sqrt(80)), counts)
        # two seeds agree on a hemisphere ~1/10 of the time
        self.assertLess(np.mean(ra == rc), 0.2)

    def test_k_random_limits_the_pool(self):
        out, _ = _mix(self.lib, self.stats, self.hemis, collision_mode="ignore", exclude_source_event=True,
                      rank_selection="random", k_random=3, mixing_seed=1)
        self.assertLessEqual(int(np.asarray(out.match_rank).max()), 2)
        with self.assertRaises(ValueError):
            _mix(self.lib, self.stats, self.hemis, rank_selection="random", k_random=11)


if __name__ == "__main__":
    unittest.main()
