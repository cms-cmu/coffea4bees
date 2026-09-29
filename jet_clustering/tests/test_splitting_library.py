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
from coffea4bees.jet_clustering.declustering import clean_ISR, get_list_of_all_sub_splittings, make_synthetic_event
from coffea4bees.jet_clustering.splitting_library import (
    build_splitting_library_rows,
    decode_flavor,
    encode_flavor,
    SplittingLibrary,
    align_children,
    library_child_flavors,
    carry_child_fields,
    random_ranks,
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


def make_toy_library(n_per_type=None, seed=11):
    """Library rows built from random children, parent = A + B."""
    rng = np.random.default_rng(seed)
    n_per_type = n_per_type or {"bb": 400, "bj": 300, "(bj)b": 200, "((bj)j)b": 3, "((jj)b)b": 30, "(jb)b": 1, "(bj)(bj)": 1, "(jj)(jj)": 1}
    rows = {k: [] for k in ["flavor", "run", "luminosityBlock", "event"] + [f"{t}_{v}" for t in "AB" for v in ("pt", "eta", "phi", "mass", "btagScore")]}
    for flavor, n in n_per_type.items():
        for i in range(n):
            rows["flavor"].append(flavor)
            rows["run"].append(1); rows["luminosityBlock"].append(1); rows["event"].append(len(rows["event"]))
            for t in "AB":
                p = rng.uniform(30, 300)
                rows[f"{t}_pt"].append(p); rows[f"{t}_eta"].append(rng.uniform(-2.4, 2.4))
                rows[f"{t}_phi"].append(rng.uniform(-np.pi, np.pi)); rows[f"{t}_mass"].append(0.12 * p)
                rows[f"{t}_btagScore"].append(rng.uniform(0, 1))
    child = {t: ak.zip({v: np.array(rows[f"{t}_{v}"]) for v in ("pt", "eta", "phi", "mass")},
                       with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior) for t in "AB"}
    parent = child["A"] + child["B"]
    cols = {k: np.array(v) for k, v in rows.items() if k != "flavor"}
    cols.update(pt=np.asarray(parent.pt), eta=np.asarray(parent.eta), phi=np.asarray(parent.phi), mass=np.asarray(parent.mass))
    cols["jet_flavor"] = ak.flatten(encode_flavor(ak.Array([rows["flavor"]])), axis=1)
    cols["in_clean_tree"] = np.ones(len(rows["flavor"]), dtype=bool)
    return ak.zip(cols, depth_limit=1)


class splittingLibraryLookupTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = make_toy_library()
        cls.lib = SplittingLibrary(cls.rows, carry_fields=["btagScore"], min_entries=10)
        rng = np.random.default_rng(3)
        n = 200
        cls.targets = dict(
            flavor=np.array(rng.choice(["bb", "bj", "(bj)b"], n), dtype=object),
            pt=rng.uniform(40, 400, n), eta=rng.uniform(-2.3, 2.3, n), phi=rng.uniform(-np.pi, np.pi, n),
            run=np.full(n, 2), luminosityBlock=np.full(n, 2), event=np.arange(n),
        )

    def _lookup(self, rank=0, **overrides):
        t = {**self.targets, **overrides}
        return self.lib.lookup(t["flavor"], t["pt"], t["eta"], t["run"], t["luminosityBlock"], t["event"], rank)

    def test_exact_type(self):
        index, level = self._lookup()
        np.testing.assert_array_equal(self.lib.flavor[index], self.targets["flavor"])
        self.assertTrue(np.all(level == 0))

    def test_alignment_reproduces_parent(self):
        index, _ = self._lookup()
        t = self.targets
        kids = align_children(self.lib, index, t["pt"], t["eta"], t["phi"])
        p4 = {c: ak.zip({v: kids[c][v] for v in ("pt", "eta", "phi", "mass")},
                        with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior) for c in "AB"}
        parent = p4["A"] + p4["B"]
        np.testing.assert_allclose(np.asarray(parent.pt), t["pt"], rtol=1e-5)
        np.testing.assert_allclose(np.asarray(parent.eta), t["eta"], rtol=1e-5, atol=1e-5)
        dphi = (np.asarray(parent.phi) - t["phi"] + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(dphi, 0, atol=1e-5)
        # scaling preserves m/pT of each child, and the carried field is the library child's
        for c in "AB":
            np.testing.assert_allclose(kids[c]["mass"] / kids[c]["pt"], 0.12, rtol=1e-5)
            np.testing.assert_allclose(kids[c]["btagScore"], self.lib.data[f"{c}_btagScore"][index])

    def test_no_scale_no_boost(self):
        index, _ = self._lookup()
        t = self.targets
        kids = align_children(self.lib, index, t["pt"], t["eta"], t["phi"], scale_pt=False, boost_z=False)
        for c in "AB":
            np.testing.assert_allclose(kids[c]["pt"], self.lib.data[f"{c}_pt"][index], rtol=1e-6)
            np.testing.assert_allclose(np.abs(kids[c]["eta"]), np.abs(self.lib.data[f"{c}_eta"][index]), rtol=1e-6)

    def test_rank_steps_outward(self):
        idx = [self._lookup(rank=r)[0] for r in range(3)]
        t = self.targets
        dist = [np.hypot(np.log(t["pt"]) - np.log(self.lib.data["pt"][i]), np.abs(t["eta"]) - np.abs(self.lib.data["eta"][i])) for i in idx]
        self.assertTrue(np.all(idx[0] != idx[1]) and np.all(idx[1] != idx[2]))
        self.assertTrue(np.all(dist[0] <= dist[1] + 1e-9) and np.all(dist[1] <= dist[2] + 1e-9))

    def test_self_match_excluded(self):
        """A target that IS a library row (same event, same kinematics) never gets itself back."""
        sel = np.where(self.lib.flavor == "bj")[0][:50]
        d = self.lib.data
        index, _ = self.lib.lookup(self.lib.flavor[sel], d["pt"][sel], d["eta"][sel],
                                   d["run"][sel], d["luminosityBlock"][sel], d["event"][sel], 0)
        self.assertTrue(np.all(index != sel))
        self.assertTrue(np.all(d["event"][index] != d["event"][sel]))
        # without the exclusion the nearest neighbour would be the row itself
        index_other, _ = self.lib.lookup(self.lib.flavor[sel], d["pt"][sel], d["eta"][sel],
                                         d["run"][sel] + 1, d["luminosityBlock"][sel], d["event"][sel], 0)
        np.testing.assert_array_equal(index_other, sel)

    def test_fallback_for_rare_type(self):
        """((bj)j)b has 3 entries (< min_entries) -> falls back to the child-content group
        (child A = 1b2j, child B = 1b0j), which also holds the 30 ((jj)b)b rows. The children
        flavors come from the library row, so each child's b/j content is preserved."""
        from coffea4bees.jet_clustering.declustering import get_splitting_summary
        n = 5
        t = self.targets
        index, level = self.lib.lookup(np.array(["((bj)j)b"] * n, dtype=object), t["pt"][:n], t["eta"][:n],
                                       np.full(n, 2), np.full(n, 2), np.arange(n), 0)
        self.assertTrue(np.all(level == 1))
        for f in self.lib.flavor[index]:
            self.assertEqual(get_splitting_summary(f), get_splitting_summary("((bj)j)b"))
        child_A, child_B = library_child_flavors(self.lib, index)
        for a, b in zip(child_A, child_B):
            self.assertEqual((a.count("b"), a.count("j"), b.count("b"), b.count("j")), (1, 2, 1, 0))

    def test_escalates_when_only_same_event(self):
        """With min_entries=1, (jb)b's exact group is its single row; a target from that row's own
        event must move to the child-content group (shared with (bj)b). (bj)(bj) moves on to the
        parent-content group 2b2j (shared with ((bj)j)b). (jj)(jj) is alone at every level ->
        last-resort self match, counted."""
        lib = SplittingLibrary(self.rows, carry_fields=["btagScore"], min_entries=1)
        d = lib.data
        for flavor, expect_level, expect_self in (("(jb)b", 1, 0), ("(bj)(bj)", 2, 0), ("(jj)(jj)", 0, 1)):
            row = np.where(lib.flavor == flavor)[0]
            index, level = lib.lookup(lib.flavor[row], d["pt"][row], d["eta"][row],
                                      d["run"][row], d["luminosityBlock"][row], d["event"][row], 0)
            self.assertEqual(int(level[0]), expect_level)
            self.assertEqual(lib.n_self_matches, expect_self)
            self.assertEqual(bool(index[0] == row[0]), bool(expect_self))

    def test_lookup_counts(self):
        lib = SplittingLibrary(self.rows, carry_fields=["btagScore"], min_entries=10)
        n = 7
        t = self.targets
        lib.lookup(np.array(["bb"] * n + ["((bj)j)b"] * 2, dtype=object), t["pt"][:n + 2], t["eta"][:n + 2],
                   np.full(n + 2, 2), np.full(n + 2, 2), np.arange(n + 2), 0)
        self.assertEqual(lib.lookup_counts, {"exact": n, "child_content": 2, "parent_content": 0, "coarse": 0, "self_match": 0})
        lib.lookup(np.array(["bb"], dtype=object), t["pt"][:1], t["eta"][:1], [2], [2], [0], 0)
        self.assertEqual(lib.lookup_counts["exact"], n + 1)          # running total

    def test_library_rank_wraps(self):
        index_big, _ = self._lookup(rank=10_000)
        self.assertTrue(np.all(index_big >= 0))


class libraryDeclusteringTestCase(unittest.TestCase):
    """make_synthetic_event with a library built from other toy events."""

    @classmethod
    def setUpClass(cls):
        lib_selev, lib_jets = make_toy_events(n_events=400, seed=21)
        lib_clustered, lib_splittings = cluster_bs(lib_jets, debug=False)
        lib_clean = clean_ISR(lib_clustered, lib_splittings)
        cls.lib_rows = build_splitting_library_rows(lib_selev, lib_jets, lib_splittings, lib_clean)
        cls.lib = SplittingLibrary(cls.lib_rows, carry_fields=["btagScore"], min_entries=5)

        cls.selev, cls.jets = make_toy_events(n_events=40, seed=5)
        clustered, splittings = cluster_bs(cls.jets, debug=False)
        cls.clustered = clean_ISR(clustered, splittings)
        for field, values in carry_child_fields(cls.clustered, cls.jets, ["btagScore"]).items():
            cls.clustered[field] = values
        cls.event_ids = np.column_stack([np.asarray(cls.selev[f]) for f in ("run", "luminosityBlock", "event")]).astype(np.int64)

    def _run(self, seed):
        return make_synthetic_event(self.clustered, None, declustering_rand_seed=seed, b_pt_threshold=30,
                                    library=self.lib, event_ids=self.event_ids)

    def test_jet_content_preserved(self):
        out = self._run(0)
        n_b_in = [f.count("b") for f in ["".join(ev) for ev in ak.to_list(self.clustered.jet_flavor)]]
        n_j_in = [f.count("j") for f in ["".join(ev) for ev in ak.to_list(self.clustered.jet_flavor)]]
        self.assertEqual([ev.count("b") for ev in ak.to_list(out.jet_flavor)], n_b_in)
        self.assertEqual([ev.count("j") for ev in ak.to_list(out.jet_flavor)], n_j_in)

    def test_btag_scores_are_real(self):
        """Every output b-tag score is either a target jet's own score (never declustered) or a
        library child's score."""
        out = self._run(0)
        lib = np.concatenate([self.lib.data["A_btagScore"], self.lib.data["B_btagScore"]]).astype(np.float64)
        allowed = np.concatenate([np.asarray(ak.flatten(self.jets.btagScore)), lib[~np.isnan(lib)]])
        scores = np.asarray(ak.flatten(out.btagScore))
        self.assertFalse(np.any(np.isnan(scores)))
        # library values are stored as float32
        self.assertLess(np.max(np.min(np.abs(scores[:, None] - allowed[None, :]), axis=1)), 1e-6)

    def _run_random(self, seed, k=5):
        return make_synthetic_event(self.clustered, None, declustering_rand_seed=seed, b_pt_threshold=30,
                                    library=self.lib, event_ids=self.event_ids,
                                    library_selection="random", library_k_neighbors=k)

    def test_random_reproducible_and_seeded(self):
        a, b, c = self._run_random(3), self._run_random(3), self._run_random(4)
        np.testing.assert_array_equal(np.asarray(ak.flatten(a.pt)), np.asarray(ak.flatten(b.pt)))
        self.assertFalse(np.allclose(np.asarray(ak.flatten(a.pt)), np.asarray(ak.flatten(c.pt))))
        self.assertEqual(ak.to_list(ak.num(a)), ak.to_list(ak.num(c)))      # same jet content per event

    def test_random_k1_is_rank0(self):
        """k_neighbors 1 draws rank 0 and steps out by the retry count: rank mode, seed 0."""
        a, b = self._run_random(7, k=1), self._run(0)
        for f in ("pt", "eta", "phi", "btagScore"):
            np.testing.assert_array_equal(np.asarray(ak.flatten(a[f])), np.asarray(ak.flatten(b[f])))

    def test_random_ranks_uniform(self):
        rng = np.random.default_rng(1)
        n = 20000
        r = random_ranks(rng.uniform(30, 300, n), rng.uniform(-2.5, 2.5, n), rng.uniform(-3, 3, n),
                         np.arange(n), 5, (0, 0, 0))
        counts = np.bincount(r, minlength=5)
        self.assertEqual(len(counts), 5)
        self.assertTrue(np.all(np.abs(counts - n / 5) < 5 * np.sqrt(n / 5)), counts)
        r2 = random_ranks(rng.uniform(30, 300, n), rng.uniform(-2.5, 2.5, n), rng.uniform(-3, 3, n),
                          np.arange(n), 5, (1, 0, 0))
        self.assertLess(np.mean(r == r2), 0.3)          # another seed -> (nearly) independent draw

    def test_seeds_differ(self):
        a, b = self._run(0), self._run(1)
        self.assertFalse(np.allclose(np.asarray(ak.flatten(a.pt)), np.asarray(ak.flatten(b.pt))))
        c = self._run(0)
        np.testing.assert_array_equal(np.asarray(ak.flatten(a.pt)), np.asarray(ak.flatten(c.pt)))


if __name__ == "__main__":
    unittest.main()
