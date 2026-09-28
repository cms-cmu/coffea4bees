"""
Library of real jet splittings for library-based ("replacement") declustering.

Instead of sampling the decluster variables from binned PDFs, the library option replaces each
clustered jet by a real splitting of the same exact type (``jet_flavor`` tree string) found by
nearest neighbour in the parent's (log pT, |eta|), and aligns the real child four-vectors onto
the target parent. See ~/ClaudeBrain/outputs/decluster-replacement/design.md.

Production side: turning the splittings of ``cluster_bs`` into flat per-splitting rows written by
the cluster processor (build_splitting_library_rows).
Consumption side: SplittingLibrary (load, rank-r nearest-neighbour lookup with same-event
exclusion) and align_children (place the real children onto a target parent).

One row per splitting (including sub-splittings):
    run, luminosityBlock, event     -- source event (self-match exclusion)
    jet_flavor                      -- exact tree string, stored as ASCII bytes (uproot cannot
                                       write string branches); decode with decode_flavor
    in_clean_tree                   -- node belongs to a clustered jet that survived clean_ISR
    pt, eta, phi, mass              -- parent
    A_pt, A_eta, A_phi, A_mass      -- child A (children ordered as by combine_particles, so the
    B_pt, B_eta, B_phi, B_mass         child flavors are children_jet_flavors(jet_flavor))
    A_<field>, B_<field>            -- carry-along fields of single-jet children (NaN for a
                                       combined child)
"""
import logging

import numpy as np
import awkward as ak

_P4 = ["pt", "eta", "phi", "mass"]


def encode_flavor(flavors):
    """Per-event jagged array of jet_flavor strings -> same structure, each string replaced by
    its list of uint8 ASCII codes."""
    flat = ak.to_list(ak.flatten(flavors))
    codes = np.frombuffer("".join(flat).encode("ascii"), dtype=np.uint8)
    encoded = ak.unflatten(codes, [len(s) for s in flat])
    return ak.unflatten(encoded, ak.num(flavors))


def decode_flavor(codes):
    """Flat array of uint8 ASCII-code lists -> numpy array of python strings."""
    offsets = np.asarray(ak.num(codes))
    buffer = np.asarray(ak.flatten(codes)).tobytes().decode("ascii")
    bounds = np.concatenate([[0], np.cumsum(offsets)])
    return np.array([buffer[bounds[i]:bounds[i + 1]] for i in range(len(offsets))], dtype=object)


def flag_clean_tree_splittings(clustered_jets_clean, splittings):
    """Boolean per splitting: True if the splitting is a node of a clustered jet that survived
    clean_ISR (i.e. a splitting the declustering will actually need to undo).

    Nodes are identified within an event by (jet_flavor, pt): cluster_bs copies the four-vectors
    of combined objects into both the parent's part_A/part_B and the splitting record.
    """
    jet_flav = ak.to_list(clustered_jets_clean.jet_flavor)
    jet_pt   = ak.to_list(clustered_jets_clean.pt)
    s_flav   = ak.to_list(splittings.jet_flavor)
    s_pt     = ak.to_list(splittings.pt)
    a_flav, a_pt = ak.to_list(splittings.part_A.jet_flavor), ak.to_list(splittings.part_A.pt)
    b_flav, b_pt = ak.to_list(splittings.part_B.jet_flavor), ak.to_list(splittings.part_B.pt)

    flags = []
    for iE in range(len(s_flav)):
        n_split = len(s_flav[iE])
        flag = [False] * n_split

        def find(flavor, pt):
            best, best_d = None, np.inf
            for iS in range(n_split):
                if s_flav[iE][iS] != flavor:
                    continue
                d = abs(s_pt[iE][iS] - pt)
                if d < best_d:
                    best, best_d = iS, d
            return best

        stack = [(f, p) for f, p in zip(jet_flav[iE], jet_pt[iE]) if len(f) > 1]
        while stack:
            flavor, pt = stack.pop()
            iS = find(flavor, pt)
            if iS is None or flag[iS]:
                continue
            flag[iS] = True
            if len(a_flav[iE][iS]) > 1:
                stack.append((a_flav[iE][iS], a_pt[iE][iS]))
            if len(b_flav[iE][iS]) > 1:
                stack.append((b_flav[iE][iS], b_pt[iE][iS]))
        flags.append(flag)

    return ak.Array(flags)


def carry_child_fields(child, input_jets, fields):
    """For each child of each splitting, look up the carry-along ``fields`` of the input jet it
    came from (single-jet children only; combined children get NaN).

    The single-jet children of cluster_bs are copies of the input jets, so the match is exact:
    the closest input jet in (pt, eta) within the same event.
    """
    d = (np.abs(child.pt[:, :, None] - input_jets.pt[:, None, :])
         + np.abs(child.eta[:, :, None] - input_jets.eta[:, None, :]))
    idx = ak.argmin(d, axis=2)
    is_single = ak.str.length(child.jet_flavor) == 1

    carried = {}
    for field in fields:
        values = ak.fill_none(input_jets[field][idx], np.nan)
        carried[field] = ak.where(is_single, ak.values_astype(values, np.float64), np.nan)
    return carried


def build_splitting_library_rows(selev, input_jets, splittings, clustered_jets_clean, carry_fields=("btagScore",)):
    """Flatten the splittings of a chunk into library rows (see module docstring)."""
    n_split = ak.num(splittings)

    columns = {
        "run":             ak.flatten(ak.broadcast_arrays(selev.run, splittings.pt)[0]),
        "luminosityBlock": ak.flatten(ak.broadcast_arrays(selev.luminosityBlock, splittings.pt)[0]),
        "event":           ak.flatten(ak.broadcast_arrays(selev.event, splittings.pt)[0]),
        "jet_flavor":      ak.flatten(encode_flavor(splittings.jet_flavor), axis=1),
        "in_clean_tree":   ak.flatten(flag_clean_tree_splittings(clustered_jets_clean, splittings)),
    }
    for var in _P4:
        columns[var] = ak.flatten(splittings[var])
    for tag, child in (("A", splittings.part_A), ("B", splittings.part_B)):
        for var in _P4:
            columns[f"{tag}_{var}"] = ak.flatten(child[var])
        for field, values in carry_child_fields(child, input_jets, carry_fields).items():
            columns[f"{tag}_{field}"] = ak.flatten(values)

    for name in columns:
        if name in ("jet_flavor", "in_clean_tree", "run", "luminosityBlock", "event"):
            continue
        columns[name] = ak.values_astype(columns[name], np.float32)

    rows = ak.zip(columns, depth_limit=1)
    assert len(rows) == int(np.sum(n_split))
    return rows


#
#  Consumption side
#

def _content_key(flavor):
    """Total (b, j) content of a jet_flavor tree, e.g. '((bj)j)b' -> '2b2j'."""
    return f"{flavor.count('b')}b{flavor.count('j')}j"


def _rapidity(pt, eta, mass):
    """Rapidity of (pt, eta, mass): y = asinh(pz / mT)."""
    pz = pt * np.sinh(eta)
    mT = np.sqrt(pt**2 + mass**2)
    return np.arcsinh(pz / mT)


def _eta_from_rapidity(pt, y, mass):
    """Inverse of _rapidity: eta = asinh(pz / pt), pz = mT sinh(y)."""
    mT = np.sqrt(pt**2 + mass**2)
    return np.arcsinh(mT * np.sinh(y) / pt)


def _wrap_phi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


class SplittingLibrary:
    """Real splittings of one era, grouped by exact jet_flavor, with a cached KD-tree per group on
    (log pT, |eta|) of the parent.

    Lookup key: the exact jet_flavor if it has >= min_entries rows; else the child (b, j) content
    summary (same b/j count per child, any tree); else the parent's total (b, j) content (any
    split). The library children bring their own flavors, so all three keep the event's b/j
    count. Last resort, counted in n_coarse: the coarse splitting_name (can change it).
    """

    #: number of extra neighbours queried to leave room for same-event exclusion
    n_extra = 4

    def __init__(self, rows, carry_fields=("btagScore",), min_entries=10, clean_tree_only=True):
        from coffea4bees.jet_clustering.declustering import get_splitting_name, get_splitting_summary

        if clean_tree_only:
            rows = rows[np.asarray(rows.in_clean_tree, dtype=bool)]

        self.carry_fields = list(carry_fields)
        self.min_entries = min_entries
        self.flavor = decode_flavor(rows.jet_flavor)

        names = ["run", "luminosityBlock", "event", "pt", "eta", "phi", "mass"]
        names += [f"{t}_{v}" for t in "AB" for v in _P4 + self.carry_fields]
        self.data = {n: np.asarray(rows[n]) for n in names}
        self.data["run"] = self.data["run"].astype(np.int64)
        self.data["luminosityBlock"] = self.data["luminosityBlock"].astype(np.int64)
        self.data["event"] = self.data["event"].astype(np.int64)

        # Lookup groups, finest first
        unique_flavors, inverse = np.unique(self.flavor, return_inverse=True)
        summary = np.array([str(get_splitting_summary(f)) for f in unique_flavors], dtype=object)[inverse]
        content = np.array([_content_key(f) for f in unique_flavors], dtype=object)[inverse]
        coarse  = np.array([get_splitting_name(f) for f in unique_flavors], dtype=object)[inverse]
        self._groups = [self._index(self.flavor), self._index(summary), self._index(content), self._index(coarse)]
        self._trees = {}
        logging.info(f"SplittingLibrary: {len(self.flavor)} splittings, {len(self._groups[0])} exact types")

    @staticmethod
    def _index(keys):
        order = np.argsort(keys, kind="stable")
        uniq, start = np.unique(keys[order], return_index=True)
        bounds = list(start) + [len(order)]
        return {k: order[bounds[i]:bounds[i + 1]] for i, k in enumerate(uniq)}

    @classmethod
    def from_files(cls, files_yaml, year, **kwargs):
        """Load from a {year: [files]} registry (local path or root:// URL, as the hemi library)."""
        import fsspec
        import uproot
        import yaml

        with fsspec.open(files_yaml, "r") as f:
            files = yaml.safe_load(f)[year]
        files = [files] if isinstance(files, str) else files
        rows = ak.concatenate([batch for batch in uproot.iterate({f: "Events" for f in files}, library="ak",
                                                                 step_size=500_000)])
        return cls(rows, **kwargs)

    def resolve_keys(self, flavor):
        """Groups to try for a target jet_flavor, finest first, as [(level, key), ...]
        (level 0 exact, 1 child-content summary, 2 parent content, 3 coarse splitting_name; see the
        class docstring). Starts at the finest
        group with >= min_entries rows (else the finest non-empty one) and continues coarser; the
        coarser groups are used when the finer one has only same-event candidates."""
        from coffea4bees.jet_clustering.declustering import get_splitting_name, get_splitting_summary

        candidates = [flavor, str(get_splitting_summary(flavor)), _content_key(flavor), get_splitting_name(flavor)]
        existing = [(level, key) for level, key in enumerate(candidates) if key in self._groups[level]]
        if not existing:
            raise KeyError(f"SplittingLibrary: no splittings compatible with {flavor}")
        # the coarse level only when nothing content-preserving exists
        if len(existing) > 1 and existing[-1][0] == 3:
            existing = existing[:-1]
        for i, (level, key) in enumerate(existing):
            if len(self._groups[level][key]) >= self.min_entries:
                return existing[i:]
        return existing

    def _tree(self, level, key):
        from scipy.spatial import cKDTree

        if (level, key) not in self._trees:
            members = self._groups[level][key]
            points = np.column_stack([np.log(self.data["pt"][members]), np.abs(self.data["eta"][members])])
            self._trees[(level, key)] = (cKDTree(points), members)
        return self._trees[(level, key)]

    def lookup(self, flavor, pt, eta, run, luminosityBlock, event, rank):
        """Library row index for each target (flat arrays), plus the lookup level used.

        Takes the rank-th nearest neighbour (0 = nearest) in (log pT, |eta|) after dropping
        neighbours from the target's own (run, luminosityBlock, event); the rank wraps modulo the
        group size, and if fewer allowed neighbours are found, the farthest allowed one is used.
        Targets whose group holds only same-event candidates move to the next coarser group; if
        none has any, the nearest (same-event) row is used and counted in self.n_self_matches.
        """
        flavor = np.asarray(flavor, dtype=object)
        pt, eta = np.asarray(pt, dtype=np.float64), np.asarray(eta, dtype=np.float64)
        rank = np.broadcast_to(np.asarray(rank, dtype=np.int64), flavor.shape)
        target_id = np.column_stack([np.asarray(run), np.asarray(luminosityBlock), np.asarray(event)]).astype(np.int64)
        points = np.column_stack([np.log(pt), np.abs(eta)])

        index = np.full(len(flavor), -1, dtype=np.int64)
        level_used = np.full(len(flavor), -1, dtype=np.int8)
        self.n_self_matches = 0

        for f in np.unique(flavor):
            pending = np.where(flavor == f)[0]
            groups = self.resolve_keys(f)
            for level, key in groups:
                tree, members = self._tree(level, key)
                n = len(members)
                r = rank[pending] % n
                k = int(min(r.max() + 1 + self.n_extra, n))
                _, nbr = tree.query(points[pending], k=list(range(1, k + 1)))
                lib_idx = members[nbr]                                        # (n_pending, k)

                lib_id = np.stack([self.data["run"][lib_idx], self.data["luminosityBlock"][lib_idx],
                                   self.data["event"][lib_idx]], axis=-1)     # (n_pending, k, 3)
                allowed = ~np.all(lib_id == target_id[pending][:, None, :], axis=-1)
                n_allowed = allowed.sum(axis=1)

                # column of the r-th allowed neighbour (or the last allowed one)
                allowed_rank = np.cumsum(allowed, axis=1) - 1
                pick = allowed & (allowed_rank == np.minimum(r, n_allowed - 1)[:, None])
                ok = n_allowed > 0
                col = np.argmax(pick, axis=1)
                index[pending[ok]] = lib_idx[np.where(ok)[0], col[ok]]
                level_used[pending[ok]] = level
                pending = pending[~ok]
                if len(pending) == 0:
                    break

            if len(pending):
                level, key = groups[0]
                tree, members = self._tree(level, key)
                _, nbr = tree.query(points[pending], k=1)
                index[pending] = members[nbr]
                level_used[pending] = level
                self.n_self_matches += len(pending)

        if self.n_self_matches:
            logging.warning(f"SplittingLibrary.lookup: {self.n_self_matches} targets had only same-event candidates")
        self.n_coarse = int(np.sum(level_used == 3))
        if self.n_coarse:
            logging.warning(f"SplittingLibrary.lookup: {self.n_coarse} targets used the coarse splitting_name group (b/j content may change)")
        return index, level_used


def align_children(library, index, pt, eta, phi, mass=None, *, scale_pt=True, boost_z=True):
    """Place the children of library rows ``index`` onto target parents (pt, eta, phi).

    1. scale both children's four-vectors by pt / pt_lib (pT exact; m/pT preserved)
    2. reflect in z if sign(eta) != sign(eta_lib) (matching is on |eta|)
    3. rotate in phi by phi - phi_lib (phi exact)
    4. boost along z so the aligned parent has the target pz (eta exact)

    The parent mass is not preserved. ``mass`` is unused (kept for symmetry with the target p4).
    Returns {"A": {pt, eta, phi, mass, <carry fields>}, "B": {...}} of flat numpy arrays.
    """
    d = library.data
    pt, eta, phi = (np.asarray(x, dtype=np.float64) for x in (pt, eta, phi))
    lib_pt, lib_eta, lib_phi, lib_mass = (d[v][index].astype(np.float64) for v in _P4)

    scale = pt / lib_pt if scale_pt else np.ones_like(pt)
    flip = np.sign(eta) != np.sign(lib_eta)
    dphi = phi - lib_phi

    if boost_z:
        parent_pt = lib_pt * scale
        parent_mass = lib_mass * scale
        y_lib = _rapidity(lib_pt, np.where(flip, -lib_eta, lib_eta), lib_mass)   # scale-invariant
        y_target = np.arcsinh(parent_pt * np.sinh(eta) / np.sqrt(parent_pt**2 + parent_mass**2))
        dy = y_target - y_lib
    else:
        dy = np.zeros_like(pt)

    out = {}
    for tag in "AB":
        c_pt   = d[f"{tag}_pt"][index].astype(np.float64) * scale
        c_mass = d[f"{tag}_mass"][index].astype(np.float64) * scale
        c_eta  = d[f"{tag}_eta"][index].astype(np.float64)
        c_eta  = np.where(flip, -c_eta, c_eta)
        c_eta  = _eta_from_rapidity(c_pt, _rapidity(c_pt, c_eta, c_mass) + dy, c_mass)
        c_phi  = _wrap_phi(d[f"{tag}_phi"][index].astype(np.float64) + dphi)
        out[tag] = {"pt": c_pt, "eta": c_eta, "phi": c_phi, "mass": c_mass}
        for field in library.carry_fields:
            out[tag][field] = d[f"{tag}_{field}"][index].astype(np.float64)
    return out


def library_child_flavors(library, index):
    """(flavor_A, flavor_B) of library rows ``index`` (numpy object arrays)."""
    from coffea4bees.jet_clustering.declustering import children_jet_flavors

    flavors = library.flavor[index]
    uniq, inverse = np.unique(flavors, return_inverse=True)
    children = [children_jet_flavors(f) for f in uniq]
    child_A = np.array([c[0] for c in children], dtype=object)[inverse]
    child_B = np.array([c[1] for c in children], dtype=object)[inverse]
    return child_A, child_B


def decluster_with_library(jets, library, event_ids, rank, *, scale_pt=True, boost_z=True):
    """Library replacement for sample_PDFs_vs_pT + decluster_combined_jets.

    jets:      jagged [event][jet] combined jets to decluster (pt, eta, phi, jet_flavor)
    event_ids: (n_events, 3) int array of (run, luminosityBlock, event) for self-match exclusion
    rank:      neighbour rank (int) for every jet
    Returns the jagged child arrays (A, B) with pt, eta, phi, mass, jet_flavor, btag_string and
    the library's carry fields (NaN for combined children).
    """
    from coffea.nanoevents.methods import vector

    counts = np.asarray(ak.num(jets))
    flat = ak.flatten(jets)
    ids = np.repeat(np.asarray(event_ids, dtype=np.int64), counts, axis=0)
    flavor = np.asarray(ak.to_list(flat.jet_flavor), dtype=object)

    index, _ = library.lookup(flavor, np.asarray(flat.pt), np.asarray(flat.eta),
                              ids[:, 0], ids[:, 1], ids[:, 2], rank)
    kids = align_children(library, index, np.asarray(flat.pt), np.asarray(flat.eta), np.asarray(flat.phi),
                          scale_pt=scale_pt, boost_z=boost_z)
    child_flavor = dict(zip("AB", library_child_flavors(library, index)))

    children = []
    for tag in "AB":
        k = kids[tag]
        is_single = np.array([len(f) == 1 for f in child_flavor[tag]], dtype=bool)
        btag = k.get("btagScore")
        btag_string = [str(round(float(b), 3)) if (s and btag is not None) else "" for b, s in
                       zip(btag if btag is not None else np.zeros(len(is_single)), is_single)]
        fields = {
            "pt": k["pt"], "eta": k["eta"], "phi": k["phi"], "mass": k["mass"],
            "jet_flavor": ak.Array(list(child_flavor[tag])),
            "btag_string": ak.Array(btag_string),
        }
        for field in library.carry_fields:
            fields[field] = k[field]
        children.append(ak.zip({n: ak.unflatten(v, counts) for n, v in fields.items()},
                               with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior))
    return children[0], children[1]
