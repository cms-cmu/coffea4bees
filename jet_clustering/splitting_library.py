"""
Library of real jet splittings for library-based ("replacement") declustering.

Instead of sampling the decluster variables from binned PDFs, the library option replaces each
clustered jet by a real splitting of the same exact type (``jet_flavor`` tree string) found by
nearest neighbour in the parent's (log pT, |eta|), and aligns the real child four-vectors onto
the target parent. See ~/ClaudeBrain/outputs/decluster-replacement/design.md.

This module holds the library *production* side: turning the splittings of ``cluster_bs`` into
flat per-splitting rows written by the cluster processor.

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
