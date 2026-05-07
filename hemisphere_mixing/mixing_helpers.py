import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector as v
import uproot
import yaml
ak.behavior.update(v.behavior)
import ast
import numba as nb
import numpy as np
from scipy.spatial import cKDTree  # "c" = C-optimized
from src.data_formats.root import Chunk, TreeReader
from coffea.nanoevents.methods import vector
from src.math_tools.random import Squares


@nb.njit(cache=True)
def _thrust_event_numba(px_i, py_i, n_steps=720):
    phis = np.linspace(0.0, 2.0*np.pi, n_steps)
    cos_t = np.cos(phis)
    sin_t = np.sin(phis)

    pTi = np.hypot(px_i, py_i)
    #denom = pTi.sum()
    #if denom == 0.0:
    #    return np.nan, 0.0, 0.0, 0.0, 0.0

    proj = np.abs(px_i[:, None]*cos_t[None,:] + py_i[:, None]*sin_t[None,:])
    sums = proj.sum(axis=0)
    best_idx = np.argmax(sums)
    best_phi = phis[best_idx]
    best_sum = sums[best_idx]

    nx, ny = np.cos(best_phi), np.sin(best_phi)
    mx, my = -ny, nx
    #T = best_sum/denom
    #T_minor = np.sum(np.abs(-px_i*ny + py_i*nx))/denom
    return best_phi, nx, ny, mx, my



def transverse_thrust_awkward_fast(p4, n_steps=720, refine_rounds=0, refine_factor=6):
    """
    Fully Awkward-1.x–compatible computation of transverse thrust (T)
    and thrust minor (T_minor) for jagged Momentum4D objects.
    Works directly on e.g. events.Jet.

    n_steps number of coarse angular steps (default: 720->0.5 degree granularity)
    refine_rounds: how many times to zoom in around the best angle (optional)
    refine_factor: how much denser each refinement grid is
    """
    #if not (hasattr(p4, "px") and hasattr(p4, "py")):
    #    raise ValueError("Input must have Momentum4D behavior (with .px/.py).")

    px, py = p4.px, p4.py

    # Precompute angular grid
    phis = np.linspace(0.0, 2.0 * np.pi, num=n_steps, endpoint=False)
    cos_t = np.cos(phis)
    sin_t = np.sin(phis)

    # loop on events
    results = []
    for px_i, py_i in zip(px, py):
        px_np, py_np = np.asarray(px_i), np.asarray(py_i)
        if len(px_np) == 0:
            results.append(dict(phi=np.nan,
                                axis=dict(nx=np.nan, ny=np.nan),
                                minor=dict(nx=np.nan, ny=np.nan)))
            continue
        th, nx, ny, mx, my = _thrust_event_numba(px_np, py_np, n_steps)
        results.append(dict(phi=th,
                            axis=dict(nx=nx, ny=ny), minor=dict(nx=mx, ny=my)))


    # pack back into an Awkward array of records
    return ak.Array(results)




def transverse_thrust_awkward(p4, n_steps=720, refine_rounds=0, refine_factor=6):
    """
    Fully Awkward-1.x–compatible computation of transverse thrust (T)
    and thrust minor (T_minor) for jagged Momentum4D objects.
    Works directly on e.g. events.Jet.

    n_steps number of coarse angular steps (default: 720->0.5 degree granularity)
    refine_rounds: how many times to zoom in around the best angle (optional)
    refine_factor: how much denser each refinement grid is
    """
    #if not (hasattr(p4, "px") and hasattr(p4, "py")):
    #    raise ValueError("Input must have Momentum4D behavior (with .px/.py).")

    px, py = p4.px, p4.py

    # Precompute angular grid
    phis = np.linspace(0.0, 2.0 * np.pi, num=n_steps, endpoint=False)
    cos_t = np.cos(phis)
    sin_t = np.sin(phis)

    # loop on events
    results = []
    for px_i, py_i in zip(px, py):
        # handle empty events
        if len(px_i) == 0:
            results.append(
                dict(T=np.nan, T_minor=np.nan, phi=np.nan,
                     axis=dict(nx=np.nan, ny=np.nan),
                     minor=dict(nx=np.nan, ny=np.nan))
            )
            continue

        pTi = np.hypot(px_i, py_i)

        #denom = np.sum(pTi)
        #if denom == 0:
        #    results.append(
        #        dict(T=np.nan, T_minor=np.nan, phi=np.nan,
        #             axis=dict(nx=np.nan, ny=np.nan),
        #             minor=dict(nx=np.nan, ny=np.nan))
        #    )
        #    continue

        # --- coarse scan ---
        proj = np.abs(px_i[:, None] * cos_t[None, :] + py_i[:, None] * sin_t[None, :])
        sums = np.sum(proj, axis=0)
        best_idx = int(np.argmax(sums))
        best_phi = phis[best_idx]
        best_sum = sums[best_idx]

        # --- optional refinement ---
        if refine_rounds > 0:
            window = 2.0 * np.pi / n_steps
            for _ in range(refine_rounds):
                half_w = 0.6 * window
                local_steps = max(24, refine_factor * 12)
                loc_phis = np.linspace(best_phi - half_w,
                                         best_phi + half_w,
                                         num=local_steps, endpoint=True)
                px_i = np.asarray(px_i)
                py_i = np.asarray(py_i)
                projL = np.abs(px_i[:, None] * np.cos(loc_phis)[None, :]
                               + py_i[:, None] * np.sin(loc_phis)[None, :]).sum(axis=0)
                best_phi = loc_phis[int(np.argmax(projL))]
                best_sum = np.max(projL)
                window *= 0.35
            best_phi = float(np.mod(best_phi, 2.0 * np.pi))

        # Axes
        nx, ny = np.cos(best_phi), np.sin(best_phi)
        mx, my = -ny, nx

        #T = best_sum / denom
        #T_minor = np.sum(np.abs(-px_i * ny + py_i * nx)) / denom

        results.append(
            #dict(T=T, T_minor=T_minor, phi=best_phi,
            #     axis=dict(nx=nx, ny=ny), minor=dict(nx=mx, ny=my))
            dict(phi=best_phi,
                 axis=dict(nx=nx, ny=ny), minor=dict(nx=mx, ny=my))
        )

    # pack back into an Awkward array of records
    return ak.Array(results)


def split_hemispheres(p4, thrust):
    """
    Split per-event jets into 'aligned' and 'anti-aligned' hemispheres
    based on the transverse thrust axis.

    Parameters
    ----------
    p4 : ak.Array (Momentum4D)
        Jagged per-event jet four-vectors.
    thrust : ak.Array
        Output of transverse_thrust_awkward() for the same events.

    Returns
    -------
    aligned, anti : ak.Array, ak.Array
        Two jagged Momentum4D arrays with jets partitioned by hemisphere.
    """
    px, py = p4.px, p4.py
    nx, ny = thrust.axis.nx, thrust.axis.ny

    # Compute projection of each jet pT onto the thrust axis
    d = px * nx[:, None] + py * ny[:, None]

    # Masks
    aligned_mask = d >= 0
    anti_mask = ~aligned_mask

    aligned = p4[aligned_mask]
    anti = p4[anti_mask]

    return aligned, anti


def compute_hemi_vars(hemis):
    hemis_sumJet = hemis.Jet.sum(axis=1)
    hemis["pz"]  = hemis_sumJet.pz
    hemis["combinedMass"] = hemis_sumJet.mass

    cos_t = np.cos(hemis.thrust_phi)
    sin_t = np.sin(hemis.thrust_phi)

    hemis["sumPt_T"]       = ak.sum(np.abs(  hemis.Jet.px * cos_t + hemis.Jet.py * sin_t), axis=1)
    hemis["sumPt_T_minor"] = ak.sum(np.abs( -hemis.Jet.px * sin_t + hemis.Jet.py * cos_t), axis=1)
    return hemis


def boost_jets_along_z(jets, pz_target, pz_matched, E_matched):
    """
    Apply Lorentz boost along z to jets so their combined pz matches target.

    Parameters
    ----------
    jets : ak.Array
        Jagged array of jets with PtEtaPhiMLorentzVector behavior
    pz_target : ak.Array
        Target pz values (one per hemisphere), shape (n_hemis,)
    pz_matched : ak.Array
        Matched hemisphere pz values (one per hemisphere), shape (n_hemis,)
    E_matched : ak.Array
        Matched hemisphere energy values (one per hemisphere), shape (n_hemis,)

    Returns
    -------
    boosted_jets : ak.Array
        Jets with boosted 4-momenta, same structure as input
    delta_rapidity : ak.Array
        Rapidity shift applied (for diagnostics), shape (n_hemis,)
    """
    # Compute rapidity shift using exact formula: arcsinh(pz / M_T)
    # where M_T = sqrt(E^2 - pz^2) is the transverse mass of the matched hemisphere.
    # This gives exact pz matching after the boost (unlike the arctanh(pz/E) approximation,
    # which is only exact for massless objects).
    M_T             = np.sqrt(E_matched**2 - pz_matched**2)
    y_matched       = np.arcsinh(pz_matched / M_T)
    y_target_approx = np.arcsinh(pz_target  / M_T)

    delta_y = y_target_approx - y_matched

    # Boost velocity: beta_z = tanh(delta_y)
    beta_z = np.tanh(delta_y)

    # Create boost vector (0, 0, beta_z) for each hemisphere
    boost_vec = ak.zip(
        {
            "x": ak.zeros_like(beta_z),
            "y": ak.zeros_like(beta_z),
            "z": beta_z,
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )

    # Apply boost to jets using coffea's built-in method
    # Need to broadcast boost_vec to match jet structure (one boost per hemisphere)
    boosted_jets = jets.boost(boost_vec)

    # Manually create pt, eta, phi, mass representation.
    # Use jets.mass (original mass) rather than boosted_jets.mass:
    # mass is Lorentz-invariant, so it is unchanged by the boost.
    # Recomputing it as sqrt(E^2 - |p|^2) after the boost causes NaN for
    # soft/forward jets where floating-point cancellation makes m^2 slightly negative.
    boosted_jets_cylind = ak.zip(
        {
            "pt": boosted_jets.pt,
            "eta": boosted_jets.eta,
            "phi": boosted_jets.phi,
            "mass": jets.mass,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    # # Copy over other jet branches (btagScore, etc.) that aren't part of the 4-vector
    # for field in jets.fields:
    #     if field not in ["pt", "eta", "phi", "mass", "x", "y", "z", "t"]:
    #         boosted_jets_cylind[field] = jets[field]

    return boosted_jets_cylind, delta_y


def split_events_into_hemispheres(event, tagged_key="tagJet"):

    #
    #  Get Thrust axis
    #
    thrust = transverse_thrust_awkward_fast(event.Jet, n_steps=720, refine_rounds=2)

    #
    # Thin out unneeded branches from jets
    #
    jets = event.Jet
    drop = {"muonIdxG", "electronIdxG","NOTTHERE",'electronIdx1G', 'electronIdx2G','muonIdx1G', 'muonIdx2G'}
    keep = [f for f in jets.fields if f not in drop]
    thinned = jets[keep]

    # the original record name, e.g. "PtEtaPhiMLorentzVector"
    record = ak.parameters(jets).get("__record__")

    # the behavior dictionary (vector mixin functions)
    behavior = jets.behavior

    # restore it
    thinned = ak.Array(thinned.layout, behavior=behavior)
    thinned_jets = ak.with_name(
        thinned,
        "PtEtaPhiMLorentzVector",
        behavior=jets.behavior
    )

    #
    #  For outputs
    #
    jet_posHemi, jet_negHemi   = split_hemispheres(thinned_jets, thrust)


    #
    #  For mutltiplicity counting
    #
    tagJet_posHemi, tagJet_negHemi = split_hemispheres(event[tagged_key], thrust)
    selJet_posHemi, selJet_negHemi = split_hemispheres(event.selJet, thrust)


    #
    #  Create hemispere objects
    #
    pos_hemi = ak.zip({"thrust_phi": thrust.phi,
                       "event": event.event,
                       "run": event.run,
                       "luminosityBlock" : event.luminosityBlock,
                       "hemisphereId": np.full(len(event.run), +1),
                       "weight": event.weight,
                       "nSelJet": ak.num(selJet_posHemi, axis=1),
                       "nTagJet": ak.num(tagJet_posHemi, axis=1),
                       "nJet" : ak.num(jet_posHemi, axis=1),
                       "Jet": jet_posHemi,
                       },
                      depth_limit=1
                      )
    pos_hemi = compute_hemi_vars(pos_hemi)

    neg_hemi = ak.zip({"thrust_phi": thrust.phi,
                       "event": event.event,
                       "run" : event.run,
                       "luminosityBlock" : event.luminosityBlock,
                       "hemisphereId": np.full(len(event.run), -1),
                       "weight": event.weight,
                       "nSelJet": ak.num(selJet_negHemi, axis=1),
                       "nTagJet": ak.num(tagJet_negHemi, axis=1),
                       "nJet" : ak.num(jet_negHemi, axis=1),
                       "Jet": jet_negHemi,
                       },
                      depth_limit=1
                      )
    neg_hemi = compute_hemi_vars(neg_hemi)

    return pos_hemi, neg_hemi



def read_hemi_files(hemi_files_yaml, year, tree_name="Events", branch_list=None):

    with open(hemi_files_yaml, 'r') as f:
        hemi_library_data = yaml.safe_load(f)
        # print("Keys",hemi_library_data.keys())
        hemi_files = hemi_library_data[year]
        # print("Hemi files:", type(hemi_files), hemi_files)


    hemi_vars = { var_name: [] for var_name in branch_list }

    if isinstance(hemi_files, str):
        for batch in uproot.iterate(
                f"{hemi_files}:{tree_name}",
                branch_list,
                step_size=200_000,  # entries per chunk
                library="np",
        ):

            if hemi_vars[branch_list[0]] is None:
                for var_name in branch_list:
                    hemi_vars[var_name] = batch[var_name]
            else:
                for var_name in branch_list:
                    hemi_vars[var_name] = np.concatenate( (hemi_vars[var_name], batch[var_name]) )
        print(f"\tread_hemi_files: Read n hemispheres: {len(hemi_vars[branch_list[0]])}")

    elif isinstance(hemi_files, list):
        #print("Reading hemisphere files:", hemifiles)
        file_spec = {f: tree_name for f in hemi_files}

        for batch in uproot.iterate(
                file_spec,
                branch_list,
                step_size=200_000,
                library="np",
        ):
            if hemi_vars[branch_list[0]] is None:
                for var_name in branch_list:
                    hemi_vars[var_name] = batch[var_name]
            else:
                for var_name in branch_list:
                    hemi_vars[var_name] = np.concatenate( (hemi_vars[var_name], batch[var_name]) )
        print(f"\tread_hemi_files: Read n hemispheres: {len(hemi_vars[branch_list[0]])}")

    return hemi_vars


def get_filter(data, key, val, low_edge=False, high_edge=False):

    if low_edge and high_edge:
        this_filter = True
    elif low_edge:
        this_filter = (data[key] <= val)
    elif high_edge:
        this_filter = (data[key] >= val)
    else:
        this_filter = (data[key] == val)

    return this_filter




def iter_hemi_filters(hemi_ranges, hemi_data):
    """
    Iterate through (tag, sel, jet) bins and yield:
        (tag, sel, jet, mask, tag_filter, sel_filter, jet_filter)
    where jet = -1 indicates the special 'no jet bins' case.
    """
    tag_keys = list(hemi_ranges.keys())

    for itag, tag in enumerate(tag_keys):


        # Tag selection
        tag_filter = get_filter(
            hemi_data, "nTagJet", tag,
            low_edge=(itag == 0),
            high_edge=(itag == len(tag_keys) - 1)
        )

        # Skip empty tag bins
        if not hemi_ranges[tag]:
            print(f"ERROR: no sel jets for tag = {tag}")
            continue

        # Selected-jet multiplicity loop
        sel_keys = list(hemi_ranges[tag].keys())
        for isel, sel in enumerate(sel_keys):

            sel_filter = get_filter(
                hemi_data, "nSelJet", sel,
                low_edge=(isel == 0),
                high_edge=(isel == len(sel_keys) - 1)
            )

            jet_bins = hemi_ranges[tag][sel]

            # Special case: no jet bins
            if not jet_bins:
                mask = tag_filter & sel_filter
                yield ( (tag, sel, -1), mask )
                continue

            # Jet loop
            for ijet, jet in enumerate(jet_bins):
                jet_filter = get_filter(
                    hemi_data, "nJet", jet,
                    low_edge=(ijet == 0),
                    high_edge=(ijet == len(jet_bins) - 1)
                )
                mask = tag_filter & sel_filter & jet_filter
                yield ( (tag, sel, jet), mask )


def get_hemispheres_data(mask, hemi_data, hemi_vars, hemi_stats=None):
    hemi_data_jet_bin = {}

    for var_name in hemi_vars:
        if hemi_stats and var_name in hemi_stats:
            this_var = hemi_stats[var_name]
            hemi_data_jet_bin[var_name] = (hemi_data[var_name][mask] - this_var["mean"]) / this_var["RMS"]
        else:
            hemi_data_jet_bin[var_name] = hemi_data[var_name][mask]
    return hemi_data_jet_bin


def get_grouped_hemispheres_data(hemi_ranges, hemi_data, hemi_vars, hemi_stats=None):
    grouped_hemi_data = {}

    for jet_mult_key, mask in iter_hemi_filters(hemi_ranges, hemi_data):

        _hemi_stats = hemi_stats[jet_mult_key] if hemi_stats else None
        grouped_hemi_data[jet_mult_key] = get_hemispheres_data(mask, hemi_data, hemi_vars, hemi_stats=_hemi_stats)


    return grouped_hemi_data




def convert_yaml_dict(raw_dict):
    output = {}
    for k, v in raw_dict.items():
        try:
            key = ast.literal_eval(k) if isinstance(k, str) else k
        except Exception:
            key = k  # leave as string if not a tuple

        #print(f"Key: {key} ({type(key)}) Value: {v} ({type(v)})")
        output[key] = convert_yaml_dict(v) if isinstance(v, dict) else v

    return output


def init_hemi_data(hemi_metadata_yaml, hemi_files_yaml, year, hemi_summary_vars, jet_branches, event_branches=["event", "run", "luminosityBlock", "thrust_phi", "hemisphereId", "weight"]):

    # Read in hemisphere library metadata
    with open(hemi_metadata_yaml, 'r') as f:
        hemi_stats_raw = yaml.safe_load(f)

    hemi_stats = convert_yaml_dict(hemi_stats_raw["hemi_summary_vars"])
    jet_ranges = convert_yaml_dict(hemi_stats_raw["jet_mult_ranges"])

    #
    #  Read in Hemisphere library data
    #
    branch_list = ["nJet", "nSelJet", "nTagJet"] + event_branches + hemi_summary_vars + jet_branches

    hemi_data = read_hemi_files(hemi_files_yaml, year, branch_list=branch_list)

    return hemi_data, jet_ranges, hemi_stats


def build_hemi_kdtrees(hemi_metadata_yaml, hemi_files_yaml, year, hemi_summary_vars, jet_branches):


    #
    #  Readin the hemisphere data
    #
    event_branches = ["event", "run", "luminosityBlock", "thrust_phi", "hemisphereId", "weight"]
    hemi_data, jet_ranges, hemi_stats = init_hemi_data(hemi_metadata_yaml, hemi_files_yaml, year, hemi_summary_vars, jet_branches, event_branches)

    #
    #  Group hemisphere data by jet multiplicity bins
    #
    grouped_hemi_data = get_grouped_hemispheres_data(jet_ranges, hemi_data, hemi_vars= event_branches + hemi_summary_vars + jet_branches, hemi_stats=hemi_stats)

    #
    #  Make the Kd-Trees
    #
    kd_trees = {}
    points   = {}
    for jet_mult_key in hemi_stats.keys():
        points  [jet_mult_key] = np.column_stack([ grouped_hemi_data[jet_mult_key][name] for name in hemi_summary_vars])
        kd_trees[jet_mult_key] = cKDTree(points[jet_mult_key])

    return kd_trees, points, jet_ranges, hemi_stats, grouped_hemi_data





def replace_hemis(*, all_hemis, hemi_kd_trees, hemi_stats, hemi_data, hemi_jet_ranges, hemi_summary_vars, jet_branches):

    mixed_hemis = []
    all_hemis["local_idx"] = ak.local_index(all_hemis, axis=0)

    #
    #  Loop on hemisphere multiplcity bins
    #
    for jet_mult_key, mask in iter_hemi_filters(hemi_jet_ranges, all_hemis):

        # print("Processing jet mult bin:", jet_mult_key, "with", np.sum(mask), "hemispheres to replace.")

        subset_hemis = all_hemis[mask]

        #
        #  Prepare the hemisphere summary variable points for kd-tree query
        #
        subset_hemis_points = np.column_stack([ (subset_hemis[name] - hemi_stats[jet_mult_key][name]["mean"]) / hemi_stats[jet_mult_key][name]["RMS"] for name in hemi_summary_vars])

        #
        #  Get the nearest neighbor hemisphere from the kd-tree
        #
        match_dist, match_idx = hemi_kd_trees[jet_mult_key].query(subset_hemis_points, k=1)

        if np.sum(mask) < 1:
            continue


        #
        # Rotate Jets to match thrust axis
        #
        new_thrust = hemi_data[jet_mult_key]["thrust_phi"][match_idx]
        dphi = subset_hemis["thrust_phi"] - new_thrust

        # determine if we need to flip the hemispheres
        do_flip_hemi = (hemi_data[jet_mult_key]["hemisphereId"][match_idx]   * subset_hemis.hemisphereId) < 0
        dphi = ak.where(do_flip_hemi, dphi + np.pi, dphi)


        #
        #  Construct the new jets
        #
        new_Jets = ak.zip(
            {
                "pt":   ak.Array(hemi_data[jet_mult_key]["Jet_pt"]  [match_idx]),
                "eta":  ak.Array(hemi_data[jet_mult_key]["Jet_eta"] [match_idx]),
                "phi": (ak.Array(hemi_data[jet_mult_key]["Jet_phi"] [match_idx]) + dphi[:, None] + np.pi) % (2 * np.pi) - np.pi,
                "mass": ak.Array(hemi_data[jet_mult_key]["Jet_mass"][match_idx]),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        # fill other jet branches
        for var_name in jet_branches:
            var_key = var_name.replace("Jet_", "")
            if var_key in ["pt", "eta", "phi", "mass"]:
                continue
            new_Jets[var_key] = ak.Array(hemi_data[jet_mult_key][var_name][match_idx])

        # fill event data
        subset_hemis_new = ak.zip({"thrust_phi":       ak.Array(hemi_data[jet_mult_key]["thrust_phi"]     [match_idx]),
                                   "event":            ak.Array(hemi_data[jet_mult_key]["event"]          [match_idx]),
                                   "run":              ak.Array(hemi_data[jet_mult_key]["run"]            [match_idx]),
                                   "luminosityBlock" : ak.Array(hemi_data[jet_mult_key]["luminosityBlock"][match_idx]),
                                   "hemisphereId":     ak.Array(hemi_data[jet_mult_key]["hemisphereId"]   [match_idx]),
                                   "weight":           ak.Array(hemi_data[jet_mult_key]["weight"]         [match_idx]),
                                   "nSelJet":          subset_hemis["nSelJet"],
                                   "nTagJet":          subset_hemis["nTagJet"],
                                   "nJet" :            ak.num(new_Jets, axis=1),
                                   "Jet":              new_Jets,
                                   "match_dist":       ak.Array(match_dist),
                                   "local_idx":        subset_hemis["local_idx"],
                                   },
                                  depth_limit=1
                                  )
        #subset_hemis_new = compute_hemi_vars(all_hemis_new)

        mixed_hemis.append(subset_hemis_new)
        all_hemis_new = ak.concatenate(mixed_hemis, axis=0)
        sort_idx = ak.argsort(all_hemis_new.local_idx)
        all_hemis_new = all_hemis_new[sort_idx]

        #all_hemis = ak.where(mask, all_hemis_new, all_hemis)

    return all_hemis_new



def replace_hemis_load_kdTrees(*, all_hemis, hemi_stats, hemi_data, hemi_jet_ranges, hemi_summary_vars, jet_branches, use_boost_corrected_matching=False, event_branches=["event", "run", "luminosityBlock", "thrust_phi", "hemisphereId", "weight"]):

    mixed_hemis = []
    all_hemis["local_idx"] = ak.local_index(all_hemis, axis=0)

    #
    #  Loop on hemisphere multiplcity bins
    #
    for (jet_mult_key, mask_3b), (jet_mult_key_4b, mask_4b) in zip(iter_hemi_filters(hemi_jet_ranges, all_hemis),
                                                                   iter_hemi_filters(hemi_jet_ranges, hemi_data)):


        # print("Processing jet mult bin:", jet_mult_key, "with", np.sum(mask), "hemispheres to replace.")

        subset_hemis = all_hemis[mask_3b]

        #
        #  Prepare the hemisphere summary variable points for kd-tree query
        #
        subset_hemis_points = np.column_stack([ (subset_hemis[name] - hemi_stats[jet_mult_key][name]["mean"]) / hemi_stats[jet_mult_key][name]["RMS"] for name in hemi_summary_vars])

        # Check for NaN values
        has_nan = np.any(np.isnan(subset_hemis_points))
        # Check for inf values (positive or negative)
        has_inf = np.any(np.isinf(subset_hemis_points))
        # Check for both NaN and inf
        has_bad_values = np.any(~np.isfinite(subset_hemis_points))

        if has_bad_values:
            print(f"Warning: Found {np.sum(np.isnan(subset_hemis_points))} NaN and {np.sum(np.isinf(subset_hemis_points))} inf values in subset_hemis_points! filtering them out.\n")
            # bad_points = subset_hemis_points[np.any(np.isnan(subset_hemis_points), axis=1)]
            # print(f"bad points:\n{bad_points}\n")
            # print(f"Nan is  {np.sum(np.isnan(subset_hemis_points))} NaN and {np.sum(np.isinf(subset_hemis_points))} inf values in subset_hemis_points! filtering them out.\n")
            subset_hemis_points = np.nan_to_num(subset_hemis_points, nan=0.0) #subset_hemis_points[~np.any(np.isnan(subset_hemis_points), axis=1)]
            # print(f"Now has_bad_values = {np.any(~np.isfinite(subset_hemis_points))}\n")
        #
        #  Get the nearest neighbor hemisphere from the kd-tree
        #
        # Build list of variables to load - always include "pz" for boost correction even if not in matching vars
        load_vars = event_branches + hemi_summary_vars + jet_branches
        if use_boost_corrected_matching and "pz" not in hemi_summary_vars:
            load_vars = load_vars + ["pz"]

        hemi_lib_data = get_hemispheres_data(mask_4b, hemi_data, load_vars, hemi_stats=hemi_stats[jet_mult_key])
        hemi_lib_points = np.column_stack([ hemi_lib_data[name] for name in hemi_summary_vars])

        # Check for NaN values
        has_nan = np.any(np.isnan(hemi_lib_points))
        # Check for inf values (positive or negative)
        has_inf = np.any(np.isinf(hemi_lib_points))
        # Check for both NaN and inf
        has_bad_values = np.any(~np.isfinite(hemi_lib_points))

        if has_bad_values:
            print(f"Warning: Found {np.sum(np.isnan(hemi_lib_points))} NaN and {np.sum(np.isinf(hemi_lib_points))} inf values in hemi_lib_points! filtering them out.\n")
            hemi_lib_points = hemi_lib_points[~np.any(np.isnan(hemi_lib_points), axis=1)]


        kd_tree = cKDTree(hemi_lib_points)
        match_dist, match_idx = kd_tree.query(subset_hemis_points, k=1)

        if np.sum(mask_3b) < 1:
            continue

        #
        # Rotate Jets to match thrust axis
        #
        new_thrust = hemi_lib_data["thrust_phi"][match_idx]
        dphi = subset_hemis["thrust_phi"] - new_thrust

        # determine if we need to flip the hemispheres
        do_flip_hemi = (hemi_lib_data["hemisphereId"][match_idx]   * subset_hemis.hemisphereId) < 0
        dphi = ak.where(do_flip_hemi, dphi + np.pi, dphi)


        #
        #  Construct the new jets
        #
        new_Jets = ak.zip(
            {
                "pt":   ak.Array(hemi_lib_data["Jet_pt"]  [match_idx]),
                "eta":  ak.Array(hemi_lib_data["Jet_eta"] [match_idx]),
                "phi": (ak.Array(hemi_lib_data["Jet_phi"] [match_idx]) + dphi[:, None] + np.pi) % (2 * np.pi) - np.pi,
                "mass": ak.Array(hemi_lib_data["Jet_mass"][match_idx]),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        #
        # Apply boost correction if enabled
        #
        if use_boost_corrected_matching:
            # Get target hemisphere pz (from 3-tag event being processed)
            pz_target = subset_hemis["pz"]
            # print("pz_target ",pz_target[0:20].tolist(),"\n")

            # Get matched hemisphere pz (from library, unnormalized)
            # Note: hemi_lib_data["pz"] is normalized, need to get raw value from hemi_data
            pz_matched_normalized = hemi_lib_data["pz"][match_idx]
            pz_matched = pz_matched_normalized * hemi_stats[jet_mult_key]["pz"]["RMS"] + hemi_stats[jet_mult_key]["pz"]["mean"]

            # print("pz_matched_normalized ",pz_matched_normalized[0:20].tolist(),"\n")
            # print("pz_matched ",pz_matched[0:20].tolist(),"\n")

            # Compute matched hemisphere energy from summed jets
            matched_hemi_sum = new_Jets.sum(axis=1)
            E_matched = matched_hemi_sum.energy

            # Apply boost
            new_Jets, delta_rapidity = boost_jets_along_z(new_Jets, pz_target, pz_matched, E_matched)

            new_Jets_sumJet = new_Jets.sum(axis=1)
            # print("pz_new ",new_Jets_sumJet.pz[0:20].tolist(),"\n")
            pz_diff = new_Jets_sumJet.pz - pz_target
            # print("pz_diff ",pz_diff[0:20].tolist(),"\n")

        else:
            delta_rapidity = np.zeros(len(subset_hemis))


        # fill other jet branches
        for var_name in jet_branches:
            var_key = var_name.replace("Jet_", "")
            if var_key in ["pt", "eta", "phi", "mass"]:
                continue
            new_Jets[var_key] = ak.Array(hemi_lib_data[var_name][match_idx])


        # fill event data
        subset_hemis_new = ak.zip({"thrust_phi":       ak.Array(hemi_lib_data["thrust_phi"]     [match_idx]),
                                   "event":            ak.Array(hemi_lib_data["event"]          [match_idx]),
                                   "run":              ak.Array(hemi_lib_data["run"]            [match_idx]),
                                   "luminosityBlock" : ak.Array(hemi_lib_data["luminosityBlock"][match_idx]),
                                   "hemisphereId":     ak.Array(hemi_lib_data["hemisphereId"]   [match_idx]),
                                   "weight":           ak.Array(hemi_lib_data["weight"]         [match_idx]),
                                   "nSelJet":          subset_hemis["nSelJet"],
                                   "nTagJet":          subset_hemis["nTagJet"],
                                   "nJet" :            ak.num(new_Jets, axis=1),
                                   "Jet":              new_Jets,
                                   "match_dist":       ak.Array(match_dist),
                                   "delta_rapidity":   ak.Array(delta_rapidity),
                                   "local_idx":      subset_hemis["local_idx"],
                                   },
                                  depth_limit=1
                                  )
        #subset_hemis_new = compute_hemi_vars(all_hemis_new)

        mixed_hemis.append(subset_hemis_new)
        all_hemis_new = ak.concatenate(mixed_hemis, axis=0)
        sort_idx = ak.argsort(all_hemis_new.local_idx)
        all_hemis_new = all_hemis_new[sort_idx]

        #all_hemis = ak.where(mask, all_hemis_new, all_hemis)

    return all_hemis_new


def replace_hemis_topk_kdTrees(
    *,
    all_hemis,
    hemi_stats,
    hemi_data,
    hemi_jet_ranges,
    hemi_summary_vars,
    jet_branches,
    k_neighbors=10,
    default_rank=0,
    collision_mode="retry",
    use_boost_corrected_matching=False,
    event_branches=["event", "run", "luminosityBlock", "thrust_phi", "hemisphereId", "weight"],
):
    """Top-K kd-tree matcher with per-hemi rank selection.

    Parallel implementation to replace_hemis_load_kdTrees. Queries the top-K
    nearest neighbors per hemi, then chooses a rank per hemi based on
    `collision_mode`. `default_rank` is either an int (same rank for both
    pos and neg hemis) or a (rp, rn) pair (independent per side); the pair
    form expands the available "alternative datasets" from K to K^2 for
    multi-rank dataset expansion runs.

    - "ignore": every hemi uses its configured default rank.
    - "drop":   pos/neg pairs whose default-rank library hemis come from the
       same source event are flagged in kept_event_mask=False.
    - "retry":  for colliding pairs, search (rp, rn) in
       [rp_default..K-1] x [rn_default..K-1] for the smallest-distance
       non-colliding pair; if none, kept=False.

    Returns
    -------
    all_hemis_new : ak.Array
        Length 2*n_event record array, one row per input hemi (pos then neg
        block, mirroring legacy). Same fields as legacy + `match_rank`.
    kept_event_mask : np.ndarray of bool, shape (n_event,)
        False for events whose pos/neg collision could not be resolved within K.
        Caller is responsible for filtering selev / pos_hemi / etc.
    """
    if k_neighbors < 1:
        raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
    if isinstance(default_rank, (tuple, list)):
        if len(default_rank) != 2:
            raise ValueError(f"default_rank as a sequence must have length 2 (rp, rn); got {default_rank!r}")
        rp_default, rn_default = int(default_rank[0]), int(default_rank[1])
    else:
        rp_default = rn_default = int(default_rank)
    if not (0 <= rp_default < k_neighbors and 0 <= rn_default < k_neighbors):
        raise ValueError(f"default_rank ({default_rank}) entries must be in [0, k_neighbors={k_neighbors})")
    if collision_mode not in ("ignore", "drop", "retry"):
        raise ValueError(f"collision_mode must be one of ignore/drop/retry, got {collision_mode!r}")

    all_hemis = ak.with_field(all_hemis, ak.local_index(all_hemis, axis=0), "local_idx")

    K = k_neighbors
    N = len(all_hemis)
    if N % 2 != 0:
        raise ValueError(f"all_hemis length must be 2*n_event (got {N}); expecting concat of pos+neg.")
    n_event = N // 2

    # ─── Stage 1: per-bin top-K query (no jet fetch) ──────────────────────
    bin_records = []

    for (jet_mult_key, mask_3b), (jet_mult_key_4b, mask_4b) in zip(
            iter_hemi_filters(hemi_jet_ranges, all_hemis),
            iter_hemi_filters(hemi_jet_ranges, hemi_data)):

        if np.sum(mask_3b) < 1:
            continue

        subset_hemis = all_hemis[mask_3b]
        subset_hemis_points = np.column_stack([
            (subset_hemis[name] - hemi_stats[jet_mult_key][name]["mean"]) / hemi_stats[jet_mult_key][name]["RMS"]
            for name in hemi_summary_vars
        ])

        if np.any(~np.isfinite(subset_hemis_points)):
            print(f"Warning (topk): bad values in subset_hemis_points for {jet_mult_key}; replacing with 0.")
            subset_hemis_points = np.nan_to_num(subset_hemis_points, nan=0.0)

        # Stage-1 only needs summary fields; jet branches deferred to Stage 3.
        load_vars_summary = list(event_branches) + list(hemi_summary_vars)
        if use_boost_corrected_matching and "pz" not in hemi_summary_vars:
            load_vars_summary = load_vars_summary + ["pz"]

        hemi_lib_data_summary = get_hemispheres_data(
            mask_4b, hemi_data, load_vars_summary, hemi_stats=hemi_stats[jet_mult_key]
        )
        hemi_lib_points = np.column_stack([hemi_lib_data_summary[name] for name in hemi_summary_vars])

        if np.any(~np.isfinite(hemi_lib_points)):
            print(f"Warning (topk): bad values in hemi_lib_points for {jet_mult_key}; filtering rows.")
            hemi_lib_points = hemi_lib_points[~np.any(np.isnan(hemi_lib_points), axis=1)]

        kd_tree = cKDTree(hemi_lib_points)
        # cap K at library size; pad later so per-hemi shape stays (N, K).
        K_eff = min(K, len(hemi_lib_points))
        match_dist, match_idx = kd_tree.query(subset_hemis_points, k=K_eff)

        if K_eff == 1:
            match_dist = match_dist[:, None]
            match_idx = match_idx[:, None]
        if K_eff < K:
            pad_d = np.tile(match_dist[:, -1:], (1, K - K_eff))
            pad_i = np.tile(match_idx[:, -1:], (1, K - K_eff))
            match_dist = np.concatenate([match_dist, pad_d], axis=1)
            match_idx = np.concatenate([match_idx, pad_i], axis=1)

        bin_records.append({
            "jet_mult_key": jet_mult_key,
            "mask_3b": mask_3b,
            "mask_4b": mask_4b,
            "subset_hemis": subset_hemis,
            "match_idx": match_idx,
            "match_dist": match_dist,
            "lib_event": np.asarray(hemi_lib_data_summary["event"])[match_idx],
            "lib_run":   np.asarray(hemi_lib_data_summary["run"])[match_idx],
            "lib_lumi":  np.asarray(hemi_lib_data_summary["luminosityBlock"])[match_idx],
        })

    if not bin_records:
        raise RuntimeError("replace_hemis_topk_kdTrees: no jet_mult bins fired (empty all_hemis or hemi_jet_ranges miss).")

    # ─── Scatter per-bin top-K identities into per-hemi arrays ────────────
    lib_event_per_hemi  = np.zeros((N, K), dtype=np.int64)
    lib_run_per_hemi    = np.zeros((N, K), dtype=np.int64)
    lib_lumi_per_hemi   = np.zeros((N, K), dtype=np.int64)
    match_dist_per_hemi = np.full((N, K), np.inf, dtype=np.float64)

    for rec in bin_records:
        m = np.asarray(rec["mask_3b"])
        lib_event_per_hemi[m]  = rec["lib_event"]
        lib_run_per_hemi[m]    = rec["lib_run"]
        lib_lumi_per_hemi[m]   = rec["lib_lumi"]
        match_dist_per_hemi[m] = rec["match_dist"]

    # ─── Stage 2: pick rank per hemi ──────────────────────────────────────
    chosen_rank = np.empty(N, dtype=np.int32)
    chosen_rank[:n_event] = rp_default
    chosen_rank[n_event:] = rn_default
    kept_event_mask = np.ones(n_event, dtype=bool)

    if collision_mode != "ignore":
        pos_e = lib_event_per_hemi[:n_event]; neg_e = lib_event_per_hemi[n_event:]
        pos_r = lib_run_per_hemi[:n_event];   neg_r = lib_run_per_hemi[n_event:]
        pos_l = lib_lumi_per_hemi[:n_event];  neg_l = lib_lumi_per_hemi[n_event:]
        pos_d = match_dist_per_hemi[:n_event]; neg_d = match_dist_per_hemi[n_event:]

        collision = (
            (pos_e[:, rp_default] == neg_e[:, rn_default])
            & (pos_r[:, rp_default] == neg_r[:, rn_default])
            & (pos_l[:, rp_default] == neg_l[:, rn_default])
        )

        if collision_mode == "drop":
            kept_event_mask = ~collision
        else:  # retry
            coll_idx = np.where(collision)[0]
            for i in coll_idx:
                best = None  # (sum_dist, rp, rn)
                for rp in range(rp_default, K):
                    pe, pr, pl = pos_e[i, rp], pos_r[i, rp], pos_l[i, rp]
                    pd = pos_d[i, rp]
                    for rn in range(rn_default, K):
                        if pe == neg_e[i, rn] and pr == neg_r[i, rn] and pl == neg_l[i, rn]:
                            continue
                        d = pd + neg_d[i, rn]
                        if best is None or d < best[0]:
                            best = (d, rp, rn)
                if best is None:
                    kept_event_mask[i] = False
                else:
                    chosen_rank[i] = best[1]
                    chosen_rank[n_event + i] = best[2]

    # ─── Stage 3: build matched hemis at chosen rank ──────────────────────
    mixed_hemis = []

    for rec in bin_records:
        m = np.asarray(rec["mask_3b"])
        subset_hemis      = rec["subset_hemis"]
        chosen_rank_sub   = chosen_rank[m]
        match_idx_sub     = rec["match_idx"]
        match_dist_sub    = rec["match_dist"]

        match_idx_chosen  = np.take_along_axis(match_idx_sub,  chosen_rank_sub[:, None], axis=1).reshape(-1)
        match_dist_chosen = np.take_along_axis(match_dist_sub, chosen_rank_sub[:, None], axis=1).reshape(-1)

        # Now fetch full library payload (jets + summary) for this bin.
        load_vars_full = list(event_branches) + list(hemi_summary_vars) + list(jet_branches)
        if use_boost_corrected_matching and "pz" not in hemi_summary_vars:
            load_vars_full = load_vars_full + ["pz"]

        hemi_lib_data = get_hemispheres_data(
            rec["mask_4b"], hemi_data, load_vars_full, hemi_stats=hemi_stats[rec["jet_mult_key"]]
        )

        new_thrust = hemi_lib_data["thrust_phi"][match_idx_chosen]
        dphi = subset_hemis["thrust_phi"] - new_thrust
        do_flip_hemi = (hemi_lib_data["hemisphereId"][match_idx_chosen] * subset_hemis.hemisphereId) < 0
        dphi = ak.where(do_flip_hemi, dphi + np.pi, dphi)

        new_Jets = ak.zip(
            {
                "pt":   ak.Array(hemi_lib_data["Jet_pt"]  [match_idx_chosen]),
                "eta":  ak.Array(hemi_lib_data["Jet_eta"] [match_idx_chosen]),
                "phi": (ak.Array(hemi_lib_data["Jet_phi"] [match_idx_chosen]) + dphi[:, None] + np.pi) % (2 * np.pi) - np.pi,
                "mass": ak.Array(hemi_lib_data["Jet_mass"][match_idx_chosen]),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        if use_boost_corrected_matching:
            pz_target = subset_hemis["pz"]
            pz_matched_normalized = hemi_lib_data["pz"][match_idx_chosen]
            pz_matched = pz_matched_normalized * hemi_stats[rec["jet_mult_key"]]["pz"]["RMS"] + hemi_stats[rec["jet_mult_key"]]["pz"]["mean"]
            E_matched = new_Jets.sum(axis=1).energy
            new_Jets, delta_rapidity = boost_jets_along_z(new_Jets, pz_target, pz_matched, E_matched)
        else:
            delta_rapidity = np.zeros(len(subset_hemis))

        for var_name in jet_branches:
            var_key = var_name.replace("Jet_", "")
            if var_key in ["pt", "eta", "phi", "mass"]:
                continue
            new_Jets[var_key] = ak.Array(hemi_lib_data[var_name][match_idx_chosen])

        subset_hemis_new = ak.zip(
            {
                "thrust_phi":      ak.Array(hemi_lib_data["thrust_phi"]     [match_idx_chosen]),
                "event":           ak.Array(hemi_lib_data["event"]          [match_idx_chosen]),
                "run":             ak.Array(hemi_lib_data["run"]            [match_idx_chosen]),
                "luminosityBlock": ak.Array(hemi_lib_data["luminosityBlock"][match_idx_chosen]),
                "hemisphereId":    ak.Array(hemi_lib_data["hemisphereId"]   [match_idx_chosen]),
                "weight":          ak.Array(hemi_lib_data["weight"]         [match_idx_chosen]),
                "nSelJet":         subset_hemis["nSelJet"],
                "nTagJet":         subset_hemis["nTagJet"],
                "nJet":            ak.num(new_Jets, axis=1),
                "Jet":             new_Jets,
                "match_dist":      ak.Array(match_dist_chosen),
                "match_rank":      ak.Array(chosen_rank_sub.astype(np.uint8)),
                "delta_rapidity":  ak.Array(delta_rapidity),
                "local_idx":       subset_hemis["local_idx"],
            },
            depth_limit=1,
        )

        mixed_hemis.append(subset_hemis_new)

    all_hemis_new = ak.concatenate(mixed_hemis, axis=0)
    sort_idx = ak.argsort(all_hemis_new.local_idx)
    all_hemis_new = all_hemis_new[sort_idx]

    return all_hemis_new, kept_event_mask


def assign_mixed_subsamples(event, n_subsamples=16):

    # Get Random Numbers uniform 0 - 1 for each event
    rng       = Squares("subsample_mixed_data")
    counter   = event.event
    rand_vals = rng.uniform(counter, low=0, high=1.0)

    for mixed_sub_sample in range(n_subsamples):

        # Set upper and lower limits based on pseudoTagWeight
        upperLimit = ((mixed_sub_sample+1) * event.pseudoTagWeight)
        lowerLimit = ( mixed_sub_sample    * event.pseudoTagWeight);

        # Handle overflow cases
        #    when overflow occurs, pick one of the samples < 9
        overflow = upperLimit > 1.0
        overflow_sub_sample_index = (event.event + mixed_sub_sample) % 9
        upperLimit = ak.where( overflow, ((overflow_sub_sample_index + 1) * event.pseudoTagWeight), upperLimit )
        lowerLimit = ak.where( overflow, ((overflow_sub_sample_index    ) * event.pseudoTagWeight), lowerLimit )

        pass_subsample = ( (rand_vals > lowerLimit) & (rand_vals < upperLimit) )
        event[f"pass_mixedSubSample_v{mixed_sub_sample}"] = pass_subsample


def update_pseudoTagWeight_of_mixed_data(event, JCM):

    event["Jet_untagged_loose"] = event.Jet[event.Jet.selected & ~event.Jet.tagged_loose]
    # num_tagged_loose_plus_one = ak.sum(event.Jet.tagged_loose, axis=1) + 1

    fourTagFilter = event['fourTag']
    fourTagEvents = event[fourTagFilter]
    new_pseudoTagWeight = np.full(len(event), event.weight)
    new_nJet_pseudotagged = np.zeros(len(event), dtype=int)

    new_pseudoTagWeight[fourTagFilter], new_nJet_pseudotagged[fourTagFilter] = JCM( ak.num(fourTagEvents['Jet_untagged_loose'], axis=1) + 1, fourTagEvents.event)
    event["nJet_pseudotagged"] = new_nJet_pseudotagged
    event["pseudoTagWeight"] = new_pseudoTagWeight
