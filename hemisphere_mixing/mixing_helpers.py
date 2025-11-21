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
    if not (hasattr(p4, "px") and hasattr(p4, "py")):
        raise ValueError("Input must have Momentum4D behavior (with .px/.py).")

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
    if not (hasattr(p4, "px") and hasattr(p4, "py")):
        raise ValueError("Input must have Momentum4D behavior (with .px/.py).")

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


def split_events_into_hemispheres(event):

    #
    #  Get Thrust axis
    #
    thrust = transverse_thrust_awkward_fast(event.Jet, n_steps=720, refine_rounds=2)

    #
    #  For outputs
    #
    jet_posHemi, jet_negHemi   = split_hemispheres(event.Jet, thrust)


    #
    #  For mutltiplicity counting
    #
    tagJet_posHemi, tagJet_negHemi = split_hemispheres(event.tagJet, thrust)
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



def read_hemi_files(hemifiles, tree_name="Events", branch_list=None):

    hemi_vars = { var_name: [] for var_name in branch_list }

    if isinstance(hemifiles, str):
        for batch in uproot.iterate(
                f"{hemifiles}:{tree_name}",
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

    elif isinstance(hemifiles, list):
        print("Reading hemisphere files:", hemifiles)
        file_spec = {f: tree_name for f in hemifiles}

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



def get_grouped_hemispheres_data(hemi_ranges, hemi_data, hemi_vars, summary_vars=None):
    grouped_hemi_data = {}

    for jet_mult_key, mask in iter_hemi_filters(hemi_ranges, hemi_data):

        grouped_hemi_data[jet_mult_key] = {}

        for var_name in hemi_vars:

            if summary_vars and var_name in summary_vars[jet_mult_key]:
                this_var = summary_vars[jet_mult_key][var_name]
                grouped_hemi_data[jet_mult_key][var_name] = (hemi_data[var_name][mask] - this_var["mean"]) / this_var["RMS"]
            else:
                grouped_hemi_data[jet_mult_key][var_name] = hemi_data[var_name][mask]

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


def build_hemi_kdtrees(hemi_metadata_yaml, hemifiles, hemi_summary_vars, jet_branches):

    # Read in hemisphere library metadata
    with open(hemi_metadata_yaml, 'r') as f:
        hemi_stats_raw = yaml.safe_load(f)

    hemi_stats = convert_yaml_dict(hemi_stats_raw["hemi_summary_vars"])
    jet_ranges = convert_yaml_dict(hemi_stats_raw["jet_mult_ranges"])

    #
    #  Read in Hemisphere library data
    #
    event_branches = ["event", "run", "luminosityBlock", "thrust_phi", "hemisphereId", "weight"]
    branch_list = ["nJet", "nSelJet", "nTagJet"] + event_branches + hemi_summary_vars + jet_branches

    hemi_data = read_hemi_files(hemifiles, branch_list=branch_list)
    #print("hemifiles is", hemifiles)
    #hemi_data = TreeReader(branch_filter=set(branch_list).intersection).concat(
    #    *(Chunk(f) for f in [hemifiles]), library="np"
    #)
    #breakpoint()

    #
    #  Group hemisphere data by jet multiplicity bins
    #
    grouped_hemi_data = get_grouped_hemispheres_data(jet_ranges, hemi_data, hemi_vars= event_branches + hemi_summary_vars + jet_branches, summary_vars=hemi_stats)

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
