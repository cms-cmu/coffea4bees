import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector as v
import uproot

ak.behavior.update(v.behavior)

import numba as nb
import numpy as np

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


def read_hemi_files(hemifiles, tree_name="Events", branch_list=None):

    hemi_vars = { var_name: [] for var_name in branch_list }

    for batch in uproot.iterate(
            f"{hemifiles}:{tree_name}",  #"coffea4bees/hemisphere_mixing/tests/*.root:Events",
            branch_list,
            step_size=200_000,  # entries per chunk
            library="np",
    ):

        #nTag2Sel2Jet2 = ( (batch["nTagJet"] == 2) & (batch["nSelJet"] == 2) & (batch["nJet"] == 2) )

        if hemi_vars[branch_list[0]] is None:
            for var_name in branch_list:
                hemi_vars[var_name] = batch[var_name]
        else:
            for var_name in branch_list:
                hemi_vars[var_name] = np.concatenate( (hemi_vars[var_name], batch[var_name]) )

    return hemi_vars


def get_filter(data, key, val, low_edge=False, high_edge=False):
    this_filter = (data[key] == val)

    if low_edge:
        this_filter |= (data[key] < val)
    elif high_edge:
        this_filter |= (data[key] > val)

    return this_filter


def get_grouped_hemispheres_data(hemi_ranges, hemi_data, hemi_vars, summary_vars=None):
    grouped_hemi_data = {}

    # Outer loop: tag multiplicity bins
    tag_keys = list(hemi_ranges.keys())
    for itag, tag in enumerate(tag_keys):

        grouped_hemi_data[tag] = {}

        # --- tag filter ----------------------------------------------------------
        tag_filter = get_filter(hemi_data, "nTagJet", tag, low_edge=(itag==0), high_edge=(itag==len(tag_keys)-1))

        # skip empty sub-ranges
        if not hemi_ranges[tag]:
            print(f"ERROR: no sel jets for tag = {tag}")
            continue

        # -------------------------------------------------------------------------
        # Middle loop: selected-jet multiplicity bins
        sel_keys = list(hemi_ranges[tag].keys())
        for isel, sel in enumerate(sel_keys):

            grouped_hemi_data[tag][sel] = {}

            # --- sel filter ------------------------------------------------------
            sel_filter = get_filter(hemi_data, "nSelJet", sel, low_edge=(isel==0), high_edge=(isel==len(sel_keys)-1))

            # ---------------------------------------------------------------------
            # Inner loop: total-jet multiplicity bins
            jet_bins = hemi_ranges[tag][sel]
            if not jet_bins:
                # special case: no jet bins defined
                grouped_hemi_data[tag][sel][-1] = {}
                for var_name in hemi_vars:
                    if summary_vars:
                        this_var = summary_vars[(tag,sel,-1)][var_name]
                        grouped_hemi_data[tag][sel][-1][var_name] = (hemi_data[var_name][tag_filter & sel_filter] - this_var["mean"]) / this_var["RMS"]
                    else:
                        grouped_hemi_data[tag][sel][-1][var_name] = hemi_data[var_name][tag_filter & sel_filter]

                continue

            for ijet, jet in enumerate(jet_bins):

                jet_filter = get_filter(hemi_data, "nJet", jet, low_edge=(ijet==0), high_edge=(ijet==len(jet_bins)-1))

                # --- final selection ---------------------------------------------
                mask = tag_filter & sel_filter & jet_filter
                grouped_hemi_data[tag][sel][jet] = {}
                for var_name in hemi_vars:
                    if summary_vars:
                        this_var = summary_vars[(tag,sel,jet)][var_name]
                        grouped_hemi_data[tag][sel][jet][var_name] = (hemi_data[var_name][tag_filter & sel_filter & jet_filter] - this_var["mean"]) / this_var["RMS"]
                    else:
                        grouped_hemi_data[tag][sel][jet][var_name] = hemi_data[var_name][tag_filter & sel_filter & jet_filter]


    return grouped_hemi_data
