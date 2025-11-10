import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector as v

ak.behavior.update(v.behavior)

import numba as nb
import numpy as np

@nb.njit(cache=True)
def _thrust_event_numba(px_i, py_i, n_steps=720):
    thetas = np.linspace(0.0, 2.0*np.pi, n_steps)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    pTi = np.hypot(px_i, py_i)
    #denom = pTi.sum()
    #if denom == 0.0:
    #    return np.nan, 0.0, 0.0, 0.0, 0.0

    proj = np.abs(px_i[:, None]*cos_t[None,:] + py_i[:, None]*sin_t[None,:])
    sums = proj.sum(axis=0)
    best_idx = np.argmax(sums)
    best_theta = thetas[best_idx]
    best_sum = sums[best_idx]

    nx, ny = np.cos(best_theta), np.sin(best_theta)
    mx, my = -ny, nx
    #T = best_sum/denom
    #T_minor = np.sum(np.abs(-px_i*ny + py_i*nx))/denom
    return best_theta, nx, ny, mx, my



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
    thetas = np.linspace(0.0, 2.0 * np.pi, num=n_steps, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # loop on events
    results = []
    for px_i, py_i in zip(px, py):
        px_np, py_np = np.asarray(px_i), np.asarray(py_i)
        if len(px_np) == 0:
            results.append(dict(theta=np.nan,
                                axis=dict(nx=np.nan, ny=np.nan),
                                minor=dict(nx=np.nan, ny=np.nan)))
            continue
        th, nx, ny, mx, my = _thrust_event_numba(px_np, py_np, n_steps)
        results.append(dict(theta=th,
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
    thetas = np.linspace(0.0, 2.0 * np.pi, num=n_steps, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # loop on events
    results = []
    for px_i, py_i in zip(px, py):
        # handle empty events
        if len(px_i) == 0:
            results.append(
                dict(T=np.nan, T_minor=np.nan, theta=np.nan,
                     axis=dict(nx=np.nan, ny=np.nan),
                     minor=dict(nx=np.nan, ny=np.nan))
            )
            continue

        pTi = np.hypot(px_i, py_i)

        #denom = np.sum(pTi)
        #if denom == 0:
        #    results.append(
        #        dict(T=np.nan, T_minor=np.nan, theta=np.nan,
        #             axis=dict(nx=np.nan, ny=np.nan),
        #             minor=dict(nx=np.nan, ny=np.nan))
        #    )
        #    continue

        # --- coarse scan ---
        proj = np.abs(px_i[:, None] * cos_t[None, :] + py_i[:, None] * sin_t[None, :])
        sums = np.sum(proj, axis=0)
        best_idx = int(np.argmax(sums))
        best_theta = thetas[best_idx]
        best_sum = sums[best_idx]

        # --- optional refinement ---
        if refine_rounds > 0:
            window = 2.0 * np.pi / n_steps
            for _ in range(refine_rounds):
                half_w = 0.6 * window
                local_steps = max(24, refine_factor * 12)
                loc_thetas = np.linspace(best_theta - half_w,
                                         best_theta + half_w,
                                         num=local_steps, endpoint=True)
                px_i = np.asarray(px_i)
                py_i = np.asarray(py_i)
                projL = np.abs(px_i[:, None] * np.cos(loc_thetas)[None, :]
                               + py_i[:, None] * np.sin(loc_thetas)[None, :]).sum(axis=0)
                best_theta = loc_thetas[int(np.argmax(projL))]
                best_sum = np.max(projL)
                window *= 0.35
            best_theta = float(np.mod(best_theta, 2.0 * np.pi))

        # Axes
        nx, ny = np.cos(best_theta), np.sin(best_theta)
        mx, my = -ny, nx

        #T = best_sum / denom
        #T_minor = np.sum(np.abs(-px_i * ny + py_i * nx)) / denom

        results.append(
            #dict(T=T, T_minor=T_minor, theta=best_theta,
            #     axis=dict(nx=nx, ny=ny), minor=dict(nx=mx, ny=my))
            dict(theta=best_theta,
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
