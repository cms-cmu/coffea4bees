#!/usr/bin/env python3
"""
Convert Phase E.5 processor .coffea files directly into a consolidated ROOT file
for the two-stage closure test (runTwoStageClosure.py), eliminating intermediate JSON files.
"""

import os
import sys
import argparse
import logging
import numpy as np
import uproot
import hist
from coffea.util import load


def slice_1d(h, process, year, tag, region):
    """Extract a 1D slice from a multi-dimensional coffea hist for given process, year, tag, region."""
    if process not in h.axes['process']:
        return None
    if year not in h.axes['year']:
        return None
    if tag not in h.axes['tag']:
        return None
    if region not in h.axes['region']:
        return None

    sel = {
        'process': process,
        'year': year,
        'tag': tag,
        'region': region,
    }
    for ax in h.axes:
        if ax.name.startswith(('pass', 'fail')) and ax.name not in sel:
            sel[ax.name] = sum
    try:
        h_slice = h[sel]
        # If any extra axes remain besides the variable axis, project to the 1D axis
        if len(h_slice.axes) > 1:
            var_axes = [ax.name for ax in h_slice.axes if ax.name not in ['process', 'year', 'tag', 'region']]
            if var_axes:
                h_slice = h_slice.project(var_axes[-1])
        return h_slice
    except Exception as e:
        logging.debug(f"Could not slice {sel}: {e}")
        return None


def to_uproot_hist(h_1d, scale=1.0):
    """Convert a 1D hist.Hist to a Weight hist suitable for uproot writing, applying optional scaling."""
    edges = np.array(h_1d.axes[0].edges, dtype=np.float64)
    values = np.array(h_1d.values(), dtype=np.float64) * scale
    variances = np.array(h_1d.variances(), dtype=np.float64) * (scale ** 2) if h_1d.variances() is not None else values * scale
    
    # Check if variable or regular binning
    widths = np.diff(edges)
    if len(edges) > 1 and not np.allclose(widths, widths[0]):
        out_h = hist.Hist.new.Var(edges, name="h").Weight()
    else:
        out_h = hist.Hist.new.Reg(len(edges) - 1, edges[0], edges[-1], name="h").Weight()
    
    out_h.view().value = values
    out_h.view().variance = variances
    return out_h


def main():
    parser = argparse.ArgumentParser(description="Directly convert Phase E subsample coffea files to ROOT for closure.")
    parser.add_argument("--closure_dir", default="output/ttHbb_mixeddata_stitched_closure/",
                        help="Base output directory containing closure_v{v} subdirectories.")
    parser.add_argument("--mix_name", default="3bDvTMix4bDvT",
                        help="Mixing name tag (e.g. 3bDvTMix4bDvT).")
    parser.add_argument("--n_subsamples", "--nMixes", type=int, default=15,
                        help="Number of mixed subsamples (default: 15).")
    parser.add_argument("--subsample_indices", nargs="+", type=int, default=None,
                        help="Explicit list of subsample indices to process (e.g. 1 2 3 ... 14). If omitted, uses range(n_subsamples).")
    parser.add_argument("--scale_mixed", type=float, default=1.0,
                        help="Scale factor for mixed data.")
    parser.add_argument("--auto_scale_mixed", action="store_true", default=False,
                        help="Auto-calculate scale_mixed to match total background prediction in SR.")
    parser.add_argument("--pure_qcd", action="store_true", default=False,
                        help="Pure QCD mode with zero ttbar.")
    parser.add_argument("--hist_keys", nargs="+",
                        default=['ps', 'ps_ttHbb', 'ps_ttHbb_fine', 'ps_zz', 'ps_zh', 'ps_hh'],
                        help="Histogram variable stems to convert.")
    parser.add_argument("--nominal_coffea", nargs="*", default=[],
                        help="Optional nominal coffea files for signal (ttHbb) and baseline background.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to output ROOT file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="[%(levelname)s] %(message)s")

    root_dict = {}

    eff_scale_mixed = args.scale_mixed
    logging.info(f"Using scale_mixed = {eff_scale_mixed}")

    closure_dir = args.closure_dir.rstrip("/")

    if args.subsample_indices is not None and len(args.subsample_indices) > 0:
        subsample_list = args.subsample_indices
    else:
        subsample_list = list(range(args.n_subsamples))

    logging.info(f"Subsamples to process ({len(subsample_list)} total): {subsample_list}")

    # Loop over subsamples and map each to a sequential closure slot m in range(len(subsamples))
    for m, v in enumerate(subsample_list):
        data_file = f"{closure_dir}/closure_v{v}/histAll_data_v{v}.coffea"
        mix_file = f"{closure_dir}/closure_v{v}/histAll_mixeddata_v{v}.coffea"

        logging.info(f"[Slot {m} <- closure_v{v}] Processing disk subsample {v} into closure model {m}...")

        # 1. Ingest Data 3b (JCM * FvT_v) and TTbar
        if os.path.exists(data_file):
            data_data = load(data_file)
            hists_dict = data_data.get("hists", {})

            for h_key, h in hists_dict.items():
                stem = h_key.replace(".", "_")
                # Only keep relevant histogram keys
                if not any(stem.endswith(k) or f"_{k}_" in stem for k in args.hist_keys):
                    continue

                for yr in h.axes['year']:
                    # Extract 3-tag Data (Multijet background model)
                    h_data_3b = slice_1d(h, 'data', yr, 'threeTag', 'SR')
                    if h_data_3b is not None:
                        u_hist = to_uproot_hist(h_data_3b)
                        # Plain nominal key: e.g. SvB_MA_ps_ttHbb_data_UL18_threeTag_SR
                        root_dict[f"{stem}_data_{yr}_threeTag_SR"] = u_hist

                        # FvT subsample key expected by runTwoStageClosure.py:
                        # SvB_MA_FvT_3bDvTMix4bDvT_v{m}_newSBDef_ps_ttHbb_data_UL18_threeTag_SR
                        idx = m
                        fvt_stem = stem
                        if "SvB_MA_ps" in stem:
                            fvt_stem = stem.replace("SvB_MA_ps", f"SvB_MA_FvT_{args.mix_name}_v{idx}_newSBDef_ps")
                        elif "SvB_ps" in stem:
                            fvt_stem = stem.replace("SvB_ps", f"SvB_FvT_{args.mix_name}_v{idx}_newSBDef_ps")
                        else:
                            fvt_stem = f"{stem}_FvT_{args.mix_name}_v{idx}_newSBDef"
                        root_dict[f"{fvt_stem}_data_{yr}_threeTag_SR"] = u_hist
                        root_dict[f"{fvt_stem}_data_3b_for_mixed_{yr}_threeTag_SR"] = u_hist

                    # Extract 3-tag TTbar
                    if not args.pure_qcd and 'TTbar4b_from_d3' in h.axes['process']:
                        h_ttbar_3b = slice_1d(h, 'TTbar4b_from_d3', yr, 'threeTag', 'SR')
                        if h_ttbar_3b is not None:
                            u_ttbar_hist = to_uproot_hist(h_ttbar_3b)
                            root_dict[f"{stem}_TTbar4b_from_d3_{yr}_threeTag_SR"] = u_ttbar_hist

                            idx = m
                            fvt_tt_stem = stem
                            if "SvB_MA_ps" in stem:
                                fvt_tt_stem = stem.replace("SvB_MA_ps", f"SvB_MA_FvT_{args.mix_name}_v{idx}_newSBDef_ps")
                            elif "SvB_ps" in stem:
                                fvt_tt_stem = stem.replace("SvB_ps", f"SvB_FvT_{args.mix_name}_v{idx}_newSBDef_ps")
                            else:
                                fvt_tt_stem = f"{stem}_FvT_{args.mix_name}_v{idx}_newSBDef"
                            root_dict[f"{fvt_tt_stem}_TTbar4b_from_d3_{yr}_threeTag_SR"] = u_ttbar_hist
                            root_dict[f"{stem}_TTbar4b_from_d3_v{idx}_{yr}_threeTag_SR"] = u_ttbar_hist
        else:
            logging.warning(f"Data file not found: {data_file}")

        # 2. Ingest Mixed Data 4b (Observation)
        if os.path.exists(mix_file):
            mix_data = load(mix_file)
            hists_dict = mix_data.get("hists", {})

            for h_key, h in hists_dict.items():
                stem = h_key.replace(".", "_")
                if not any(stem.endswith(k) or f"_{k}_" in stem for k in args.hist_keys):
                    continue

                mix_proc = f"mix_v{v}"
                if mix_proc not in h.axes['process']:
                    # Fallback if process axis has a generic name
                    for p in h.axes['process']:
                        if 'mix' in p:
                            mix_proc = p
                            break

                for yr in h.axes['year']:
                    h_mix_4b = slice_1d(h, mix_proc, yr, 'fourTag', 'SR')
                    if h_mix_4b is not None:
                        u_hist = to_uproot_hist(h_mix_4b, scale=eff_scale_mixed)
                        idx = m
                        root_dict[f"{stem}_mix_v{idx}_{yr}_fourTag_SR"] = u_hist
                        root_dict[f"{stem}_{args.mix_name}_v{idx}_{yr}_fourTag_SR"] = u_hist
        else:
            logging.warning(f"Mixed data file not found: {mix_file}")

    # 3. Ingest Nominal Signal & Datasets (if provided)
    for nom_f in args.nominal_coffea:
        if os.path.exists(nom_f):
            logging.info(f"Loading nominal file {nom_f}...")
            nom_data = load(nom_f)
            hists_dict = nom_data.get("hists", {})
            for h_key, h in hists_dict.items():
                stem = h_key.replace(".", "_")
                if not any(stem.endswith(k) or f"_{k}_" in stem for k in args.hist_keys):
                    continue
                for proc in ['ttHbb', 'GluGluToHHTo4B_cHHH1', 'ZH4b', 'ZZ4b']:
                    if proc in h.axes['process']:
                        for yr in h.axes['year']:
                            h_sig = slice_1d(h, proc, yr, 'fourTag', 'SR')
                            if h_sig is not None:
                                root_dict[f"{stem}_{proc}_{yr}_fourTag_SR"] = to_uproot_hist(h_sig)

    logging.info(f"Total histograms to write to ROOT: {len(root_dict)}")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with uproot.recreate(args.output) as f_out:
        for k, v in root_dict.items():
            f_out[k] = v

    logging.info(f"Successfully saved {args.output}")


if __name__ == "__main__":
    main()
