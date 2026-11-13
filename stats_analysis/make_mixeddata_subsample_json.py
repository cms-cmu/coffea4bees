#!/usr/bin/env python3
"""
Pseudo-data JSON for the bkg_syst F_4 mock analysis from ONE mixed subsample and ONE per-subsample
background model (config f4_observation: subsample).

  observation  `data` four-tag  <- mix_v<k> four-tag (Stage A_3 histograms; includes its ttbar pseudodata)
  background   every process of the F_1 per-subsample histogram <clf>_v<j>.<var> that the nominal JSON
               also has (same process, tag, region) <- model j: the background model fitted to subsample
               j (FvT method: 3b data x JCM_j x FvT_j and TTbar4b_from_d3; MvD method: the mixed_all
               sample x JCM_j x MvD_j and TTbar4b_from_MvD). Four-tag `data` is never taken from F_1.
  signal       unchanged (the nominal JSON's)

With j = k the background is the model fitted to the observed subsample itself; j != k uses an
independently fitted model. make_mixeddata_ave_json.py (f4_observation: average) is the other mode:
the 16-subsample average against the NOMINAL background, which is fitted to real four-tag data.
"""
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import json
import argparse
from coffea.util import load
from src.tools.convert_hist_to_json import extract_hist_data

STANDARD_AXES = {'process', 'year', 'tag', 'region', 'variation'}


def region_dicts(h, process, year, tag):
    """{region+suffix: json hist} for one process/year/tag of a coffea hist (as make_mixeddata_ave_json)."""
    axis_indices = {ax.name: idx for idx, ax in enumerate(h.axes)}
    custom_axes = [ax.name for ax in h.axes[:-1] if ax.name not in STANDARD_AXES]
    var_axis = h.axes[-1]
    edges, centers = var_axis.edges.tolist(), var_axis.centers.tolist()
    out = {}
    for region in (list(h.axes['region']) if 'region' in axis_indices else ['SR', 'SB']):
        sel = {'process': process, 'year': year, 'tag': tag, 'region': region}
        for suffix, d in extract_hist_data(h, sel, custom_axes, axis_indices, edges, centers).items():
            out[f"{region}{suffix}"] = d
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--nominal_json", required=True, help="nominal JSON (F_3's, from inputs.nominal_hists)")
    p.add_argument("--bkg_coffea", required=True, help="F_1 histAll_mixeddata_bkgs.coffea (per-subsample models)")
    p.add_argument("--mixed_coffea", required=True, help="A_3 histAll_<channel>_mixeddata_v<k>.coffea")
    p.add_argument("--subsample", type=int, required=True, help="k: the observed subsample")
    p.add_argument("--model_subsample", type=int, default=None, help="j: the background model (default: k)")
    p.add_argument("--variables", nargs="+", required=True, help="fit variables, e.g. SvB_MA.ps_hh_fine")
    p.add_argument("--years", nargs="+", required=True)
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()
    k = args.subsample
    j = k if args.model_subsample is None else args.model_subsample

    with open(args.nominal_json) as f:
        out = json.load(f)
    bkg = load(args.bkg_coffea)["hists"]
    mix = load(args.mixed_coffea)["hists"]

    for var in args.variables:
        if var not in out:
            print(f"Warning: {var} not in {args.nominal_json}, skipping")
            continue
        clf, stem = var.split(".", 1)
        h_model = bkg.get(f"{clf}_v{j}.{stem}")
        if h_model is None:
            raise KeyError(f"{clf}_v{j}.{stem} not in {args.bkg_coffea}: no per-subsample model {j}")
        h_mix = mix.get(var)
        if h_mix is None:
            raise KeyError(f"{var} not in {args.mixed_coffea}")
        mix_proc = f"mix_v{k}"
        if mix_proc not in h_mix.axes['process']:
            raise KeyError(f"process {mix_proc} not in {args.mixed_coffea} ({list(h_mix.axes['process'])})")

        # background: like-for-like replacement of every process/tag the nominal JSON and model j share
        replaced = []
        for proc in h_model.axes['process']:
            if proc not in out[var]:
                continue
            for yr in args.years:
                if yr not in out[var][proc] or yr not in h_model.axes['year']:
                    continue
                for tag in list(out[var][proc][yr]):
                    if (proc == 'data' and tag == 'fourTag') or tag not in h_model.axes['tag']:
                        continue
                    for region, d in region_dicts(h_model, proc, yr, tag).items():
                        if region in out[var][proc][yr][tag]:
                            out[var][proc][yr][tag][region] = d
                    replaced.append(f"{proc}/{tag}/{yr}")
        if not replaced:
            raise RuntimeError(f"{var}: no process of model {j} matches the nominal JSON ({list(h_model.axes['process'])})")
        print(f"{var}: background from model v{j}: {sorted(set(replaced))}")

        # observation: mix_v<k> four-tag as `data` (also kept as process mix_v<k>)
        out[var].setdefault(mix_proc, {})
        for yr in args.years:
            if yr not in h_mix.axes['year']:
                continue
            obs = region_dicts(h_mix, mix_proc, yr, 'fourTag')
            out[var][mix_proc][yr] = {'fourTag': obs}
            if 'data' in out[var] and yr in out[var]['data'] and 'fourTag' in out[var]['data'][yr]:
                for region, d in obs.items():
                    if region in out[var]['data'][yr]['fourTag']:
                        out[var]['data'][yr]['fourTag'][region] = d
        print(f"{var}: observation = {mix_proc} four-tag")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
