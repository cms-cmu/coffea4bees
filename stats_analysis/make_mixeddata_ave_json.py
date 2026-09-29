#!/usr/bin/env python3
"""
Average 4-tag histograms across all regions from 15 mixed data subsamples and inject into nominal JSON
to create pseudo-data observation for unblinded Combine testing.
"""
import os
import sys
import copy
import json
import argparse
import numpy as np
from coffea.util import load


def extract_1d_dict(h, process, year, tag, region):
    sel = {'process': process, 'year': year, 'tag': tag, 'region': region}
    for ax in h.axes:
        if ax.name.startswith(('pass', 'fail')) and ax.name not in sel:
            sel[ax.name] = sum
    h_1d = h[sel]
    if len(h_1d.axes) > 1:
        coord_axes = [ax.name for ax in h_1d.axes if ax.name not in ['process', 'year', 'tag', 'region']]
        h_1d = h_1d.project(coord_axes[-1])

    coord_axis = h_1d.axes[-1]
    edges = np.array(coord_axis.edges, dtype=float).tolist()
    centers = np.array(coord_axis.centers, dtype=float).tolist()
    vals = np.array(h_1d.values(flow=True), dtype=float)
    vars_ = h_1d.variances(flow=True)
    if vars_ is None:
        vars_ = vals.copy()
    else:
        vars_ = np.array(vars_, dtype=float)

    return {
        'edges': edges,
        'centers': centers,
        'values': vals[1:-1],
        'variances': vars_[1:-1],
        'underflow_value': float(vals[0]),
        'underflow_variance': float(vars_[0]),
        'overflow_value': float(vals[-1]),
        'overflow_variance': float(vars_[-1])
    }


def main():
    parser = argparse.ArgumentParser(description="Create average mixed data JSON for unblinded Combine tests.")
    parser.add_argument("-i", "--nominal_json", default="output/ttHbb_stitched/histAll_ttHbb_stitched.json",
                        help="Path to baseline nominal json.")
    parser.add_argument("-c", "--closure_dir", default="output/ttHbb_mixeddata_stitched_closure",
                        help="Directory containing closure_v{v}/histAll_mixeddata_v{v}.coffea")
    parser.add_argument("-o", "--output", default="output/ttHbb_mixeddata_stitched_closure/histAll_ttHbb_mixeddata_ave.json",
                        help="Output JSON path.")
    parser.add_argument("--subsamples", type=int, nargs="+", default=list(range(15)),
                        help="Subsample indices to average.")
    parser.add_argument("--variables", nargs="+", default=["SvB_MA.ps_ttHbb", "SvB_MA.ps_ttHbb_gt6"],
                        help="Variables to average.")
    parser.add_argument("--years", nargs="+", default=["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"],
                        help="Years to process.")
    args = parser.parse_args()

    print(f"Loading baseline JSON: {args.nominal_json}")
    with open(args.nominal_json, "r") as f:
        out_json = json.load(f)

    print(f"Averaging {len(args.subsamples)} subsamples: {args.subsamples}")
    for var in args.variables:
        if var not in out_json:
            print(f"Warning: {var} not found in {args.nominal_json}, skipping.")
            continue

        for yr in args.years:
            if "data" not in out_json[var] or yr not in out_json[var]["data"] or "fourTag" not in out_json[var]["data"][yr]:
                continue
            regions = list(out_json[var]["data"][yr]["fourTag"].keys())

            for region in regions:
                val_list = []
                var_list = []
                ref_dict = None

                for v in args.subsamples:
                    fpath = os.path.join(args.closure_dir, f"closure_v{v}", f"histAll_mixeddata_v{v}.coffea")
                    if not os.path.exists(fpath):
                        raise FileNotFoundError(f"Missing mixeddata coffea file: {fpath}")
                    h_data = load(fpath)["hists"][var]
                    mix_proc = f"mix_v{v}"
                    d_1d = extract_1d_dict(h_data, mix_proc, yr, "fourTag", region)
                    val_list.append(d_1d["values"])
                    var_list.append(d_1d["variances"])
                    if ref_dict is None:
                        ref_dict = d_1d

                N = len(args.subsamples)
                ave_values = np.mean(val_list, axis=0)
                ave_variances = np.sum(var_list, axis=0) / (N ** 2)

                ave_json_hist = {
                    "edges": ref_dict["edges"],
                    "centers": ref_dict["centers"],
                    "values": ave_values.tolist(),
                    "variances": ave_variances.tolist(),
                    "underflow_value": 0.0,
                    "underflow_variance": 0.0,
                    "overflow_value": 0.0,
                    "overflow_variance": 0.0
                }

                # 1. Replace 'data' in fourTag region with the average mixed data
                out_json[var]["data"][yr]["fourTag"][region] = ave_json_hist

                # 2. Also register 'mix_ave' as a distinct process
                if "mix_ave" not in out_json[var]:
                    out_json[var]["mix_ave"] = {}
                if yr not in out_json[var]["mix_ave"]:
                    out_json[var]["mix_ave"][yr] = {"fourTag": {}}
                out_json[var]["mix_ave"][yr]["fourTag"][region] = ave_json_hist

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Writing output to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(out_json, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
