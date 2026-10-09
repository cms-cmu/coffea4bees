#!/usr/bin/env python3
"""
Average 4-tag histograms across all regions from 15 mixed data subsamples and inject into nominal JSON
to create pseudo-data observation for unblinded Combine testing.
"""
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import copy
import json
import argparse
import numpy as np
from collections import defaultdict
from coffea.util import load
from src.tools.convert_hist_to_json import extract_hist_data


def main():
    parser = argparse.ArgumentParser(description="Create average mixed data JSON for unblinded Combine tests.")
    parser.add_argument("-i", "--nominal_json", default="output/ttHbb_stitched/histAll_ttHbb_stitched.json",
                        help="Path to baseline nominal json.")
    parser.add_argument("-c", "--closure_dir", default="output/ttHbb_mixeddata_stitched_closure",
                        help="Directory containing the subsample coffea files (see --file_template)")
    parser.add_argument("--file_template", default="closure_v{v}/histAll_mixeddata_v{v}.coffea",
                        help="Subsample coffea file under --closure_dir; {v} is the subsample index.")
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

        print(f"Processing variable: {var}")
        accumulated = defaultdict(lambda: defaultdict(list))
        ref_dicts = {}

        for v in args.subsamples:
            fpath = os.path.join(args.closure_dir, args.file_template.format(v=v))
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing mixeddata coffea file: {fpath}")

            cdata = load(fpath)
            h = cdata["hists"][var]
            axis_indices = {ax.name: idx for idx, ax in enumerate(h.axes)}
            standard_axes = {'process', 'year', 'tag', 'region', 'variation'}
            custom_axes = [ax.name for ax in h.axes[:-1] if ax.name not in standard_axes]
            var_axis = h.axes[-1]
            edges = var_axis.edges.tolist()
            centers = var_axis.centers.tolist()
            base_regions = list(h.axes['region']) if 'region' in axis_indices else ['SR', 'SB']

            for yr in args.years:
                for iregion in base_regions:
                    sel = {'process': f'mix_v{v}', 'year': yr, 'tag': 'fourTag', 'region': iregion}
                    region_hists = extract_hist_data(h, sel, custom_axes, axis_indices, edges, centers)
                    for suffix, d in region_hists.items():
                        region_name = f"{iregion}{suffix}"
                        accumulated[yr][region_name].append(d)
                        if region_name not in ref_dicts:
                            ref_dicts[region_name] = d

        for yr in args.years:
            for region_name, d_list in accumulated[yr].items():
                if not d_list:
                    continue
                N = len(d_list)
                val_list = [np.array(d["values"], dtype=float) for d in d_list]
                var_list = [np.array(d["variances"], dtype=float) for d in d_list]

                ave_values = np.mean(val_list, axis=0)
                ave_variances = np.sum(var_list, axis=0) / (N ** 2)

                ref = ref_dicts[region_name]
                ave_json_hist = {
                    "edges": ref["edges"],
                    "centers": ref["centers"],
                    "values": ave_values.tolist(),
                    "variances": ave_variances.tolist(),
                    "underflow_value": 0.0,
                    "underflow_variance": 0.0,
                    "overflow_value": 0.0,
                    "overflow_variance": 0.0
                }

                # 1. Replace 'data' in fourTag region with the average mixed data
                if "data" in out_json[var] and yr in out_json[var]["data"] and "fourTag" in out_json[var]["data"][yr]:
                    out_json[var]["data"][yr]["fourTag"][region_name] = ave_json_hist

                # 2. Also register 'mix_ave' as a distinct process
                if "mix_ave" not in out_json[var]:
                    out_json[var]["mix_ave"] = {}
                if yr not in out_json[var]["mix_ave"]:
                    out_json[var]["mix_ave"][yr] = {"fourTag": {}}
                out_json[var]["mix_ave"][yr]["fourTag"][region_name] = ave_json_hist

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Writing output to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(out_json, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
