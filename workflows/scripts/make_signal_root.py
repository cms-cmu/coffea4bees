#!/usr/bin/env python3
import os
import json
import array
import argparse
import ROOT

def main():
    parser = argparse.ArgumentParser(description="Create signal ROOT histograms from stitched JSON")
    parser.add_argument("-i", "--input", default="output/ttHbb_stitched/histAll_ttHbb_stitched.json",
                        help="Input JSON file containing ttHbb")
    parser.add_argument("-o", "--output", default="output/ttHbb_mixeddata_stitched_closure/root_inputs/hist_signal_ttHbb.root",
                        help="Output ROOT file for closure test")
    parser.add_argument("--var", default="SvB_MA.ps", help="Variable name in JSON")
    parser.add_argument("--years", nargs="+", default=["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"],
                        help="List of years to export")
    parser.add_argument("--dummy", action="store_true", default=False,
                        help="Generate fallback/dummy signal histograms if input JSON is missing")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    var_prefix = args.var.replace(".", "_")
    f_out = ROOT.TFile(args.output, "RECREATE")

    if os.path.exists(args.input) and not args.dummy:
        with open(args.input) as f:
            d = json.load(f)

        if args.var not in d:
            raise KeyError(f"Variable {args.var} not found in {args.input}. Keys: {list(d.keys())}")

        svb = d[args.var]
        tot_integral = 0.0
        for y in args.years:
            if "ttHbb" not in svb or y not in svb["ttHbb"]:
                continue
            dat = svb["ttHbb"][y]["fourTag"]["SR"]
            edges = dat["edges"]
            vals = dat["values"]
            vars_ = dat["variances"]

            h = ROOT.TH1F(f"{var_prefix}_ttHbb_{y}_fourTag_SR", f"{var_prefix}_ttHbb_{y}_fourTag_SR",
                          len(edges) - 1, array.array("d", edges))
            for b in range(1, len(edges)):
                h.SetBinContent(b, vals[b - 1])
                h.SetBinError(b, vars_[b - 1] ** 0.5)
            h.Write()
            tot_integral += h.Integral()
            print(f"  {var_prefix}_ttHbb_{y}_fourTag_SR: {h.Integral():.2f}")
        print(f"Successfully created {args.output} from {args.input} (integral: {tot_integral:.2f})")
    else:
        print(f"Input JSON '{args.input}' not found or --dummy requested; writing fallback dummy signal histograms.")
        nbins = 30
        edges = [float(i) / nbins for i in range(nbins + 1)]
        for y in args.years:
            h = ROOT.TH1F(f"{var_prefix}_ttHbb_{y}_fourTag_SR", f"{var_prefix}_ttHbb_{y}_fourTag_SR",
                          nbins, array.array("d", edges))
            h.Write()
        print(f"Successfully created fallback {args.output} for years {args.years}")

    f_out.Close()

if __name__ == "__main__":
    main()
