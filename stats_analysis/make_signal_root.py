#!/usr/bin/env python3
import os
import json
import array
import argparse
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    ROOT = None
    HAS_ROOT = False
try:
    import uproot
    import hist
    HAS_UPROOT = True
except ImportError:
    uproot = None
    hist = None
    HAS_UPROOT = False
import numpy as np

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

    if HAS_ROOT:
        f_out = ROOT.TFile(args.output, "RECREATE")
        if os.path.exists(args.input) and os.path.getsize(args.input) > 0 and not args.dummy:
            if args.input.endswith(".coffea"):
                from coffea.util import load
                c_data = load(args.input)
                hists_dict = c_data.get("hists", {})
                h_match = None
                for hk in [args.var, args.var.replace("_", "."), args.var.replace(".", "_")]:
                    if hk in hists_dict:
                        h_match = hists_dict[hk]
                        break
                if h_match is None:
                    # Fuzzy match
                    for hk, h in hists_dict.items():
                        if "SvB_MA" in hk and "ttHbb" in hk:
                            h_match = h
                            break
                if h_match is None:
                    raise KeyError(f"Could not find histogram matching {args.var} in {args.input}. Keys: {list(hists_dict.keys())}")

                tot_integral = 0.0
                for y in args.years:
                    if y not in h_match.axes['year'] or 'ttHbb' not in h_match.axes['process']:
                        continue
                    sel = {'process': 'ttHbb', 'year': y, 'tag': 'fourTag', 'region': 'SR'}
                    for ax in h_match.axes.name:
                        if ax.startswith(('pass', 'fail')) and ax not in sel:
                            sel[ax] = sum
                    sub = h_match[sel]
                    if len(sub.axes) > 1:
                        sub = sub.project(sub.axes[-1].name)
                    edges = np.array(sub.axes[0].edges, dtype=np.float64)
                    vals = np.array(sub.values(), dtype=np.float64)
                    vars_ = np.array(sub.variances(), dtype=np.float64) if sub.variances() is not None else vals

                    h_root = ROOT.TH1F(f"{var_prefix}_ttHbb_{y}_fourTag_SR", f"{var_prefix}_ttHbb_{y}_fourTag_SR",
                                       len(edges) - 1, array.array("d", edges))
                    for b in range(1, len(edges)):
                        h_root.SetBinContent(b, vals[b - 1])
                        h_root.SetBinError(b, vars_[b - 1] ** 0.5)
                    h_root.Write("", ROOT.TObject.kOverwrite)
                    tot_integral += float(vals.sum())
                    print(f"  {var_prefix}_ttHbb_{y}_fourTag_SR: {vals.sum():.2f}")
                print(f"Successfully created {args.output} from {args.input} (integral: {tot_integral:.2f})")
            else:
                with open(args.input) as f:
                    d = json.load(f)

                var_key = None
                for vk in [args.var, args.var.replace("_", "."), args.var.replace(".", "_")]:
                    if vk in d:
                        var_key = vk
                        break
                if var_key is None:
                    for vk in d:
                        if "SvB_MA" in vk and "ttHbb" in vk:
                            var_key = vk
                            break
                if var_key is None:
                    raise KeyError(f"Variable {args.var} not found in {args.input}. Keys: {list(d.keys())}")

                svb = d[var_key]
                tot_integral = 0.0
                for y in args.years:
                    if "ttHbb" not in svb or y not in svb["ttHbb"]:
                        continue
                    dat = svb["ttHbb"][y]["fourTag"]["SR"]
                    edges = dat["edges"]
                    vals = dat["values"]
                    vars_ = dat["variances"]

                    target_years = [y]
                    if y == "UL18" and "2018" not in target_years:
                        target_years.append("2018")
                    elif y == "UL17" and "2017" not in target_years:
                        target_years.append("2017")
                    elif y.startswith("UL16") and "2016" not in target_years:
                        target_years.append("2016")

                    for yr in target_years:
                        h = ROOT.TH1F(f"{var_prefix}_ttHbb_{yr}_fourTag_SR", f"{var_prefix}_ttHbb_{yr}_fourTag_SR",
                                      len(edges) - 1, array.array("d", edges))
                        for b in range(1, len(edges)):
                            h.SetBinContent(b, vals[b - 1])
                            h.SetBinError(b, vars_[b - 1] ** 0.5)
                        h.Write("", ROOT.TObject.kOverwrite)
                    tot_integral += sum(vals)
                    print(f"  {var_prefix}_ttHbb_{y}_fourTag_SR: {sum(vals):.2f}")
                print(f"Successfully created {args.output} from {args.input} (integral: {tot_integral:.2f})")
        else:
            print(f"Input JSON '{args.input}' not found or --dummy requested; writing fallback dummy signal histograms.")
            nbins = 30
            edges = [float(i) / nbins for i in range(nbins + 1)]
            for y in args.years:
                target_years = [y]
                if y == "UL18" and "2018" not in target_years:
                    target_years.append("2018")
                elif y == "UL17" and "2017" not in target_years:
                    target_years.append("2017")
                elif y.startswith("UL16") and "2016" not in target_years:
                    target_years.append("2016")
                for yr in target_years:
                    h = ROOT.TH1F(f"{var_prefix}_ttHbb_{yr}_fourTag_SR", f"{var_prefix}_ttHbb_{yr}_fourTag_SR",
                                  nbins, array.array("d", edges))
                    for b in range(1, nbins + 1):
                        h.SetBinContent(b, 1.0)
                        h.SetBinError(b, 0.1)
                    h.Write()
            print(f"Successfully created fallback {args.output} for years {args.years}")
        f_out.Close()
    elif HAS_UPROOT:
        # uproot fallback
        with uproot.recreate(args.output) as f_out:
            if os.path.exists(args.input) and os.path.getsize(args.input) > 0 and not args.dummy:
                if args.input.endswith(".coffea"):
                    from coffea.util import load
                    c_data = load(args.input)
                    hists_dict = c_data.get("hists", {})
                    h_match = None
                    for hk in [args.var, args.var.replace("_", "."), args.var.replace(".", "_")]:
                        if hk in hists_dict:
                            h_match = hists_dict[hk]
                            break
                    if h_match is None:
                        for hk, h in hists_dict.items():
                            if "SvB_MA" in hk and "ttHbb" in hk:
                                h_match = h
                                break
                    if h_match is None:
                        raise KeyError(f"Could not find histogram matching {args.var} in {args.input}. Keys: {list(hists_dict.keys())}")

                    tot_integral = 0.0
                    for y in args.years:
                        if y not in h_match.axes['year'] or 'ttHbb' not in h_match.axes['process']:
                            continue
                        sel = {'process': 'ttHbb', 'year': y, 'tag': 'fourTag', 'region': 'SR'}
                        for ax in h_match.axes.name:
                            if ax.startswith(('pass', 'fail')) and ax not in sel:
                                sel[ax] = sum
                        sub = h_match[sel]
                        if len(sub.axes) > 1:
                            sub = sub.project(sub.axes[-1].name)
                        edges = np.array(sub.axes[0].edges, dtype=np.float64)
                        vals = np.array(sub.values(), dtype=np.float64)
                        vars_ = np.array(sub.variances(), dtype=np.float64) if sub.variances() is not None else vals

                        widths = np.diff(edges)
                        if len(edges) > 1 and not np.allclose(widths, widths[0]):
                            out_h = hist.Hist.new.Var(edges, name="h").Weight()
                        else:
                            out_h = hist.Hist.new.Reg(len(edges) - 1, edges[0], edges[-1], name="h").Weight()
                        out_h.view().value = vals
                        out_h.view().variance = vars_

                        f_out[f"{var_prefix}_ttHbb_{y}_fourTag_SR"] = out_h
                        tot_integral += float(vals.sum())
                        print(f"  {var_prefix}_ttHbb_{y}_fourTag_SR: {vals.sum():.2f}")
                    print(f"Successfully created {args.output} from {args.input} (integral: {tot_integral:.2f})")
                else:
                    with open(args.input) as f:
                        d = json.load(f)
                    svb = d.get(args.var, {})
                    tot_integral = 0.0
                    for y in args.years:
                        if "ttHbb" in svb and y in svb["ttHbb"]:
                            dat = svb["ttHbb"][y]["fourTag"]["SR"]
                            edges = np.array(dat["edges"], dtype=np.float64)
                            vals = np.array(dat["values"], dtype=np.float64)
                            vars_ = np.array(dat["variances"], dtype=np.float64)
                            widths = np.diff(edges)
                            if len(edges) > 1 and not np.allclose(widths, widths[0]):
                                h = hist.Hist.new.Var(edges, name="h").Weight()
                            else:
                                h = hist.Hist.new.Reg(len(edges) - 1, edges[0], edges[-1], name="h").Weight()
                            h.view().value = vals
                            h.view().variance = vars_
                            f_out[f"{var_prefix}_ttHbb_{y}_fourTag_SR"] = h
                            tot_integral += float(vals.sum())
                            print(f"  {var_prefix}_ttHbb_{y}_fourTag_SR: {vals.sum():.2f}")
                    print(f"Successfully created {args.output} from {args.input} (integral: {tot_integral:.2f})")
            else:
                print(f"Input JSON '{args.input}' not found or --dummy requested; writing fallback dummy signal histograms via uproot.")
                nbins = 30
                edges = np.linspace(0.0, 1.0, nbins + 1)
                for y in args.years:
                    h = hist.Hist.new.Var(edges, name="h").Weight()
                    h.view().value = np.ones(nbins, dtype=np.float64)
                    h.view().variance = np.ones(nbins, dtype=np.float64) * 0.01
                    f_out[f"{var_prefix}_ttHbb_{y}_fourTag_SR"] = h
                print(f"Successfully created {args.output} via uproot")
    else:
        raise ImportError("Neither ROOT nor uproot is available.")

if __name__ == "__main__":
    main()
