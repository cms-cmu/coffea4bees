#!/usr/bin/env python3
"""Signal histograms for the two-stage closure (bkg_syst F_2, runTwoStageClosure --input_file_sig).

Writes <var>_<process>_<year>_fourTag_<region> for every --vars x --processes x --years (x --regions
for a .coffea input; SR only, plus 2016/2017/2018 aliases, for a JSON input) to a ROOT file, with
PyROOT when available and uproot otherwise. --dummy (or a missing input) writes flat placeholders.
"""
import os
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "coffea4bees")
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

YEAR_ALIASES = {"UL18": "2018", "UL17": "2017", "UL16_preVFP": "2016", "UL16_postVFP": "2016"}


def find_key(var, keys):
    """The hist key for `var`: as given, dotted, underscored, or the coffea '<classifier>.<rest>' form
    of an underscored name (SvB_MA_ps_hh -> SvB_MA.ps_hh); else the first SvB key with the ps suffix."""
    prefix = "SvB_MA" if "SvB_MA" in var else "SvB"
    candidates = [var, var.replace("_", "."), var.replace(".", "_")]
    if var.startswith(prefix + "_"):
        candidates.append(prefix + "." + var[len(prefix) + 1:])
    for k in candidates:
        if k in keys:
            return k
    suffix = "ps_ttHbb" if "ttHbb" in var else "ps"
    return next((k for k in keys if prefix in k and suffix in k), None)


class Writer:
    """One output ROOT file, written with PyROOT or uproot."""

    def __init__(self, path):
        if HAS_ROOT:
            self.f = ROOT.TFile(path, "RECREATE")
        elif HAS_UPROOT:
            self.f = uproot.recreate(path)
        else:
            raise ImportError("Neither ROOT nor uproot is available.")

    def write(self, name, edges, vals, variances):
        edges = np.asarray(edges, dtype=np.float64)
        vals = np.asarray(vals, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        if HAS_ROOT:
            h = ROOT.TH1F(name, name, len(edges) - 1, array.array("d", edges))
            for b in range(1, len(edges)):
                h.SetBinContent(b, vals[b - 1])
                h.SetBinError(b, variances[b - 1] ** 0.5)
            h.Write("", ROOT.TObject.kOverwrite)
        else:
            widths = np.diff(edges)
            if len(edges) > 1 and not np.allclose(widths, widths[0]):
                h = hist.Hist.new.Var(edges, name="h").Weight()
            else:
                h = hist.Hist.new.Reg(len(edges) - 1, edges[0], edges[-1], name="h").Weight()
            h.view().value = vals
            h.view().variance = variances
            self.f[name] = h

    def close(self):
        if HAS_ROOT:
            self.f.Close()
        else:
            self.f.close()


def from_coffea(args, out):
    from coffea.util import load
    hists = load(args.input).get("hists", {})
    total = 0.0
    for var in args.var:
        key = find_key(var, list(hists))
        if key is None:
            print(f"Warning: Could not find histogram matching {var} in {args.input}.")
            continue
        h = hists[key]
        prefix = var.replace(".", "_")
        for proc in args.processes:
            if proc not in h.axes['process']:
                print(f"Warning: process {proc} not in {key} ({args.input}).")
                continue
            for reg in args.regions:
                for y in args.years:
                    if y not in h.axes['year'] or reg not in h.axes['region']:
                        continue
                    sel = {'process': proc, 'year': y, 'tag': 'fourTag', 'region': reg}
                    for ax in h.axes.name:
                        if ax.startswith(('pass', 'fail')) and ax not in sel:
                            sel[ax] = sum
                    sub = h[sel]
                    if len(sub.axes) > 1:
                        sub = sub.project(sub.axes[-1].name)
                    vals = np.array(sub.values(), dtype=np.float64)
                    variances = np.array(sub.variances(), dtype=np.float64) if sub.variances() is not None else vals
                    name = f"{prefix}_{proc}_{y}_fourTag_{reg}"
                    out.write(name, sub.axes[0].edges, vals, variances)
                    total += float(vals.sum())
                    print(f"  {name}: {vals.sum():.2f}")
    return total


def from_json(args, out):
    with open(args.input) as f:
        d = json.load(f)
    total = 0.0
    for var in args.var:
        key = find_key(var, list(d))
        if key is None:
            print(f"Warning: Variable {var} not found in {args.input}.")
            continue
        prefix = var.replace(".", "_")
        for proc in args.processes:
            for y in args.years:
                if proc not in d[key] or y not in d[key][proc]:
                    continue
                dat = d[key][proc][y]["fourTag"]["SR"]
                for yr in dict.fromkeys([y, YEAR_ALIASES.get(y, y)]):
                    out.write(f"{prefix}_{proc}_{yr}_fourTag_SR", dat["edges"], dat["values"], dat["variances"])
                total += float(sum(dat["values"]))
                print(f"  {prefix}_{proc}_{y}_fourTag_SR: {sum(dat['values']):.2f}")
    return total


def dummy(args, out):
    nbins = 30
    edges = np.linspace(0.0, 1.0, nbins + 1)
    for var in args.var:
        prefix = var.replace(".", "_")
        for proc in args.processes:
            for y in args.years:
                for yr in dict.fromkeys([y, YEAR_ALIASES.get(y, y)]):
                    out.write(f"{prefix}_{proc}_{yr}_fourTag_SR", edges, np.ones(nbins), np.full(nbins, 0.01))


def main():
    parser = argparse.ArgumentParser(description="Create signal ROOT histograms for the two-stage closure")
    parser.add_argument("-i", "--input", default="output/ttHbb_stitched/histAll_ttHbb_stitched.json",
                        help="Input .coffea or JSON file containing the signal process(es)")
    parser.add_argument("-o", "--output", default="output/ttHbb_mixeddata_stitched_closure/root_inputs/hist_signal_ttHbb.root",
                        help="Output ROOT file for closure test")
    parser.add_argument("--var", "--vars", nargs="+", default=["SvB_MA.ps_ttHbb"],
                        help="Variable name(s) in JSON/coffea")
    parser.add_argument("--processes", nargs="+", default=["ttHbb"],
                        help="Signal process(es) to export, one histogram each (default: ttHbb)")
    parser.add_argument("--years", nargs="+", default=["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"],
                        help="List of years to export")
    parser.add_argument("--regions", nargs="+", default=["SR", "SB"],
                        help="List of regions to export (default: SR SB)")
    parser.add_argument("--dummy", action="store_true", default=False,
                        help="Generate fallback/dummy signal histograms if input JSON is missing")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out = Writer(args.output)
    if os.path.exists(args.input) and os.path.getsize(args.input) > 0 and not args.dummy:
        total = from_coffea(args, out) if args.input.endswith(".coffea") else from_json(args, out)
        print(f"Successfully created {args.output} from {args.input} (integral: {total:.2f})")
    else:
        print(f"Input '{args.input}' not found or --dummy requested; writing fallback dummy signal histograms.")
        dummy(args, out)
        print(f"Successfully created fallback {args.output} for years {args.years}")
    out.close()


if __name__ == "__main__":
    main()
