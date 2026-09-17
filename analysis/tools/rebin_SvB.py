#!/usr/bin/env python3
"""
===============================================================================
Automated Right-to-Left Variable Binning Tool for SvB Discriminators
===============================================================================

Purpose:
  Derives quantile binning schemes that produce a strictly flat signal event yield
  in the Signal Region (SR) across N bins, anchored from the purest right side
  (ps = 1.0 down to ps = ps_min, e.g. 0.01), while verifying background statistics.

Usage Instructions:
  1. Extract unbinned SR events (ps, weights, nSelJets):
     ./run_container python coffea4bees/analysis/tools/extract_unbinned_SR.py

  2. Run binning derivation for Inclusive selection:
     ./run_container python coffea4bees/analysis/tools/rebin_SvB.py \
         -s output/ttHbb/unbinned_SR_ttHbb.npz \
         -n 20 --ps-min 0.01 --sweep right_to_left --selection inclusive \
         -o output/ttHbb/rebin_flat_SR_20bins_inclusive.png

  3. Run binning derivation for nSelJets > 6 selection:
     ./run_container python coffea4bees/analysis/tools/rebin_SvB.py \
         -s output/ttHbb/unbinned_SR_ttHbb.npz \
         -n 20 --ps-min 0.01 --sweep right_to_left --selection gt6 \
         -o output/ttHbb/rebin_flat_SR_20bins_gt6.png

  4. Copy the printed array of bin edges into `coffea4bees/analysis/helpers/hist_templates.py`
     under `ttHbbSvBHists.var_binning_ps_ttHbb` and `var_binning_ps_ttHbb_gt6`.
===============================================================================
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hist import Hist

def load_signal_data(input_path, variable="ps", selection="inclusive"):
    """Load unbinned signal ps and weight arrays from .npz or ROOT files."""
    if input_path.endswith(".npz"):
        data = np.load(input_path)
        ps = data["ps"]
        weight = data["weight"]
        if selection == "gt6" and "nSelJets" in data:
            mask = data["nSelJets"] > 6
            ps = ps[mask]
            weight = weight[mask]
        return ps, weight
    elif input_path.endswith(".root"):
        import uproot
        with uproot.open(input_path) as f:
            t = f["Events"]
            ps = t[variable].array(library="np")
            weight = t["weight"].array(library="np")
            if selection == "gt6" and "nSelJets" in t:
                mask = t["nSelJets"].array(library="np") > 6
                ps = ps[mask]
                weight = weight[mask]
        return ps, weight
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

def load_bkg_evaluator(bkg_path, bkg_type="total"):
    """
    Returns a function get_bkg_yield(x_low, x_high) that computes the background
    content in any arbitrary interval [x_low, x_high] using the fine-binned distribution.
    """
    if bkg_path.endswith(".coffea"):
        import coffea.util
        data = coffea.util.load(bkg_path)
        hists = data.get("hists", data)
        # Use fine histogram if available
        hist_name = "SvB_MA.ps_ttHbb_fine" if "SvB_MA.ps_ttHbb_fine" in hists else "SvB_MA.ps"
        h = hists[hist_name]
        
        # Multijet: data threeTag SR
        mj_1d = np.sum(h[{"process": "data", "tag": "threeTag", "region": "SR"}].values(), axis=(0, 1))
        # ttbar: TTbar4b_from_d3 threeTag SR
        tt_1d = np.sum(h[{"process": "TTbar4b_from_d3", "tag": "threeTag", "region": "SR"}].values(), axis=(0, 1))
        
        bkg_arr = (mj_1d + tt_1d) if bkg_type == "total" else mj_1d
        edges = np.array(h.axes[-1].edges)
        
        def get_bkg(x_low, x_high):
            mask = (edges[:-1] >= x_low - 1e-6) & (edges[1:] <= x_high + 1e-6)
            return float(np.sum(bkg_arr[mask]))
            
        return get_bkg, edges, bkg_arr
    elif bkg_path.endswith(".npz"):
        bdata = np.load(bkg_path)
        b_ps = bdata["ps"]
        b_wt = bdata["weight"]
        def get_bkg(x_low, x_high):
            mask = (b_ps >= x_low) & (b_ps < x_high)
            return float(np.sum(b_wt[mask]))
        return get_bkg, None, None
    else:
        raise ValueError(f"Unsupported background format: {bkg_path}")

def optimize_binning(
    sig_ps,
    sig_weight,
    get_bkg_yield,
    target_bins=20,
    min_bkg=3.0,
    min_ps=0.01,
):
    """
    Sweeps from x = 1.0 down to min_ps through unbinned signal events,
    accumulating until target signal is reached for each bin.
    """
    mask = (sig_ps >= min_ps) & (sig_ps <= 1.0) & (~np.isnan(sig_ps))
    ps_clean = sig_ps[mask]
    wt_clean = sig_weight[mask]
    
    total_sig = np.sum(wt_clean)
    target_sig = total_sig / float(target_bins)
    
    # Sort descending (from 1.0 down to min_ps)
    sort_idx = np.argsort(ps_clean)[::-1]
    sorted_ps = ps_clean[sort_idx]
    sorted_wt = wt_clean[sort_idx]
    
    bin_edges = [1.0]
    curr_sig = 0.0
    
    for i in range(len(sorted_ps)):
        curr_sig += sorted_wt[i]
        curr_ps = sorted_ps[i]
        
        # Check if target signal reached and we still need bins
        if curr_sig >= target_sig and len(bin_edges) < target_bins:
            bin_edges.append(curr_ps)
            curr_sig = 0.0
                
    if bin_edges[-1] != min_ps:
        bin_edges.append(min_ps)
        
    bin_edges = np.sort(np.unique(bin_edges))
    return bin_edges, target_sig, total_sig

def plot_validation(bin_edges, sig_ps, sig_weight, get_bkg_yield, target_sig, output_path):
    """Generate diagnostic plot verifying signal flatness and background counts."""
    # Compute contents per bin
    sig_hist = Hist.new.Variable(bin_edges, name="ps").Double()
    mask = (sig_ps > 0) & (sig_ps <= 1.0) & (~np.isnan(sig_ps))
    sig_hist.fill(sig_ps[mask], weight=sig_weight[mask])
    
    sig_per_bin = sig_hist.values()
    bkg_per_bin = [get_bkg_yield(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]
    bin_indices = np.arange(len(sig_per_bin))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
    
    # Signal Flatness
    ax1.bar(bin_indices, sig_per_bin, width=0.8, color="crimson", alpha=0.85, edgecolor="darkred", label=f"Unbinned ttHbb Signal (Mean: {np.mean(sig_per_bin[1:-1]):.2f} ev/bin)")
    ax1.axhline(target_sig, color="black", linestyle="--", lw=1.5, label=f"Target S = {target_sig:.2f} events/bin")
    ax1.set_ylabel("Signal Events / bin", fontsize=11)
    ax1.set_title(f"Optimized Flat Binning ({len(sig_per_bin)} Bins) — Signal Flatness Check", fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(sig_per_bin)*1.35)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Background Counts
    ax2.bar(bin_indices, bkg_per_bin, width=0.8, color="royalblue", alpha=0.85, edgecolor="darkblue", label="Background Events / bin")
    ax2.axhline(3.0, color="red", linestyle=":", lw=2, label="Threshold = 3.0 Events")
    ax2.set_yscale("log")
    ax2.set_xlabel("Rebinned Bin Index (0: Low Score $\\rightarrow$ High Score)", fontsize=11)
    ax2.set_ylabel("Background Events (log)", fontsize=11)
    ax2.set_title("Background Event Count per Bin (Preserving $\\geq 3$ Events)", fontsize=12, fontweight='bold')
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nValidation plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Derive flat variable binning from unbinned events.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--sig-input", default="output/ttHbb/unbinned_SR_ttHbb.npz", help="Signal unbinned NPZ or ROOT file")
    parser.add_argument("-b", "--bkg-input", default="output/ttHbb/histAll_ttHbb.coffea", help="Background coffea or NPZ file")
    parser.add_argument("--bkg-type", default="total", choices=["total", "multijet"], help="Background definition ('total' = multijet + ttbar, 'multijet' = data 3-tag only)")
    parser.add_argument("-n", "--target-bins", type=int, default=20, help="Target number of bins (~20)")
    parser.add_argument("-t", "--min-bkg", type=float, default=3.0, help="Minimum background events per bin")
    parser.add_argument("--min-ps", type=float, default=0.01, help="Minimum ps threshold")
    parser.add_argument("--selection", default="inclusive", choices=["inclusive", "gt6"], help="Category selection")
    parser.add_argument("-o", "--output-plot", default="output/ttHbb/rebin_flat_SR.png", help="Path to save verification plot")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  OPTIMIZING UNBINNED VARIABLE BINNING SCHEME")
    print("=" * 70)
    print(f"Signal input:     {args.sig_input}")
    print(f"Background input: {args.bkg_input} (type: {args.bkg_type})")
    print(f"Category:         {args.selection}")
    print(f"Min ps cut:       >= {args.min_ps}")
    print(f"Target bins:      ~{args.target_bins}")
    print(f"Min background:   >= {args.min_bkg} events/bin")
    print("=" * 70)
    
    sig_ps, sig_wt = load_signal_data(args.sig_input, selection=args.selection)
    get_bkg, _, _ = load_bkg_evaluator(args.bkg_input, bkg_type=args.bkg_type)
    
    bin_edges, target_sig, total_sig = optimize_binning(
        sig_ps, sig_wt, get_bkg, target_bins=args.target_bins, min_bkg=args.min_bkg, min_ps=args.min_ps
    )
    
    n_bins = len(bin_edges) - 1
    print(f"\nOptimization Results:")
    print(f"  Total Signal Yield:      {total_sig:.2f}")
    print(f"  Target Signal per Bin:   {target_sig:.2f}")
    print(f"  Achieved Number of Bins: {n_bins}")
    
    print("\n" + "=" * 70)
    print("PYTHON CODE FOR hist_templates.py:")
    print("=" * 70)
    edges_str = np.array2string(bin_edges, precision=6, separator=", ", max_line_width=85)
    print(f"    var_binning_ps_flat = np.array({edges_str})\n")
    print("=" * 70)
    
    plot_validation(bin_edges, sig_ps, sig_wt, get_bkg, target_sig, args.output_plot)

if __name__ == "__main__":
    main()
