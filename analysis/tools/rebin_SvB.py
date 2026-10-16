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
  1. Automated / Pipeline mode (from evaluated friend tree & classifier inputs):
     ./run_container python coffea4bees/analysis/tools/rebin_SvB.py \\
         --friends output/SvB_MA_ttHbb_stitched/result.json \\
         --classifier-inputs coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json \\
         --key SvB_MA -n 240 --output-json output/SvB_MA_ttHbb_stitched/svb_quantiles.json

  2. Interactive mode (from pre-extracted unbinned .npz):
     ./run_container python coffea4bees/analysis/tools/rebin_SvB.py \\
         -s output/ttHbb/unbinned_SR_ttHbb.npz \\
         -n 20 --ps-min 0.01 --selection inclusive \\
         -o output/ttHbb/rebin_flat_SR_20bins_inclusive.png
===============================================================================
"""

import argparse
import json
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

def extract_signal_from_friends(
    friends_path,
    classifier_inputs,
    year="UL18",
    svb_key="SvB_MA",
    sample_pattern="ttHbb",
    selection="inclusive",
    max_workers=16,
):
    """Extract unbinned SR signal events directly from friend trees and classifier inputs."""
    import glob
    import yaml
    import uproot
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Resolve friend tree URL / path
    if friends_path.endswith(".json") or "result.json" in friends_path:
        url_svb_json = friends_path.split("@@")[0]
    else:
        with open(friends_path) as f:
            friends_cfg = yaml.safe_load(f)
        url_svb_json = friends_cfg["friends"][year][svb_key].split("@@")[0]

    # Load SvB mapping
    if url_svb_json.startswith("root://"):
        try:
            import XRootD.client
            client = XRootD.client.File()
            client.open(url_svb_json, XRootD.client.flags.OpenFlags.READ)
            content = b""
            while True:
                st, d = client.read(len(content), 65536)
                if not d:
                    break
                content += d
            client.close()
            svb_data = json.loads(content.decode("utf-8"))
        except Exception:
            import fsspec
            with fsspec.open(url_svb_json, "r") as f:
                svb_data = json.load(f)
    else:
        with open(url_svb_json) as f:
            svb_data = json.load(f)

    svb_map = {}
    for entry in svb_data["analysis"][0]["merged"]["data"]:
        pico_path = entry[0]["path"]
        svb_chunk = entry[1][0]["chunk"]["path"]
        svb_map[pico_path] = svb_chunk

    # Load classifier inputs
    if classifier_inputs.startswith("root://"):
        import fsspec
        with fsspec.open(classifier_inputs, "r") as f:
            cfg_list = [json.load(f)]
    else:
        cfg_files = sorted(glob.glob(classifier_inputs)) if any(c in classifier_inputs for c in "*?[]") else [classifier_inputs]
        cfg_list = []
        for cf in cfg_files:
            if os.path.exists(cf):
                with open(cf) as f:
                    cfg_list.append(json.load(f))

    tasks = []
    for cfg in cfg_list:
        data_entries = cfg.get("HCR_input", {}).get("data", []) if isinstance(cfg, dict) else []
        for entry in data_entries:
            pico_path = entry[0]["path"]
            if sample_pattern and sample_pattern not in pico_path.split("/")[-2]:
                continue
            hcr_path = entry[1][0]["chunk"]["path"]
            svb_path = svb_map.get(pico_path)
            if svb_path:
                tasks.append((pico_path, hcr_path, svb_path))

    if not tasks:
        raise RuntimeError(f"No matching signal chunks found for pattern '{sample_pattern}'")

    print(f"Extracting signal events from {len(tasks)} chunks across friend trees...")

    def _process_chunk(p, h, s):
        try:
            with uproot.open(h) as f_hcr, uproot.open(s) as f_svb:
                t_hcr = f_hcr["Events"]
                t_svb = f_svb["Events"]
                sr = t_hcr["SR"].array(library="np")
                four_tag = t_hcr["fourTag"].array(library="np")
                mask = sr & four_tag
                if selection == "gt6" and "nSelJets" in t_hcr:
                    mask = mask & (t_hcr["nSelJets"].array(library="np") > 6)
                if np.sum(mask) == 0:
                    return np.array([]), np.array([])
                w = t_hcr["weight"].array(library="np")[mask]
                p_sig = t_svb["p_ttHbb"].array(library="np")[mask]
                p_mj = t_svb["p_multijet"].array(library="np")[mask]
                p_tt = t_svb["p_ttbar"].array(library="np")[mask]
                ps = p_sig / np.maximum(p_sig + p_mj + p_tt, 1e-10)
                return ps, w
        except Exception as e:
            print(f"Error processing {p}: {e}")
            return np.array([]), np.array([])

    all_ps, all_wt = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_chunk, p, h, s): (p, h, s) for p, h, s in tasks}
        for fut in as_completed(futures):
            ps, w = fut.result()
            if len(ps) > 0:
                all_ps.append(ps)
                all_wt.append(w)

    if not all_ps:
        raise RuntimeError("No signal events extracted from friends / classifier inputs")
    return np.concatenate(all_ps), np.concatenate(all_wt)

def load_bkg_evaluator(bkg_path, bkg_type="total"):
    """
    Returns a function get_bkg_yield(x_low, x_high) that computes the background
    content in any arbitrary interval [x_low, x_high] using the fine-binned distribution.
    """
    if not bkg_path or not os.path.exists(bkg_path):
        return None, None, None

    if bkg_path.endswith(".coffea"):
        import coffea.util
        data = coffea.util.load(bkg_path)
        hists = data.get("hists", data)
        hist_name = "SvB_MA.ps"
        if hist_name not in hists:
            return None, None, None
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
        return None, None, None

def optimize_binning(
    sig_ps,
    sig_weight,
    get_bkg_yield=None,
    target_bins=20,
    min_bkg=3.0,
    min_ps=0.01,
):
    """
    Sweeps through unbinned signal events to derive flat quantile binning.
    For fine binning (target_bins >= 50, e.g. 240 bins), computes smooth weighted
    quantiles via cumulative sum interpolation, strictly guaranteeing target_bins + 1 edges.
    """
    mask = (sig_ps >= min_ps) & (sig_ps <= 1.0) & (~np.isnan(sig_ps))
    ps_clean = sig_ps[mask]
    wt_clean = sig_weight[mask]

    total_sig = float(np.sum(wt_clean))
    target_sig = total_sig / float(target_bins)

    if target_bins >= 50:
        order = np.argsort(ps_clean)
        sorted_ps = ps_clean[order]
        sorted_wt = wt_clean[order]
        cum_w = np.cumsum(sorted_wt)
        cdf = cum_w / cum_w[-1]
        q = np.linspace(0.0, 1.0, target_bins + 1)
        bin_edges = np.interp(q, cdf, sorted_ps)
        bin_edges[0] = min_ps
        bin_edges[-1] = 1.0
        return bin_edges, target_sig, total_sig

    # Right-to-left sweep (coarse binning mode)
    sort_idx = np.argsort(ps_clean)[::-1]
    sorted_ps = ps_clean[sort_idx]
    sorted_wt = wt_clean[sort_idx]

    bin_edges = [1.0]
    curr_sig = 0.0

    for i in range(len(sorted_ps)):
        curr_sig += sorted_wt[i]
        curr_ps = sorted_ps[i]

        if curr_sig >= target_sig and len(bin_edges) < target_bins:
            bin_edges.append(curr_ps)
            curr_sig = 0.0

    if bin_edges[-1] != min_ps:
        bin_edges.append(min_ps)

    bin_edges = np.sort(np.unique(bin_edges))
    return bin_edges, target_sig, total_sig

def plot_validation(bin_edges, sig_ps, sig_weight, get_bkg_yield, target_sig, output_path):
    """Generate diagnostic plot verifying signal flatness and background counts."""
    sig_hist = Hist.new.Variable(bin_edges, name="ps").Double()
    mask = (sig_ps > 0) & (sig_ps <= 1.0) & (~np.isnan(sig_ps))
    sig_hist.fill(sig_ps[mask], weight=sig_weight[mask])

    sig_per_bin = sig_hist.values()
    bin_indices = np.arange(len(sig_per_bin))

    has_bkg = get_bkg_yield is not None
    n_plots = 2 if has_bkg else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 8 if has_bkg else 5), sharex=True,
                             gridspec_kw={'height_ratios': [1.2, 1]} if has_bkg else None)
    ax1 = axes[0] if has_bkg else axes

    # Signal Flatness
    ax1.bar(bin_indices, sig_per_bin, width=0.8, color="crimson", alpha=0.85, edgecolor="darkred",
            label=f"Signal (Mean: {np.mean(sig_per_bin):.2f} ev/bin)")
    ax1.axhline(target_sig, color="black", linestyle="--", lw=1.5, label=f"Target S = {target_sig:.2f} events/bin")
    ax1.set_ylabel("Signal Events / bin", fontsize=11)
    ax1.set_title(f"Optimized Flat Binning ({len(sig_per_bin)} Bins) — Signal Flatness Check", fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(sig_per_bin) * 1.35 if len(sig_per_bin) > 0 and max(sig_per_bin) > 0 else 1.0)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3, axis='y')

    if has_bkg:
        ax2 = axes[1]
        bkg_per_bin = [get_bkg_yield(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
        ax2.bar(bin_indices, bkg_per_bin, width=0.8, color="royalblue", alpha=0.85, edgecolor="darkblue", label="Background Events / bin")
        ax2.axhline(3.0, color="red", linestyle=":", lw=2, label="Threshold = 3.0 Events")
        ax2.set_yscale("log")
        ax2.set_xlabel("Rebinned Bin Index (0: Low Score $\\rightarrow$ High Score)", fontsize=11)
        ax2.set_ylabel("Background Events (log)", fontsize=11)
        ax2.set_title("Background Event Count per Bin (Preserving $\\geq 3$ Events)", fontsize=12, fontweight='bold')
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel("Rebinned Bin Index (0: Low Score $\\rightarrow$ High Score)", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Validation plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Derive flat variable binning from unbinned events.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--sig-input", default=None, help="Signal unbinned NPZ or ROOT file")
    parser.add_argument("--friends", default=None, help="Friend tree result.json or friends.yml file for on-the-fly extraction")
    parser.add_argument("--classifier-inputs", default=None, help="Classifier input JSON manifest or pattern for on-the-fly extraction")
    parser.add_argument("--sample-pattern", default="ttHbb", help="Sample pattern for on-the-fly extraction")
    parser.add_argument("--year", default="UL18", help="Year key inside friends YAML")
    parser.add_argument("--svb-key", default="SvB_MA", help="Friend key for SvB")
    parser.add_argument("--key", default=None, help="Key name to write under in JSON (default: same as svb-key)")
    parser.add_argument("-b", "--bkg-input", default=None, help="Background coffea or NPZ file (optional)")
    parser.add_argument("--bkg-type", default="total", choices=["total", "multijet"], help="Background definition ('total' = multijet + ttbar, 'multijet' = data 3-tag only)")
    parser.add_argument("-n", "--target-bins", type=int, default=240, help="Target number of bins (default: 240 for quantile mapping)")
    parser.add_argument("-t", "--min-bkg", type=float, default=3.0, help="Minimum background events per bin")
    parser.add_argument("--min-ps", type=float, default=0.01, help="Minimum ps threshold")
    parser.add_argument("--selection", default="inclusive", choices=["inclusive", "gt6"], help="Category selection")
    parser.add_argument("-o", "--output-plot", default=None, help="Path to save verification plot (optional)")
    parser.add_argument("-j", "--output-json", default=None, help="Path to save bin edges JSON dictionary")

    args = parser.parse_args()

    print("=" * 70)
    print("  OPTIMIZING UNBINNED VARIABLE BINNING SCHEME")
    print("=" * 70)
    print(f"Signal input:     {args.sig_input}")
    print(f"Friends input:    {args.friends}")
    print(f"Classifier input: {args.classifier_inputs}")
    print(f"Category:         {args.selection}")
    print(f"Min ps cut:       >= {args.min_ps}")
    print(f"Target bins:      {args.target_bins}")
    print("=" * 70)

    # 1. Load or extract signal data
    if args.sig_input and os.path.exists(args.sig_input):
        sig_ps, sig_wt = load_signal_data(args.sig_input, selection=args.selection)
    elif args.friends and args.classifier_inputs:
        sig_ps, sig_wt = extract_signal_from_friends(
            friends_path=args.friends,
            classifier_inputs=args.classifier_inputs,
            year=args.year,
            svb_key=args.svb_key,
            sample_pattern=args.sample_pattern,
            selection=args.selection,
        )
    else:
        # Fallback to default NPZ location if exists
        default_npz = "output/ttHbb/unbinned_SR_ttHbb.npz"
        if os.path.exists(default_npz):
            sig_ps, sig_wt = load_signal_data(default_npz, selection=args.selection)
        else:
            raise ValueError("Must provide either valid --sig-input or both --friends and --classifier-inputs")

    # 2. Evaluate background if provided
    get_bkg = None
    if args.bkg_input and os.path.exists(args.bkg_input):
        get_bkg, _, _ = load_bkg_evaluator(args.bkg_input, bkg_type=args.bkg_type)

    # 3. Optimize binning
    bin_edges, target_sig, total_sig = optimize_binning(
        sig_ps, sig_wt, get_bkg, target_bins=args.target_bins, min_bkg=args.min_bkg, min_ps=args.min_ps
    )

    n_bins = len(bin_edges) - 1
    print(f"\nOptimization Results:")
    print(f"  Total Signal Yield:      {total_sig:.2f}")
    print(f"  Target Signal per Bin:   {target_sig:.2f}")
    print(f"  Achieved Number of Bins: {n_bins}")

    key_name = args.key or args.svb_key or "SvB_MA"

    # 4. Save to JSON if requested
    if args.output_json:
        out_dict = {}
        if os.path.exists(args.output_json):
            try:
                with open(args.output_json, "r") as f:
                    out_dict = json.load(f)
            except Exception:
                out_dict = {}
        out_dict[key_name] = [round(float(x), 6) for x in bin_edges]
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"\nSaved bin edges to JSON: {args.output_json} under key '{key_name}'")

    print("\n" + "=" * 70)
    print("PYTHON CODE FOR hist_templates.py:")
    print("=" * 70)
    edges_str = np.array2string(bin_edges, precision=6, separator=", ", max_line_width=85)
    print(f"    var_binning_ps_flat = np.array({edges_str})\n")
    print("=" * 70)

    # 5. Validation plot
    if args.output_plot:
        plot_validation(bin_edges, sig_ps, sig_wt, get_bkg, target_sig, args.output_plot)

if __name__ == "__main__":
    main()
