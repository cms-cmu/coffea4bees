#!/usr/bin/env python3
"""
Generate multi-subsample overlay comparison plots across the 16 background variants
(v0..v15) from Stage F_1 histAll_mixeddata_bkgs.coffea.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coffea.util import load

# 16-color cycle for subsamples
COLORS = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#bcbd22"
]


def slice_1d(h, process, region, tag='threeTag'):
    """Extract 1D slice from coffea Hist summed across years."""
    if h is None:
        return None
    if process not in h.axes['process']:
        return None
    if region not in h.axes['region']:
        return None
    if tag not in h.axes['tag']:
        return None

    sel = {
        'process': process,
        'tag': tag,
        'region': region,
    }
    for ax in h.axes:
        if ax.name.startswith(('pass', 'fail')) and ax.name not in sel:
            sel[ax.name] = sum
    try:
        h_sel = h[sel]
        # Sum over year
        if 'year' in h_sel.axes.name:
            h_sel = h_sel[{'year': sum}]
        # Project to the remaining 1D variable axis
        var_axes = [ax.name for ax in h_sel.axes if ax.name not in ['process', 'year', 'tag', 'region']]
        if var_axes:
            return h_sel.project(var_axes[-1])
        return h_sel
    except Exception as e:
        return None


def plot_overlay(hists_per_subsample, var_name, region, out_dir, xlabel="Variable", yscale="linear"):
    """Plot overlaid curves for all subsamples with a ratio-to-v0 bottom panel."""
    os.makedirs(out_dir, exist_ok=True)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}
    )

    ref_vals = None
    ref_edges = None

    for m, h_1d in enumerate(hists_per_subsample):
        if h_1d is None:
            continue
        edges = np.array(h_1d.axes[0].edges)
        vals = np.array(h_1d.values())
        color = COLORS[m % len(COLORS)]

        if m == 0:
            ref_vals = vals
            ref_edges = edges

        # Plot step on top axis
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_top.stairs(vals, edges, color=color, label=f"v{m}", linewidth=1.2, alpha=0.85)

        # Plot ratio on bottom axis
        if ref_vals is not None and len(ref_vals) == len(vals):
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(ref_vals > 0, vals / ref_vals, np.nan)
            ax_bot.stairs(ratio, edges, color=color, linewidth=1.2, alpha=0.85)

    ax_top.set_ylabel("Events / bin", fontsize=12)
    ax_top.set_yscale(yscale)
    ax_top.set_title(f"{var_name} [{region}] (16 Subsamples Background Model)", fontsize=13)
    ax_top.grid(True, linestyle="--", alpha=0.4)
    ax_top.legend(ncol=4, fontsize=8, loc="upper right")

    ax_bot.set_xlabel(xlabel, fontsize=12)
    ax_bot.set_ylabel("v / v0", fontsize=11)
    ax_bot.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax_bot.set_ylim(0.7, 1.3)
    ax_bot.grid(True, linestyle="--", alpha=0.4)

    out_file = os.path.join(out_dir, f"overlay_{var_name}_{region}.png")
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"[Overlay] Saved {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate subsample overlay plots from Stage F_1 background coffea.")
    parser.add_argument("-i", "--input", required=True, help="Input histAll_mixeddata_bkgs.coffea")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for plots")
    parser.add_argument("--channel", default="ttHbb", help="Channel name")
    parser.add_argument("--n_subsamples", type=int, default=16, help="Number of subsamples")
    parser.add_argument("--regions", nargs="+", default=["SR", "SB"], help="Regions to plot")
    args = parser.parse_args()

    data = load(args.input)
    hists = data.get("hists", {})

    subsamples = [f"v{m}" for m in range(args.n_subsamples)]

    targets = [
        {"key_template": "SvB_MA_{sub}.ps_ttHbb", "name": "SvB_MA_ps_ttHbb", "xlabel": "SvB MA P(Signal) (Quantile bins)", "yscale": "log"},
        {"key_template": "SvB_MA_{sub}.ps", "name": "SvB_MA_ps", "xlabel": "SvB MA P(Signal)", "yscale": "log"},
        {"key_template": "FvT_{sub}.FvT", "name": "FvT", "xlabel": "FvT weight", "yscale": "log"},
        {"key_template": "selJets_{sub}.pt", "name": "selJets_pt", "xlabel": "Selected Jet $p_T$ [GeV]", "yscale": "log"},
        {"key_template": "selJets_{sub}.n", "name": "selJets_n", "xlabel": "Number of Selected Jets", "yscale": "linear"},
    ]

    for reg in args.regions:
        for tgt in targets:
            sub_hists = []
            for sub in subsamples:
                k = tgt["key_template"].format(sub=sub)
                h = hists.get(k)
                if h is not None:
                    # Multijet data
                    h_data = slice_1d(h, "data", reg, tag="threeTag")
                    # Add ttbar if available
                    h_tt = slice_1d(h, "TTbar4b_from_d3", reg, tag="threeTag")
                    if h_data is not None and h_tt is not None:
                        try:
                            h_tot = h_data + h_tt
                            sub_hists.append(h_tot)
                        except Exception:
                            sub_hists.append(h_data)
                    elif h_data is not None:
                        sub_hists.append(h_data)
                    else:
                        sub_hists.append(None)
                else:
                    sub_hists.append(None)

            if any(h is not None for h in sub_hists):
                plot_overlay(sub_hists, tgt["name"], reg, args.output_dir, xlabel=tgt["xlabel"], yscale=tgt["yscale"])


if __name__ == "__main__":
    main()
