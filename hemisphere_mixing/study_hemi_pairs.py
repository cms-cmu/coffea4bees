"""
Interactive plotting utility for HH4b analysis.

This module provides functions for creating and customizing 1D and 2D plots
from histogram data, with support for multiple variables, regions, and processes.
"""

import os
import sys
from typing import Optional, Union, List, Tuple, Dict, Any

# Third-party imports
import hist
import matplotlib.pyplot as plt

# Local imports
sys.path.insert(0, os.getcwd())
from coffea4bees.plots.plots import load_config_4b
from src.plotting.plots import (
    makePlot, make2DPlot, load_hists,
    read_axes_and_cuts, parse_args, print_cfg
)
from src.plotting.iPlot_config import plot_config
cfg = plot_config()

# Constants
DEFAULT_OUTPUT_FILE = "test.pdf"

from itertools import combinations
import matplotlib.pyplot as plt
from collections import Counter





def initialize_config() -> None:
    """Initialize the configuration from command line arguments."""
    args = parse_args()
    cfg.plotConfig = load_config_4b(args.metadata)
    cfg.outputFolder = args.outputFolder
    cfg.combine_input_files = args.combine_input_files

    if cfg.outputFolder and not os.path.exists(cfg.outputFolder):
        os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    cfg.set_hist_key("hists")


def study_region(year, region):

    year_str = year

    if type(year) is list:
        year_str = "_".join(year)
        for iy, _y in enumerate(year):
            if not iy:
                events       = cfg.hists[0][f"mixeddata_all_{_y}"][f"hemi_pair_{region}_event"]
                runs         = cfg.hists[0][f"mixeddata_all_{_y}"][f"hemi_pair_{region}_run"]
                hemisphereId = cfg.hists[0][f"mixeddata_all_{_y}"][f"hemi_pair_{region}_hemisphereId"]
            else:
                events       += cfg.hists[0][f"mixeddata_all_{_y}"][f"hemi_pair_{region}_event"]
                runs         += cfg.hists[0][f"mixeddata_all_{_y}"][f"hemi_pair_{region}_run"]
                hemisphereId += cfg.hists[0][f"mixeddata_all_{_y}"][f"hemi_pair_{region}_hemisphereId"]

    else:
        events = cfg.hists[0][f"mixeddata_all_{year}"][f"hemi_pair_{region}_event"]
        runs   = cfg.hists[0][f"mixeddata_all_{year}"][f"hemi_pair_{region}_run"]
        hemisphereId   = cfg.hists[0][f"mixeddata_all_{year}"][f"hemi_pair_{region}_hemisphereId"]


    # Flatten across hemisphere
    hemis = [
        (e, r, h)
        for event, run,  hemi in zip(events, runs, hemisphereId)
        for e, r, h in zip(event, run, hemi)
    ]

    hemi_counts = Counter(hemis)
    hemi_ave_usage = sum(hemi_counts.values()) / len(hemi_counts)

    # Step 3: make histogram of "number of triples used N times"
    hist_hemi = Counter(hemi_counts.values())

    #for n_uses, n_triples in sorted(hist.items()):
    #    print(f"{n_triples} triples appear {n_uses} times")
    print(f"Average hemisphere usage in {region} {hemi_ave_usage:.2f}")
    plt.bar(hist_hemi.keys(), hist_hemi.values())
    plt.title(f"Average usage frequency {hemi_ave_usage:.2f}")
    plt.xlabel("Number of times a hemisphere is used")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.savefig(f"hemi_usage_{year_str}_{region}.pdf")
    plt.close()
    #breakpoint()

    # Build list of triples per event
    hemi_pairs_per_event = [
        [(r, e, h) for r, e, h in zip(run, event, hemi)]
        for run, event, hemi in zip(runs, events, hemisphereId)
    ]

    # Count unordered pairs of triples across all events
    pair_counts = Counter()

    for triples in hemi_pairs_per_event:
        for a, b in combinations(triples, 2):
            # make unordered key (sorted) so (A,B) == (B,A)
            key = tuple(sorted((a, b)))
            pair_counts[key] += 1

    pair_ave_usage = sum(pair_counts.values()) / len(pair_counts)
    #for (a, b), n in pair_counts.items():
    #    if n >1:
    #        print(f"{a} paired with {b}: {n} times")
    print(f"Average hemisphere pair usage in {region} {pair_ave_usage:.5f}")
    hist_pairs = Counter(pair_counts.values())
    plt.bar(hist_pairs.keys(), hist_pairs.values())
    plt.title(f"Average pair frequency {pair_ave_usage:.2f}")
    plt.xlabel("Number of times a hemisphere pair is used")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.savefig(f"hemi_pair_usage_{year_str}_{region}.pdf")
    plt.close()


    #
    #  Now making 3b subsample corrlations
    #
    if region == "SR":
        if type(year) is list:
            for iy, _y in enumerate(year):
                if not iy:
                    subsample_counts = cfg.hists[0][f"mixeddata_all_{_y}"]["subsample_counts"]
                else:
                    _subsample_counts = cfg.hists[0][f"mixeddata_all_{_y}"]["subsample_counts"]
                    for key in subsample_counts.keys():
                        subsample_counts[key] += _subsample_counts[key]


        else:
            subsample_counts = cfg.hists[0][f"mixeddata_all_{year}"]["subsample_counts"]

        li = list(subsample_counts.keys())
        n_sub_samples = max([int(el.split("_")[0]) for el in li]) + 1

        top_line = ""
        for i in range(n_sub_samples):
            top_line += f"{i:10d} "
        print(top_line)

        for i in range(n_sub_samples):
            line = f"{i} "
            for j in range(n_sub_samples):
                if j < i:
                    line += f"{'':10s} "
                else:
                    line += f"{subsample_counts[f'{i}_{j}']:10d} "
            print(line)


if __name__ == '__main__':
    initialize_config()
    #print_cfg(cfg)

    for year in ["UL17", "UL18", ["UL16_preVFP", "UL16_postVFP"]]:
        print("\n\nDoing year", year)
        for region in ["SR", "SB"]:
            study_region(year= year, region=region)
