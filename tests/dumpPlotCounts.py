import argparse
from coffea.util import load
import sys
import os
import numpy as np
sys.path.insert(0, os.getcwd())
from coffea4bees.plots.plots import load_config_4b
from src.plotting.plots import makePlot, load_hists, read_axes_and_cuts
import src.plotting.helpers as plot_helpers
from src.plotting.iPlot_config import plot_config
cfg = plot_config()
import matplotlib.pyplot as plt

def print_counts_yaml(var, cut, region, counts):

    key_parts = [var] + ([cut] if cut is not None else []) + [region]
    outputFile.write(f"{'_'.join(key_parts)}:\n")
    outputFile.write(f"    var:\n")
    outputFile.write(f"        {var}\n")
    outputFile.write(f"    cut:\n")
    outputFile.write(f"        {cut}\n")
    outputFile.write(f"    region:\n")
    outputFile.write(f"           {region}\n")
    outputFile.write(f"    counts:\n")
    outputFile.write(f"           {counts.tolist()}\n")
    outputFile.write("\n\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-o','--outputFile', default='knownCounts.yml', help='Input File. Default: hists.pkl')
    args = parser.parse_args()

    outputFile = open(f'{args.outputFile}', 'w')

    metadata = "coffea4bees/plots/metadata/plotsAll.yml"
    cfg.plotConfig = load_config_4b(metadata)
    cfg.hists = load_hists([args.inputFile])
    cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    cfg.set_hist_key("hists")

    default_args = {"doRatio":0, "rebin":4, "norm":0, "process":"Multijet"}

    test_vectors = [("SvB_MA.ps", None, "region_SR"),
                    ("SvB_MA.ps", None, "region_SB"),

                    ("SvB_MA.ps_hh", None, "region_SR"),
                    ("SvB_MA.ps_hh", None, "region_SB"),

                    ("SvB_MA.ps_zh", None, "region_SR"),
                    ("SvB_MA.ps_zh", None, "region_SB"),

                    ("SvB_MA.ps_zz", None, "region_SR"),
                    ("SvB_MA.ps_zz", None, "region_SB"),

                    ]

    for tv in test_vectors:

        var    = tv[0]
        cut    = tv[1]
        region = tv[2]
        print(f"testing {var}, {cut}, {region}")
        fig, axes = makePlot(cfg, var=var, cut=cut, region=region,
                             outputFolder=cfg.outputFolder, **default_args)

        ax = axes[0]
        counts = np.array([])
        for line in ax.lines:
            if hasattr(line, "get_label") and line.get_label() == '_nolegend_':
                counts = line.get_ydata()
                break
        if len(counts) == 0 and len(ax.lines) > 0:
            counts = ax.lines[0].get_ydata()
        if len(counts) == 0 and len(ax.patches) > 0:
            for patch in ax.patches:
                if hasattr(patch, "get_data"):
                    counts = np.array(patch.get_data()[0])
                    break

        print_counts_yaml(var, cut, region, counts)
        plt.close()
