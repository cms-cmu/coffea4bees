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

    outputFile.write(f"{'_'.join([var,cut,region])}:\n")
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

    test_vectors = [("SvB_MA.ps", "passPreSel", "region_SR"),
                    ("SvB_MA.ps", "passPreSel", "region_SB"),

                    ("SvB_MA.ps_hh", "passPreSel", "region_SR"),
                    ("SvB_MA.ps_hh", "passPreSel", "region_SB"),

                    ("SvB_MA.ps_zh", "passPreSel", "region_SR"),
                    ("SvB_MA.ps_zh", "passPreSel", "region_SB"),

                    ("SvB_MA.ps_zz", "passPreSel", "region_SR"),
                    ("SvB_MA.ps_zz", "passPreSel", "region_SB"),

                    ]

    for tv in test_vectors:

        var    = tv[0]
        cut    = tv[1]
        region = tv[2]
        print(f"testing {var}, {cut}, {region}")
        fig, axes = makePlot(cfg, var=var, cut=cut, region=region,
                             outputFolder=cfg.outputFolder, **default_args)

        ax = axes[0]
        for i in range(len(ax.lines)):

            if hasattr(ax.lines[i], "get_label") and ax.lines[i].get_label() == '_nolegend_':
                counts = ax.lines[i].get_ydata()
                break

        print_counts_yaml(var, cut, region, counts)
        plt.close()
