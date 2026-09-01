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

def print_counts_yaml(var, cut, region, counts, out_handle):
    out_handle.write(f"{var}_{region}:\n")
    out_handle.write("    var:\n")
    out_handle.write(f"        {var}\n")
    out_handle.write("    cut:\n")
    if cut is not None:
        out_handle.write(f"        {cut}\n")
    out_handle.write("    region:\n")
    out_handle.write(f"           {region}\n")
    out_handle.write("    counts:\n")
    out_handle.write(f"           {list(counts.tolist())}\n")
    out_handle.write("\n\n")


def main():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("-i", "--inputFile", help="Input file")
    parser.add_argument("-o", "--output", help="Output file")
    args = parser.parse_args()

    cfg.plotConfig = load_config_4b("coffea4bees/plots/metadata/plotsAll.yml")
    cfg.hists = load_hists([args.inputFile])
    cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    cfg.set_hist_key("hists")

    default_args = {"doRatio":0, "rebin":4, "norm":0, "process":"Multijet"}

    test_vectors = [("SvB_MA.ps", None, "SR"),
                    ("SvB_MA.ps", None, "SB"),

                    ("SvB_MA.ps_hh", None, "SR"),
                    ("SvB_MA.ps_hh", None, "SB"),

                    ("SvB_MA.ps_zh", None, "SR"),
                    ("SvB_MA.ps_zh", None, "SB"),

                    ("SvB_MA.ps_zz", None, "SR"),
                    ("SvB_MA.ps_zz", None, "SB"),

                    ]

    out_handle = open(args.output, "w") if args.output else sys.stdout

    for tv in test_vectors:

        var    = tv[0]
        cut    = tv[1]
        region = tv[2]
        print(f"testing {var}, {cut}, {region}")
        fig, axes = makePlot(cfg, var=var, cut=cut, axis_opts={"region": region},
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

        print_counts_yaml(var, cut, region, counts, out_handle)
        plt.close()

    if args.output:
        out_handle.close()


if __name__ == "__main__":
    main()
