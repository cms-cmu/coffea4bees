# From
# https://github.com/patrickbryant/ZZ4b/blob/master/nTupleAnalysis/scripts/makeAccxEff.py
#


import os
import time
import sys
import yaml
import hist
import argparse
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
from coffea.util import load
import numpy as np
import yaml
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl

sys.path.insert(0, os.getcwd())
from coffea4bees.plots.plots import load_config_4b
from src.plotting.plots import makePlot, make2DPlot, load_hists, read_axes_and_cuts, parse_args
from src.plotting.iPlot_config import plot_config
cfg = plot_config()

import src.plotting.helpers as plot_helpers
import copy
np.seterr(divide='ignore', invalid='ignore')


colors = ["DFDAFDSADS",
          "xkcd:pinkish purple",
          "blue",
          "green",
          "red",
          "black",
          "xkcd:light purple"]


cuts_flow = [("passCleanGenWeight",0, "Denominator"),
             ("passJetMult",0, "$\geq4$ selected jets" ), # purple
             ("passPreSel_woTrig",1, "4 b-tagged jets"),  # blue
             ("passDiJetMass_woTrig",1, "m(j,j)"), # green
             ("SR_woTrig",1 , "Signal region"),     #
             ("SR",1, "Trigger")
             ]



def _get_hist_data(input_hist):
    config = {}
    config["values"]     = input_hist.values().tolist()
    #config["variances"]  = input_hist.variances().tolist()
    config["variances"]  = input_hist.values().tolist()
    config["centers"]    = input_hist.axes[0].centers.tolist()
    config["edges"]      = input_hist.axes[0].edges.tolist()
    config["x_label"]    = input_hist.axes[0].label
    #config["under_flow"] = float(input_hist.view(flow=True)["value"][0])
    #config["over_flow"]  = float(input_hist.view(flow=True)["value"][-1])
    return config


def get_hist_data(in_file, hist_name, hist_key, rebin):
    input_hist = None
    for hk in hist_key:
        if input_hist is None:
            input_hist = copy.deepcopy(in_file['cutflow_hists'][hk][hist_name])
        else:
            input_hist += in_file['cutflow_hists'][hk][hist_name]

    input_hist = input_hist[::hist.rebin(rebin)]

    return _get_hist_data(input_hist)


def _makeMhhPlot(name, data_to_plot, output_dir, **kwargs):
    size = 7
    fig = plt.figure()   # figsize=(size,size/_phi))
    fig.add_axes((0.1, 0.15, 0.85, 0.8))
    main_ax = fig.gca()
    ratio_ax = None

    year_str = plot_helpers.get_year_str(kwargs.get("year","Run2"))

    hep.cms.label("Internal", data=False,
                  year=year_str, loc=0, ax=main_ax)

    bin_centers = np.array(data_to_plot["centers"])
    bin_width = bin_centers[1] - bin_centers[0]
    bin_centers = np.array(data_to_plot["centers"])

    bin_edges = [bin_centers[0] - bin_width / 2] + [center + bin_width / 2 for center in bin_centers]

    signal_shape = np.array(data_to_plot["values"])
    signal_shape_norm = np.sum(signal_shape)
    signal_shape /= signal_shape_norm

    #plt.bar(bin_centers, signal_shape, width=bin_width, align='center', color="gray", edgecolor=None, alpha=0.3)

    plt.hist(
        bin_centers,
        bins=bin_edges,
        weights=signal_shape,
        histtype='stepfilled',
        edgecolor='gray',
        color="gray",
        alpha=0.3,
        linewidth=2,
        label="Inclusive HH signal shape",
    )

    plt.hist(
        bin_centers,
        bins=bin_edges,
        weights=signal_shape,
        histtype='step',
        edgecolor='gray',
        linewidth=2,
        #label="Inclusive HH signal shape",
    )

    #plt.text(350, 1.25e-3, 'Inclusive HH signal shape (Normalization Arbitrary)', fontsize=20, color='k', ha='left', va='bottom')



    xlim = kwargs.get("xlim",[200, 1200])
    plt.plot(xlim, [1,1], color="k", linestyle=":")
    plt.xlim(xlim)
    plt.ylim(kwargs.get("ylim",[0,1.3]))
    plt.yscale(kwargs.get("yscale","linear"))
    plt.xlabel("$m_{4b}^{gen}$ [GeV]")
    plt.ylabel("Normalized")
    plt.legend(loc="upper left", ncol=2,
               bbox_to_anchor=(0.025, .975), # Moves the legend outside and centers it
               fontsize = "large"
               )


    plt.savefig(f"{output_dir}/{name}_{year_str}.pdf")



def makeMhhPlots(cfg, year, output_dir):

    process = "GluGluToHHTo4B_cHHH1"

    if year in ["Run2","RunII"]:
        hist_key = [f"{process}_{y}" for y in ["UL18", "UL17", "UL16_preVFP", "UL16_postVFP"]]
    else:
        hist_key = [f"{process}_{year}"]

    rebin = 1

    tot_eff = {}
    rel_eff = {}

    data_all   = get_hist_data(cfg.hists[0], "all", hist_key, rebin)
    data_clean = get_hist_data(cfg.hists[0], "passCleanGenWeight", hist_key, rebin)

    _makeMhhPlot("mHH_all",   data_all,   year=year, output_dir=output_dir, ylim=[1e-3, 1e-1], xlim=[200, 800])
    _makeMhhPlot("mHH_clean", data_clean, year=year, output_dir=output_dir, ylim=[1e-3, 1e-1], xlim=[200, 800])
    #_makeMhhPlot("mHH_clean", data_clean, year=year, output_dir=output_dir, ylim=[1e-3, 1e-1], xlim=[200, 800])



def makeEffPlot(name, data_to_plot, cuts_flow, output_dir, **kwargs):
    size = 7
    fig = plt.figure()   # figsize=(size,size/_phi))
    fig.add_axes((0.1, 0.15, 0.85, 0.8))
    main_ax = fig.gca()
    ratio_ax = None

    year_str = plot_helpers.get_year_str(kwargs.get("year","Run2"))

    hep.cms.label("Internal", data=False,
                  year=year_str, loc=0, ax=main_ax)


    for ic in range(1,len(cuts_flow)):
        cut_name = cuts_flow[ic][0]

        plot_mask = (np.array(data_to_plot[cut_name]["centers"]) > 200)
        plt.plot(np.array(data_to_plot[cut_name]["centers"])[plot_mask],
                 np.array(data_to_plot[cut_name]["ratio"])[plot_mask]
                 , marker='o', markersize=8, linestyle='-', linewidth=2, color=colors[ic], label=cuts_flow[ic][2])


        #print("errors",0.5* np.sum(data_to_plot[cut_name]["error"], axis=0)[plot_mask])
        #plt.errorbar(
        #    np.array(data_to_plot[cut_name]["centers"])[plot_mask],
        #    np.array(data_to_plot[cut_name]["ratio"])[plot_mask],
        #    yerr = 0.5* np.sum(data_to_plot[cut_name]["error"], axis=0)[plot_mask],
        #    fmt='o',             # marker style (same as marker='o')
        #    color=colors[ic],
        #    linestyle='-',       # connect points with a line
        #    ecolor=colors[ic],  # color for error bars
        #    capsize=2            # length of the caps on the error bars
        #)

    if kwargs.get("signal_shape", None):
        bin_centers = np.array(data_to_plot[cut_name]["centers"])[plot_mask]
        bin_width = bin_centers[1] - bin_centers[0]
        bin_centers = np.array(data_to_plot[cut_name]["centers"])[plot_mask]

        bin_edges = [bin_centers[0] - bin_width / 2] + [center + bin_width / 2 for center in bin_centers]

        signal_shape = np.array(kwargs.get("signal_shape", None))[plot_mask]
        signal_shape_norm = np.sum(signal_shape)
        signal_shape /= signal_shape_norm

        #plt.bar(bin_centers, signal_shape, width=bin_width, align='center', color="gray", edgecolor=None, alpha=0.3)

        plt.hist(
            bin_centers,
            bins=bin_edges,
            weights=signal_shape,
            histtype='stepfilled',
            edgecolor='gray',
            color="gray",
            alpha=0.3,
            linewidth=2
        )

        plt.hist(
            bin_centers,
            bins=bin_edges,
            weights=signal_shape,
            histtype='step',
            edgecolor='gray',
            linewidth=2
        )

        plt.text(350, 1.25e-3, 'Inclusive HH signal shape (Normalization Arbitrary)', fontsize=20, color='k', ha='left', va='bottom')



    xlim = kwargs.get("xlim",[200, 1200])
    plt.plot(xlim, [1,1], color="k", linestyle=":")
    plt.xlim(xlim)
    plt.ylim(kwargs.get("ylim",[0,1.3]))
    plt.yscale(kwargs.get("yscale","linear"))
    plt.xlabel("$m_{4b}^{gen}$ [GeV]")
    plt.ylabel("Acceptance x Efficiency")
    plt.legend(loc="upper left", ncol=2,
               bbox_to_anchor=(0.025, .975), # Moves the legend outside and centers it
               fontsize = "large"
               )

    plt.savefig(f"{output_dir}/{name}_{year_str}.pdf")





def calculate_ratios(num_data, den_data, thisSF):
    """Calculate ratios and uncertainties for efficiency plots.

    Args:
        num_data: Numerator data
        den_data: Denominator data
        thisSF: Scale factor to apply

    Returns:
        Tuple of (ratios, ratio_uncertainties)
    """
    ratios, ratio_uncert = plot_helpers.make_ratio(
        np.array(num_data["values"]) * thisSF,
        np.array(num_data["variances"]) * thisSF**2,
        np.array(den_data["values"]),
        np.array(den_data["variances"])
    )
    return ratios, ratio_uncert

def calculate_total_ratios(num_data, den_data, thisSF):
    """Calculate total ratios and uncertainties for efficiency plots.

    Args:
        num_data: Numerator data
        den_data: Denominator data
        thisSF: Scale factor to apply

    Returns:
        Tuple of (ratios, ratio_uncertainties)
    """
    ratios_tot, ratio_tot_uncert = plot_helpers.make_ratio(
        np.array(num_data["values"]) * thisSF,
        np.array(num_data["variances"]) * thisSF**2,
        np.array(den_data["values"]),
        np.array(den_data["variances"])
    )
    return ratios_tot, ratio_tot_uncert

def makePlot(cfg, year, output_dir, debug=False):


    process = "GluGluToHHTo4B_cHHH1"

    if year in ["Run2","RunII"]:
        hist_key = [f"{process}_{y}" for y in ["UL18", "UL17", "UL16_preVFP", "UL16_postVFP"]]
    else:
        hist_key = [f"{process}_{year}"]

    rebin = 10

    tot_eff = {}
    rel_eff = {}

    den_tot_data = get_hist_data(cfg.hists[0], cuts_flow[0][0], hist_key, rebin)

    #
    # Compute the SF between two input files
    #
    data_0 = get_hist_data(cfg.hists[0], "SR_woTrig", hist_key, rebin)
    data_1 = get_hist_data(cfg.hists[1], "SR_woTrig", hist_key, rebin)

    print(f"\tnorm file0 {np.sum(data_0['values'])} vs file1: {np.sum(data_1['values'])} ratio {np.sum(data_0['values'])/np.sum(data_1['values'])}")

    scalefactor = np.sum(data_0['values'])/np.sum(data_1['values'])

    for ic in range(1,len(cuts_flow)):

        #print(ic, cuts_flow[ic])
        #print(year)
        den_cut_name = cuts_flow[ic - 1][0]
        den_file_idx = cuts_flow[ic - 1][1]
        den_data = get_hist_data(cfg.hists[den_file_idx], den_cut_name, hist_key, rebin)

        num_cut_name = cuts_flow[ic][0]
        num_file_idx = cuts_flow[ic][1]
        num_data = get_hist_data(cfg.hists[num_file_idx], num_cut_name, hist_key, rebin)

        #if ic == 5:
        #    print("\tnum:", num_data["values"])
        #    print("\tden:", den_data["values"])

        #
        # Relative Efficiencies
        #
        thisSF = 1.0
        if not den_file_idx == num_file_idx:
            thisSF = scalefactor

        ratios, ratio_uncert = calculate_ratios(num_data, den_data, thisSF)
        rel_eff[cuts_flow[ic][0]] = {"ratio":ratios, "error":ratio_uncert, "centers":num_data["centers"]}

        #
        # Total Efficiencies
        #
        if not num_file_idx == 0:
            thisSF = scalefactor

        ratios_tot, ratio_tot_uncert = calculate_total_ratios(num_data, den_data, thisSF)
        tot_eff[cuts_flow[ic][0]] = {"ratio":ratios_tot, "error":ratio_tot_uncert, "centers":num_data["centers"]}



    #
    #
    #
    makeEffPlot("total_eff",    tot_eff, cuts_flow, output_dir=output_dir, yscale="log", year=year, ylim=[1e-3, 10], signal_shape=den_tot_data["values"])
    makeEffPlot("relative_eff", rel_eff, cuts_flow, output_dir=output_dir, year=year)


    return




if __name__ == '__main__':

    args = parse_args()

    cfg.plotConfig = load_config_4b(args.metadata)
    cfg.outputFolder = args.outputFolder
    cfg.combine_input_files = args.combine_input_files
    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    cfg.set_hist_key("hists")

    for y in ["UL18", "UL17","UL16_preVFP", "UL16_postVFP", "RunII"]:
    #for y in ["UL18"]:
        makePlot(cfg, year=y, debug=args.debug, output_dir=cfg.outputFolder)

        makeMhhPlots(cfg, year=y, output_dir=cfg.outputFolder)
