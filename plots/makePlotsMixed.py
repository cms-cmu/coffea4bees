import os
import time
import sys
import yaml
import hist
import argparse
import tempfile
import copy
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
from coffea.util import load
import numpy as np

sys.path.insert(0, os.getcwd())
from src.plotting.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args, get_value_nested_dict, get_hist, _plot, _savefig, _colors, _plot_ratio
import src.plotting.iPlot_config as cfg

np.seterr(divide='ignore', invalid='ignore')

def doPlots(debug=False):
    region = "SR"
    cut = None
    rebin = 2

    var_to_plot = "SvB_MA_FvT_3bDvTMix4bDvT_vXXX_newSBDef.ps_hh"
    kfold = True
    if kfold:
        var_to_plot = var_to_plot.replace("newSBDef","newSBDefSeedAve")

    process_config = get_value_nested_dict(cfg.plotConfig, "Multijet")
    hist_objs = []
    hist_sum = None
    for sub_sample in range(15):
        _var_to_plot = var_to_plot.replace("vXXX", f"v{sub_sample}")
        _hist = get_hist(cfg, process_config, var=_var_to_plot, region=region, cut=cut, rebin=rebin, year="RunII")
        hist_objs.append( copy.copy(_hist))

        if hist_sum:
            hist_sum += _hist
        else:
            hist_sum = _hist

    hist_ave = hist_sum * 1/15

    n_sub_samples = 8

    hists = []
    for i in range(n_sub_samples):
        hist_config_vX = copy.copy(get_value_nested_dict(cfg.plotConfig, "Multijet"))
        hist_config_vX["name"] = f"bkg_v{i}"
        hist_config_vX["fillcolor"] = _colors[i],
        hist_config_vX["histtype"] = "errorbar"
        hist_config_vX["label"] = f"bkg_v{i}"

        hists.append( (hist_objs[i], hist_config_vX) )

    hist_config_ave = copy.copy(get_value_nested_dict(cfg.plotConfig, "Multijet"))
    hist_config_ave["name"] = "bkg_ave"
    hist_config_ave["label"] = "bkg_ave"
    stack_dict = {}
    stack_dict["bkg_ave"] = ( (hist_ave, hist_config_ave) )

    kwargs = {"year" : "RunII",
              "outputFolder" : args.outputFolder,
              #"histtype" : "errorbar",
              "yscale" : "linear",
              "rlim": [0.8,1.2],
              }

    ratio_plots = []

    denValues = hist_ave.values()

    denValues[denValues == 0] = plot_helpers.EPSILON
    denCenters = hist_ave.axes[0].centers

    for iH, _h in enumerate(hists):
        numValues = _h[0].values()

        ratio_config = {"color": _colors[iH],
                        "marker": "o",
                        }
        ratios, ratio_uncert = plot_helpers.make_ratio(numValues, denValues, **kwargs)
        ratio_plots.append((denCenters, ratios, ratio_uncert, ratio_config))

    fig, main_ax, ratio_ax = _plot_ratio(hists, stack_dict, ratio_plots, **kwargs)
    ax = (main_ax, ratio_ax)

    _savefig(fig, var_to_plot, kwargs.get("outputFolder"), kwargs["year"], cut, "threeTag", region)

    var_list = ["SvB_MA.ps_hh", "SvB_MA.ps_zh", "SvB_MA.ps_zz", "SvB_MA.ps_hh_fine", "SvB_MA.ps_zh_fine", "SvB_MA.ps_zz_fine", ]
    rebin_list = [20, 10, 8, 8, 10, 8]

    for var, rebin in zip(var_list, rebin_list):
        print(f"{var} (rebin {rebin})" )
        var_bkg = var.replace("SvB_MA","SvB_MA_FvT_3bDvTMix4bDvT_v1_newSBDef")
        if kfold:
            var_bkg = var_bkg.replace("newSBDef","newSBDefSeedAve")

        plot_args = {"var":var,
                     "var_over_ride":{"Multijet": var_bkg},
                     }

        plot_args = plot_args | {"region":"SR", "cut":None, "doRatio":1, "rebin":rebin}

        plot_args["outputFolder"] = args.outputFolder

        fig = makePlot(cfg, **plot_args)
        plt.close()

if __name__ == '__main__':

    args = parse_args()

    cfg.plotConfig = load_config(args.metadata)
    cfg.outputFolder = args.outputFolder
    cfg.combine_input_files = args.combine_input_files

    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    doPlots(debug=args.debug)
