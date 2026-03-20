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
import src.plotting.helpers as plot_helpers
import src.plotting.helpers_make_plot_dict as plot_helpers_make_plot_dict
import src.plotting.helpers_make_plot as plot_helpers_make_plot

from src.plotting.plots import parse_args, load_config, load_hists, read_axes_and_cuts
import src.plotting.iPlot_config as cfg

np.seterr(divide='ignore', invalid='ignore')

def get_average_over_mixed_data(plotConfig, **kwargs):
    rebin   = kwargs.get("rebin", 1)
    year    = kwargs.get("year", "RunII")
    debug   = kwargs.get("debug", False)
    region   = kwargs.get("region", "SR")
    var    = kwargs.get("var", None)

    process_config = plot_helpers.get_value_nested_dict(cfg.plotConfig, "mix_v0")
    hist_objs = []
    hist_sum = None
    for sub_sample in range(15):
        _proc_config_sub = copy.copy(process_config)
        _proc_config_sub["process"] = f"mix_v{sub_sample}"
        _hist = plot_helpers_make_plot_dict.get_hist_data(process=_proc_config_sub["process"], cfg=cfg, config=_proc_config_sub, var=var, region=region, rebin=rebin, year=year, cut=None)
        hist_objs.append( copy.copy(_hist))

        if hist_sum:
            hist_sum += _hist
        else:
            hist_sum = _hist

    hist_ave = hist_sum * 1/15

    return hist_ave


def plotVar(var, region, year, rebin, yscale, **kwargs):

    #
    # Single plots
    #
    plot_args = {"var":var, "yscale":yscale}
    plot_args = plot_args | {"region":region, "doRatio":1, "rebin":rebin, "year":year}
    plot_args["outputFolder"] = args.outputFolder

#    fig = makePlot(cfg, **plot_args)
#    plt.close()

    cut = None

    #
    #  Plot same but using the Average over the mixed datasets
    #
    plot_data_ave = {}
    plot_data_ave["hists"] = {}
    plot_data_ave["stack"] = {}
    plot_data_ave["ratio"] = {}
    plot_data_ave["var"] = var
    plot_data_ave["cut"] = cut
    plot_data_ave["region"] = region

    #if region == "SR":
    #    ylim = [10, 2_000] if yscale == "log" else [0, 1_250]
    #else:
    #    ylim = [1e-1, 8_000] if yscale == "log" else [0, 3_250]

    plot_data_ave["kwargs"] = kwargs | {"outputFolder" : args.outputFolder} | {"yscale": yscale,
                                                                               "xlim": [0, 1],
                                                                               "rlabel" : 'Data / Model',
                                                                               "ylabel" : 'Events / bin',
                                                                               #"ylim" : ylim,
                                                                               "write_yaml" : True,
                                                                               }


    #
    # Add data
    #
    hist_config_data = copy.copy(plot_helpers.get_value_nested_dict(cfg.plotConfig, "data"))
    plot_helpers_make_plot_dict.add_hist_data(cfg=cfg, config=hist_config_data,
                                              var=var, region=region, cut=cut, rebin=rebin, year=year,
                                              debug=False)

    plot_data_ave["hists"]["data"] = hist_config_data


    #
    # Add Average over mixed to hists
    #
    hist_ave = get_average_over_mixed_data(cfg.plotConfig, var=var, region=region, rebin=rebin, year=year)

    hist_config_ave = copy.copy(plot_helpers.get_value_nested_dict(cfg.plotConfig, "mix_v0"))
    hist_config_ave["name"] = "mix_vAve"
    hist_config_ave["fillcolor"] = "r"
    hist_config_ave["histtype"] = "step"
    hist_config_ave["linewidth"] = 2
    hist_config_ave["label"] = "Average of mixes"
    hist_config_ave["values"]     = hist_ave.values().tolist()
    hist_config_ave["variances"]  = hist_ave.variances().tolist()
    hist_config_ave["centers"]    = hist_ave.axes[0].centers.tolist()
    hist_config_ave["edges"]      = hist_ave.axes[0].edges.tolist()
    hist_config_ave["x_label"]    = '$P_{HH}$'
    hist_config_ave["xlim"]    = [0, 1]
    hist_config_ave["rlabel"]    = 'Data / Model'
    hist_config_ave["under_flow"] = float(hist_ave.view(flow=True)["value"][0])
    hist_config_ave["over_flow"]  = float(hist_ave.view(flow=True)["value"][-1])
    plot_data_ave["hists"]["mix_vAve"] = hist_config_ave
    plot_data_ave["file_name"]  = var+"_ave"

    #
    # Get all the stacks and add to stack_dict
    #
    stack_config = cfg.plotConfig.get("stack", {})
    plot_data_ave["stack"] = plot_helpers_make_plot_dict.load_stack_config(cfg=cfg, stack_config=stack_config, var=var, cut=cut, region=region, rebin=rebin,  **kwargs)

    #
    #  Config Ratios
    #
    ratio_config = copy.deepcopy(cfg.plotConfig["ratios"])
    ratio_config["mixToBkg"]["denominator"]["key"] = "mix_vAve"
    plot_helpers_make_plot_dict.add_ratio_plots(ratio_config, plot_data_ave, **kwargs)

    plot_helpers_make_plot.make_plot_from_dict(plot_data_ave, **kwargs)
    plt.close()



def doPlots(debug=False):
    #
    # A single Mixed model vs 3b vs data
    #
    var_list =  ["SvB_MA_noFvT.ps_hh", "SvB_MA_noFvT.ps_zh", "SvB_MA_noFvT.ps_zz", "SvB_MA_noFvT.ps_hh_fine", "SvB_MA_noFvT.ps_zh_fine", "SvB_MA_noFvT.ps_zz_fine", ]
    rebin_list = [1, 1, 1, 8, 8, 8]

    var_list += ["SvB_noFvT.ps_hh", "SvB_noFvT.ps_zh", "SvB_noFvT.ps_zz", "SvB_noFvT.ps_hh_fine", "SvB_noFvT.ps_zh_fine", "SvB_noFvT.ps_zz_fine", ]
    rebin_list += [1, 1, 1, 8, 8, 8]

    region = "SR"
    year   = "RunII"

    for region_data in [("SB","log"),("SB","linear"),
                        ("SR","log"),("SR","linear")]:
        region = region_data[0]
        yscale = region_data[1]
        for var, rebin in zip(var_list, rebin_list):
            print(f"{region}: {var} (rebin {rebin})" )
            plotVar(var, region, year, rebin, yscale)






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
