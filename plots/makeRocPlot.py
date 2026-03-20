

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
from src.plotting.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args, get_plot_dict_from_config
import src.plotting.iPlot_config as cfg
import src.plotting.helpers as plot_helpers
import copy
np.seterr(divide='ignore', invalid='ignore')


colors = ["blue",
          "DFDAFDSADS",
          "xkcd:pinkish purple",

          "green",
          "red",
          "black",
          "xkcd:light purple"]



def _makeRocPlot(roc_data, **kwargs):

    for _v in roc_data:

        # Example input arrays (replace these with your actual histogram bin contents)
        sig_bins = np.array(roc_data[_v]["sig_values"])
        bkg_bins = np.array(roc_data[_v]["bkg_values"])

        # Create cumulative sums from histograms
        sig_cumsum = np.cumsum(sig_bins[::-1])[::-1]
        bkg_cumsum = np.cumsum(bkg_bins[::-1])[::-1]

        # Compute true positive rate (TPR) and false positive rate (FPR)
        TPR = sig_cumsum / sig_cumsum[0]
        FPR = bkg_cumsum / bkg_cumsum[0]

#        # Append (0,0) and (1,1) to complete the ROC curve
#        TPR = np.append(TPR, 0)
#        FPR = np.append(FPR, 0)
#        TPR = np.insert(TPR, 0, 1)
#        FPR = np.insert(FPR, 0, 1)

        # Compute AUC manually
        roc_auc = np.trapz(TPR, FPR)

        roc_data[_v]["FPR"] = FPR
        roc_data[_v]["TPR"] = TPR
        roc_data[_v]["auc"] = roc_auc

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    for _v in roc_data:
        plt.plot(roc_data[_v]["FPR"], roc_data[_v]["TPR"], color=roc_data[_v]["color"], lw=2, label=f'{roc_data[_v]["label"]} (area = {roc_data[_v]["auc"]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('Background Efficiency')
    plt.ylabel('Signal Efficiency')
    #plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()

    outputFolder = kwargs.get('outputFolder','./')
    if outputFolder:
        plot_name = kwargs.get('plot_name','test')
        plt.savefig(f"{outputFolder}/{plot_name}.pdf")

        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f"{outputFolder}/{plot_name}_log.pdf")
        #plt.show()


def get_signal_plot(plot_data, s):

    try:
        return plot_helpers.get_value_nested_dict(plot_data, s)
    except:
        print(f"Cant find {s}. Assuming its an alternative kl... Will try to construct on the fly")
        return plot_helpers.make_klambda_hist(s, plot_data)

def makeRocPlot(cfg, vars_to_plot, **kwargs):

    roc_data = {}

    for _v in vars_to_plot:

        plot_data = get_plot_dict_from_config(cfg, _v["var"], cut=None, region="SR", **kwargs)

        #
        # Sum signal
        #
        sig_data = None
        for _s in _v["sig"]:
            if sig_data is None:
                sig_data = get_signal_plot(plot_data, _s)
            else:
                _s_data = get_signal_plot(plot_data, _s)
                for _k in ["values", "variances", "under_flow", "over_flow"]:
                    sig_data[_k] += np.array(_s_data[_k])


        #
        # Sum Background
        #
        bkg_data = None
        for _b in _v["bkg"]:
            if bkg_data is None:
                bkg_data = plot_helpers.get_value_nested_dict(plot_data, _b)
            else:
                _b_data = plot_helpers.get_value_nested_dict(plot_data, _b)
                for _k in ["values", "variances", "under_flow", "over_flow"]:
                    bkg_data[_k] += np.array(_b_data[_k])



        #
        # Add underflow
        #
        sig_data["values"][0] += sig_data["under_flow"]
        bkg_data["values"][0] += bkg_data["under_flow"]

        roc_data[_v["name"]] = {"bkg_values": bkg_data["values"],
                        "sig_values": sig_data["values"],
                        "label": _v["name"],
                        "color": _v["color"],
                        }


    _makeRocPlot(roc_data, **kwargs)




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


    makeRocPlot(cfg, plot_name="Test_SvB_MA",
                vars_to_plot=[{"var":"SvB_MA.ps_hh_fine","name":"ps_hh", "color": "blue",   "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps_zz_fine","name":"ps_zz", "color": "orange", "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps_zh_fine","name":"ps_zh", "color": "red",    "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              ],
                outputFolder=cfg.outputFolder)



    makeRocPlot(cfg, plot_name="SvB_vs_SvB_MA",
                vars_to_plot=[{"var":"SvB_MA.ps_hh_fine","name":"SvB_MA", "color": "blue",   "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB.ps_hh_fine",   "name":"SvB",    "color": "orange", "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              ],
                outputFolder=cfg.outputFolder)


    makeRocPlot(cfg, plot_name="SvB_MA_binning",
                vars_to_plot=[{"var":"SvB_MA.ps_hh_fine","name":"hh fine",     "color": "blue",   "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps_hh",     "name":"hh rebinned", "color": "orange", "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              ],
                outputFolder=cfg.outputFolder)


    makeRocPlot(cfg, plot_name="SvB_MA_hh_vs_ps",
                vars_to_plot=[{"var":"SvB_MA.ps_hh_fine","name":"ps hh",  "color": "blue",   "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps",        "name":"ps",     "color": "orange", "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              ],
                outputFolder=cfg.outputFolder)

    makeRocPlot(cfg, plot_name="SvB_MA_hh_vs_phh_hh",
                vars_to_plot=[{"var":"SvB_MA.ps_hh_fine",  "name":"ps hh",  "color": "blue",   "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.phh_hh_fine", "name":"phh hh", "color": "orange", "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              ],
                outputFolder=cfg.outputFolder)


    makeRocPlot(cfg, plot_name="SvB_MA_ps_vs_phh",
                vars_to_plot=[{"var":"SvB_MA.ps",       "name":"ps",  "color": "blue",   "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.phh_fine", "name":"phh", "color": "orange", "sig":["HH4b"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              ],
                outputFolder=cfg.outputFolder)





    #
    #  K-lambda
    #
    cfg.plotConfig = load_config("coffea4bees/plots/metadata/plotsAll_klambda.yml")

    makeRocPlot(cfg, plot_name="SvB_MA_ps_hh_kl",
                vars_to_plot=[{"var":"SvB_MA.ps_hh_fine","name":"k-lambda -5",  "color": "blue",     "sig":["HH4b_kl-5"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps_hh_fine","name":"k-lambda 0",   "color": "orange",   "sig":["HH4b_kl0"],  "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps_hh_fine","name":"k-lambda 1",   "color": "green",    "sig":["HH4b_kl1"],  "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps_hh_fine","name":"k-lambda 3",   "color": "red",      "sig":["HH4b_kl3"],  "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps_hh_fine","name":"k-lambda 10",  "color": "pink",     "sig":["HH4b_kl10"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              ],
                outputFolder=cfg.outputFolder)

    makeRocPlot(cfg, plot_name="SvB_MA_ps_kl",
                vars_to_plot=[{"var":"SvB_MA.ps","name":"k-lambda -5",  "color": "blue",     "sig":["HH4b_kl-5"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps","name":"k-lambda 0",   "color": "orange",   "sig":["HH4b_kl0"],  "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps","name":"k-lambda 1",   "color": "green",    "sig":["HH4b_kl1"],  "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps","name":"k-lambda 3",   "color": "red",      "sig":["HH4b_kl3"],  "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB_MA.ps","name":"k-lambda 10",  "color": "pink",     "sig":["HH4b_kl10"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              ],
                outputFolder=cfg.outputFolder)


    makeRocPlot(cfg, plot_name="SvB_kl",
                vars_to_plot=[{"var":"SvB.ps_hh_fine","name":"k-lambda 1",   "color": "blue",    "sig":["HH4b_kl1"],    "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB.ps_hh_fine","name":"k-lambda 0",   "color": "orange",  "sig":["HH4b_kl0"],    "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB.ps_hh_fine","name":"k-lambda 2.45","color": "green",   "sig":["HH4b_kl2p45"], "bkg":["TTbar","Multijet"],"year":"RunII"},
                              {"var":"SvB.ps_hh_fine","name":"k-lambda 5",   "color": "red",     "sig":["HH4b_kl5"],    "bkg":["TTbar","Multijet"],"year":"RunII"},
                              ],
                outputFolder=cfg.outputFolder)
