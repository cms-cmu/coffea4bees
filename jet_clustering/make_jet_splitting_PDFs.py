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
import re

sys.path.insert(0, os.getcwd())
from coffea4bees.plots.plots import load_config_4b
from src.plotting.plots import load_hists, read_axes_and_cuts, init_arg_parser
import src.plotting.helpers as plot_helpers
from src.plotting.iPlot_config import plot_config
cfg = plot_config()

from coffea4bees.jet_clustering.declustering import get_splitting_summary, get_splitting_name

np.seterr(divide='ignore', invalid='ignore')

# def plot(var, **kwargs):
#     fig, ax = makePlot(cfg, var, outputFolder= args.outputFolder, **kwargs)
#     plt.close()
#     return fig, ax


def write_1D_pdf(output_file, varName, bin_centers, probs, total_counts, n_spaces=4):
    spaces = " " * n_spaces
    output_file.write(f"{spaces}{varName}:\n")
    output_file.write(f"{spaces}    bin_centers:  {bin_centers.tolist()}\n")
    output_file.write(f"{spaces}    probs:  {probs.tolist()}\n")
    output_file.write(f"{spaces}    counts:  {total_counts}\n")


def write_2D_pdf(output_file, varName, hist, n_spaces=4):

    counts = hist.view(flow=False)

    xedges = hist.axes[0].edges
    yedges = hist.axes[1].edges
    probabilities = counts.value / counts.value.sum()

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    probabilities_flat = probabilities.flatten()

    # Hack for empty histograms
    if any(np.isnan(probabilities_flat)):
        probabilities_flat[np.isnan(probabilities_flat)] = 0
        probabilities_flat[0] = 1

    spaces = " " * n_spaces
    output_file.write(f"{spaces}{varName}:\n")
    output_file.write(f"{spaces}    xcenters:  {xcenters.tolist()}\n")
    output_file.write(f"{spaces}    ycenters:  {ycenters.tolist()}\n")
    output_file.write(f"{spaces}    probabilities_flat:  {probabilities_flat.tolist()}\n")


def make_PDFs_vs_Pt(config, output_file_name_vs_pT, year, doBoosted=False):

    splittings = list(config.keys())
    varNames   = list(config[splittings[0]].keys())


    if doBoosted:
        #pt_bins = [300, 533, 766, np.inf]
        pt_bins = [300, 400, 500, 600, 700, 800, 900, np.inf]
    else:
        pt_bins = [0, 140, 230, 320, 410, np.inf]

    with open(output_file_name_vs_pT, 'w') as output_file_vs_pT:

        output_file_vs_pT.write("varNames:\n")
        output_file_vs_pT.write(f"    {varNames}\n\n")

        output_file_vs_pT.write("splittings:\n")
        output_file_vs_pT.write(f"    {splittings}\n\n")

        output_file_vs_pT.write("pt_bins:\n")
        output_file_vs_pT.write(f"    {pt_bins}\n\n")


        for _s in splittings:
            output_file_vs_pT.write(f"\n{_s}:\n")
            print(f"Writing {_s}")
            for _v in varNames:
                var_config = config[_s][_v]
                #splitting_{_s}.{_v}_pT"
                #_hist_name = f"splitting_{_s.replace('/','_')}.{var_config[0]}_pT"
                _hist_name = f"splitting_{_s}.{var_config[0]}_pT"
                #print(f"\t var {_hist_name}")

                output_file_vs_pT.write(f"    {_v}:\n")

                if _v.find("_vs_") == -1:
                    is_1d_hist = True
                    plt.figure(figsize=(6, 6))
                    #x_axis_name = var_config[
                else:
                    is_1d_hist = False
                    plt.figure(figsize=(18, 12))


                for _iPt in range(len(pt_bins) - 1):

                    if doBoosted:
                        cut_dict = plot_helpers.get_cut_dict("passNFatJets", cfg.cutList)
                    else:
                        cut_dict = plot_helpers.get_cut_dict("passPreSel", cfg.cutList)

                    plot_dict = {"process":"data", "year":year, "tag":sum,"region":sum, "pt":_iPt}
                    plot_dict = plot_dict | cut_dict


                    if is_1d_hist:
                        _hist = cfg.hists[0]['hists'][_hist_name][plot_dict]
                        counts = _hist.view(flow=False)
                        bin_edges = _hist.axes[-1].edges

                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        probs = counts.value / counts.value.sum()

                        total_counts = np.sum(counts).value


                        # Hack for empty histograms
                        if any(np.isnan(probs)):
                            probs[np.isnan(probs)] = 0
                            probs[0] = 1


                        write_1D_pdf(output_file_vs_pT, _iPt, bin_centers, probs, total_counts, n_spaces=8)
                    else:

                        hist_to_plot = cfg.hists[0]['hists'][f"{_hist_name}"]
                        _hist = hist_to_plot[plot_dict]

                        write_2D_pdf(output_file_vs_pT, _iPt, _hist, n_spaces=8)




def centers_to_edges(centers):
    bin_width = centers[1] - centers[0]

    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = (centers[1:] + centers[:-1]) / 2
    edges[0] = centers[0] - bin_width / 2
    edges[-1] = centers[-1] + bin_width / 2
    return edges


def get_bins_xMin_xMax_from_centers(centers):
    nBins = len(centers)
    bin_half_width = 0.5*(centers[1]  - centers[0])
    xMin  = centers[0]  - bin_half_width
    xMax  = centers[-1] + bin_half_width

    return nBins, xMin, xMax


def test_PDFs_vs_Pt(config, output_file_name, year):
    splittings = list(config.keys())
    varNames   = list(config[splittings[0]].keys())

    #
    #  test the Pdfs
    #
    with open(output_file_name, 'r') as input_file:

        input_pdfs = yaml.safe_load(input_file)
        nPt_bins = len(input_pdfs["pt_bins"]) - 1

        counts_vs_pt = {}
        total_counts = {}

        for _s in splittings:

            print(f"Doing splitting {_s}")
            total_counts[_s] = 0

            for _v in varNames:
                #print(f"\tDoing var  {_v}")

                if _v.find("_vs_") == -1:
                    is_1d_hist = True
                else:
                    is_1d_hist = False


                if is_1d_hist:


                    for _iPt in range(nPt_bins):

                        # Only read the total counts from mA
                        if _v == "decay_phi":
                            _counts = input_pdfs[_s][_v][_iPt]["counts"]
                            total_counts[_s] +=_counts
                            counts_vs_pt[f"{_s}_{_iPt}"] = _counts


                        probs   = np.array(input_pdfs[_s][_v][_iPt]["probs"],       dtype=float)
                        centers = np.array(input_pdfs[_s][_v][_iPt]["bin_centers"], dtype=float)

                        nBins, xMin, xMax = get_bins_xMin_xMax_from_centers(centers)

                        num_samples = 10000
                        samples = np.random.choice(centers, size=num_samples, p=probs)

                        sample_hist = hist.Hist.new.Reg(nBins, xMin, xMax).Double()
                        sample_hist.fill(samples)

                        sample_pdf  = hist.Hist.new.Reg(nBins, xMin, xMax).Double()
                        sample_pdf[...] = probs * num_samples

                        sample_hist.plot(label="samples")
                        sample_pdf.plot(label="pdf")

                    plt.xlabel(_v)
                    plt.legend()
                    plt.savefig(args.outputFolder+f"/test_sampling_pt_{_s.replace('/','_')}_{_v}.pdf")
                    plt.yscale("log")
                    plt.savefig(args.outputFolder+f"/test_sampling_pt_{_s.replace('/','_')}_{_v}_log.pdf")

                    plt.close()


                else:
                    pass

        ## splittings
        print("Total Counts\n")
        print(total_counts)

        #all_splitting_names = [get_splitting_name(i) for i in splittings]
        #all_splitting_names = set(all_splitting_names) # make unique
        #breakpoint()

        year_str = year
        if year_str == sum:
            year_str = "RunII"
        sorted_counts = dict(sorted(total_counts.items(), key=lambda item: item[1], reverse=True) )
        with open(args.outputFolder+f'/all_splittings_multiplicities_{year_str}.yml', 'w') as splitting_mult_file:
            yaml.dump(sorted_counts, splitting_mult_file, default_flow_style=False)


        with open(args.outputFolder+f"/all_splittings_multiplicities_{year_str}.txt", "w") as splitting_mult_file:
            for k, v, in sorted_counts.items():
                #nJets, nbs = get_splitting_summary(k)

                _s_info = f"{k:25}   {v:10}"
                print(_s_info)
                splitting_mult_file.write(f"{_s_info}\n")


#        #
#        # Now the grouped splittings
#        #
#        total_counts_grouped_splittings = {}
#        for k, v in sorted_counts.items():
#            _split_name = get_splitting_name(k)
#            total_counts_grouped_splittings[_split_name] = total_counts_grouped_splittings.get(_split_name,0) + v
#
#        sorted_counts_grouped = dict(sorted(total_counts_grouped_splittings.items(), key=lambda item: item[1], reverse=True) )
#        with open(args.outputFolder+"/all_grouped_splittings_multiplicities.txt", "w") as splitting_group_mult_file:
#            for k, v, in sorted_counts_grouped.items():
#                _s_info = f"{k:25}   {v:10} "
#                print(_s_info)
#                splitting_group_mult_file.write(f"{_s_info}\n")
#

        #print("Total Counts vs pt\n")
        #print(counts_vs_pt)
        #counts_vs_pt[f"{_s}_{_iPt}"] = _counts





def doPlots(year, doBoosted=False, debug=False):

    #
    #  config Setup
    #
    splitting_config = {}

    zA_mA_mB         = { "mA":("mA_r",    1),  "mB":("mB_r",    1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_vs_thetaA",   1), "rhoA": ("rhoA", 1), "rhoB": ("rhoB", 1)}
    zA_l_mA_mB       = { "mA":("mA_r",    1),  "mB":("mB_r",    1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_l_vs_thetaA", 1), "rhoA": ("rhoA", 1), "rhoB": ("rhoB", 1)}
    #zA_mA_mB         = { "mA":("mA",    1),  "mB":("mB",    1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_vs_thetaA",   1), "rhoA": ("rhoA", 1), "rhoB": ("rhoB", 1)}
    #zA_l_mA_mB       = { "mA":("mA",    1),  "mB":("mB",    1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_l_vs_thetaA", 1), "rhoA": ("rhoA", 1), "rhoB": ("rhoB", 1)}
    zA_l_mA_l_mB     = { "mA":("mA_l",  1),  "mB":("mB",    1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_l_vs_thetaA", 1), "rhoA": ("rhoA", 1), "rhoB": ("rhoB", 1)}
    zA_mA_l_mB_l     = { "mA":("mA_l",  1),  "mB":("mB_l",  1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_vs_thetaA",   1), "rhoA": ("rhoA", 1), "rhoB": ("rhoB", 1)}
    zA_l_mA_l_mB_l   = { "mA":("mA_l",  1),  "mB":("mB_l",  1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_l_vs_thetaA", 1), "rhoA": ("rhoA", 1), "rhoB": ("rhoB", 1)}
    zA_l_mA_vl_mB    = { "mA":("mA_vl", 1),  "mB":("mB",    1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_l_vs_thetaA", 1), "rhoA": ("rhoA", 1), "rhoB": ("rhoB", 1)}
    zA_l_mA_vl_mB_vl = { "mA":("mA_vl", 1),  "mB":("mB_vl", 1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_l_vs_thetaA", 1), "rhoA": ("rhoA", 1), "rhoB": ("rhoB", 1)}


    #
    # Define the regex pattern
    #
    #pattern = r'[01]b[01]j(/[01]b[01]j)?'

    patterns = { '1b0j/1b0j' : zA_mA_mB,
                 '0b1j/0b1j' : zA_mA_mB,
                 "1b0j/0b1j" : zA_l_mA_mB,

                 "1b1j/1b0j" : zA_l_mA_l_mB,
                 "0b2j/0b1j" : zA_l_mA_l_mB,

                 "0b2j/1b0j" : zA_l_mA_l_mB,
                 "1b1j/0b1j" : zA_l_mA_l_mB,

                 "0b2j/0b2j" : zA_l_mA_l_mB_l,
                 "1b1j/1b1j" : zA_l_mA_l_mB_l,

                 "1b1j/0b2j" : zA_l_mA_l_mB_l,

                 "1b2j/1b0j" : zA_l_mA_l_mB,
                 "0b3j/0b1j" : zA_l_mA_l_mB,

                 "1b2j/0b1j" : zA_l_mA_l_mB,
                 "0b3j/1b0j" : zA_l_mA_l_mB,

                 "4/1" : zA_l_mA_vl_mB,
                 "X/1" : zA_l_mA_vl_mB,

                 "4/2" : zA_l_mA_vl_mB_vl,
                 "3/3" : zA_l_mA_vl_mB_vl,
                 "X/2" : zA_l_mA_vl_mB_vl,
                 "3/2" : zA_l_mA_vl_mB_vl,
                 "X/X" : zA_l_mA_vl_mB_vl,

                }

    # If doing boosted
    if doBoosted:
         patterns["1b0j/1b0j"] = zA_l_mA_l_mB_l


    #
    #  Get All splittings
    #
    all_splittings = [i.replace("splitting_","").replace(".pt_l","").replace("_","/") for i in cfg.hists[0]["hists"].keys() if not i.find("pt_l") == -1 and i.find("detailed") == -1]

    unconfig_splitting = []


    for _s in all_splittings:

        any_match = False

        for _p, _hists in patterns.items():
            if re.fullmatch(_p, _s):
                print(f"Matched: {_s}")
                any_match = True
                splitting_config[_s]    = _hists
            #else:
            #    print(f"Not matched: {_s}")

        if not any_match:
            unconfig_splitting.append(_s)
            #print(f" {_s} unmatched")

    n_splittings = len(all_splittings)
    n_configured_splittings = len(splitting_config.keys())

    print(f" Total Splittings {n_splittings}")
    print(f"   nConfigured    {n_configured_splittings}")

    if len(unconfig_splitting):
        print(f"Unconfigured splittings are {unconfig_splitting}")

    #print(len(splitting_config.)

    #
    #splitting_config["(bj)(bj)"] = s_XX_XX
    #
    #splitting_config["((jj)b)b"] = s_XX_X_X
    #splitting_config["((bj)j)b"] = s_XX_X_X

    year_str = year
    if year_str == sum:
        year_str = "RunII"

    with open(args.outputFolder+f"/all_splittings_{year_str}.txt", "w") as splitting_out_file:
        all_splittings.sort(reverse=True)
        for _s in all_splittings:

            splitting_out_file.write(f"{_s}\n")

    output_file_name_vs_pT = args.outputFolder+f"/clustering_pdfs_vs_pT_{year_str}.yml"
    make_PDFs_vs_Pt(splitting_config, output_file_name_vs_pT, year, doBoosted=doBoosted)
    test_PDFs_vs_Pt(splitting_config, output_file_name_vs_pT, year)




if __name__ == '__main__':

    parser = init_arg_parser()
    parser.add_argument('--years', default=["RunII"], nargs="+",help='years to process.')
    parser.add_argument('--doBoosted', action="store_true", help='If set, will do the boosted declustering PDFs.')

    args = parser.parse_args()
    print(f" Doing years {args.years}")

    cfg.plotConfig = load_config_4b(args.metadata)
    cfg.outputFolder = args.outputFolder

    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    cfg.set_hist_key("hists")

    #varList = [ h for h in cfg.hists[0].keys() if not h in args.skip_hists ]



    years = args.years

    if years == ["Run3"]:
        years = ["2022_preEE", "2022_EE", "2023_preBPix", "2023_BPix"]


    if years in [ ["Run2"], ["RunII"]]:
        years = ["UL18", "UL17", "UL16_preVFP", "UL16_postVFP"]

    if args.doTest:
        years = [sum]

    for y in years:
        doPlots(year=y, doBoosted=args.doBoosted, debug=args.debug)
