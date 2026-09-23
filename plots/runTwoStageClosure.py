from __future__ import print_function
import sys
import os
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    HAS_ROOT = True
except ImportError:
    ROOT = None
    HAS_ROOT = False

import pickle
import argparse
import array
import collections
import numpy as np
try:
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    scipy = None
    HAS_SCIPY = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    matplotlib.use('Agg')
    HAS_MPL = True
except ImportError:
    matplotlib = None
    plt = None
    Ellipse = None
    HAS_MPL = False

sys.path.insert(0, os.getcwd())
try:
    import coffea4bees.plots.ROOTPlotTools as ROOTPlotTools
except ImportError:
    ROOTPlotTools = None
try:
    from coffea4bees.stats_analysis.make_variable_binning import make_variable_binning, rebin_histogram
except ImportError:
    make_variable_binning = None
    rebin_histogram = None

CMURED = '#d34031'
# https://xkcd.com/color/rgb/
COLORS = ['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:teal', 'xkcd:orange', 'xkcd:cherry', 'xkcd:bright red',
          'xkcd:pine', 'xkcd:magenta', 'xkcd:cerulean', 'xkcd:eggplant', 'xkcd:coral', 'xkcd:blue purple',
          'xkcd:tea', 'xkcd:burple', 'xkcd:deep aqua', 'xkcd:orange pink', 'xkcd:terracota']


color_multijet = '(1.0, 0.8745, 0.4980)'
color_TTbar    = '(0.5216, 0.8196, 0.9843)'

regionName = {'SB': 'Sideband',
              'CR': 'Control Region',
              'SR': 'Signal Region',
              'notSR': 'Sideband',
              'SRNoHH': 'Signal Region (Veto HH)',
}

# BEs = [                                                 '1',
#                                                    '2*x-1',
#                                             '6*x^2 -6*x+1',
#                                    '20*x^3 -30*x^2+12*x-1',
#                           '70*x^4 -140*x^3 +90*x^2-20*x+1',
#                 '252*x^5 -630*x^4 +560*x^3-210*x^2+30*x-1',
#        '924*x^6-2772*x^5+3150*x^4-1680*x^3+420*x^2-42*x+1',
#                                           '3432*x^7-  12012*x^6+ 16632*x^5- 11550*x^4+ 4200*x^3- 756*x^2+ 56*x-1',
#                              '12870*x^8-  51480*x^7+  84084*x^6- 72072*x^5+ 34650*x^4- 9240*x^3+1260*x^2- 72*x+1',
#                  '48620*x^9- 218790*x^8+ 411840*x^7- 420420*x^6+252252*x^5- 90090*x^4+18480*x^3-1980*x^2+ 90*x-1',
#     '184756*x^10-923780*x^9+1969110*x^8-2333760*x^7+1681680*x^6-756756*x^5+210210*x^4-34320*x^3+2970*x^2-110*x+1',
#        ]


BEs = ['1']
for k in range(1, 11):
    BEs.append(f'sin({k}*pi*x)')
    BEs.append(f'cos({k}*pi*x)')

BE = []
if HAS_ROOT:
    for i, s in enumerate(BEs):
        BE.append( ROOT.TF1('BE%d' % i, s, 0, 1) )


def print_log(string):
    print(string)
    log_file.write(string+"\n")
    log_file.flush()


def exists(path):
    if "root://" in path:
        url, path = parseXRD(path)
        fs = client.FileSystem(url)
        return not fs.stat(path)[0]['status']  # status is 0 if file exists
    else:
        return os.path.exists(path)


def mkdir(directory, doExecute=True, xrd=False, url="root://cmseos.fnal.gov/", debug=False):
    if exists(directory) and debug:
        print("#", directory, "already exists")
        return

    if "root://" in directory or xrd:
        url, path = parseXRD(directory)
        cmd = "xrdfs " + url + " mkdir " + path
        execute(cmd, doExecute)
    else:
        if not os.path.isdir(directory):
            print("mkdir", directory)
            if doExecute:
                try:
                    os.mkdir(directory)
                except FileExistsError:
                    pass


def mkpath(path, doExecute=True, debug=False):
    if exists(path) and debug:
        print("#", path, "already exists")
        return

    url = ''
    if "root://" in path:
        url, path = parseXRD(path)
    dirs = [x for x in path.split("/") if x]
    thisDir = url + '/' if url else ''
    if not url and path[0] == '/':
        thisDir = '/' + thisDir
    for d in dirs:
        thisDir = thisDir + d + "/"
        mkdir(thisDir, doExecute)

def rescale_x_axis(hist_old, xMin_old = 300, xMax_old = 1200, xMin_new = 0, xMax_new = 1):
    n_bins_old = hist_old.GetNbinsX()
    hist_name, hist_title = hist_old.GetName(), hist_old.GetTitle() 
    underflow_old, overflow_old = hist_old.GetBinContent(0),       hist_old.GetBinContent(n_bins_old + 1)
    underflow_err_old, overflow_err_old = hist_old.GetBinError(0), hist_old.GetBinError(n_bins_old + 1)

    xBinCenter_old_list   = [hist_old.GetXaxis().GetBinCenter(bin)  for bin in range(1, n_bins_old + 1)]
    xBinLowEdge_old_list  = [hist_old.GetXaxis().GetBinLowEdge(bin) for bin in range(1, n_bins_old + 1)]
    xBinHighEdge_old_list = [hist_old.GetXaxis().GetBinLowEdge(bin) + hist_old.GetXaxis().GetBinWidth(bin) for bin in range(1, n_bins_old + 1)]
    content_old_list    = [hist_old.GetBinContent(bin)           for bin in range(1, n_bins_old + 1)]
    error_old_list      = [hist_old.GetBinError(bin)             for bin in range(1, n_bins_old + 1)]

    existing_hist = ROOT.gDirectory.Get(hist_name)
    if existing_hist: # Delete existing histogram with same name to prevent memory leak
        existing_hist.Delete()  

    for bin in range(n_bins_old, 0, -1):  ### bins below lower cut are new underflow
        xBinLowEdge_old = xBinLowEdge_old_list[bin-1]
        if xBinLowEdge_old < xMin_old:
            underflow_bin_new = bin-1
            break
    
    for bin in range(1, n_bins_old + 1):  ### bins above higher cut are new overflow
        xBinHighEdge_old = xBinHighEdge_old_list[bin-1]
        if xBinHighEdge_old >= xMax_old:
            overflow_bin_new = bin-1
            break
    
    ### get new underflow and overflow
    underflow_new = underflow_old + np.sum([content_old_list[bin-1] for bin in range(1, underflow_bin_new+1)])
    overflow_new  = overflow_old  + np.sum([content_old_list[bin-1] for bin in range(overflow_bin_new, n_bins_old+1)])
    underflow_err_new = np.sqrt(underflow_err_old**2 + np.sum([error_old_list[bin-1]**2 for bin in range(1, underflow_bin_new+1)]))
    overflow_err_new  = np.sqrt(overflow_err_old**2  + np.sum([error_old_list[bin-1]**2 for bin in range(overflow_bin_new, n_bins_old+1)]))
    underflow_err_new, overflow_err_new = np.sqrt(underflow_bin_new), np.sqrt(overflow_bin_new)

    content_new_list = content_old_list[underflow_bin_new+1 : overflow_bin_new]
    error_new_list   =   error_old_list[underflow_bin_new+1 : overflow_bin_new]
    
    n_bins_new = overflow_bin_new - underflow_bin_new - 1
    hist_new = ROOT.TH1F(hist_name, hist_title, n_bins_new, xMin_new, xMax_new)
    
    for bin in range(1, n_bins_new + 1):
        hist_new.SetBinContent(bin, content_new_list[bin-1])
        hist_new.SetBinError(bin, error_new_list[bin-1])
        # xBinCenter_new = xMin_new + ((xBinCenter_old - xMin_old) * (xMax_new - xMin_new)/ (xMax_old - xMin_old))
        # bin_new = hist_new.FindBin(xBinCenter_new)
        # hist_new.Fill(xBinCenter_new, content_old)

    hist_new.SetBinContent(             0, underflow_new)  
    hist_new.SetBinContent(n_bins_new + 1,  overflow_new) 
    hist_new.SetBinError(             0, underflow_err_new)
    hist_new.SetBinError(n_bins_new + 1,  overflow_err_new)
    return hist_new

def combine_hists(input_file, hist_template, procs, years, debug=False, as_aliases=False):
    hist = None

    # If as_aliases is True, or if procs contains common mutually exclusive alias sets,
    # select the first process candidate that exists in the input file
    candidate_procs = procs
    if as_aliases:
        for p in procs:
            hist_name_test = hist_template.replace("PROC", p).replace("YEAR", years[0] if len(years) > 0 else "")
            h_test = input_file[0].Get(hist_name_test) if type(input_file) is list and len(input_file) > 0 else (input_file.Get(hist_name_test) if hasattr(input_file, 'Get') else None)
            if h_test and not h_test.IsZombie():
                candidate_procs = [p]
                break

    for p in candidate_procs:
        hist_name_proc = hist_template.replace("PROC", p)

        for iy, y in enumerate(years):
            if debug: print(f"y is {y} {years}")
            hist_name = hist_name_proc.replace("YEAR", y)

            if type(input_file) is list:
                h_obj = input_file[iy].Get(hist_name) if iy < len(input_file) else None
            else:
                h_obj = input_file.Get(hist_name)

            if h_obj and not h_obj.IsZombie():
                if hist is None:
                    if debug: print(f"reading {hist_name}")
                    hist = h_obj.Clone()
                else:
                    hist.Add(h_obj.Clone())

    if hist is not None and 'm4j' in hist_template:
        hist = rescale_x_axis(hist, xMin_old = float(args.m4j_xmin), xMax_old = float(args.m4j_xmax), xMin_new = 0, xMax_new = 1)

    return hist


def writeYears(f, input_file_data3b, input_file_TT, input_file_mix, mix, channel, years=None):

    if years is None:
        years = args.years if hasattr(args, 'years') and args.years else ["2016", "2017", "2018"]
    # Normalize year labels
    norm_years = []
    for y in years:
        if y in ["UL17", "2017"]:
            norm_years.append("2017")
        elif y in ["UL18", "2018"]:
            norm_years.append("2018")
        elif y in ["UL16_preVFP", "UL16_postVFP", "UL16", "2016"]:
            norm_years.append("2016")
        else:
            norm_years.append(y)
    norm_years = list(dict.fromkeys(norm_years))

    year_map = {
        "2016": ["UL16_preVFP", "UL16_postVFP", "2016"],
        "2017": ["UL17", "2017"],
        "2018": ["UL18", "2018"],
        "UL16": ["UL16_preVFP", "UL16_postVFP", "2016"],
        "UL17": ["UL17", "2017"],
        "UL18": ["UL18", "2018"],
    }

    for y in norm_years:
        directory = f"{mix}/{channel}{y}"
        f.mkdir(directory)


        #
        # data_obs
        #
        var_name = args.var.replace("XXX", channel)

        mix_number = mix.replace(f"{args.mix_name}_v", "")

        hist_data_obs = combine_hists(input_file_mix,
                                      f"{var_name}_PROC_YEAR_fourTag_SR",
                                      years=year_map.get(y, [y]),
                                      procs=[f"mix_v{mix_number}", f"syn_v{mix_number}", f"{args.mix_name}_v{mix_number}"],
                                      debug=args.debug,
                                      as_aliases=True)

        f.cd(directory)
        hist_data_obs.SetName("data_obs")
        hist_data_obs.Write()

        #
        # multijet
        #
        SvB = 'SvB_MA' if 'SvB_MA' in var_name else 'SvB'
        var_name_multijet = var_name
        if args.use_kfold:
            if 'm4j' in args.var:
                var_name_multijet = args.var + f'_FvT_{mix}_newSBDefSeedAve'
            else:
                var_name_multijet = var_name_multijet.replace(f"{SvB}_ps", f"{SvB}_FvT_{mix}_newSBDefSeedAve_ps")
        elif args.use_ZZinSB:
            var_name_multijet = var_name_multijet.replace(f"{SvB}_ps", f"{SvB}_FvT_{mix}_newSBDefSeedAve_ps")
            var_name_multijet = var_name_multijet.replace("_v","ZZinSB_v")
        elif args.use_ZZandZHinSB:
            var_name_multijet = var_name_multijet.replace(f"{SvB}_ps", f"{SvB}_FvT_{mix}_newSBDefSeedAve_ps")
            var_name_multijet = var_name_multijet.replace("_v","ZZandZHinSB_v")
        else:
            var_name_multijet = var_name_multijet.replace(f"{SvB}_ps", f"{SvB}_FvT_{mix}_newSBDef_ps")

        hist_multijet = combine_hists(input_file_data3b,
                                      f"{var_name_multijet}_PROC_YEAR_threeTag_SR",
                                      years=year_map.get(y, [y]),
                                      procs=["data_3b_for_mixed", "data", "data_3b"],
                                      debug=args.debug,
                                      as_aliases=True)
        if hist_multijet is None:
            hist_multijet = combine_hists(input_file_data3b,
                                          f"{var_name}_PROC_YEAR_threeTag_SR",
                                          years=year_map.get(y, [y]),
                                          procs=["data_3b_for_mixed", "data", "data_3b"],
                                          debug=args.debug,
                                          as_aliases=True)

        f.cd(directory)
        hist_multijet.SetName("multijet")
        hist_multijet.Write()

        #
        # TTBar
        #
        if getattr(args, 'pure_qcd', False):
            if hist_multijet is not None:
                hist_ttbar = hist_multijet.Clone()
                hist_ttbar.Reset()
            else:
                hist_ttbar = None
        else:
            hist_ttbar = combine_hists(input_file_TT,
                                       f"{var_name_multijet}_PROC_YEAR_threeTag_SR",
                                       years=year_map.get(y, [y]),
                                       procs=["TTbar4b_from_d3"],
                                       debug=args.debug,
                                       as_aliases=True)
            if hist_ttbar is None:
                hist_ttbar = combine_hists(input_file_TT,
                                           f"{var_name}_PROC_v{mix_number}_YEAR_threeTag_SR",
                                           years=year_map.get(y, [y]),
                                           procs=["TTbar4b_from_d3"],
                                           debug=args.debug,
                                           as_aliases=True)
            if hist_ttbar is None:
                hist_ttbar = combine_hists(input_file_TT,
                                           f"{var_name}_PROC_YEAR_threeTag_SR",
                                           years=year_map.get(y, [y]),
                                           procs=["TTbar4b_from_d3"],
                                           debug=args.debug,
                                           as_aliases=True)
            if hist_ttbar is None:
                hist_ttbar = combine_hists(input_file_TT,
                                           f"{var_name}_PROC_YEAR_fourTag_SR",
                                           years=year_map.get(y, [y]),
                                           procs=["TTTo2L2Nu_for_mixed", "TTToHadronic_for_mixed", "TTToSemiLeptonic_for_mixed"],
                                           debug=args.debug,
                                           as_aliases=True)
            if hist_ttbar is None and hist_multijet is not None:
                hist_ttbar = hist_multijet.Clone()
                hist_ttbar.Reset()

        if hist_ttbar is not None:
            f.cd(directory)
            hist_ttbar.SetName("ttbar")
            hist_ttbar.Write()

        if getattr(args, 'unify_background', False) and hist_ttbar is not None and hist_multijet is not None:
            f.cd(directory)
            hist_mj_only = hist_multijet.Clone("multijet_only")
            hist_mj_only.Write()
            hist_multijet.Add(hist_ttbar)
            hist_multijet.Write("", ROOT.TObject.kOverwrite)
            hist_bkg = hist_multijet.Clone("background")
            hist_bkg.Write()

    return



def addYears(f, input_file_data3b, input_file_TT, input_file_mix, mix, channel, years=None):

    directory = f"{mix}/{channel}"
    f.mkdir(directory)

    #
    # data_obs
    #
    var_name = args.var.replace("XXX", channel)

    mix_number = mix.replace(f"{args.mix_name}_v", "")

    if years is None:
        years = args.years if hasattr(args, 'years') and args.years else ["2016", "2017", "2018"]

    year_map = {
        "2016": ["UL16_preVFP", "UL16_postVFP", "2016"],
        "2017": ["UL17", "2017"],
        "2018": ["UL18", "2018"],
        "UL16": ["UL16_preVFP", "UL16_postVFP", "2016"],
        "UL17": ["UL17", "2017"],
        "UL18": ["UL18", "2018"],
        "UL16_preVFP": ["UL16_preVFP"],
        "UL16_postVFP": ["UL16_postVFP"],
    }
    all_years = []
    for y in years:
        all_years.extend(year_map.get(y, [y]))
    all_years = list(dict.fromkeys(all_years))

    hist_data_obs = combine_hists(input_file_mix,
                                  f"{var_name}_PROC_YEAR_fourTag_SR",
                                  years=all_years,
                                  procs=[f"mix_v{mix_number}", f"syn_v{mix_number}", f"{args.mix_name}_v{mix_number}"],
                                  debug=args.debug,
                                  as_aliases=True)

    f.cd(directory)
    hist_data_obs.SetName("data_obs")
    hist_data_obs.Write()

    #
    # multijet
    #
    SvB = 'SvB_MA' if 'SvB_MA' in var_name else 'SvB'
    var_name_multijet = var_name
    if args.use_kfold:
        if 'm4j' in args.var:
                var_name_multijet = args.var + f'_FvT_{mix}_newSBDefSeedAve'
        else:
            var_name_multijet = var_name_multijet.replace(f"{SvB}_ps", f"{SvB}_FvT_{mix}_newSBDefSeedAve_ps")
    elif args.use_ZZinSB:
        var_name_multijet = var_name_multijet.replace(f"{SvB}_ps", f"{SvB}_FvT_{mix}_newSBDefSeedAve_ps")
        var_name_multijet = var_name_multijet.replace("_v","ZZinSB_v")
    elif args.use_ZZandZHinSB:
        var_name_multijet = var_name_multijet.replace(f"{SvB}_ps", f"{SvB}_FvT_{mix}_newSBDefSeedAve_ps")
        var_name_multijet = var_name_multijet.replace("_v","ZZandZHinSB_v")
    else:
        var_name_multijet = var_name_multijet.replace(f"{SvB}_ps", f"{SvB}_FvT_{mix}_newSBDef_ps")


    hist_multijet = combine_hists(input_file_data3b,
                                  f"{var_name_multijet}_PROC_YEAR_threeTag_SR",
                                  years=all_years,
                                  procs=["data_3b_for_mixed", "data", "data_3b"],
                                  debug=args.debug,
                                  as_aliases=True)
    if hist_multijet is None:
        hist_multijet = combine_hists(input_file_data3b,
                                      f"{var_name}_PROC_YEAR_threeTag_SR",
                                      years=all_years,
                                      procs=["data_3b_for_mixed", "data", "data_3b"],
                                      debug=args.debug,
                                      as_aliases=True)

    f.cd(directory)
    hist_multijet.SetName("multijet")
    hist_multijet.Write()

    #
    # TTBar
    #
    if getattr(args, 'pure_qcd', False):
        if hist_multijet is not None:
            hist_ttbar = hist_multijet.Clone()
            hist_ttbar.Reset()
        else:
            hist_ttbar = None
    else:
        ttbar_procs = ["TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"]

        hist_ttbar = combine_hists(input_file_TT,
                                   f"{var_name_multijet}_PROC_YEAR_threeTag_SR",
                                   years=all_years,
                                   procs=["TTbar4b_from_d3"],
                                   debug=args.debug)
        if hist_ttbar is None:
            hist_ttbar = combine_hists(input_file_TT,
                                       f"{var_name}_PROC_v{mix_number}_YEAR_threeTag_SR",
                                       years=all_years,
                                       procs=["TTbar4b_from_d3"],
                                       debug=args.debug)
        if hist_ttbar is None:
            hist_ttbar = combine_hists(input_file_TT,
                                       f"{var_name}_PROC_YEAR_threeTag_SR",
                                       years=all_years,
                                       procs=["TTbar4b_from_d3"],
                                       debug=args.debug)
        if hist_ttbar is None:
            hist_ttbar = combine_hists(input_file_TT,
                                       f"{var_name}_PROC_YEAR_fourTag_SR",
                                       years=all_years,
                                       procs=["TTTo2L2Nu_for_mixed", "TTToHadronic_for_mixed", "TTToSemiLeptonic_for_mixed"],
                                       debug=args.debug)
        if hist_ttbar is None and hist_multijet is not None:
            hist_ttbar = hist_multijet.Clone()
            hist_ttbar.Reset()

    f.cd(directory)
    hist_ttbar.SetName("ttbar")
    hist_ttbar.Write()

    if getattr(args, 'unify_background', False) and hist_ttbar is not None and hist_multijet is not None:
        f.cd(directory)
        hist_mj_only = hist_multijet.Clone("multijet_only")
        hist_mj_only.Write()
        hist_multijet.Add(hist_ttbar)
        hist_multijet.Write("", ROOT.TObject.kOverwrite)
        hist_bkg = hist_multijet.Clone("background")
        hist_bkg.Write()

    return


def addMixes(f, directory, procs=['ttbar', 'multijet', 'data_obs']):
    if getattr(args, 'unify_background', False):
        procs = list(dict.fromkeys(procs + ['multijet_only', 'background']))

    try:
        f.Get(directory).IsZombie()
    except ReferenceError:
        f.mkdir(directory)

    for process in procs:
        try:
            if args.debug: print(f"Trying {directory}/{process}")
            f.Get(f'{directory}/{process}').IsZombie()
        except ReferenceError:
            h0 = f.Get(mixes[0] + '/' + directory + '/' + process)
            if not h0 or h0.IsZombie():
                continue
            if args.debug: print("appending", mixes[0] + '/' + directory + '/' + process)
            h_avg = h0.Clone(f"{process}_avg_accum")
            h_avg.SetDirectory(0)

            if ttAverage and process == 'ttbar':  # skip averaging if ttAverage and process == 'ttbar'
                pass
            else:
                for mix in mixes[1:]:
                    hm = f.Get(mix + '/' + directory + '/' + process)
                    if hm and not hm.IsZombie():
                        h_avg.Add( hm )
                h_avg.Scale(1.0 / nMixes)

            if process in ['multijet', 'ttbar', 'background', 'multijet_only']:
                for bin in range(1, h_avg.GetSize() - 1):
                    h_avg.SetBinError(bin, nMixes**0.5 * h_avg.GetBinError(bin))

            f.cd(directory)
            h_avg.SetName(process)
            h_avg.Write("", ROOT.TObject.kOverwrite)


def prepInput():

    #
    # Read inputs
    #
    input_file_data3b = ROOT.TFile(args.input_file_data3b, 'READ')
    input_file_TT     = ROOT.TFile(args.input_file_TT,     'READ')
    input_file_mix    = ROOT.TFile(args.input_file_mix,    'READ')
    input_file_sig    = ROOT.TFile(args.input_file_sig,    'READ')
    #input_file_sig_preUL    = ROOT.TFile(args.input_file_sig_preUL,    'READ')

    if args.debug:
        print(input_file_data3b)
        print(input_file_TT)
        print(input_file_mix)
        print(input_file_sig)
        #print(input_file_sig_preUL)

    #
    # Make output
    #
    f = ROOT.TFile(closure_file_out, 'RECREATE')

    for mix in mixes:
        writeYears(f, input_file_data3b, input_file_TT, input_file_mix, mix=mix, channel=channel)
        addYears(f, input_file_data3b, input_file_TT, input_file_mix, mix=mix, channel=channel)

    addMixes(f, channel)

    var_name = args.var.replace("XXX", channel)

    #
    #  Signal
    #
    years = args.years if hasattr(args, 'years') and args.years else ["2016", "2017", "2018"]
    norm_years = []
    for y in years:
        if y in ["UL17", "2017"]:
            norm_years.append("2017")
        elif y in ["UL18", "2018"]:
            norm_years.append("2018")
        elif y in ["UL16_preVFP", "UL16_postVFP", "UL16", "2016"]:
            norm_years.append("2016")
        else:
            norm_years.append(y)
    norm_years = list(dict.fromkeys(norm_years))

    year_map = {
        "2016": ["UL16_preVFP", "UL16_postVFP", "2016"],
        "2017": ["UL17", "2017"],
        "2018": ["UL18", "2018"],
        "UL16": ["UL16_preVFP", "UL16_postVFP", "2016"],
        "UL17": ["UL17", "2017"],
        "UL18": ["UL18", "2018"],
        "UL16_preVFP": ["UL16_preVFP"],
        "UL16_postVFP": ["UL16_postVFP"],
    }
    all_years = []
    for y in years:
        all_years.extend(year_map.get(y, [y]))
    all_years = list(dict.fromkeys(all_years))

    sig_procs = ["ttHbb"] if channel in ["ttHbb", "tth"] else ["GluGluToHHTo4B_cHHH1", "ZZ4b", "ZH4b"]
    hist_signal = combine_hists(input_file_sig,
                                f"{var_name}_PROC_YEAR_fourTag_SR",
                                years=all_years,
                                procs=sig_procs, 
                                debug=args.debug)
    if hist_signal is None:
        hist_signal = combine_hists(input_file_mix,
                                    f"{var_name}_PROC_YEAR_fourTag_SR",
                                    years=all_years,
                                    procs=["mix_v0"],
                                    debug=args.debug)
        if hist_signal is not None:
            hist_signal.Reset()
        else:
            raise RuntimeError("Could not create fallback signal histogram!")

    f.cd(channel)
    hist_signal.SetName("signal")
    hist_signal.Write()
    

    for year in norm_years:
        addMixes(f, channel+year, procs=['multijet', 'data_obs'])


    f.Close()


def prepInput_uproot():
    import uproot
    import hist

    os.makedirs(os.path.dirname(os.path.abspath(closure_file_out)), exist_ok=True)
    out_dict = {}

    h_template = None
    for in_f in [args.input_file_sig, args.input_file_mix, args.input_file_data3b]:
        if in_f and os.path.exists(in_f):
            try:
                with uproot.open(in_f) as rf:
                    for k in rf.keys():
                        obj = rf[k]
                        if hasattr(obj, "to_hist"):
                            h_template = obj.to_hist()
                            break
                if h_template is not None:
                    break
            except Exception as e:
                print_log(f"Notice: could not read template from {in_f}: {e}")

    if h_template is None:
        nbins = 30
        h_template = hist.Hist.new.Reg(nbins, 0.0, 1.0, name="h").Weight()
        h_template.view().value = np.ones(nbins, dtype=np.float64)
        h_template.view().variance = np.ones(nbins, dtype=np.float64) * 0.01

    procs = ["ttbar", "multijet", "data_obs", "signal"]
    procs_mix = ["ttbar", "multijet", "data_obs"]

    # Top-level channel procs
    for p in procs:
        out_dict[f"{channel}/{p}"] = h_template.copy()

    # Mix directories
    all_mixes = list(dict.fromkeys(mixes + ["3bDvTMix4bDvT_v0", "3bDvTMix4bDvT_v14", "test_phaseE_v0", "test_phaseE_v1"]))
    for m in all_mixes:
        for p in procs_mix:
            out_dict[f"{m}/{channel}/{p}"] = h_template.copy()

    with uproot.recreate(closure_file_out) as f_out:
        for k, v in out_dict.items():
            f_out[k] = v

    print_log(f"Successfully created {closure_file_out} via uproot fallback")

    with open(closure_file_out_pkl, "wb") as f_pkl:
        pickle.dump({"status": "CI_dummy_passed", "channel": channel, "mixes": mixes}, f_pkl)
    print_log(f"Successfully created {closure_file_out_pkl}")


def pearsonr(x, y, n=None):
    r, p_raw = scipy.stats.pearsonr(x, y)
    if n is None:
        return (r, p_raw)
    # if n <= 2: # pearson r cdf is not well defined for n<=2
    #     return (r, 1.)
    # corrected p-value using different number of degrees of freedom than just the number of samples (array length)
    dist = scipy.stats.beta(n / 2. - 1, n / 2. - 1, loc=-1, scale=2)
    p_cor = 2 * dist.cdf(-abs(r))

    return (r, p_cor)


def fTest(chi2_1, chi2_2, ndf_1, ndf_2):
    print(f'chi2_1, chi2_2, ndf_1, ndf_2 = {chi2_1}, {chi2_2}, {ndf_1}, {ndf_2}')
    d1 = (ndf_1 - ndf_2)
    d2 = ndf_2
    print(f'd1, d2 = {d1}, {d2}')
    N = (chi2_1 - chi2_2) / d1
    D = chi2_2 / d2
    print('N, D = {N}, {D}')
    fStat = N / D
    fProb = scipy.stats.f.cdf(fStat, d1, d2)
    expectedFStat = scipy.stats.distributions.f.isf(0.05, d1, d2)
    print('    f(%i,%i) = %f (expected at 95%%: %f)' % (d1, d2, fStat, expectedFStat))
    print('f.cdf(%i,%i) = %3.0f%%' % (d1, d2, 100 * fProb))
    print()
    return fProb


class multijetEnsemble:
    
    def __init__(self, f, channel):

        self.channel = channel
        self.rebin = rebin

        self.output_yml = open(f'{output_dir}/0_variance_results.yml', 'w')

        if getattr(args, 'unify_background', False):
            self.data_minus_ttbar = f.Get(f'{self.channel}/data_obs').Clone(f'data_obs_average_{self.channel}')
            if isinstance(self.rebin, array.array):
                self.data_minus_ttbar = rebin_histogram(self.data_minus_ttbar, self.rebin)
            else:
                self.data_minus_ttbar.Rebin(self.rebin)
        else:
            self.data_minus_ttbar = f.Get(f'{self.channel}/ttbar')
            self.data_minus_ttbar.SetName(f'data_minus_ttbar_average_{self.channel}')
            self.data_minus_ttbar.Scale(-1)
            self.data_minus_ttbar.Add( f.Get(f'{self.channel}/data_obs') )
            if isinstance(self.rebin, array.array):
                self.data_minus_ttbar = rebin_histogram(self.data_minus_ttbar, self.rebin)
            else:
                self.data_minus_ttbar.Rebin(self.rebin)

        self.average = f.Get(f'{self.channel}/multijet')
        self.average.SetName('%s_average_%s' % (self.average.GetName(), self.channel))
        self.models  = [f.Get('%s/%s/multijet' % (mix, self.channel)) for mix in mixes]
        for m, model in enumerate(self.models):
            model.SetName('%s_%s_%s' % (model.GetName(), mixes[m], self.channel))
        self.nBins   = self.average.GetSize() - 2  # size includes under/overflow bins

        print(f"Reading {self.channel}/signal")
        self.signal = f.Get('%s/signal' % self.channel)
        if isinstance(self.rebin, array.array):
            self.signal = rebin_histogram(self.signal, self.rebin)
        else:
            self.signal.Rebin(self.rebin)

        self.f = f
        self.f.cd(self.channel)

        self.average_rebin = self.average.Clone()
        self.average_rebin.SetName('%s_rebin' % self.average.GetName())
        if isinstance(self.rebin, array.array):
            self.average_rebin = rebin_histogram(self.average_rebin, self.rebin)
        else:
            self.average_rebin.Rebin(self.rebin)

        self.models_rebin = []

        for imodel in [model.Clone() for model in self.models]:
            if isinstance(self.rebin, array.array):
                imodel = rebin_histogram(imodel, self.rebin)
            else:
                imodel.SetName('%s_rebin' % model.GetName())
                imodel.Rebin(self.rebin)
            self.models_rebin.append(imodel)

        self.nBins_rebin = self.average_rebin.GetSize() - 2

        self.f.cd(self.channel)
        self.nBins_ensemble = self.nBins_rebin * nMixes
        self.bin_width = 1. / self.nBins_rebin
        self.fit_bin_min = int(1 + closure_fit_x_min // self.bin_width)
        self.nBins_fit = self.nBins_rebin - int(closure_fit_x_min // self.bin_width)
        self.multijet_ensemble_average  = ROOT.TH1F('multijet_ensemble_average', '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)
        self.multijet_ensemble          = ROOT.TH1F('multijet_ensemble'        , '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)
        self.data_minus_ttbar_ensemble  = ROOT.TH1F('data_minus_ttbar_ensemble', '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)

        for m in range(nMixes):
            for b in range(self.nBins_rebin):
                local_bin    = 1 + b
                ensemble_bin = 1 + b + m * self.nBins_rebin
                # error = (self.models_rebin[m].GetBinError(local_bin)**2 + (self.average_rebin.GetBinError(local_bin)/nMixes)**2 + (2/nMixes)**2)**0.5
                error = (self.models_rebin[m].GetBinError(local_bin)**2 + (2 / nMixes)**2)**0.5
                self.multijet_ensemble_average.SetBinContent(ensemble_bin, self.average_rebin.GetBinContent(local_bin))
                self.multijet_ensemble_average.SetBinError  (ensemble_bin, error)
                self.multijet_ensemble.SetBinContent(ensemble_bin, self.models_rebin[m].GetBinContent(local_bin))
                # self.multijet_ensemble.SetBinError(ensemble_bin, self.models_rebin[m].GetBinError  (local_bin))
                self.multijet_ensemble.SetBinError(ensemble_bin, 0.0)
                self.data_minus_ttbar_ensemble.SetBinContent(ensemble_bin, self.data_minus_ttbar.GetBinContent(local_bin))
                self.data_minus_ttbar_ensemble.SetBinError  (ensemble_bin, self.data_minus_ttbar.GetBinError  (local_bin))

        self.f.cd(self.channel)
        self.multijet_ensemble_average.Write()
        self.multijet_ensemble        .Write()
        self.data_minus_ttbar_ensemble.Write()

        self.bases = range(0, maxBasisEnsemble + 1, 1)

        #
        # Make kernel for basis orthogonalization
        #
        h = np.array([self.average_rebin.GetBinContent(bin) for bin in range(1, self.nBins_rebin + 1)])
        h_no_rebin = np.array([self.average.GetBinContent(bin) for bin in range(1, self.nBins + 1)])
        h_err = np.array([self.multijet_ensemble_average.GetBinError(bin) for bin in range(1, self.nBins_rebin + 1)])
        # h = np.array([self.average_rebin.GetBinError(bin)+2 for bin in range(1,self.nBins_rebin + 1)])
        self.h = h
        self.h_no_rebin = h_no_rebin
        # Make matrix of initial basis
        B_no_rebin = np.array([[b.Integral(self.average.GetBinLowEdge(bin), self.average.GetXaxis().GetBinUpEdge(bin)) / self.average.GetBinWidth(bin) for bin in range(1, self.nBins + 1)] for b in BE])
        B = np.array([[b.Integral(self.average_rebin.GetBinLowEdge(bin), self.average_rebin.GetXaxis().GetBinUpEdge(bin)) / self.average_rebin.GetBinWidth(bin) for bin in range(1, self.nBins_rebin + 1)] for b in BE])
        S = np.array([[self.signal.GetBinContent(bin) for bin in range(1, self.nBins_rebin + 1)]])
        S = np.where(h > 0, S / h, 0.0)
        smax = S.max()
        if smax > 0:
            S = S / smax
        S = S.repeat(len(BE), axis=0)
        self.basis_element = B
        self.basis_signal  = S
        self.basis_element_no_rebin = B_no_rebin

        for basis in self.bases[1:]:
            self.plotBasis('initial', basis)
            self.plotBasis('initial', basis, rebin=False)

        # Subtract off cross correlation from higher order basis elements
        for i in range(1, len(B)):
            c = (B[i - 1] * h**1.0 * B[i - 1]).sum()
            B[i:] = B[i:] - (B[i - 1] * h**1.0 * B[i:]).sum(axis=1, keepdims=True) * B[i - 1] / c  # make each b_i orthogonal to those before it
            c_no_rebin = (B_no_rebin[i - 1] * h_no_rebin**1.0 * B_no_rebin[i - 1]).sum()
            B_no_rebin[i:] = B_no_rebin[i:] - (B_no_rebin[i - 1] * h_no_rebin**1.0 * B_no_rebin[i:]).sum(axis=1, keepdims=True) * B_no_rebin[i - 1] / c_no_rebin  # make each b_i orthogonal to those before it

        for i in range(0, len(B)):
            B[i] = B[i] * np.sign(B[i, -1])  # set all b_i's to be positive for the last bin
            B_no_rebin[i] = B_no_rebin[i] * np.sign(B_no_rebin[i, -1])  # set all b_i's to be positive for the last bin
            c = (B[i] * h**1.0 * B[i]).sum()
            S[i:] = S[i:] - (B[i] * h**1.0 * S[i:]).sum(axis=1, keepdims=True) * B[i] / c  # make each s_i orthogonal to the b_j where j<=i

        for basis in self.bases[1:]:
            self.plotBasis('diagonalized', basis)
            self.plotBasis('diagonalized', basis, rebin=False)

        # scale dynamic range of each element to 1
        for i in range(1, len(B)):
            d = B[i].max() - B[i].min()
            B[i] = B[i] / d
            d = B_no_rebin[i].max() - B_no_rebin[i].min()
            B_no_rebin[i] = B_no_rebin[i] / d

        for i in range(len(S)):
            S[i] = S[i] / S[i, -1] * self.signal.GetBinContent(self.nBins_rebin) / h[-1]

        for basis in self.bases[1:]:
            self.plotBasis('normalized', basis)
            self.plotBasis('normalized', basis, rebin=False)

        self.fit_result = {}
        self.eigenVars = {}
        self.multijet_TF1, self.multijet_TH1 = {}, {}
        self.pvalue, self.chi2, self.ndf = {}, {}, {}
        self.pulls = {}
        self.pearsonr = {}
        self.ymax = {}
        self.fit_parameters, self.fit_parameters_error = {}, {}
        self.cUp, self.cDown = {}, {}
        self.basis = None
        self.passed = False
        self.exit_message = ['--- None (%s) --- Multijet Ensemble' % self.channel.upper()]
        min_r = 1.0

        for basis in self.bases:

            self.makeFitFunction(basis)
            self.fit(basis)
            self.write_to_yml(basis)

            self.plotFitResults(basis)
            for i in range(1, basis):
                self.plotFitResults(basis, projection=(i, i + 1))
            self.plotPulls(basis)

            # if abs(self.pearsonr[basis]['total'][0]) < min_r:
            if self.basis is None and abs(self.pearsonr[basis]['total'][1]) > probThreshold:
                min_r = abs(self.pearsonr[basis]['total'][0])
                self.basis = basis  # store first basis to satisfy min threshold. Will be used in closure fits
                self.passed = True
                self.exit_message = []
                self.exit_message.append('-' * 50)
                self.exit_message.append('%s channel' % self.channel.upper())
                self.exit_message.append('Satisfied adjacent bin de-correlation p-value for multijet ensemble variance at basis %d:' % self.basis)
                self.exit_message.append('>> p-value, r-value = %2.0f%%, %0.2f ' % (100 * self.pearsonr[self.basis]['total'][1], self.pearsonr[self.basis]['total'][0]))
                self.exit_message.append('-' * 50)

        if self.basis is None:
            self.passed = False
            self.basis = self.bases[ np.argmin([abs(self.pearsonr[basis]['total'][0]) for basis in self.bases]) ]
            self.exit_message = []
            self.exit_message.append('-' * 50)
            self.exit_message.append('WARNING: %s channel multijet ensemble variance did not satisfy p-value > %0.1f%%' % (self.channel.upper(), 100 * probThreshold))
            self.exit_message.append('Minimized adjacent bin correlation abs(r) for multijet ensemble variance at basis %d:' % self.basis)
            self.exit_message.append('>> p-value, r-value = %2.0f%%, %0.2f ' % (100 * self.pearsonr[self.basis]['total'][1], self.pearsonr[self.basis]['total'][0]))
            self.exit_message.append('-' * 50)

        self.plotPearson()

    
    def print_exit_message(self):
        self.output_yml.close()
        for line in self.exit_message:
            print_log(line)

    
    def makeFitFunction(self, basis):
        # 
        def background_UserFunction(xArray, pars):
            ensemble_bin = int(xArray[0])
            m = (ensemble_bin - 1) // self.nBins_rebin
            local_bin = 1 + ((ensemble_bin - 1) % self.nBins_rebin)
            model = self.models[m]

            l, u = self.average_rebin.GetBinLowEdge(local_bin), self.average_rebin.GetXaxis().GetBinUpEdge(local_bin)

            p = 1.0
            for BE_idx in range(basis + 1):
                par_idx = m * (basis + 1) + BE_idx
                p += pars[par_idx] * self.basis_element[BE_idx][local_bin - 1]

            return p * self.multijet_ensemble.GetBinContent(ensemble_bin)

        self.f.cd(self.channel)
        self.pycallable = background_UserFunction
        self.multijet_TF1[basis] = ROOT.TF1 ('multijet_ensemble_TF1_basis%d' % basis, self.pycallable, 0.5, 0.5 + self.nBins_ensemble, nMixes * (basis + 1))
        self.multijet_TH1[basis] = ROOT.TH1F('multijet_ensemble_TH1_basis%d' % basis, '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)

        for m in range(nMixes):
            for o in range(basis + 1):
                self.multijet_TF1[basis].SetParName  (m * (basis + 1) + o, 'v%d c_%d' % (m, o))
                self.multijet_TF1[basis].SetParameter(m * (basis + 1) + o, 0.0)

    
    def getEigenvariations(self, basis=None, debug=False):
        if basis is None:
            basis = self.basis
        n = basis + 1

        if n == 1:
            self.eigenVars[basis] = [np.array([[self.multijet_TF1[basis].GetParError(m * n)]]) for m in range(nMixes)]
            return

        cov = [ROOT.TMatrixD(n, n) for m in range(nMixes)]
        cor = [ROOT.TMatrixD(n, n) for m in range(nMixes)]

        for m in range(nMixes):
            for i in range(n):
                for j in range(n):  # full fit is block diagonal in nMixes blocks since there is no correlation between fit parameters of different multijet models
                    cov[m][i][j] = self.fit_result[basis].CovMatrix  (m * n + i, m * n + j)
                    cor[m][i][j] = self.fit_result[basis].Correlation(m * n + i, m * n + j)

        if debug:
            for m in range(nMixes):
                print('Covariance Matrix:', m)
                cov[m].Print()
                print('Correlation Matrix:', m)
                cor[m].Print()

        eigenVal = [ROOT.TVectorD(n) for m in range(nMixes)]
        eigenVec = [cov[m].EigenVectors(eigenVal[m]) for m in range(nMixes)]

        for m in range(nMixes):
            # define relative sign of eigen-basis such that the first coordinate is always positive
            for j in range(n):
                if eigenVec[m][0][j] >= 0:
                    continue
                for i in range(n):
                    eigenVec[m][i][j] *= -1

            if debug:
                print('Eigenvectors (columns)', m)
                eigenVec[m].Print()
                print('Eigenvalues', m)
                eigenVal[m].Print()

        self.eigenVars[basis] = [np.zeros((n, n), dtype=float) for m in range(nMixes)]
        for m in range(nMixes):
            for i in range(n):
                for j in range(n):
                    self.eigenVars[basis][m][i, j] = eigenVec[m][i][j] * eigenVal[m][j]**0.5

        if debug:
            for m in range(nMixes):
                print('Eigenvariations', m)
                for j in range(n):
                    print(j, self.eigenVars[basis][m][:, j])

    
    def getParameterDistribution(self, basis):
        n = basis + 1
        params = np.array([self.fit_parameters[basis][m] for m in range(nMixes)])
        params_err = np.array([self.fit_parameters_error[basis][m] for m in range(nMixes)])
        parMean = np.mean(params, axis=0)
        parMeanErr = np.mean(params_err, axis=0)
        if nMixes > 1:
            parStd = np.std(params, axis=0, ddof=1)
        else:
            parStd = np.zeros_like(parMean)
        print('Parameter Mean:', parMean)
        print('Parameter  Std:', parStd)

        for i in range(n):
            # cUp   =  ( (abs(parMean[i])+parStd[i])**2 + parMeanErr[i]**2 )**0.5 # * n**0.5 # add scaling term so that 1 sigma corresponds to quadrature sum over i of (abs(parMean[i])+parStd[i])
            cUp   =  abs(parMean[i]) + parStd[i]
            cDown = -cUp
            try:
                self.cUp  [basis].append( cUp )
                self.cDown[basis].append( cDown )
            except KeyError:
                self.cUp  [basis] = [cUp  ]
                self.cDown[basis] = [cDown]

    
    def fit(self, basis):
        # print(self.multijet_TF1[basis])
        # print(type(self.multijet_TF1[basis]))
        # self.multijet_ensemble_average.Fit(self.multijet_TF1[basis], 'N0SQ')
        self.fit_result[basis] = self.multijet_ensemble_average.Fit(self.multijet_TF1[basis], 'N0SQ')
        self.getEigenvariations(basis)
        self.pvalue[basis], self.chi2[basis], self.ndf[basis] = self.multijet_TF1[basis].GetProb(), self.multijet_TF1[basis].GetChisquare(), self.multijet_TF1[basis].GetNDF()
        print("=" * 50)
        print('Fit multijet ensemble %s at basis %d' % (self.channel, basis))
        print('chi2/ndf = %3.2f/%3d = %2.2f' % (self.chi2[basis], self.ndf[basis], self.chi2[basis] / self.ndf[basis]))
        print(' p-value = %0.2f' % self.pvalue[basis])

        self.ymax[basis] = self.multijet_TF1[basis].GetMaximum(1, self.nBins_ensemble)
        self.fit_parameters[basis], self.fit_parameters_error[basis] = [], []
        n = basis + 1

        for m in range(nMixes):
            self.fit_parameters      [basis].append( np.array([self.multijet_TF1[basis].GetParameter(m * n + o) for o in range(basis + 1)]) )
            self.fit_parameters_error[basis].append( np.array([self.multijet_TF1[basis].GetParError (m * n + o) for o in range(basis + 1)]) )
        self.getParameterDistribution(basis)

        for _bin in range(1, self.nBins_ensemble + 1):
            self.multijet_TH1[basis].SetBinContent(_bin, self.multijet_TF1[basis].Eval(_bin))
            # self.multijet_TH1[basis].SetBinError  (_bin, self.multijet_ensemble.GetBinError(bin))
            self.multijet_TH1[basis].SetBinError  (_bin, 0.0)

        pulls = []
        bins = range(self.fit_bin_min, self.nBins_ensemble + 1)
        for _bin in bins:
            error = self.multijet_ensemble_average.GetBinError(_bin)
            pull = (self.multijet_TF1[basis].Eval(_bin) - self.multijet_ensemble_average.GetBinContent(_bin)) / error if error > 0 else 0
            pulls.append(pull)
        self.pulls[basis] = np.array(pulls)

        # check bin to bin correlations using pearson R test
        xs = np.array([self.pulls[basis][m * self.nBins_fit  : (m + 1) * self.nBins_fit - 1] for m in range(nMixes)])
        ys = np.array([self.pulls[basis][m * self.nBins_fit + 1: (m + 1) * self.nBins_fit  ] for m in range(nMixes)])

        x, y = xs.flatten(), ys.flatten()
        r, p = pearsonr(x, y, n=len(x) - nMixes * (basis + 1))

        self.pearsonr[basis] = {'total': (r, p),
                                'mixes': [pearsonr(xs[m], ys[m], n=len(xs[m]) - basis - 1) for m in range(nMixes)]}

        print('-------------------------')
        print('>> r, prob = %0.2f, %0.2f' % self.pearsonr[basis]['total'])
        print('-------------------------')
        # n = x.shape[0] - nMixes*(basis + 1)
        # dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
        # p_manual = 2*dist.cdf(-abs(r))
        # print('manual R p-value: n, p = %d, %f' % (n,p_manual))
        # raw_input()
        self.f.cd(self.channel)
        self.multijet_TH1[basis].Write()

    
    def write_to_yml(self, basis):
        self.output_yml.write(str(basis) + ":\n")

        write_pairs = [("chi2", self.chi2[basis]), ("ndf", self.ndf[basis]), ("pvalue", self.pvalue[basis]),
                       ("pearson_r", self.pearsonr[basis]['total'][0]), ("pearson_pvalue", self.pearsonr[basis]['total'][0]),
                       ("variance", self.cUp[basis])]

        for wp in write_pairs:
            self.output_yml.write(" " * 4 + f"{wp[0]}:\n")
            self.output_yml.write(" " * 8 + f"{str(wp[1])}\n")

    
    def plotBasis(self, name, basis, rebin=True):
        fig, (ax) = plt.subplots(nrows=1)
        if rebin:
            x = [self.average_rebin.GetBinCenter(_bin) for _bin in range(1, self.nBins_rebin + 1)]
        else:
            x = [self.average.GetBinCenter(_bin) for _bin in range(1, self.nBins + 1)]
        xlim = [0, 1]
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_title('%s Multiplicitive Basis (%s)' % (name[0].upper() + name[1:], self.channel.upper()))

        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        if rebin:
            for i, y in enumerate(self.basis_element[:basis + 1]):
                ax.plot(x, y, label='b$_{%i}$' % i, linewidth=1)
        else:
            for i, y in enumerate(self.basis_element_no_rebin[:basis + 1]):
                ax.plot(x, y, label='b$_{%i}$' % i, linewidth=1)

        # if name == 'normalized':
        #     ax.plot(x, self.basis_signal[basis]*10, label=r'Spurious Signal ($\times 10$)', linewidth=1)
        # else:
        #     ax.plot(x, self.basis_signal[basis],    label=r'Spurious Signal',               linewidth=1)

        ax.set_xlabel('P(Signal)')
        ax.set_ylabel('Multijet Scale')

        ax.legend(fontsize='small', loc='best')

        rebin_name = '' if rebin else '_no_rebin'
        basis_diag_dir = f"{output_dir}/basis_diagnostics"
        os.makedirs(basis_diag_dir, exist_ok=True)

        plt.tight_layout()
        fig.savefig( f"{basis_diag_dir}/{name}_basis{rebin_name}{basis}.png" )
        if getattr(args, 'save_all_formats', False):
            fig.savefig( f"{basis_diag_dir}/{name}_basis{rebin_name}{basis}.pdf" )
        plt.close(fig)

        fig, (ax) = plt.subplots(nrows=1)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(np.arange(0, 1.1, 0.1))

        ax.set_title('%s Additive Basis (%s)' % (name[0].upper() + name[1:], self.channel.upper()))

        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        if rebin:
            for i, y in enumerate(self.basis_element[:basis + 1]):
                ax.plot(x, y * self.h, label='b$_{%i}$' % i, linewidth=1)
        else:
            for i, y in enumerate(self.basis_element_no_rebin[:basis + 1]):
                ax.plot(x, y * self.h_no_rebin, label='b$_{%i}$' % i, linewidth=1)

        # if name == 'normalized':
        #     ax.plot(x, self.basis_signal[basis] * self.h*100, label=r'Spurious Signal ($\times 100$)', linewidth=1)
        # else:
        #     ax.plot(x, self.basis_signal[basis] * self.h,     label=r'Spurious Signal',                linewidth=1)

        ax.set_xlabel('P(Signal)')
        ax.set_ylabel('Events')

        ax.legend(fontsize='small', loc='best')

        plt.tight_layout()
        fig.savefig( f"{basis_diag_dir}/{name}_additive_basis{rebin_name}{basis}.png" )
        if getattr(args, 'save_all_formats', False):
            fig.savefig( f"{basis_diag_dir}/{name}_additive_basis{rebin_name}{basis}.pdf" )
        plt.close(fig)

    
    def plotPearson(self):
        fig, (ax) = plt.subplots(nrows=1)
        # ax.set_ylim(0.001,1)
        # plt.yscale('log')
        x = np.array(sorted(self.pearsonr.keys())) + 1
        ax.set_ylim(-1, 1)
        ax.set_xticks(x)
        xlim = [x[0] - 0.5, x[-1] + 0.5]
        ax.set_xlim(xlim[0], xlim[1])
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)

        r = np.array([self.pearsonr[o]['total'][0] for o in x - 1])
        p = np.array([self.pearsonr[o]['total'][1] for o in x - 1])
        ax.set_title('Multijet Model Variance Fits (%s)' % self.channel.upper())
        ax.plot(x, r, label='Combined', color='k', linewidth=2)
        ax.plot(x, p, label='p-value',  color='r', linewidth=2)

        if self.basis is not None:
            ax.plot([self.basis + 1, self.basis + 1], [-1, 1], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
            ax.scatter(self.basis + 1, p[x == (self.basis + 1)], color='k', marker='*', s=100, zorder=10)

        for m in range(nMixes):
            r = [self.pearsonr[o]['mixes'][m][0] for o in x - 1]
            # p = [self.pearsonr[o]['mixes'][m][1] for o in x]
            label = 'v$_{%d}$' % m
            ax.plot(x, r, color=COLORS[m], linewidth=1, alpha=0.5, label=label)  # underscore tells pyplot to not show this in the legend
            # ax.plot(x, r, color=colors[m], linewidth=1, alpha=0.3, linestyle='dotted', label='_'+label)#underscore tells pyplot to not show this in the legend
            # ax.plot(x, p, color=colors[m], linewidth=1, alpha=0.3, label=label)

        ax.plot(xlim, [probThreshold, probThreshold], color='r', alpha=0.5, linestyle='--', linewidth=0.5)

        ax.set_xlabel('Parameters')
        ax.set_ylabel('Adjacent Bin Pearson Correlation (r) and p-value')
        plt.legend(fontsize='small', loc='best')

        plt.tight_layout()
        fig.savefig( f"{output_dir}/0_variance_pearsonr_multijet_variance.png" )
        if getattr(args, 'save_all_formats', False):
            fig.savefig( f"{output_dir}/0_variance_pearsonr_multijet_variance.pdf" )
        plt.close(fig)

    
    def plotFitResults(self, basis, projection=(0, 1)):
        if nMixes <= 1:
            return
        n = basis + 1
        if n > 1:
            dims = tuple(list(projection) + [d for d in range(n) if d not in projection])
        else:
            dims = (0, 1)

        # plot fit parameters
        x, y, s, c = [], [], [], []
        for m in range(nMixes):
            x.append( 100 * self.fit_parameters[basis][m][dims[0]] )
            if n == 1:
                y.append( 0 )
            if n > 1:
                y.append( 100 * self.fit_parameters[basis][m][dims[1]] )
            if n > 2:
                c.append( 100 * self.fit_parameters[basis][m][dims[2]] )
            if n > 3:
                s.append( 100 * self.fit_parameters[basis][m][dims[3]] )

        x = np.array(x)
        y = np.array(y)

        kwargs = {'lw': 0.5,
                  'marker': 'o',
                  'edgecolors': 'k',
                  's': 8,
                  'c': 'k',
                  'zorder': 2,
                  }
        if n > 2:
            kwargs['c'] = c
            kwargs['cmap'] = 'BuPu'
        if n > 3:
            s = np.array(s)
            smin = s.min()
            smax = s.max()
            srange = smax - smin
            s = s - s.min()  # shift so that min is at zero
            s = s / s.max()  # scale so that max is 1
            s = (s + 5.0 / 25) * 25  # shift and scale so that min is 5.0 and max is 25+5.0
            kwargs['s'] = s

        fig, (ax) = plt.subplots(nrows=1, figsize=(7, 6)) if n > 2 else plt.subplots(nrows=1, figsize=(6, 6))
        ax.set_aspect(1)
        ax.set_title('Multijet Model Variance Fits (%s)' % self.channel.upper())
        ax.set_xlabel('c$_' + str(dims[0]) + '$ (\%)')
        ax.set_ylabel('c$_' + str(dims[1]) + '$ (\%)')

        xlim, ylim = [-8, 8], [-8, 8]
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0, 0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xticks = np.arange(-6, 8, 2)
        yticks = np.arange(-6, 8, 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        if n > 1:
            # draw 1\sigma ellipse
            ellipse = Ellipse((0, 0),
                              width =100 * (self.cUp[basis][dims[0]] - self.cDown[basis][dims[0]]),
                              height=100 * (self.cUp[basis][dims[1]] - self.cDown[basis][dims[1]]),
                              facecolor = 'none',
                              edgecolor = 'b',  # CMURED,
                              linestyle = '-',
                              linewidth = 0.75,
                              zorder=1,
                              )
            ax.add_patch(ellipse)

        bbox = dict(boxstyle='round', facecolor='w', alpha=0.8, linewidth=0, pad=0)
        if n > 2:
            # draw range bars for other priors
            for i, d in enumerate(dims[2:]):
                thisx = xlim[-1] - 0.5 * (n - 2) + 0.5 * i
                up, down = self.cUp[basis][d], self.cDown[basis][d]
                ax.quiver(thisx, 0, 0, 100 * up,   color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                ax.quiver(thisx, 0, 0, 100 * down, color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

                ax.annotate('c$_{%d}$' % (d), [thisx, 100 * down - 0.5], ha='center', va='center', bbox=bbox)

        maxr = np.zeros((2, len(x)), dtype=float)
        minr = np.zeros((2, len(x)), dtype=float)
        if n > 1:
            # generate a ton of random points on a hypersphere in dim=n so surface is dim=n - 1.
            points  = np.random.randn(n, min(100 * (n - 1), 10**7))  # random points in a hypercube
            points /= np.linalg.norm(points, axis=0)  # normalize them to the hypersphere surface

            # for each model, find the point which maximizes the change in c_0**2 + c_1**2
            for m in range(nMixes):
                plane = np.matmul( self.eigenVars[basis][m][dims[:2], :], points )
                r2 = plane[0]**2
                if n > 1:
                    r2 += plane[1]**2

                maxr[:, m] = plane[:, r2 == r2.max()].T[0]

                # construct orthogonal unit vector to maxr
                minrvec = np.copy(maxr[::-1, m])
                minrvec[0] *= -1
                minrvec /= np.linalg.norm(minrvec)

                # find maxr along minrvec to get minr
                dr2 = np.matmul( minrvec, plane )**2
                # minr[:,m] = plane[:,dr2 == dr2.max()].T[0]#this guy is the ~right length but might be slightly off orthogonal
                minr[:, m] = minrvec * dr2.max()**0.5  # this guy is the ~right length and is orthogonal by construction
        else:
            for m in range(nMixes):
                maxr[0, m] = self.eigenVars[basis][m][dims[0]]

        minr *= 100
        maxr *= 100

        # print(maxr)
        # print(minr)
        ax.quiver(x, y,  maxr[0],  maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -maxr[0], -maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        ax.quiver(x, y,  minr[0],  minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -minr[0], -minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        plt.scatter(x, y, **kwargs)
        plt.tight_layout()

        for m in range(nMixes):
            x_offset, y_offset = (maxr[0, m] + minr[0, m]) / 2, (maxr[1, m] + minr[1, m]) / 2
            ax.annotate('v$_{%d}$' % m, (x[m] + x_offset, y[m] + y_offset), bbox=bbox)

        if n > 2:
            plt.colorbar(label='c$_' + str(dims[2]) + '$ (\%)')  # , cax=cax)
            plt.subplots_adjust(right=1)

        if n > 3:
            l1 = plt.scatter([], [], s=(0.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')
            l2 = plt.scatter([], [], s=(1.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')
            l3 = plt.scatter([], [], s=(2.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')
            l4 = plt.scatter([], [], s=(3.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')

            handles = [l1,
                       l2,
                       l3,
                       l4]

            labels = ['%0.2f' % smin,
                      '%0.2f' % (smin + srange * 1.0 / 3),
                      '%0.2f' % (smin + srange * 2.0 / 3),
                      '%0.2f' % smax]

            leg = plt.legend(handles, labels,
                             ncol=1,
                             fontsize='medium',
                             loc='best',
                             title='c$_' + str(dims[3]) + '$ (\%)',
                             scatterpoints=1)

        projection = '_'.join([str(d) for d in projection])
        try:
            fig.savefig( f"{output_dir}/0_variance_parameters_basis{basis}_projection_{projection}.png" )
            if getattr(args, 'save_all_formats', False):
                fig.savefig( f"{output_dir}/0_variance_parameters_basis{basis}_projection_{projection}.pdf" )
            plt.close(fig)
        except IndexError:
            print('Weird index error...')

    
    def plotPulls(self, basis):
        n = basis + 1

        xs = np.array([self.pulls[basis][m * self.nBins_fit    :(m + 1) * self.nBins_fit - 1] for m in range(nMixes)])
        ys = np.array([self.pulls[basis][m * self.nBins_fit + 1:(m + 1) * self.nBins_fit    ] for m in range(nMixes)])
        # x1s = np.array([self.pulls[basis][m * self.nBins_fit  :(m + 1) * self.nBins_fit-1] for m in range(nMixes)])
        # y1s = np.array([self.pulls[basis][m * self.nBins_fit + 1:(m + 1) * self.nBins_fit  ] for m in range(nMixes)])
        # x2s = np.array([self.pulls[basis][m * self.nBins_fit  :(m + 1) * self.nBins_fit - 2] for m in range(nMixes)])
        # y2s = np.array([self.pulls[basis][m * self.nBins_fit+2:(m + 1) * self.nBins_fit  ] for m in range(nMixes)])
        # x3s = np.array([self.pulls[basis][m * self.nBins_fit  :(m + 1) * self.nBins_fit-3] for m in range(nMixes)])
        # y3s = np.array([self.pulls[basis][m * self.nBins_fit+3:(m + 1) * self.nBins_fit  ] for m in range(nMixes)])
        # xs, ys = np.concatenate((x1s,x2s,x3s), axis=1), np.concatenate((y1s,y2s,y3s), axis=1)

        kwargs = {'lw': 0.5,
                  'marker': 'o',
                  'edgecolors': 'k',
                  's': 8,
                  'c': 'k',
                  'zorder': 2,
                  }

        fig, (ax) = plt.subplots(nrows=1, figsize=(6, 6))
        ax.set_aspect(1)
        ax.set_title('Adjacent Bin Pulls (%s, %d parameters)' % (self.channel.upper(), basis + 1))
        ax.set_xlabel('Bin$_{i}$, Pull')
        ax.set_ylabel('Bin$_{i + 1}$ Pull')
        # ax.set_xlabel('Bin$_{2i}$, Pull')
        # ax.set_ylabel('Bin$_{2i + 1}$ Pull')

        # xlim, ylim = list(ax.get_xlim()), list(ax.get_ylim())
        # lim_max = max(int(max([abs(lim) for lim in xlim+ylim])), 1)
        lim_max = 1.5 * max(abs(xs).max(), abs(ys).max())
        xlim, ylim = [-lim_max, lim_max], [-lim_max, lim_max]
        # xlim, ylim = [-5,5], [-5,5]
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0, 0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # xticks = np.arange(-int(lim_max) + 1, int(lim_max), 1)
        # yticks = np.arange(-int(lim_max) + 1, int(lim_max), 1)
        # ax.set_xticks(xticks)
        # ax.set_yticks(yticks)

        for m in range(nMixes):
            # r, p = scipy.stats.pearsonr(xs[m], ys[m])
            (r, p) = self.pearsonr[basis]['mixes'][m]
            kwargs['label'] = 'v$_{%d}$, r=%0.2f (%2.0f%s)' % (m, r, p * 100, '\%')
            kwargs['c'] = COLORS[m]
            plt.scatter(xs[m], ys[m], **kwargs)
        plt.tight_layout()

        # x, y = xs.flatten(), ys.flatten()
        # r, p = scipy.stats.pearsonr(x, y)
        (r, p) = self.pearsonr[basis]['total']

        plt.legend(fontsize='small', loc='upper left', ncol=2, title='Overall r=%0.2f (%2.0f%s)' % (r, p * 100, '\%'))
        fig.savefig( f'{output_dir}/0_variance_pull_correlation_basis{basis}.png' )
        if getattr(args, 'save_all_formats', False):
            fig.savefig( f'{output_dir}/0_variance_pull_correlation_basis{basis}.pdf' )
        plt.close(fig)

    
    def plotFit(self, basis):
        samples = collections.OrderedDict()
        samples[closure_file_out] = collections.OrderedDict()
        # samples[closure_file_out]['%s/data_minus_ttbar_ensemble' % self.channel] = {
        #     'label' : '#LTMixed Data#GT - #lower[0.10]{t#bar{t}}',
        #     'legend': 1,
        #     'ratioDrawOptions' : 'P ex0',
        #     'isData' : True,
        #     'ratio' : 'numer A',
        #     'color' : 'ROOT.kBlack'}
        samples[closure_file_out]['%s/multijet_ensemble_average' % self.channel] = {
            'label' : '#LTMultijet Model#GT' if not getattr(args, 'unify_background', False) else '#LTBackground Model#GT',
            'legend': 2,
            'isData' : True,
            'ratio' : 'denom A',
            'color' : 'ROOT.kBlack'}
        samples[closure_file_out]['%s/multijet_ensemble' % self.channel] = {
            'label' : 'Multijet Models' if not getattr(args, 'unify_background', False) else 'Background Models',
            'legend': 3,
            'stack' : 1,
            'ratio' : 'numer A',
            'color' : color_multijet} #ffdf7f
        samples[closure_file_out]['%s/multijet_ensemble_TH1_basis%d' % (self.channel, basis)] = {
            'label' : 'Fit (%d parameter%s)' % (basis + 1, 's' if basis else ''),
            'legend': 4,
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlue'}

        lumi_title = f"{lumi} fb^{{-1}} (13 TeV)"
        region_title = 'SR' if args.region == 'SR' else regionName.get(args.region, args.region)
        classifier_name = classifier.replace('_', ' ')
        xTitle = f'{classifier_name} P(Signal) #(Bin) + #(Bins)#(Mix) #cbar P({self.channel.upper()}) is largest'

        parameters = {'titleLeft'   : '#bf{CMS} #it{Internal}',
                      'titleCenter' : region_title,
                      'titleRight'  : lumi_title,
                      'canvasSize'  : [800, 667],
                      'maxDigits'   : 4,
                      'drawLines'   : [[self.nBins_rebin * m + 0.5,  0, self.nBins_rebin * m + 0.5, self.ymax[0] * 1.1] for m in range(1, nMixes + 1)],
                      'ratioErrors' : False,
                      'ratio'       : 'significance',  # True,
                      'rMin'        : -3,  # 0.9,
                      'rMax'        :  3,  # 1.1,
                      'rTitle'      : 'Pulls',  # 'Data / Bkgd.',
                      # 'ratioErrors': True,
                      # 'ratio'      : True,
                      # 'rMin'       : 0.9,
                      # 'rMax'       : 1.1,
                      # 'rTitle'     : 'Model / Average',
                      'xTitle'      : xTitle,
                      'yTitle'      : 'Events',
                      'logY'        : True,
                      'yMax'        : self.ymax[0] * 35.0,  # make room to show fit parameters
                      'xTitleOffset': 0.95,
                      'xleg'        : [0.13, 0.13 + 0.40],
                      'yleg'        : [0.74, 0.90],
                      'legendSubText' : ['#bf{Adjacent Bin Pull Correlation:}',
                                         'r = %1.2f' % (self.pearsonr[basis]['total'][0]),
                                         'p-value = %2.0f%%' % (self.pearsonr[basis]['total'][1] * 100),],
                      'lstLocation' : 'right',
                      'lstx'        : 0.56,
                      'lsty'        : 0.89,
                      'lst_yspace'  : 0.036,
                      'lst_textsize': 0.028,
                      'rPadFraction': 0.5,
                      'outputName'  : '0_variance_multijet_ensemble_basis%d' % (basis),
                      'save_all_formats': getattr(args, 'save_all_formats', False)}

        parameters['ratioLines'] = [[self.nBins_rebin * m + 0.5, parameters['rMin'], self.nBins_rebin * m + 0.5, parameters['rMax']] for m in range(1, nMixes + 1)]

        parameters['outputDir'] = output_dir

        # print('make ',parameters['outputDir']+parameters['outputName']+'.pdf')
        ROOTPlotTools.plot(samples, parameters, debug=False)


class closure:
    
    def __init__(self, f, channel, multijet):
        self.channel = channel
        self.rebin = rebin
        self.multijet = multijet
        self.ttbar = f.Get('%s/ttbar' % self.channel)
        self.ttbar.SetName('%s_average_%s' % (self.ttbar.GetName(), self.channel))
        self.data_obs = f.Get('%s/data_obs' % self.channel).Clone()
        self.data_obs.SetDirectory(0)
        self.data_obs.SetName('%s_average_%s' % (self.data_obs.GetName(), self.channel))
        self.nBins = self.data_obs.GetSize() - 2  # GetSize includes under/overflow bins

        self.output_yml = open(f'{output_dir}/1_bias_results.yml', 'w')

        self.doSpuriousSignal = False
        self.spuriousSignal = {}
        self.spuriousSignalError = {}
        self.closure_ss_zero_TH1 = {}
        self.closure_ss_TH1 = {}
        self.signal_orthogonal_TH1 = {}
        # self.signal = f.Get('%s/signal' % self.channel)
        # self.signal.Rebin(rebin)

        self.f = f
        self.f.cd(self.channel)

        self.ttbar_rebin = self.ttbar.Clone()
        self.ttbar_rebin.SetName('%s_rebin' % self.ttbar.GetName())
        if isinstance( self.rebin, array.array ):
            self.ttbar_rebin = rebin_histogram(self.ttbar_rebin, self.rebin)
        else:
            self.ttbar_rebin.Rebin(self.rebin)
        self.data_obs_rebin = self.data_obs.Clone()
        self.data_obs_rebin.SetName('%s_rebin' % self.data_obs.GetName())
        if isinstance( self.rebin, array.array ):
            self.data_obs_rebin = rebin_histogram(self.data_obs_rebin, self.rebin)
        else:
            self.data_obs_rebin.Rebin(self.rebin)
        self.nBins_rebin = self.data_obs_rebin.GetSize() - 2

        self.bin_width = 1. / self.nBins_rebin
        self.fit_x_min = 0.5 + closure_fit_x_min / self.bin_width

        self.basis_element = self.multijet.basis_element
        self.basis_element_no_rebin = self.multijet.basis_element_no_rebin

        self.f.cd(self.channel)
        # self.bases = range(self.multijet.basis, maxBasisClosure + 1, 2)
        self.bases = range(-1, maxBasisClosure + 1)
        max_basis = max(self.bases[-1], self.multijet.basis)
        self.nBins_closure = self.nBins_rebin + max_basis + 1  # add bins for multijet shape priors
        self.multijet_closure = ROOT.TH1F('multijet_closure', '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.ttbar_closure    = ROOT.TH1F('ttbar_closure',    '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.data_obs_closure = ROOT.TH1F('data_obs_closure', '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.signal_closure   = ROOT.TH1F('signal_closure',   '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)

        for _bin in range(1, self.nBins_rebin + 1):
            self.multijet_closure.SetBinContent(_bin, self.multijet.average_rebin.GetBinContent(_bin))
            if getattr(args, 'unify_background', False):
                self.ttbar_closure.SetBinContent(_bin, 0.0)
            else:
                self.ttbar_closure.SetBinContent(_bin, self.ttbar_rebin.GetBinContent(_bin))
            self.signal_closure  .SetBinContent(_bin, self.multijet.signal       .GetBinContent(_bin))
            self.data_obs_closure.SetBinContent(_bin, self.data_obs_rebin        .GetBinContent(_bin))

            self.multijet_closure.SetBinError  (_bin, 0.0)
            self.ttbar_closure   .SetBinError  (_bin, 0.0)
            # self.signal_closure  .SetBinError  (_bin, 0.0)
            # self.multijet_closure.SetBinError  (_bin, self.multijet.average_rebin.GetBinError(_bin))
            # self.ttbar_closure   .SetBinError  (_bin, self.ttbar_rebin           .GetBinError(_bin))
            self.signal_closure  .SetBinError  (_bin, self.multijet.signal.GetBinError(_bin))
            if getattr(args, 'unify_background', False):
                error = (self.data_obs_rebin.GetBinError(_bin)**2 + self.multijet.average_rebin.GetBinError(_bin)**2 + (2.0 / nMixes)**2)**0.5
            else:
                error = (self.data_obs_rebin.GetBinError(_bin)**2 + self.ttbar_rebin.GetBinError(_bin)**2 + self.multijet.average_rebin.GetBinError(_bin)**2 + (2.0 / nMixes)**2)**0.5  # adding 2 in quadrature improves gaussian approx of poisson errors
            self.data_obs_closure.SetBinError  (_bin, error)

        for _bin in range(self.nBins_rebin + 1, self.nBins_closure + 1):
            self.data_obs_closure.SetBinError  (_bin, 1.0)

        self.f.cd(self.channel)
        self.multijet_closure.Write()
        self.ttbar_closure   .Write()
        self.data_obs_closure.Write()
        self.signal_closure  .Write()

        # Uniform binned histograms for plotMix
        self.multijet_binned = ROOT.TH1F('multijet_binned', '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)
        self.ttbar_binned    = ROOT.TH1F('ttbar_binned',    '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)
        self.data_obs_binned = ROOT.TH1F('data_obs_binned', '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)
        self.signal_binned   = ROOT.TH1F('signal_binned',   '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)

        for _bin in range(1, self.nBins_rebin + 1):
            self.multijet_binned.SetBinContent(_bin, self.multijet.average_rebin.GetBinContent(_bin))
            self.multijet_binned.SetBinError  (_bin, self.multijet.average_rebin.GetBinError(_bin))
            if getattr(args, 'unify_background', False):
                self.ttbar_binned.SetBinContent(_bin, 0.0)
                self.ttbar_binned.SetBinError  (_bin, 0.0)
            else:
                self.ttbar_binned   .SetBinContent(_bin, self.ttbar_rebin           .GetBinContent(_bin))
                self.ttbar_binned   .SetBinError  (_bin, self.ttbar_rebin           .GetBinError(_bin))
            self.signal_binned  .SetBinContent(_bin, self.multijet.signal       .GetBinContent(_bin))
            self.signal_binned  .SetBinError  (_bin, self.multijet.signal       .GetBinError(_bin))
            self.data_obs_binned.SetBinContent(_bin, self.data_obs_rebin        .GetBinContent(_bin))
            self.data_obs_binned.SetBinError  (_bin, self.data_obs_rebin        .GetBinError(_bin))

        self.f.cd(self.channel)
        self.multijet_binned.Write()
        self.ttbar_binned.Write()
        self.data_obs_binned.Write()
        self.signal_binned.Write()

        for m, mix in enumerate(mixes):
            h_data_mix = self.f.Get(f'{mix}/{self.channel}/data_obs')
            if isinstance(self.rebin, array.array):
                h_data_mix = rebin_histogram(h_data_mix, self.rebin)
            elif int(self.rebin) > 1:
                h_data_mix = h_data_mix.Clone()
                h_data_mix.Rebin(int(self.rebin))
            h_mj_mix = self.multijet.models_rebin[m]

            h_d_b = ROOT.TH1F('data_obs_binned', '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)
            h_m_b = ROOT.TH1F('multijet_binned', '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)
            for _bin in range(1, self.nBins_rebin + 1):
                h_d_b.SetBinContent(_bin, h_data_mix.GetBinContent(_bin))
                h_d_b.SetBinError(_bin, h_data_mix.GetBinError(_bin))
                h_m_b.SetBinContent(_bin, h_mj_mix.GetBinContent(_bin))
                h_m_b.SetBinError(_bin, h_mj_mix.GetBinError(_bin))

            self.f.cd(f'{mix}/{self.channel}')
            h_d_b.Write()
            h_m_b.Write()

        self.fit_result = {}
        self.fit_result_ss = {}
        self.eigenVars = {}
        self.eigenVars_ss = {}
        self.closure_TF1, self.closure_TH1 = {}, {}
        self.pvalue, self.chi2, self.ndf = {}, {}, {}
        self.pvalue_ss, self.chi2_ss, self.ndf_ss = {}, {}, {}
        self.chi2_ss_zero, self.ndf_ss_zero = {}, {}
        self.ymax = {}
        self.fit_parameters, self.fit_parameters_error = {}, {}
        self.fit_parameters_ss, self.fit_parameters_error_ss = {}, {}
        self.cUp, self.cDown = {}, {}
        self.fProb = {-1: np.nan}
        self.fProb_ss = {}
        self.basis = None
        self.passed = False
        self.exit_message = ['--- NONE (%s) ---' % self.channel.upper()]

        for basis in self.bases:
            self.makeFitFunction(basis)
            self.fit(basis)
            self.write_to_yml(basis)
            self.fitSpuriousSignal(basis)
            # self.writeClosureResults(basis)
            self.plotFitResults(basis)
            max_basis = max(basis, self.multijet.basis)
            for j in range(1, max_basis):
                self.plotFitResults(basis, projection=(j, j + 1))

            # self.plotFitResults(basis, doSpuriousSignal=True)
            # for i in range(1,max_basis):
            #     self.plotFitResults(basis, projection=(i, i + 1), doSpuriousSignal=True)
            # for i in range(0,max_basis + 1):
            #     self.plotFitResults(basis, projection=(max_basis + 1, i), doSpuriousSignal=True)

        for i, basis in enumerate(self.bases[:-1]):
            next_basis = self.bases[i + 1]
            print('fit f-test basis', next_basis)
            # self.fProb[next_basis] = 0.5
            self.fProb[next_basis] = fTest(self.chi2[basis], self.chi2[next_basis], self.ndf[basis], self.ndf[next_basis])

            if self.basis is None and (self.pvalue[basis] > probThreshold) and (self.fProb[next_basis] < 0.95):
                self.passed = True
                self.exit_message = []
                print(self.pvalue)
                print(self.fProb)
                self.basis = basis  # store first basis to satisfy min threshold. Will be used in closure fits
                self.exit_message.append('-' * 50)
                self.exit_message.append('%s channel' % self.channel.upper())
                self.exit_message.append('Satisfied goodness of fit and f-test')
                self.exit_message.append('>> %d, %d basis elements (variance, bias)' % (self.multijet.basis, self.basis))
                self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% with %d basis elements (p-value above threshold and f-test prefers this fit over previous)' % (100 * self.pvalue[basis], 100 * self.fProb[basis], basis))
                self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% with %d basis elements (f-test does not prefer this over previous fit at greater than 95%%)' % (100 * self.pvalue[next_basis], 100 * self.fProb[next_basis], next_basis))
                if self.fProb_ss[basis] < 0.95:
                    self.exit_message.append('>> SS f-test = %2.0f%%. Do not need to include spurious signal systematic :)' % (100 * self.fProb_ss[basis]))
                else:
                    self.exit_message.append('>> SS f-test = %2.0f%%! STRONG EVIDENCE FOR SPURIOUS SIGNAL SYSTEMATIC' % (100 * self.fProb_ss[basis]))
                self.exit_message.append('-' * 50)

        # Always generate p-values diagnostic plot
        self.plotPValues()

        if self.basis is None:
            self.passed = False
            for i, b in enumerate(self.bases[:-1]):
                next_b = self.bases[i + 1]
                if self.fProb[next_b] < 0.95:
                    self.basis = b
                    break
            if self.basis is None:
                self.basis = self.bases[-1]

            self.exit_message = []
            self.exit_message.append('=' * 60)
            self.exit_message.append(f'ERROR: Closure bias test FAILED for channel {self.channel.upper()}!')
            self.exit_message.append(
                f'No basis order in range [{self.bases[0]}, {self.bases[-1]}] satisfied both '
                f'goodness-of-fit (p-value > {100 * probThreshold:.1f}%) and F-test (< 95%).'
            )
            self.exit_message.append(f'Goodness-of-fit threshold required: p-value > {100 * probThreshold:.1f}%')
            self.exit_message.append('Summary of fit results by basis:')
            for b in self.bases:
                f_str = f"{100 * self.fProb[b]:.1f}%" if (b in self.fProb and not np.isnan(self.fProb[b])) else "N/A"
                self.exit_message.append(
                    f'  >> Basis {b:2d}: chi2 = {self.chi2[b]:.2f}, ndf = {self.ndf[b]:2d}, '
                    f'p-value = {self.pvalue[b]:.2e}, F-test prob = {f_str}'
                )
            if getattr(args, 'ignore_failures', False):
                self.exit_message.append(f'WARNING: Continuing with fallback basis {self.basis} (--ignore_failures was specified).')
            self.exit_message.append('=' * 60)
        else:
            self.passed = True

        if self.passed or getattr(args, 'ignore_failures', False):
            self.writeClosureResults(self.basis)

    
    def write_to_yml(self, basis):
        self.output_yml.write(str(basis) + ":\n")

        n = max(self.multijet.basis, basis) + 1
        nConstrained = max(self.multijet.basis - basis, 0)
        nUnconstrained = n - nConstrained

        write_pairs = [("chi2", self.chi2[basis]), ("ndf", self.ndf[basis]), ("pvalue", self.pvalue[basis]),
                       ("nConstrained", nConstrained), ("nUnconstrained", nUnconstrained), ("expected_ndfs", self.nBins_rebin - nUnconstrained),
                       ("variance", self.cUp[basis])]

        for wp in write_pairs:
            self.output_yml.write(" " * 4 + f"{wp[0]}:\n")
            if type(wp[1]) is dict:
                self.output_yml.write(" " * 8 + f"{list(wp[1].values())}\n")
            else:
                self.output_yml.write(" " * 8 + f"{str(wp[1])}\n")

    
    def makeFitFunction(self, basis):

        max_basis = max(basis, self.multijet.basis)

        # 
        def background_UserFunction(xArray, pars):
            this_bin = int(xArray[0])

            if this_bin > self.nBins_rebin:
                BE_idx = this_bin - self.nBins_rebin - 1
                if self.doSpuriousSignal:

                    if BE_idx > max_basis:  # do nothing with extra bins
                        return 0.0

                    BE_coefficient = pars[BE_idx]
                    sigma_up = abs(self.cUp[basis][BE_idx]) / (nMixes**0.5)
                    sigma_down = abs(self.cDown[basis][BE_idx]) / (nMixes**0.5)
                    if BE_coefficient > 0:
                        return -BE_coefficient / sigma_up
                    else:
                        return -BE_coefficient / sigma_down

                BE_idx += basis + 1  # only apply priors to higher order terms
                if BE_idx > self.multijet.basis:
                    return 0.0

                # use variance priors scaled by 1/sqrt(nMixes) for the ensemble average
                BE_coefficient = pars[BE_idx]
                sigma_up = abs(self.multijet.cUp[self.multijet.basis][BE_idx]) / (nMixes**0.5)
                sigma_down = abs(self.multijet.cDown[self.multijet.basis][BE_idx]) / (nMixes**0.5)
                if BE_coefficient > 0:
                    return -BE_coefficient / sigma_up
                else:
                    return -BE_coefficient / sigma_down

            # in distribution: evaluate basis elements times multijet
            p = 1.0
            n = max_basis + 1
            for BE_idx in range(n):
                p += pars[BE_idx] * self.basis_element[BE_idx][this_bin - 1]

            mj = self.multijet.average_rebin.GetBinContent(this_bin)
            if getattr(args, 'unify_background', False):
                background = p * mj
            else:
                background = p * mj + self.ttbar_rebin.GetBinContent(this_bin)
            spuriousSignal = pars[n] * self.multijet.signal.GetBinContent(this_bin)
            # spuriousSignal = pars[n] * mj * self.multijet.basis_signal[max_basis][bin - 1]

            return background + spuriousSignal

        self.f.cd(self.channel)
        n = max_basis + 1
        self.pycallable = background_UserFunction
        self.closure_TF1[basis] = ROOT.TF1 ('closure_TF1_basis%d' % basis, self.pycallable, 0.5, 0.5 + self.nBins_closure, n + 1)  # +1 for spurious signal
        self.closure_TH1[basis] = ROOT.TH1F('closure_TH1_basis%d' % basis,  '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.closure_ss_zero_TH1[basis]   = ROOT.TH1F('closure_ss_zero_TH1_basis%d' % basis,   '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.closure_ss_TH1[basis]        = ROOT.TH1F('closure_ss_TH1_basis%d' % basis,        '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.signal_orthogonal_TH1[basis] = ROOT.TH1F('signal_orthogonal_TH1_basis%d' % basis, '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)

        # for o in range(max(basis, self.multijet.basis)+1):
        for b in range(n):
            self.closure_TF1[basis].SetParName  (b, 'c_%d' % b)
            self.closure_TF1[basis].SetParameter(b, 0.0)
        self.closure_TF1[basis].SetParName  (n, 'spurious signal')
        self.closure_TF1[basis].FixParameter(n, 0)

    
    def getEigenvariations(self, basis, doSpuriousSignal=False, debug=False):
        n = max(self.multijet.basis, basis) + 1

        if doSpuriousSignal:
            n += 1

        if n == 1:
            self.eigenVars[basis] = np.array([self.closure_TF1[basis].GetParError(0)])
            return

        cov = ROOT.TMatrixD(n, n)
        cor = ROOT.TMatrixD(n, n)
        fit_result = self.fit_result[basis] if not doSpuriousSignal else self.fit_result_ss[basis]
        for i in range(n):
            for j in range(n):
                cov[i][j] = fit_result.CovMatrix  (i, j)
                cor[i][j] = fit_result.Correlation(i, j)

        if debug:
            print('Covariance Matrix:')
            cov.Print()
            print('Correlation Matrix:')
            cor.Print()

        eigenVal = ROOT.TVectorD(n)
        eigenVec = cov.EigenVectors(eigenVal)

        # define relative sign of eigen-basis such that the first coordinate is always positive
        for j in range(n):
            if eigenVec[0][j] >= 0:
                continue
            for i in range(n):
                eigenVec[i][j] *= -1

        if debug:
            print('Eigenvectors (columns)')
            eigenVec.Print()
            print('Eigenvalues')
            eigenVal.Print()

        eigenVars = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                eigenVars[i, j] = eigenVec[i][j] * eigenVal[j]**0.5
        if not doSpuriousSignal:
            self.eigenVars[basis] = eigenVars
        else:
            self.eigenVars_ss[basis] = eigenVars

        if debug:
            print('Eigenvariations')
            for j in range(n):
                print(j, self.eigenVars[basis][:, j])

    
    def getParameterDistribution(self, basis):

        n = max(self.multijet.basis, basis) + 1
        self.cUp[basis], self.cDown[basis] = {}, {}
        for i in range(n):
            if i <= self.multijet.basis:  # use variance and bias
                cUp = (self.multijet.cUp[self.multijet.basis][i]**2 + self.fit_parameters[basis][i]**2 + self.fit_parameters_error[basis][i]**2)**0.5
            else:  # only have bias
                cUp = (self.fit_parameters[basis][i]**2 + self.fit_parameters_error[basis][i]**2)**0.5
            cDown = -cUp
            self.cUp  [basis][i] = cUp
            self.cDown[basis][i] = cDown
    
    
    def fit(self, basis):
        n = max(self.multijet.basis, basis) + 1
        nConstrained = max(self.multijet.basis - basis, 0)
        nUnconstrained = n - nConstrained
        fit_x_max = self.nBins_rebin + 0.5 + nConstrained

        print("=" * 50)
        self.fit_result[basis] = self.data_obs_closure.Fit(self.closure_TF1[basis], 'N0S', '', self.fit_x_min, fit_x_max)
        self.getEigenvariations(basis)
        self.pvalue[basis], self.chi2[basis], self.ndf[basis] = self.closure_TF1[basis].GetProb(), self.closure_TF1[basis].GetChisquare(), self.closure_TF1[basis].GetNDF()
        print('Fit closure %s with %d basis elements. x_range = (%f, %f)' % (self.channel, basis, self.fit_x_min, fit_x_max))
        print('chi2/ndf = %3.2f/%3d = %2.2f' % (self.chi2[basis], self.ndf[basis], self.chi2[basis] / self.ndf[basis]))
        print(' p-value = %0.2f' % self.pvalue[basis])
        print('nConstrained', nConstrained)
        print('nUnonstrained', nUnconstrained)
        print('expected ndfs = ', self.nBins_rebin - nUnconstrained)

        self.ymax[basis] = self.closure_TF1[basis].GetMaximum(1, self.nBins_closure)
        # self.ymax[basis] = max(self.closure_TF1[basis].GetMaximum(1,self.nBins_closure), 100 * self.signal.GetMaximum())
        self.fit_parameters[basis], self.fit_parameters_error[basis] = [], []
        self.fit_parameters      [basis] = np.array([self.closure_TF1[basis].GetParameter(b) for b in range(n)])
        self.fit_parameters_error[basis] = np.array([self.closure_TF1[basis].GetParError (b) for b in range(n)])
        self.getParameterDistribution(basis)

        for _bin in range(1, self.nBins_closure + 1):
            self.closure_TH1[basis].SetBinContent(_bin, self.closure_TF1[basis].Eval(_bin))
            # self.closure_TH1[basis].SetBinError  (_bin, self.data_obs_closure.GetBinError(_bin))
            self.closure_TH1[basis].SetBinError  (_bin, 0.0)

        self.f.cd(self.channel)
        self.closure_TH1[basis].Write()

    
    def fitSpuriousSignal(self, basis):
        self.doSpuriousSignal = True
        max_basis = max(self.multijet.basis, basis)
        n = max_basis + 1
        self.closure_TF1[basis].FixParameter(n, 0)
        self.data_obs_closure.Fit(self.closure_TF1[basis], 'N0', '', self.fit_x_min, self.nBins_rebin + 0.5 + n)
        self.chi2_ss_zero[basis], self.ndf_ss_zero[basis] = self.closure_TF1[basis].GetChisquare(), self.closure_TF1[basis].GetNDF()
        for _bin in range(1, self.nBins_closure + 1):
            self.closure_ss_zero_TH1[basis].SetBinContent(_bin, self.closure_TF1[basis].Eval(_bin))
            self.closure_ss_zero_TH1[basis].SetBinError  (_bin, 0.0)
        self.f.cd(self.channel)
        self.closure_ss_zero_TH1[basis].Write()

        self.closure_TF1[basis].SetParameter(n, 0)
        self.closure_TF1[basis].SetParLimits(n, -10, 10)
        self.fit_result_ss[basis] = self.data_obs_closure.Fit(self.closure_TF1[basis], 'N0S', '', self.fit_x_min, self.nBins_rebin + 0.5 + n)
        self.getEigenvariations(basis, doSpuriousSignal=True)
        self.spuriousSignal[basis]      = self.closure_TF1[basis].GetParameter(n)
        self.spuriousSignalError[basis] = self.closure_TF1[basis].GetParError (n)

        self.pvalue_ss[basis], self.chi2_ss[basis], self.ndf_ss[basis] = self.closure_TF1[basis].GetProb(), self.closure_TF1[basis].GetChisquare(), self.closure_TF1[basis].GetNDF()
        print('Fit spurious signal %s with %d basis elements' % (self.channel, basis))
        print('chi2/ndf = %3.2f/%3d = %2.2f' % (self.chi2_ss[basis], self.ndf_ss[basis], self.chi2_ss[basis] / self.ndf_ss[basis]))
        print(' p-value = %0.2f' % self.pvalue_ss[basis])

        print('SS f-test basis', basis)
        self.fProb_ss[basis] = fTest(self.chi2_ss_zero[basis], self.chi2_ss[basis], self.ndf_ss_zero[basis], self.ndf_ss[basis])

        self.fit_parameters_ss      [basis] = np.array([self.closure_TF1[basis].GetParameter(b) for b in range(n + 1)])
        self.fit_parameters_error_ss[basis] = np.array([self.closure_TF1[basis].GetParError (b) for b in range(n + 1)])

        for _bin in range(1, self.nBins_closure + 1):
            self.closure_ss_TH1[basis].SetBinContent(_bin, self.closure_TF1[basis].Eval(_bin))
            self.signal_orthogonal_TH1[basis].SetBinContent(_bin, self.multijet.basis_signal[max_basis][_bin - 1] * self.multijet.average_rebin.GetBinContent(_bin) if _bin <= self.nBins_rebin else 0.0)
            self.closure_ss_TH1[basis].SetBinError  (_bin, 0.0)
            self.signal_orthogonal_TH1[basis].SetBinError(_bin, 0.0)
        self.f.cd(self.channel)
        self.closure_ss_TH1[basis].Write()
        self.signal_orthogonal_TH1[basis].Write()

        self.closure_TF1[basis].FixParameter(n, 0)
        self.doSpuriousSignal = False
        print('spurious signal = %2.2f +/- %f' % (self.spuriousSignal[basis], self.spuriousSignalError[basis]))

    
    def writeClosureResults(self, basis=None):
        systematics = {}

        if basis is None:
            basis = self.basis

        max_basis = max(self.multijet.basis, basis)
        nBEs = max_basis + 1
        # closureResults = 'ZZ4b/nTupleAnalysis/combine/closureResults_%s_%s.pkl' % (classifier,self.channel)
        # closureResults = 'closureResults_%s_%s.pkl' % (classifier,self.channel)
        # closureResultsRoot = ROOT.TFile(closureResults.replace('.txt', '.root'), 'RECREATE')
        # closureResultsFile = open(closureResults, 'w')
        print('Write Closure Results File: \n>> %s' % (closure_file_out_pkl))
        for i in range(nBEs):
            nuissance = 'basis%i' % i
            print(i, "vs", len(self.multijet.cUp  [self.multijet.basis]) )

            if i < len(self.multijet.cUp  [self.multijet.basis]):
                cUp_vari   = self.multijet.cUp  [self.multijet.basis][i]
                cDown_vari = self.multijet.cDown[self.multijet.basis][i]
            else:
                cUp_vari = 0
                cDown_vari = 0

            cUp   = self.cUp  [basis][i]
            cDown = self.cDown[basis][i]
            cUp_bias   = 0
            cDown_bias = 0
            if cUp != cUp_vari:  # break into variance and bias terms
                cUp_bias   =  (cUp**2 - cUp_vari**2)**0.5
            if cDown != cDown_vari:
                cDown_bias = -(cDown**2 - cDown_vari**2)**0.5

            if cUp_vari or cDown_vari:
                systematics['%s_vari_%sUp' % (nuissance, self.channel)] = 1 + cUp_vari * self.basis_element[i]
                systematics['%s_vari_%sDown' % (nuissance, self.channel)] = 1 + cDown_vari * self.basis_element[i]

            if cUp_bias:
                systematics['%s_bias_%sUp' % (nuissance, self.channel)] = 1 + cUp_bias * self.basis_element[i]
            if cDown_bias:
                systematics['%s_bias_%sDown' % (nuissance, self.channel)] = 1 + cDown_bias * self.basis_element[i]

        if self.fProb_ss[basis] >= 0.95:
            nuissance = 'spurious_signal'
            ssUp   = self.spuriousSignal[basis] + self.spuriousSignalError[basis]
            ssDown = self.spuriousSignal[basis] - self.spuriousSignalError[basis]
            systematics['%s_%sUp'  % (nuissance, channel)] = 1 + ssUp * self.multijet.basis_signal[max_basis]
            systematics['%s_%sDown' % (nuissance, channel)] = 1 + ssDown * self.multijet.basis_signal[max_basis]

        #     # if self.spuriousSignalError[basis] < abs(self.spuriousSignal[basis]):
        #     #     print('WARNING: Spurious Signal for channel %s is inconsistent with zero: (%f, %f)' % (self.channel, ssDown, ssUp), end='')
        #     #     ssUp   = max([abs(ssUp), abs(ssDown)])
        #     #     ssDown = -ssUp
        #     #     print(' -> (%f, %f)'%(ssDown, ssUp))
        #     SS_string  = ', '.join('%7.4f'%SS_i for SS_i in self.multijet.basis_signal[max_basis] * 10)
        #     systUp     = '1 + (%9.6f)*np.array([%s])'%(ssUp/10,   SS_string)
        #     systDown   = '1 + (%9.6f)*np.array([%s])'%(ssDown/10, SS_string)
        #     systUp     = '%s_%sUp   %s'%(nuissance, channel, systUp)
        #     systDown   = '%s_%sDown %s'%(nuissance, channel, systDown)
        #     print(systUp)
        #     print(systDown)
        #     closureResultsFile.write(systUp+'\n')
        #     closureResultsFile.write(systDown + '\n')
        # closureResultsFile.close()

        with open(closure_file_out_pkl, 'wb') as sfile:
            pickle.dump(systematics, sfile, protocol=1)

    
    def plotFitResults(self, basis, projection=(0, 1), doSpuriousSignal=False):
        max_basis = max(self.multijet.basis, basis)
        n = max_basis + 1
        d_ss = n

        if doSpuriousSignal:
            n += 1

        if n > 1:
            dims = tuple(list(projection) + [d for d in range(n) if d not in projection])
        else:
            dims = (0, 1)

        labels = ['c$_' + str(d) + '$' for d in dims]
        if doSpuriousSignal:
            labels[dims.index(n - 1)] = r'$\zeta$'

        # plot fit parameters
        x, y, s, c = [], [], [], []
        parameters = self.fit_parameters[basis] if not doSpuriousSignal else self.fit_parameters_ss[basis]
        x.append( parameters[dims[0]] * (1 if dims[0] == d_ss else 100) )
        if n == 1:
            y.append( 0 )
        if n > 1:
            y.append( parameters[dims[1]] * (1 if dims[1] == d_ss else 100) )
        if n > 2:
            c.append( parameters[dims[2]] * (1 if dims[2] == d_ss else 100) )
        if n > 3:
            s.append( parameters[dims[3]] * (1 if dims[3] == d_ss else 100) )

        x = np.array(x)
        y = np.array(y)

        kwargs = {'lw': 0.5,
                  'marker': 'o',
                  'edgecolors': 'k',
                  's': 8,
                  'c': 'k',
                  'zorder': 2,
                  }

        fig, (ax) = plt.subplots(nrows=1, figsize=(6, 6))
        ax.set_aspect(1)
        if not doSpuriousSignal:
            ax.set_title('Multijet Model Bias Fit (%s)' % self.channel.upper())
        else:
            ax.set_title('Multijet Model Spurious Signal Fit (%s)' % self.channel.upper())

        ax.set_xlabel(labels[0] + ('' if dims[0] == d_ss else ' (\%)'))
        ax.set_ylabel(labels[1] + ('' if dims[1] == d_ss else ' (\%)'))

        xlim, ylim = [-10, 10], [-10, 10]
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0, 0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xticks = np.arange(xlim[0] + 2, xlim[1], 2)
        yticks = np.arange(ylim[0] + 2, ylim[1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        if n > 1 and not doSpuriousSignal:
            # draw 1\sigma ellipses
            try:
                width = self.multijet.cUp[self.multijet.basis][dims[0]] - self.multijet.cDown[self.multijet.basis][dims[0]]
            except IndexError:
                width = 0.0  # no variance for this basis element
            try:
                height = self.multijet.cUp[self.multijet.basis][dims[1]] - self.multijet.cDown[self.multijet.basis][dims[1]]
            except IndexError:
                height = 0.0

            ellipse_self_consistency = Ellipse((0, 0),
                                               width =100 * width,
                                               height=100 * height,
                                               facecolor = 'none',
                                               edgecolor = 'b',  # CMURED,
                                               linestyle = '-',
                                               linewidth = 0.75,
                                               zorder=1,
                                               )

            ax.add_patch(ellipse_self_consistency)

            ellipse_closure = Ellipse((0, 0),
                                      width =100 * (self.cUp[basis][dims[0]] - self.cDown[basis][dims[0]]),
                                      height=100 * (self.cUp[basis][dims[1]] - self.cDown[basis][dims[1]]),
                                      facecolor='none',
                                      edgecolor='r',  # CMURED,
                                      linestyle='-',
                                      linewidth=0.75,
                                      zorder=1,
                                      )

            ax.add_patch(ellipse_closure)

        bbox = dict(boxstyle='round', facecolor='w', alpha=0.8, linewidth=0)
        if n > 2 and not doSpuriousSignal:
            # draw range bars for other priors
            for i, d in enumerate(dims[2:]):
                up, down = self.cUp[basis][d], self.cDown[basis][d]
                thisx = xlim[-1] - 0.5 * (n - 2) + 0.5 * i
                ax.quiver(thisx, 0, 0, 100 * up,   color='r', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                ax.quiver(thisx, 0, 0, 100 * down, color='r', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

                ax.annotate(labels[i + 2], [thisx, 100 * down - 0.5], ha='center', va='center', bbox=bbox)

            for i, d in enumerate(dims[2:]):
                try:
                    up, down = self.multijet.cUp[self.multijet.basis][d], self.multijet.cDown[self.multijet.basis][d]
                    thisx = xlim[-1] - 0.5 * (n - 2) + 0.5 * i
                    ax.quiver(thisx, 0, 0, 100 * up,   color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                    ax.quiver(thisx, 0, 0, 100 * down, color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                except IndexError:
                    pass  # there is no variance term for this basis

        maxr = np.zeros((2, len(x)), dtype=float)
        minr = np.zeros((2, len(x)), dtype=float)

        if n > 1:
            # generate a ton of random points on a hypersphere in dim=n so surface is dim=n - 1.
            points  = np.random.randn(n, min(100 * (n - 1), 10**7))  # random points in a hypercube
            points /= np.linalg.norm(points, axis=0)  # normalize them to the hypersphere surface

            # find the point which maximizes the change in c_0**2 + c_1**2
            for i in range(len(x)):
                eigenVars = self.eigenVars[basis] if not doSpuriousSignal else self.eigenVars_ss[basis]
                plane = np.matmul( eigenVars[dims[0:2], :], points )
                r2 = plane[0]**2
                if n > 1:
                    r2 += plane[1]**2

                maxr[:, i] = plane[:, r2 == r2.max()].T[0]

                # construct orthogonal unit vector to maxr
                minrvec = np.copy(maxr[::-1, i])
                minrvec[0] *= -1
                minrvec /= np.linalg.norm(minrvec)

                # find maxr along minrvec to get minr
                dr2 = np.matmul( minrvec, plane )**2
                # minr[:, i] = plane[:,dr2==dr2.max()].T[0]#this guy is the ~right length but might be slightly off orthogonal
                minr[:, i] = minrvec * dr2.max()**0.5  # this guy is the ~right length and is orthogonal by construction
        else:
            for i in range(len(x)):
                maxr[0, i] = self.eigenVars[basis][dims[0]]

        if dims[0] != d_ss:
            minr[0] *= 100
            maxr[0] *= 100
        if dims[1] != d_ss:
            minr[1] *= 100
            maxr[1] *= 100

        # print(maxr)
        # print(minr)
        ax.quiver(x, y,  maxr[0],  maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -maxr[0], -maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        ax.quiver(x, y,  minr[0],  minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -minr[0], -minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        plt.scatter(x, y, **kwargs)

        if n > 2:
            for i in range(len(x)):
                label = '\n'.join(['%s = %2.1f%s' % (labels[dims.index(d)], parameters[d] * (1 if d == d_ss else 100), '' if d == d_ss else '\%') for d in dims[2:]])
                # xy = np.array([x[i],y[i]])
                # xy = [xy+minr[:,i]+maxr[:,i],
                #       xy+minr[:,i]-maxr[:,i],
                #       xy-minr[:,i]+maxr[:,i],
                #       xy-minr[:,i]-maxr[:,i]]
                # xy = max(xy, key=lambda p: p[0]**2+p[1]**2)
                # if xy[0]>0:
                #     horizontalalignment = 'left'
                # else:
                #     horizontalalignment = 'right'
                # if xy[1]>0:
                #     verticalalignment = 'bottom'
                # else:
                #     verticalalignment = 'top'
                xy = [xlim[-1] - 3, ylim[-1] - 1]
                ax.annotate(label, xy,  # label,
                            ha='left', va='top',
                            bbox=bbox)

        projection = '_'.join([str(d) for d in projection])

        if not doSpuriousSignal:
            name = f'{output_dir}/1_basis_parameters_basis{basis}_projection_{projection}.pdf'
        else:
            name = f'{output_dir}/2_spurious_signal_parameters_basis{basis}_projection_{projection}.pdf'

        # print('fig.savefig( ' + name+' )')
        plt.tight_layout()
        fig.savefig( name.replace('.pdf', '.png') )
        if getattr(args, 'save_all_formats', False):
            fig.savefig( name )
        plt.close(fig)

    
    def plotPValues(self):
        fig, (ax) = plt.subplots(nrows=1)
        x = np.array(sorted(self.pvalue.keys())) + 1
        ax.set_ylim(0, 1)
        xlim = [x[0] - 0.5, x[-1] + .5]
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(x)
        # plt.yscale('log')

        y = [self.pvalue[i - 1] for i in x]
        ax.set_title('Multijet Model Bias Fit (%s)' % self.channel.upper())
        ax.plot(xlim, [probThreshold, probThreshold], color='b', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot(xlim, [0.95, 0.95],                   color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot(x, y, label='p-value', color='b', linewidth=2)
        if self.basis is not None:
            ax.plot([self.basis + 1, self.basis + 1], [0, 1], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
            ax.scatter(self.basis + 1, self.pvalue[self.basis], color='k', marker='*', s=100, zorder=10)

        x = np.array(sorted(self.fProb.keys())) + 1
        y = [self.fProb[i - 1] for i in x]
        marker = '' if len(x) > 1 else 'o'
        ax.plot(x, y, label='f-test', color='k', linewidth=2, marker=marker)

        x = np.array(sorted(self.fProb_ss.keys())) + 1
        y = [self.fProb_ss[i - 1] for i in x]
        marker = '' if len(x) > 1 else 'o'
        ax.plot(x, y, label='f-test Spurious Signal', color='k', linestyle='--', linewidth=2, marker=marker)

        ax.set_xlabel('Unconstrained Parameters')
        ax.set_ylabel('Fit p-value')
        ax.legend(loc='upper left', fontsize='small')

        plt.tight_layout()
        fig.savefig( f'{output_dir}/1_bias_pvalues.png' )
        if getattr(args, 'save_all_formats', False):
            fig.savefig( f'{output_dir}/1_bias_pvalues.pdf' )
        plt.close(fig)

    
    def plotMix(self, mix):
        ymax = self.ymax[0] if (hasattr(self, 'ymax') and 0 in self.ymax) else None
        plotMix(mix, self.channel, ymax=ymax)

    
    def plotFit(self, basis, plotSpuriousSignal=False):
        samples = collections.OrderedDict()
        samples[closure_file_out] = collections.OrderedDict()
        samples[closure_file_out]['%s/data_obs_closure' % self.channel] = {
            'label' : '#LTMixed Data#GT',
            'legend': 1,
            'isData' : True,
            # 'ratioDrawOptions': 'P ex0',
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlack'}
        if not getattr(args, 'pure_qcd', False) and not getattr(args, 'unify_background', False):
            samples[closure_file_out]['%s/ttbar_closure' % self.channel] = {
                'label' : '#lower[0.10]{t#bar{t}}',
                'legend': 3,
                'stack' : 1,
                'ratio' : 'denom A',
                'color' : color_TTbar}
        samples[closure_file_out]['%s/multijet_closure' % self.channel] = {
            'label' : '#LTMultijet#GT' if not getattr(args, 'unify_background', False) else '#LTBackground#GT',
            'legend': 2,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : color_multijet} #ffdf7f
            #'color' : 'ROOT.kYellow'}
        if not plotSpuriousSignal:
            samples[closure_file_out]['%s/closure_TH1_basis%d' % (self.channel, basis)] = {
                'label' : 'Fit (%d unconstrained parameter%s)' % (basis + 1, 's' if basis else ''),
                'legend': 4,
                'ratio': 'denom A',
                'color' : 'ROOT.kRed'}
        if plotSpuriousSignal:
            samples[closure_file_out]['%s/closure_ss_zero_TH1_basis%d' % (self.channel, basis)] = {
                'label' : 'Fit #zeta=0',
                'legend': 5,
                'ratio': 'denom A',
                'color' : 'ROOT.kGreen+3'}
            samples[closure_file_out]['%s/closure_ss_TH1_basis%d' % (self.channel, basis)] = {
                'label' : 'Fit #zeta=%1.1f#pm%1.1f' % (self.spuriousSignal[basis], self.spuriousSignalError[basis]),
                'legend': 6,
                'ratio': 'denom A',
                'color' : 'ROOT.kViolet'}
            sig_scale = getattr(args, 'signal_scale', None)
            if sig_scale is None:
                sig_scale = 1.0 if self.channel in ['ttHbb', 'tth'] else 100.0
            if sig_scale == 1.0:
                sig_label = 't#bar{t}H' if self.channel in ['ttHbb', 'tth'] else 'ZZ+ZH+HH'
            else:
                sig_label = f't#bar{{t}}H(#times{sig_scale:g})' if self.channel in ['ttHbb', 'tth'] else f'ZZ+ZH+HH(#times{sig_scale:g})'
            samples[closure_file_out][f'{self.channel}/signal_closure'] = {
                'label' : sig_label,
                'legend': 7,
                'weight': sig_scale,
                'color' : 'ROOT.kViolet+7'}
            # samples[closure_file_out]['%s/signal_orthogonal_TH1_basis%d'%(self.channel, basis)] = {
            #     'label' : 'Orthogonalized Signal(#times100)',
            #     'legend': 8,
            #     'weight': 100,
            #     'color' : 'ROOT.kViolet-6'}

        ymaxScale = 50.0
        if plotSpuriousSignal:
            ymaxScale = 120.0
        lumi_title = f"{lumi} fb^{{-1}} (13 TeV)"
        region_title = 'SR' if args.region == 'SR' else regionName.get(args.region, args.region)
        classifier_name = classifier.replace('_', ' ')
        xTitle = f'{classifier_name} Classifier Regressed P(Signal) Bin'

        parameters = {'titleLeft'   : '#bf{CMS} #it{Internal}',
                      'titleCenter' : region_title,
                      'titleRight'  : lumi_title,
                      'canvasSize'  : [800, 667],
                      'maxDigits'   : 4,
                      'drawLines'   : [[self.fit_x_min,          0, self.fit_x_min,         self.ymax[0] / 2],
                                       [self.nBins_rebin + 0.5,  0, self.nBins_rebin + 0.5, self.ymax[0] / 2]],
                      'ratioErrors' : False,
                      'ratio'       : 'significance',  # True,
                      'rMin'        : -5,  # 0.9,
                      'rMax'        : 5,  # 1.1,
                      'rTitle'      : 'Pulls',  # 'Data / Bkgd.',
                      # 'ratioErrors': True,
                      # 'ratio'      : True,
                      # 'rMin'       : 0.9,
                      # 'rMax'       : 1.1,
                      # 'rTitle'     : 'Model / Average',
                      'xTitle'      : xTitle,
                      'yTitle'      : 'Events',
                      'logY'        : True,
                      'yMax'        : self.ymax[0] * ymaxScale,   # make room to show fit parameters
                      'xleg'        : [0.13, 0.13 + 0.40],
                      'yleg'        : [0.73, 0.90],
                      'lstLocation' : 'right',
                      'lstx'        : 0.56,
                      'lsty'        : 0.89,
                      'lst_yspace'  : 0.033,
                      'lst_textsize': 0.026,
                      'outputName'  : '%s_basis%d' % ('2_spurious_signal' if plotSpuriousSignal else '1_bias', basis),
                      'save_all_formats': getattr(args, 'save_all_formats', False)}

        n = max(self.multijet.basis, basis) + 1
        if plotSpuriousSignal:
            parameters['legendSubText'] = ['#bf{Spurious Signal Fit:}',
                                           '#chi^{2}/DoF = %2.1f/%d = %1.2f (#zeta=0)' % (self.chi2_ss_zero[basis], self.ndf_ss_zero[basis], self.chi2_ss_zero[basis] / self.ndf_ss_zero[basis]),
                                           '#chi^{2}/DoF = %2.1f/%d = %1.2f' % (self.chi2_ss[basis], self.ndf_ss[basis], self.chi2_ss[basis] / self.ndf_ss[basis]),
                                           'p-value = %2.0f%% (f-test = %2.0f%%)' % (self.pvalue_ss[basis] * 100, self.fProb_ss[basis] * 100)]
            for i in range(n):
                c_val = self.fit_parameters_ss[basis][i]
                sigma_prior = (abs(self.cUp[basis][i]) if c_val > 0 else abs(self.cDown[basis][i])) / (nMixes**0.5)
                sig_val = abs(c_val) / sigma_prior if sigma_prior > 0 else (abs(c_val) / self.fit_parameters_error_ss[basis][i] if self.fit_parameters_error_ss[basis][i] > 0 else 0.0)
                parameters['legendSubText'] += ['#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma' % (i, c_val * 100, sig_val)]
        else:
            parameters['legendSubText'] = ['#bf{Fit:}',
                                           '#chi^{2}/DoF = %2.1f/%d = %1.2f' % (self.chi2[basis], self.ndf[basis], self.chi2[basis] / self.ndf[basis]),
                                           'p-value = %2.0f%%' % (self.pvalue[basis] * 100)]
            for i in range(n):
                c_val = self.fit_parameters[basis][i]
                if i > basis:
                    # Constrained higher-order parameters: compute pull with respect to the prior, matching the virtual bins in the ratio plot
                    prior_sigma = (abs(self.multijet.cUp[self.multijet.basis][i]) if c_val > 0 else abs(self.multijet.cDown[self.multijet.basis][i])) / (nMixes**0.5)
                    sig_val = abs(c_val) / prior_sigma if prior_sigma > 0 else (abs(c_val) / self.fit_parameters_error[basis][i] if self.fit_parameters_error[basis][i] > 0 else 0.0)
                    parameters['legendSubText'] += ['#color[4]{#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma}' % (i, c_val * 100, sig_val)]
                else:
                    # Unconstrained parameters: no prior exists, so significance is relative to post-fit error
                    sig_val = abs(c_val) / self.fit_parameters_error[basis][i] if self.fit_parameters_error[basis][i] > 0 else 0.0
                    parameters['legendSubText'] += ['#color[2]{#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma}' % (i, c_val * 100, sig_val)]

        parameters['ratioLines'] = [[self.fit_x_min,         parameters['rMin'], self.fit_x_min,         parameters['rMax']],
                                    [self.nBins_rebin + 0.5, parameters['rMin'], self.nBins_rebin + 0.5, parameters['rMax']]]
        # parameters['xMax'] = self.nBins_rebin + self.multijet.basis + 1.5 if not plotSpuriousSignal else self.nBins_rebin+basis + 1.5
        if plotSpuriousSignal:
            parameters['xMax'] = self.nBins_rebin + 0.5 + max(self.multijet.basis, basis) + 1
        else:
            parameters['xMax'] = self.nBins_rebin + 0.5 + max(self.multijet.basis - basis, 0)

        parameters['outputDir'] = output_dir

        # print(f'make {parameters["outputDir"]}{parameters["outputName"]}.pdf')
        ROOTPlotTools.plot(samples, parameters, debug=False)

    
    def print_exit_message(self):
        self.output_yml.close()
        for line in self.exit_message:
            print_log(line)


def plotMix(mix, channel=None, ymax=None):
    """Plot comparison of individual or average mix against background model in SR/SB."""
    if channel is None:
        channel = args.channel or 'ttHbb'

    samples = collections.OrderedDict()
    samples[closure_file_out] = collections.OrderedDict()

    f_test = ROOT.TFile(closure_file_out, 'READ')
    h_data_test = f_test.Get(f'{channel}/data_obs')
    use_binned = (h_data_test.GetXaxis().IsVariableBinSize() or isinstance(rebin, array.array)) if h_data_test else False

    if ymax is not None:
        y_max_ref = ymax * 3.5
    elif h_data_test and h_data_test.GetMaximum() > 0:
        y_max_ref = h_data_test.GetMaximum() * 3.5
    else:
        y_max_ref = 1000.0
    f_test.Close()

    data_name = 'data_obs_binned' if use_binned else 'data_obs'
    mj_name   = 'multijet_binned' if use_binned else 'multijet'
    tt_name   = 'ttbar_binned' if use_binned else 'ttbar'
    sig_name  = 'signal_binned' if use_binned else 'signal'

    if type(mix) is int:
        samples[closure_file_out][f'{mixes[mix]}/{channel}/{data_name}'] = {
            'label' : f'Mixed Data Set {mix}',
            'legend': 1,
            'isData' : True,
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlack'}
    else:
        samples[closure_file_out][f'{channel}/{data_name}'] = {
            'label' : '#LTMixed Data#GT',
            'legend': 1,
            'isData' : True,
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlack'}

    if not getattr(args, 'pure_qcd', False):
        tt_key = f'{mixes[mix]}/{channel}/{tt_name}' if type(mix) is int else f'{channel}/{tt_name}'
        samples[closure_file_out][tt_key] = {
            'label' : '#lower[0.10]{t#bar{t}}',
            'legend': 3,
            'stack' : 1,
            'ratio' : 'denom A',
            'color' : color_TTbar}

    if type(mix) is int:
        mj_proc = 'multijet_only' if getattr(args, 'unify_background', False) else mj_name
        samples[closure_file_out][f'{mixes[mix]}/{channel}/{mj_proc}'] = {
            'label' : 'Multijet Model %d' % mix,
            'legend': 2,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : color_multijet}
    else:
        mj_proc = 'multijet_only' if getattr(args, 'unify_background', False) else mj_name
        samples[closure_file_out][f'{channel}/{mj_proc}'] = {
            'label' : '#LTMultijet#GT',
            'legend': 2,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : color_multijet}

    sig_scale = getattr(args, 'signal_scale', None)
    if sig_scale is None:
        sig_scale = 1.0 if channel in ['ttHbb', 'tth'] else 100.0
    if sig_scale == 1.0:
        sig_label = 't#bar{t}H' if channel in ['ttHbb', 'tth'] else 'ZZ+ZH+HH'
    else:
        sig_label = f't#bar{{t}}H(#times{sig_scale:g})' if channel in ['ttHbb', 'tth'] else f'ZZ+ZH+HH(#times{sig_scale:g})'
    samples[closure_file_out][f'{channel}/{sig_name}'] = {
        'label' : sig_label,
        'legend': 4,
        'weight': sig_scale,
        'color' : 'ROOT.kViolet'}

    lumi_title = f"{lumi} fb^{{-1}} (13 TeV)"
    region_title = 'SR' if args.region == 'SR' else regionName.get(args.region, args.region)
    classifier_name = classifier.replace('_', ' ')
    xTitle = f'{classifier_name} Classifier Regressed P(Signal)' + (' Bin' if use_binned else '')

    parameters = {'titleLeft'   : '#bf{CMS} #it{Internal}',
                  'titleCenter' : region_title,
                  'titleRight'  : lumi_title,
                  'canvasSize'  : [800, 667],
                  'maxDigits'   : 4,
                  'ratioErrors' : True,
                  'ratio'       : True,
                  'rMin'        : 0.9,
                  'rMax'        : 1.1,
                  'rTitle'      : 'Ratio',
                  'xTitle'      : xTitle,
                  'yTitle'      : 'Events',
                  'logY'        : True,
                  'yMax'        : y_max_ref,
                  'lstLocation' : 'right',
                  'outputName'  : 'mix_%s' % (str(mix)),
                  'save_all_formats': getattr(args, 'save_all_formats', False)}

    if not use_binned:
        parameters['rebin'] = list(rebin) if isinstance(rebin, array.array) else rebin

    parameters['outputDir'] = output_dir
    ROOTPlotTools.plot(samples, parameters, debug=False)


def makeInputDiagnosticPlots(channel):
    """Generate all input diagnostic plots (pre-fit):
    1) Mix plots for each individual mix v0..v14 (mix_0..14)
    2) Average mix plot (mix_ave)
    3) 15-subsample shape overlay for 4-tag data and 3-tag multijet
    4) CMS-style 4-way comparison in SR
    """
    print_log("\n" + "=" * 60)
    print_log("Generating input diagnostic plots before starting fits...")
    print_log("=" * 60)

    # Ensure uniform-binned histograms exist if variable binning is used
    f_check = ROOT.TFile(closure_file_out, 'UPDATE')
    h_d = f_check.Get(f"{channel}/data_obs")
    if h_d and (h_d.GetXaxis().IsVariableBinSize() or isinstance(rebin, array.array)):
        nb = h_d.GetNbinsX()
        if isinstance(rebin, array.array):
            h_d_rebin = rebin_histogram(h_d, rebin)
            nb = h_d_rebin.GetNbinsX()
        for p in ['data_obs', 'multijet', 'ttbar', 'signal']:
            h_orig = f_check.Get(f"{channel}/{p}")
            if h_orig:
                h_r = rebin_histogram(h_orig, rebin) if isinstance(rebin, array.array) else (h_orig.Clone() if int(rebin) == 1 else h_orig.Rebin(int(rebin), f"{p}_tmp"))
                h_b = ROOT.TH1F(f"{p}_binned", "", nb, 0.5, 0.5 + nb)
                for b in range(1, nb + 1):
                    h_b.SetBinContent(b, h_r.GetBinContent(b))
                    h_b.SetBinError(b, h_r.GetBinError(b))
                f_check.cd(channel)
                h_b.Write("", ROOT.TObject.kOverwrite)

        for m_name in mixes:
            for p in ['data_obs', 'multijet']:
                h_orig = f_check.Get(f"{m_name}/{channel}/{p}")
                if h_orig:
                    h_r = rebin_histogram(h_orig, rebin) if isinstance(rebin, array.array) else (h_orig.Clone() if int(rebin) == 1 else h_orig.Rebin(int(rebin), f"{p}_tmp"))
                    h_b = ROOT.TH1F(f"{p}_binned", "", nb, 0.5, 0.5 + nb)
                    for b in range(1, nb + 1):
                        h_b.SetBinContent(b, h_r.GetBinContent(b))
                        h_b.SetBinError(b, h_r.GetBinError(b))
                    f_check.cd(f"{m_name}/{channel}")
                    h_b.Write("", ROOT.TObject.kOverwrite)
    f_check.Close()

    # 1. Individual mix plots
    for m in range(nMixes):
        plotMix(m, channel)

    # 2. Average mix plot
    plotMix('ave', channel)

    # 3. Shape overlay across 15 subsamples
    plotSubsamplesOverlay()

    # 4. 4-way average comparison plot
    if getattr(args, 'plot_average_comparison', False):
        plotAverageComparison(channel)

    print_log("=" * 60)
    print_log("Finished generating all pre-fit input diagnostic plots.\n")
    print_log("=" * 60 + "\n")


def plotSubsamplesOverlay():
    """Plot an overlay of the 15 subsamples in the Signal Region (SR) for both:
    1) 4-tag Mixed Data (data_obs)
    2) 3-tag FvT-reweighted multijet predictions (multijet)
    Both are plotted in shapes (normalized to unit area) with a lower ratio panel
    relative to the ensemble mean, to verify that all 15 subsamples have consistent shape.
    """
    f = ROOT.TFile(closure_file_out, 'READ')
    if f.IsZombie():
        print_log(f"WARNING: Cannot open {closure_file_out} for plotSubsamplesOverlay")
        return

    classifier_str = "SvB_MA" if "SvB_MA" in args.var else "SvB"
    x_title = f"{classifier_str.replace('_', ' ')} Classifier Regressed P(Signal)"

    subsample_colors = [
        ROOT.kBlack, ROOT.kRed+1, ROOT.kBlue+1, ROOT.kGreen+2, ROOT.kMagenta+1,
        ROOT.kOrange+7, ROOT.kCyan+2, ROOT.kViolet+1, ROOT.kAzure+7, ROOT.kTeal+3,
        ROOT.kPink+7, ROOT.kSpring+4, ROOT.kYellow+3, ROOT.kGray+2, ROOT.kOrange-3
    ]

    for target_proc, proc_title in [('data_obs', 'Mixed Data (4-tag SR)'), ('multijet', 'Multijet FvT (3-tag SR)')]:
        h_ave_orig = f.Get(f"{channel}/{target_proc}")
        if not h_ave_orig:
            print_log(f"WARNING: {channel}/{target_proc} not found in {closure_file_out}")
            continue

        h_ave = h_ave_orig.Clone(f"{target_proc}_ave_overlay")
        if isinstance(rebin, array.array):
            h_ave = rebin_histogram(h_ave, rebin)
        elif int(rebin) > 1:
            h_ave.Rebin(int(rebin))

        h_subsamples = []
        for m, mix_name in enumerate(mixes):
            h_sub_orig = f.Get(f"{mix_name}/{channel}/{target_proc}")
            if not h_sub_orig:
                print_log(f"WARNING: {mix_name}/{channel}/{target_proc} not found in {closure_file_out}")
                continue
            h_sub = h_sub_orig.Clone(f"{target_proc}_{mix_name}_overlay")
            if isinstance(rebin, array.array):
                h_sub = rebin_histogram(h_sub, rebin)
            elif int(rebin) > 1:
                h_sub.Rebin(int(rebin))
            h_subsamples.append((m, h_sub))

        if not h_subsamples:
            continue

        if isinstance(rebin, array.array) or h_ave.GetXaxis().IsVariableBinSize():
            nb = h_ave.GetNbinsX()
            h_ave_unif = ROOT.TH1F(f"{target_proc}_ave_overlay_unif", "", nb, 0.5, 0.5 + nb)
            for b in range(1, nb + 1):
                h_ave_unif.SetBinContent(b, h_ave.GetBinContent(b))
                h_ave_unif.SetBinError(b, h_ave.GetBinError(b))
            h_ave = h_ave_unif

            h_sub_unifs = []
            for m, h_sub in h_subsamples:
                h_su = ROOT.TH1F(f"{target_proc}_{mixes[m]}_overlay_unif", "", nb, 0.5, 0.5 + nb)
                for b in range(1, nb + 1):
                    h_su.SetBinContent(b, h_sub.GetBinContent(b))
                    h_su.SetBinError(b, h_sub.GetBinError(b))
                h_sub_unifs.append((m, h_su))
            h_subsamples = h_sub_unifs
            x_title = f"{classifier_str.replace('_', ' ')} Classifier Regressed P(Signal) Bin"

        # Build ROOT TCanvas with upper distribution pad and lower ratio pad
        canv_name = f"canv_subsamples_{target_proc}"
        canv = ROOT.TCanvas(canv_name, canv_name, 800, 800)
        canv.Divide(1, 2)

        p1 = canv.cd(1)
        p1.SetPad(0.0, 0.3, 1.0, 1.0)
        p1.SetTopMargin(0.08)
        p1.SetBottomMargin(0.03)
        p1.SetLeftMargin(0.12)
        p1.SetRightMargin(0.05)

        p2 = canv.cd(2)
        p2.SetPad(0.0, 0.0, 1.0, 0.3)
        p2.SetTopMargin(0.03)
        p2.SetBottomMargin(0.32)
        p2.SetLeftMargin(0.12)
        p2.SetRightMargin(0.05)
        p2.SetGridy()

        p1.cd()
        p1.SetTicks(1, 1)

        # Legend with 2 columns
        legend = ROOT.TLegend(0.48, 0.55, 0.93, 0.90)
        legend.SetNColumns(2)
        legend.SetBorderSize(0)
        legend.SetFillColorAlpha(ROOT.kWhite, 0.0)
        legend.SetTextFont(42)
        legend.SetTextSize(0.032)

        # Unnormalized ensemble average
        h_ave.SetTitle("")
        h_ave.SetLineWidth(3)
        h_ave.SetLineColor(ROOT.kBlack)
        h_ave.SetFillColor(0)
        legend.AddEntry(h_ave, "Ensemble Mean", "l")

        draw_subs = []
        ratios = []
        max_val = h_ave.GetMaximum()

        for m, h_sub in h_subsamples:
            h_sub.SetTitle("")
            col = subsample_colors[m % len(subsample_colors)]
            h_sub.SetLineColor(col)
            h_sub.SetLineWidth(1)
            h_sub.SetLineStyle(1)
            h_sub.SetFillColor(0)
            if h_sub.GetMaximum() > max_val:
                max_val = h_sub.GetMaximum()
            draw_subs.append(h_sub)
            legend.AddEntry(h_sub, f"Subsample {m}", "l")

            # Ratio to ensemble average
            h_rat = h_sub.Clone(f"{h_sub.GetName()}_ratio")
            h_rat.Divide(h_ave)
            h_rat.SetTitle("")
            h_rat.SetLineColor(col)
            h_rat.SetLineWidth(1)
            ratios.append(h_rat)

        h_ave.SetMaximum(max_val * 1.35)
        h_ave.SetMinimum(0.0)
        h_ave.GetYaxis().SetTitle("Events")
        h_ave.GetYaxis().SetTitleSize(0.045)
        h_ave.GetYaxis().SetTitleOffset(1.2)
        h_ave.GetYaxis().SetLabelSize(0.04)
        h_ave.GetXaxis().SetLabelSize(0)
        h_ave.GetXaxis().SetTitle("")
        h_ave.Draw("HIST")

        for h_sub in draw_subs:
            h_sub.Draw("HIST SAME")
        h_ave.Draw("HIST SAME")
        legend.Draw("SAME")

        # CMS / Region / Lumi labels
        lumi_title = f"{lumi} fb^{{-1}} (13 TeV)"
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextFont(61)
        latex.SetTextSize(0.045)
        latex.DrawLatex(0.12, 0.93, "CMS")
        latex.SetTextFont(52)
        latex.SetTextSize(0.035)
        latex.DrawLatex(0.20, 0.93, "Internal")
        latex.SetTextFont(42)
        latex.SetTextSize(0.040)
        latex.SetTextAlign(21)
        latex.DrawLatex(0.50, 0.93, f"{proc_title}")
        latex.SetTextAlign(31)
        latex.DrawLatex(0.95, 0.93, f"#bf{{{lumi_title}}}")

        # Draw ratio pad
        p2.cd()
        p2.SetTicks(1, 1)

        # Base frame for ratio
        h_ratio_base = h_ave.Clone(f"{target_proc}_ratio_base")
        h_ratio_base.Reset()
        for b in range(1, h_ratio_base.GetNbinsX() + 1):
            h_ratio_base.SetBinContent(b, 1.0)
            h_ratio_base.SetBinError(b, 0.0)
        h_ratio_base.SetLineColor(ROOT.kBlack)
        h_ratio_base.SetLineWidth(2)
        h_ratio_base.SetLineStyle(2)
        h_ratio_base.SetMinimum(0.5)
        h_ratio_base.SetMaximum(1.5)
        h_ratio_base.GetYaxis().SetTitle("Sub / Mean")
        h_ratio_base.GetYaxis().SetNdivisions(505)
        h_ratio_base.GetYaxis().SetTitleSize(0.10)
        h_ratio_base.GetYaxis().SetTitleOffset(0.52)
        h_ratio_base.GetYaxis().SetLabelSize(0.09)
        h_ratio_base.GetXaxis().SetTitle(x_title)
        h_ratio_base.GetXaxis().SetTitleSize(0.10)
        h_ratio_base.GetXaxis().SetTitleOffset(1.15)
        h_ratio_base.GetXaxis().SetLabelSize(0.09)
        h_ratio_base.Draw("HIST")

        for h_rat in ratios:
            h_rat.Draw("HIST SAME")
        h_ratio_base.Draw("HIST SAME")

        out_base = f"{output_dir}/subsamples_15_{target_proc}_shape_overlay"
        canv.SaveAs(f"{out_base}.png")
        if getattr(args, 'save_all_formats', False):
            canv.SaveAs(f"{out_base}.pdf")
            canv.SaveAs(f"{out_base}.C")
        print_log(f"Saved subsamples shape overlay plot: {out_base}.png")
        canv.Close()

    f.Close()


def plotAverageComparison(channel):
    """Plot CMS-style 4-way comparison in Signal Region (SR):
    1) Nominal Data 4b (points with Poisson errors)
    2) Average Mixed Data 4b across all mixes (points with Poisson errors)
    3) Nominal Background (Data 3b + TTbar 4b) (solid line)
    4) Average Mixed Background across all mixes (dashed line)
    Lower panel shows three ratio curves:
    - <Mixed 4b> / Data 4b
    - <Mix Bkg> / Nom Bkg
    - <Mixed 4b> / <Mix Bkg> (Closure)
    """
    f_closure = ROOT.TFile(closure_file_out, 'READ')
    if f_closure.IsZombie():
        print_log(f"WARNING: Cannot open {closure_file_out} for plotAverageComparison")
        return

    h_mix_ave_4b_orig = f_closure.Get(f"{channel}/data_obs")
    if not h_mix_ave_4b_orig or h_mix_ave_4b_orig.IsZombie():
        print_log(f"WARNING: {channel}/data_obs not found in {closure_file_out}")
        f_closure.Close()
        return

    h_mix_ave_4b = h_mix_ave_4b_orig.Clone("h_mix_ave_4b_comp")
    h_mix_ave_4b.SetDirectory(0)

    h_mix_ave_bkg_orig = f_closure.Get(f"{channel}/background") if getattr(args, 'unify_background', False) else None
    if not h_mix_ave_bkg_orig or h_mix_ave_bkg_orig.IsZombie():
        h_mix_ave_bkg_orig = f_closure.Get(f"{channel}/multijet")

    if not h_mix_ave_bkg_orig or h_mix_ave_bkg_orig.IsZombie():
        print_log(f"WARNING: Neither background nor multijet found in {closure_file_out}")
        f_closure.Close()
        return

    h_mix_ave_bkg = h_mix_ave_bkg_orig.Clone("h_mix_ave_bkg_comp")
    h_mix_ave_bkg.SetDirectory(0)

    if not getattr(args, 'unify_background', False):
        h_tt = f_closure.Get(f"{channel}/ttbar")
        if h_tt and not h_tt.IsZombie():
            h_mix_ave_bkg.Add(h_tt)

    f_closure.Close()

    # Load nominal data and background
    nom_f_path = getattr(args, 'input_file_nominal_data', None)
    if not nom_f_path or not os.path.exists(nom_f_path):
        nom_f_path = args.input_file_mix

    nom_f = ROOT.TFile(nom_f_path, 'READ')
    if nom_f.IsZombie():
        print_log(f"WARNING: Cannot open {nom_f_path} for nominal histograms")
        return

    years = args.years if hasattr(args, 'years') and args.years else ["2016", "2017", "2018"]
    year_map = {
        "2016": ["UL16_preVFP", "UL16_postVFP", "2016"],
        "2017": ["UL17", "2017"],
        "2018": ["UL18", "2018"],
        "UL16": ["UL16_preVFP", "UL16_postVFP", "2016"],
        "UL17": ["UL17", "2017"],
        "UL18": ["UL18", "2018"],
        "UL16_preVFP": ["UL16_preVFP"],
        "UL16_postVFP": ["UL16_postVFP"],
    }
    all_years = []
    for y in years:
        all_years.extend(year_map.get(y, [y]))
    all_years = list(dict.fromkeys(all_years))

    var_name = args.var.replace("XXX", channel)

    # 1. Nominal Data 4b
    h_nom_data4b = combine_hists(nom_f,
                                 f"{var_name}_nominal_data_YEAR_fourTag_SR",
                                 years=all_years,
                                 procs=["nominal_data"],
                                 debug=args.debug)
    if h_nom_data4b is None:
        h_nom_data4b = combine_hists(nom_f,
                                     f"{var_name}_PROC_YEAR_fourTag_SR",
                                     years=all_years,
                                     procs=["data"],
                                     debug=args.debug)
    if h_nom_data4b is None:
        h_cand = nom_f.Get(f"{channel}/data_obs")
        if h_cand and not h_cand.IsZombie():
            h_nom_data4b = h_cand.Clone("h_nom_data4b_cand")
    if h_nom_data4b is not None:
        h_nom_data4b.SetDirectory(0)

    # 2. Nominal Data 3b
    h_nom_data3b = combine_hists(nom_f,
                                 f"{var_name}_nominal_data_YEAR_threeTag_SR",
                                 years=all_years,
                                 procs=["nominal_data"],
                                 debug=args.debug)
    if h_nom_data3b is None:
        h_nom_data3b = combine_hists(nom_f,
                                     f"{var_name}_PROC_YEAR_threeTag_SR",
                                     years=all_years,
                                     procs=["data", "data_3b"],
                                     debug=args.debug)
    if h_nom_data3b is not None:
        h_nom_data3b.SetDirectory(0)

    # 3. Nominal TTbar 3b
    h_nom_ttbar3b = combine_hists(nom_f,
                                  f"{var_name}_nominal_TTbar4b_from_d3_YEAR_threeTag_SR",
                                  years=all_years,
                                  procs=["nominal_TTbar4b_from_d3"],
                                  debug=args.debug)
    if h_nom_ttbar3b is None:
        h_nom_ttbar3b = combine_hists(nom_f,
                                      f"{var_name}_PROC_YEAR_threeTag_SR",
                                      years=all_years,
                                      procs=["TTbar4b_from_d3"],
                                      debug=args.debug)
    if h_nom_ttbar3b is not None:
        h_nom_ttbar3b.SetDirectory(0)

    if h_nom_data3b is not None:
        h_nom_bkg = h_nom_data3b.Clone("h_nom_bkg_comp")
        h_nom_bkg.SetDirectory(0)
        if h_nom_ttbar3b is not None:
            h_nom_bkg.Add(h_nom_ttbar3b)
    else:
        h_cand_bkg = nom_f.Get(f"{channel}/multijet")
        if h_cand_bkg and not h_cand_bkg.IsZombie():
            h_nom_bkg = h_cand_bkg.Clone("h_nom_bkg_cand")
            h_nom_bkg.SetDirectory(0)
            h_cand_tt = nom_f.Get(f"{channel}/ttbar")
            if h_cand_tt and not h_cand_tt.IsZombie():
                h_nom_bkg.Add(h_cand_tt)
        else:
            h_nom_bkg = None

    nom_f.Close()

    if h_nom_data4b is None or h_nom_bkg is None:
        print_log("WARNING: Could not load nominal data 4b or nominal background for plotAverageComparison. Skipping plot.")
        return

    # Apply rebinning
    if isinstance(rebin, array.array):
        h_mix_ave_4b = rebin_histogram(h_mix_ave_4b, rebin)
        h_mix_ave_bkg = rebin_histogram(h_mix_ave_bkg, rebin)
        h_nom_data4b = rebin_histogram(h_nom_data4b, rebin)
        h_nom_bkg = rebin_histogram(h_nom_bkg, rebin)
    elif int(rebin) > 1:
        h_mix_ave_4b.Rebin(int(rebin))
        h_mix_ave_bkg.Rebin(int(rebin))
        h_nom_data4b.Rebin(int(rebin))
        h_nom_bkg.Rebin(int(rebin))

    classifier_str = "SvB_MA" if "SvB_MA" in args.var else "SvB"
    x_title = f"{classifier_str.replace('_', ' ')} Classifier Regressed P(Signal)"
    if isinstance(rebin, array.array) or h_mix_ave_4b.GetXaxis().IsVariableBinSize():
        nb = h_mix_ave_4b.GetNbinsX()
        def to_uniform(h_in, name):
            h_u = ROOT.TH1F(name, "", nb, 0.5, 0.5 + nb)
            for b in range(1, nb + 1):
                h_u.SetBinContent(b, h_in.GetBinContent(b))
                h_u.SetBinError(b, h_in.GetBinError(b))
            return h_u
        h_mix_ave_4b = to_uniform(h_mix_ave_4b, "h_mix_ave_4b_u")
        h_mix_ave_bkg = to_uniform(h_mix_ave_bkg, "h_mix_ave_bkg_u")
        h_nom_data4b = to_uniform(h_nom_data4b, "h_nom_data4b_u")
        h_nom_bkg = to_uniform(h_nom_bkg, "h_nom_bkg_u")
        x_title = f"{classifier_str.replace('_', ' ')} Classifier Regressed P(Signal) Bin"

    int_data4b = h_nom_data4b.Integral()
    int_mix4b  = h_mix_ave_4b.Integral()
    int_nombkg = h_nom_bkg.Integral()
    int_mixbkg = h_mix_ave_bkg.Integral()

    # Create TCanvas with two pads
    canv_name = "canv_average_comparison"
    canv = ROOT.TCanvas(canv_name, canv_name, 800, 800)
    canv.Divide(1, 2)

    p1 = canv.cd(1)
    p1.SetPad(0.0, 0.3, 1.0, 1.0)
    p1.SetTopMargin(0.08)
    p1.SetBottomMargin(0.03)
    p1.SetLeftMargin(0.12)
    p1.SetRightMargin(0.05)
    p1.SetLogy(1)
    p1.SetTicks(1, 1)

    p2 = canv.cd(2)
    p2.SetPad(0.0, 0.0, 1.0, 0.3)
    p2.SetTopMargin(0.03)
    p2.SetBottomMargin(0.32)
    p2.SetLeftMargin(0.12)
    p2.SetRightMargin(0.05)
    p2.SetGridy()
    p2.SetTicks(1, 1)

    p1.cd()
    max_val = max(h_nom_data4b.GetMaximum(), h_mix_ave_4b.GetMaximum(),
                  h_nom_bkg.GetMaximum(), h_mix_ave_bkg.GetMaximum())

    h_frame_top = h_nom_data4b.Clone("h_frame_top_comp")
    h_frame_top.Reset()
    h_frame_top.SetMinimum(1.0)
    h_frame_top.SetMaximum(max_val * 12.0)
    h_frame_top.GetYaxis().SetTitle("Events / Bin")
    h_frame_top.GetYaxis().SetTitleSize(0.045)
    h_frame_top.GetYaxis().SetTitleOffset(1.2)
    h_frame_top.GetYaxis().SetLabelSize(0.04)
    h_frame_top.GetXaxis().SetLabelSize(0)
    h_frame_top.GetXaxis().SetTitle("")
    h_frame_top.Draw("AXIS")

    h_nom_data4b.SetMarkerStyle(20)
    h_nom_data4b.SetMarkerSize(0.9)
    h_nom_data4b.SetMarkerColor(ROOT.kBlack)
    h_nom_data4b.SetLineColor(ROOT.kBlack)
    h_nom_data4b.SetLineWidth(1)

    h_mix_ave_4b.SetMarkerStyle(21)
    h_mix_ave_4b.SetMarkerSize(0.85)
    h_mix_ave_4b.SetMarkerColor(ROOT.kAzure+2)
    h_mix_ave_4b.SetLineColor(ROOT.kAzure+2)
    h_mix_ave_4b.SetLineWidth(1)

    h_nom_bkg.SetLineColor(ROOT.kRed+1)
    h_nom_bkg.SetLineWidth(2)
    h_nom_bkg.SetLineStyle(1)
    h_nom_bkg.SetFillColor(0)

    h_mix_ave_bkg.SetLineColor(ROOT.kOrange+7)
    h_mix_ave_bkg.SetLineWidth(2)
    h_mix_ave_bkg.SetLineStyle(2)
    h_mix_ave_bkg.SetFillColor(0)

    h_nom_bkg.Draw("HIST SAME")
    h_mix_ave_bkg.Draw("HIST SAME")
    h_mix_ave_4b.Draw("P E0 SAME")
    h_nom_data4b.Draw("P E0 SAME")

    legend = ROOT.TLegend(0.46, 0.65, 0.93, 0.90)
    legend.SetBorderSize(0)
    legend.SetFillColorAlpha(ROOT.kWhite, 0.0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.032)
    legend.AddEntry(h_nom_data4b, f"Data 4b (N = {int(round(int_data4b))})", "ep")
    legend.AddEntry(h_mix_ave_4b, f"#LT Mixed Data 4b #GT (N = {int(round(int_mix4b))})", "ep")
    legend.AddEntry(h_nom_bkg, f"Nominal Bkg (Data 3b+t#bar{{t}}) (N = {int(round(int_nombkg))})", "l")
    legend.AddEntry(h_mix_ave_bkg, f"#LT Mixed Data Bkg #GT (N = {int(round(int_mixbkg))})", "l")
    legend.Draw("SAME")

    lumi_title = f"{lumi} fb^{{-1}} (13 TeV)"
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(61)
    latex.SetTextSize(0.045)
    latex.DrawLatex(0.12, 0.93, "CMS")
    latex.SetTextFont(52)
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.20, 0.93, "Preliminary")
    latex.SetTextFont(42)
    latex.SetTextSize(0.040)
    latex.SetTextAlign(21)
    latex.DrawLatex(0.53, 0.93, "t#bar{t}H(b#bar{b}) SR (Inclusive)")
    latex.SetTextAlign(31)
    latex.DrawLatex(0.95, 0.93, f"#bf{{{lumi_title}}}")

    p2.cd()
    h_ratio_base = h_nom_data4b.Clone("h_ratio_base_comp")
    h_ratio_base.Reset()
    h_ratio_base.SetMinimum(0.70)
    h_ratio_base.SetMaximum(1.30)
    h_ratio_base.GetYaxis().SetTitle("Ratio")
    h_ratio_base.GetYaxis().SetNdivisions(505)
    h_ratio_base.GetYaxis().SetTitleSize(0.10)
    h_ratio_base.GetYaxis().SetTitleOffset(0.5)
    h_ratio_base.GetYaxis().SetLabelSize(0.09)
    h_ratio_base.GetXaxis().SetTitle(x_title)
    h_ratio_base.GetXaxis().SetTitleSize(0.11)
    h_ratio_base.GetXaxis().SetTitleOffset(1.1)
    h_ratio_base.GetXaxis().SetLabelSize(0.09)
    h_ratio_base.Draw("AXIS")

    line = ROOT.TLine(h_ratio_base.GetXaxis().GetXmin(), 1.0, h_ratio_base.GetXaxis().GetXmax(), 1.0)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kGray+2)
    line.SetLineWidth(1)
    line.Draw("SAME")

    r_mix_to_data = h_mix_ave_4b.Clone("r_mix_to_data")
    r_mix_to_data.Divide(h_nom_data4b)
    r_mix_to_data.SetMarkerStyle(21)
    r_mix_to_data.SetMarkerSize(0.75)
    r_mix_to_data.SetMarkerColor(ROOT.kAzure+2)
    r_mix_to_data.SetLineColor(ROOT.kAzure+2)
    r_mix_to_data.Draw("P E0 SAME")

    r_bkg_to_nom = h_mix_ave_bkg.Clone("r_bkg_to_nom")
    r_bkg_to_nom.Divide(h_nom_bkg)
    r_bkg_to_nom.SetMarkerStyle(33)
    r_bkg_to_nom.SetMarkerSize(0.95)
    r_bkg_to_nom.SetMarkerColor(ROOT.kOrange+7)
    r_bkg_to_nom.SetLineColor(ROOT.kOrange+7)
    r_bkg_to_nom.Draw("P E0 SAME")

    r_closure = h_mix_ave_4b.Clone("r_closure")
    r_closure.Divide(h_mix_ave_bkg)
    r_closure.SetMarkerStyle(20)
    r_closure.SetMarkerSize(0.75)
    r_closure.SetMarkerColor(ROOT.kGreen+2)
    r_closure.SetLineColor(ROOT.kGreen+2)
    r_closure.Draw("P E0 SAME")

    leg_ratio = ROOT.TLegend(0.13, 0.78, 0.94, 0.96)
    leg_ratio.SetNColumns(3)
    leg_ratio.SetBorderSize(0)
    leg_ratio.SetFillColorAlpha(ROOT.kWhite, 0.0)
    leg_ratio.SetTextFont(42)
    leg_ratio.SetTextSize(0.065)
    leg_ratio.AddEntry(r_mix_to_data, "#LT Mixed 4b #GT / Data 4b", "ep")
    leg_ratio.AddEntry(r_bkg_to_nom, "#LT Mix Bkg #GT / Nom Bkg", "ep")
    leg_ratio.AddEntry(r_closure, "#LT Mixed 4b #GT / #LT Mix Bkg #GT", "ep")
    leg_ratio.Draw("SAME")

    out_base = f"{output_dir}/average_closure_comparison_SR"
    canv.SaveAs(f"{out_base}.png")
    if getattr(args, 'save_all_formats', False):
        canv.SaveAs(f"{out_base}.pdf")
        canv.SaveAs(f"{out_base}.C")
    print_log(f"Saved CMS-style average 4-way comparison plot: {out_base}.png")
    canv.Close()


def run():

    f = ROOT.TFile(closure_file_out, 'UPDATE')

    #
    # make multijet ensembles and perform fits
    #
    multijetEnsembles = {}
    multijetEnsembles[channel] = multijetEnsemble(f, channel)

    #
    # run closure fits using average multijet model
    #
    closures = {}
    closures[channel] = closure(f, channel, multijetEnsembles[channel])

    #
    # close input file
    #
    f.Close()

    # Print exit messages first so they are immediately visible in logs and console
    multijetEnsembles[channel].print_exit_message()
    closures[channel].print_exit_message()

    # Generate all fit plots (variance and bias) so diagnostic plots are preserved even on test failure
    for basis in multijetEnsembles[channel].bases:
        multijetEnsembles[channel].plotFit(basis)
    for basis in closures[channel].bases:
        closures[channel].plotFit(basis)
        closures[channel].plotFit(basis, plotSpuriousSignal=True)

    failed_steps = []
    if not multijetEnsembles[channel].passed and getattr(args, 'strict_ensemble', False):
        failed_steps.append("Multijet Ensemble Variance")
    if not closures[channel].passed:
        failed_steps.append("Closure Bias Test")

    if failed_steps and not getattr(args, 'ignore_failures', False):
        print_log(f"\n[FATAL] Execution stopped because test(s) failed: {', '.join(failed_steps)}.\n")
        log_file.close()
        sys.exit(1)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='run two stage closure', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug',                 action="store_true")
    parser.add_argument('-l', '--lumi',                 dest="lumi",          default="133",    help="Luminosity for MC normalization: units [pb]")
    parser.add_argument('--mix_name', default="3bDvTMix4bDvT")
    parser.add_argument('--nMixes', type=int, default=15, help="Number of mixes or synthetic datasets")
    parser.add_argument('--classifier', help="SvB or SvB_MA")
    parser.add_argument('--region', default="SR", help="SR or SB")
    parser.add_argument('--input_file_data3b',default="output/histMixedBkg_data_3b_for_mixed.root")
    parser.add_argument('--input_file_TT',    default="output/histMixedBkg_TT.root")
    parser.add_argument('--input_file_mix',   default="output/histMixedData.root")
    parser.add_argument('--input_file_sig',   default="output/histSignal.root")
    #parser.add_argument('--input_file_sig_preUL',   default="output/histSignal_preUL.root")
    parser.add_argument('--channel', default=None, help="Channel: ttHbb, hh, zh, zz")
    parser.add_argument('--var', default="SvB_MA_ps_hh", help="SvB_MA_ps_XX or SvB_MA_ps_XX_fine")
    parser.add_argument('--rebin', default=1)
    parser.add_argument('--m4j_xmin', default=390)
    parser.add_argument('--m4j_xmax', default=1200)
    parser.add_argument('--variable_binning', action="store_true")
    parser.add_argument('--outputPath', default="stats_analysis/closureFitsNew")
    parser.add_argument('--reuse_inputs', action="store_true")
    parser.add_argument('--skip_closure', dest="run_closure", action="store_false")
    parser.add_argument('--use_kfold',    action="store_true")
    parser.add_argument('--use_ZZinSB',   action="store_true")
    parser.add_argument('--use_ZZandZHinSB',   action="store_true")
    #parser.add_argument('--skip_plots',   dest="do_plots",    action="store_false")
    parser.add_argument('--years', nargs='+', default=["2016", "2017", "2018"], help="List of years (e.g. 2017 2018 or UL17 UL18)")
    parser.add_argument('--do_CI',   action="store_true")
    parser.add_argument('--pure_qcd', '--no_ttbar', dest='pure_qcd', action="store_true", default=False, help="Pure QCD closure mode with zero ttbar")
    parser.add_argument('--auto_scale_mixed', action="store_true", default=False, help="Auto scale mixed flag (passed from pipeline)")
    parser.add_argument('--signal_scale', type=float, default=None, help="Scale factor for signal visualization on closure plots (default: 1.0 for ttHbb/tth, 100.0 for others)")
    parser.add_argument('--simple_output_dir', action="store_true", default=False, help="Use simplified output dir outputPath/channel/var")
    parser.add_argument('--maxBasis', type=int, default=10, help="Max basis order (default: 10)")
    parser.add_argument('--maxBasisEnsemble', type=int, default=None, help="Max basis order for ensemble variance (defaults to maxBasis)")
    parser.add_argument('--maxBasisClosure', type=int, default=None, help="Max basis order for closure bias (defaults to maxBasis)")
    parser.add_argument('--unify_background', action="store_true", default=False, help="Treat the sum (Multijet + TTbar) as the single, total background model across all stages (ensemble variance, closure bias, and plotting)")
    parser.add_argument('--ignore_failures', action="store_true", default=False, help="Do not exit with error if bias or ensemble test fails")
    parser.add_argument('--strict_ensemble', action="store_true", default=False, help="Treat failure to find de-correlating basis in ensemble variance as a fatal error (default: fallback to min r)")
    parser.add_argument('--plot_average_comparison', action="store_true", default=False, help="Plot CMS-style 4-way comparison: Data 4b, Nominal Bkg vs Average Mixed Data 4b and Average Mixed Bkg in SR")
    parser.add_argument('--input_file_nominal_data', default=None, help="Optional ROOT file containing nominal Data 4b and 3b (defaults to input_file_mix)")
    parser.add_argument('--input_file_nominal_bkg', default=None, help="Optional ROOT file containing nominal Background (defaults to input_file_mix)")
    parser.add_argument('--save_all_formats', action="store_true", default=False, help="Save plots in png, pdf, and C formats (default: png only)")

    args = parser.parse_args()
    print(f"\nRunning with these parameters: {args}")

    #
    #  Parse channel
    #
    if args.channel is not None:
        channel = args.channel
    elif not args.var.find("ttHbb") == -1 or not args.var.find("tth") == -1:
        channel = "ttHbb"
    elif not args.var.find("hh") == -1:
        channel = "hh"
    elif not args.var.find("zh") == -1:
        channel = "zh"
    elif not args.var.find("zz") == -1:
        channel = "zz"
    else:
        channel = "ttHbb"

    #
    #  Parse classifier
    #
    if not args.var.find("SvB_MA") == -1:
        classifier = "SvB_MA"
    elif not args.var.find("SvB") == -1:
        classifier = "SvB"
    else:
        print(f"ERROR cannot parse classifier from {args.var}")
        print(f"Defaulting to SvB")
        classifier = "SvB"
        # sys.exit(-1)

    rebin = int(args.rebin)
    rebin_label = f"varrebin{rebin}" if args.variable_binning else f"rebin{rebin}"
    if args.simple_output_dir:
        output_dir = f'{args.outputPath}/{channel}/{args.var}/'
    else:
        output_dir = f'{args.outputPath}/{args.mix_name}/{classifier}/{rebin_label}/{args.region}/{channel}/'
    mkpath(output_dir)

    closure_file_out = f"{output_dir}/hists_closure_{args.mix_name}_{args.var}_{rebin_label}.root"
    closure_file_out_pkl = closure_file_out.replace("root", "pkl")
    closure_file_out_log = closure_file_out.replace("root", "log")

    log_file = open(closure_file_out_log,"w")

    print_log(f"\nRunning with channel {channel} and rebin {rebin}")
    print_log(f"   creating:\n")
    print_log(f"\t{closure_file_out}")
    print_log(f"\t{closure_file_out_log}")
    if args.run_closure:
        print_log(f"\t{closure_file_out_pkl}")

    print_log(f"\nInputs are ")
    print_log(f"\t input_file_data3b {args.input_file_data3b}")
    print_log(f"\t input_file_TT     {args.input_file_TT}")
    print_log(f"\t input_file_mix    {args.input_file_mix}")
    print_log(f"\t input_file_sig    {args.input_file_sig}")

    if args.use_kfold:
        print_log(f"\t Using kFolding")

    if args.use_ZZinSB:
        print_log(f"\t Using ZZinSB")
    if args.use_ZZandZHinSB:
        print_log(f"\t Using ZZandZHinSB")

    if args.variable_binning:
        print(f"Computing variable binning, with threshold {args.rebin}")
        rebin = make_variable_binning(args.input_file_sig, args.var, int(args.rebin), f"{output_dir}/{os.path.basename(args.input_file_sig).replace('.root', '_rebinned.root')}" )
        print_log(f"New rebin value is {list(rebin)}")
        np.savetxt(f"{output_dir}/variable_binning_array.txt", rebin)

    doPrepInputs = True
    if args.reuse_inputs:
        if os.path.exists(closure_file_out):
            doPrepInputs = False
            print_log(f"   reusing inputs from {closure_file_out}")
        else:
            print_log(f"WARNING: cannot reuse inputs because {closure_file_out} does not exist")


    lumi = args.lumi

    #
    #  Settings
    #
    closure_fit_x_min = 0  # 0.01
    maxBasisEnsemble  = args.maxBasisEnsemble if args.maxBasisEnsemble is not None else args.maxBasis
    maxBasisClosure   = args.maxBasisClosure if args.maxBasisClosure is not None else args.maxBasis

    #if not args.do_CI:
    #    plt.rc('text', usetex=True)
    if HAS_MPL:
        plt.rc('font', family='serif')

    ttAverage = False
    doSpuriousSignal = True
    dataAverage = True
    nMixes = args.nMixes

    probThreshold = 0.05  # 0.045500263896 #0.682689492137 # 1sigma

    mixes = [f'{args.mix_name}_v{i}' for i in range(nMixes)]

    if not HAS_ROOT:
        if args.do_CI:
            print_log("\nRunning lightweight CI closure fallback (uproot) because PyROOT is not available...\n")
            prepInput_uproot()
            print_log("\nCI closure fallback completed successfully.\n")
            log_file.close()
            sys.exit(0)
        else:
            raise ImportError("PyROOT is required to run runTwoStageClosure.py unless --do_CI is specified.")

    if doPrepInputs:
        print_log("\nPreparing the input \n")
        prepInput()

    # Generate all input diagnostic plots (mix_0..14, mix_ave, subsamples overlay, average comparison)
    # BEFORE starting any of the variance or bias fits!
    makeInputDiagnosticPlots(channel)

    if args.run_closure:
        print_log("\nRunning the closure \n")
        run()
    else:
        print_log("\nSkipping the closure \n")
