from __future__ import print_function
import os
import sys
import collections
import array

from . import config
from .io import print_log, combine_hists

def plotMix(mix, channel=None, ymax=None):
    """Plot comparison of individual or average mix against background model in SR/SB."""
    ROOT = config.ROOT
    args = config.args
    closure_file_out = config.closure_file_out
    rebin = config.rebin
    color_TTbar = config.color_TTbar
    color_multijet = config.color_multijet
    mixes = config.mixes
    lumi = config.lumi
    regionName = config.regionName
    classifier = config.classifier
    output_dir = config.output_dir
    ROOTPlotTools = config.ROOTPlotTools

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
        mj_proc = ('multijet_only_binned' if use_binned else 'multijet_only') if getattr(args, 'unify_background', False) else mj_name
        samples[closure_file_out][f'{mixes[mix]}/{channel}/{mj_proc}'] = {
            'label' : 'Multijet Model %d' % mix,
            'legend': 2,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : color_multijet}
    else:
        mj_proc = ('multijet_only_binned' if use_binned else 'multijet_only') if getattr(args, 'unify_background', False) else mj_name
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
    ROOT = config.ROOT
    args = config.args
    closure_file_out = config.closure_file_out
    rebin = config.rebin
    rebin_histogram = config.rebin_histogram
    mixes = config.mixes
    nMixes = config.nMixes

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
        # every histogram the diagnostic plots read in binned form (multijet_only: unify_background)
        for p in ['data_obs', 'multijet', 'multijet_only', 'ttbar', 'signal']:
            h_orig = f_check.Get(f"{channel}/{p}")
            if h_orig:
                h_r = rebin_histogram(h_orig, rebin) if isinstance(rebin, array.array) else (h_orig.Clone() if int(rebin) == 1 else h_orig.Clone().Rebin(int(rebin)))
                h_b = ROOT.TH1F(f"{p}_binned", "", nb, 0.5, 0.5 + nb)
                for b in range(1, nb + 1):
                    h_b.SetBinContent(b, h_r.GetBinContent(b))
                    h_b.SetBinError(b, h_r.GetBinError(b))
                f_check.cd(channel)
                h_b.Write("", ROOT.TObject.kOverwrite)

        for m_name in mixes:
            for p in ['data_obs', 'multijet', 'multijet_only', 'ttbar']:
                h_orig = f_check.Get(f"{m_name}/{channel}/{p}")
                if h_orig:
                    h_r = rebin_histogram(h_orig, rebin) if isinstance(rebin, array.array) else (h_orig.Clone() if int(rebin) == 1 else h_orig.Clone().Rebin(int(rebin)))
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

    # 5. Diagnostic plot: Data vs <Mixed Data> vs Bkg (Data 3b + ttbar) in SR and SB
    for reg in ['SR', 'SB']:
        plotDataVsMixedVsBkg(channel, region=reg)

    print_log("=" * 60)
    print_log("Finished generating all pre-fit input diagnostic plots.\n")
    print_log("=" * 60 + "\n")


def plotSubsamplesOverlay():
    """Plot an overlay of the 15 subsamples in the Signal Region (SR) for both:
    1) 4-tag Mixed Data (data_obs)
    2) 3-tag FvT-reweighted multijet predictions (multijet)
    """
    ROOT = config.ROOT
    args = config.args
    closure_file_out = config.closure_file_out
    channel = config.channel
    rebin = config.rebin
    rebin_histogram = config.rebin_histogram
    mixes = config.mixes
    lumi = config.lumi
    output_dir = config.output_dir

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

        legend = ROOT.TLegend(0.48, 0.55, 0.93, 0.90)
        legend.SetNColumns(2)
        legend.SetBorderSize(0)
        legend.SetFillColorAlpha(ROOT.kWhite, 0.0)
        legend.SetTextFont(42)
        legend.SetTextSize(0.032)

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

        p2.cd()
        p2.SetTicks(1, 1)

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
    1) Nominal Data 4b
    2) Average Mixed Data 4b across all mixes
    3) Nominal Background (Data 3b + TTbar 4b)
    4) Average Mixed Background across all mixes
    """
    ROOT = config.ROOT
    args = config.args
    closure_file_out = config.closure_file_out
    rebin = config.rebin
    rebin_histogram = config.rebin_histogram
    lumi = config.lumi
    output_dir = config.output_dir

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

    nom_f_path = getattr(args, 'input_file_nominal_data', None)
    if not nom_f_path or not os.path.exists(nom_f_path):
        nom_f_path = args.input_file_mix

    nom_f = ROOT.TFile(nom_f_path, 'READ')
    if nom_f.IsZombie():
        print_log(f"WARNING: Cannot open {nom_f_path} for nominal histograms")
        return

    years = args.years if hasattr(args, 'years') and args.years else ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"]
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
    region_tag = getattr(args, 'region', 'SR')

    h_nom_data4b = combine_hists(nom_f,
                                 f"{var_name}_nominal_data_YEAR_fourTag_{region_tag}",
                                 years=all_years,
                                 procs=["nominal_data"],
                                 debug=args.debug)
    if h_nom_data4b is None:
        h_nom_data4b = combine_hists(nom_f,
                                     f"{var_name}_PROC_YEAR_fourTag_{region_tag}",
                                     years=all_years,
                                     procs=["data"],
                                     debug=args.debug)
    if h_nom_data4b is None:
        h_cand = nom_f.Get(f"{channel}/data_obs")
        if h_cand and not h_cand.IsZombie():
            h_nom_data4b = h_cand.Clone("h_nom_data4b_cand")
    if h_nom_data4b is not None:
        h_nom_data4b.SetDirectory(0)

    h_nom_data3b = combine_hists(nom_f,
                                 f"{var_name}_nominal_data_YEAR_threeTag_{region_tag}",
                                 years=all_years,
                                 procs=["nominal_data"],
                                 debug=args.debug)
    if h_nom_data3b is None:
        h_nom_data3b = combine_hists(nom_f,
                                     f"{var_name}_PROC_YEAR_threeTag_{region_tag}",
                                     years=all_years,
                                     procs=["data", "data_3b"],
                                     debug=args.debug)
    if h_nom_data3b is not None:
        h_nom_data3b.SetDirectory(0)

    h_nom_ttbar3b = combine_hists(nom_f,
                                  f"{var_name}_nominal_TTbar4b_from_d3_YEAR_threeTag_{region_tag}",
                                  years=all_years,
                                  procs=["nominal_TTbar4b_from_d3"],
                                  debug=args.debug)
    if h_nom_ttbar3b is None:
        h_nom_ttbar3b = combine_hists(nom_f,
                                      f"{var_name}_PROC_YEAR_threeTag_{region_tag}",
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


def plotDataVsMixedVsBkg(channel, region="SR"):
    """Plot comparison of:
    a) data 3b with data JCM (yellow fill) + ttbar 4b (blue fill)
    b) mixed data (red line histogram)
    c) real data (black points with Poisson error bars)
    """
    ROOT = config.ROOT
    args = config.args
    closure_file_out = config.closure_file_out
    rebin = config.rebin
    rebin_histogram = config.rebin_histogram
    rebin_label = config.rebin_label
    nMixes = config.nMixes
    lumi = config.lumi
    output_dir = config.output_dir

    ROOT.gStyle.SetOptStat(0)

    nom_f_path = getattr(args, 'input_file_nominal_data', None)
    if not nom_f_path or not os.path.exists(nom_f_path):
        nom_f_path = args.input_file_mix

    nom_f = None
    if nom_f_path and os.path.exists(nom_f_path):
        nom_f = ROOT.TFile(nom_f_path, 'READ')
        if nom_f.IsZombie():
            nom_f = None

    years = args.years if hasattr(args, 'years') and args.years else ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"]
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

    h_nom_data4b = None
    h_nom_data3b = None
    h_nom_ttbar3b = None
    h_mix_ave_4b = None

    if nom_f:
        h_nom_data4b = combine_hists(nom_f,
                                     f"{var_name}_nominal_data_YEAR_fourTag_{region}",
                                     years=all_years,
                                     procs=["nominal_data"],
                                     debug=args.debug)
        if h_nom_data4b is None:
            h_nom_data4b = combine_hists(nom_f,
                                         f"{var_name}_PROC_YEAR_fourTag_{region}",
                                         years=all_years,
                                         procs=["data", "nominal_data"],
                                         debug=args.debug)
        if h_nom_data4b is None and region == "SR":
            h_cand = nom_f.Get(f"{channel}/data_obs")
            if h_cand and not h_cand.IsZombie():
                h_nom_data4b = h_cand.Clone("h_nom_data4b_cand")
        if h_nom_data4b is not None:
            h_nom_data4b.SetDirectory(0)

        h_nom_data3b = combine_hists(nom_f,
                                     f"{var_name}_nominal_data_YEAR_threeTag_{region}",
                                     years=all_years,
                                     procs=["nominal_data"],
                                     debug=args.debug)
        if h_nom_data3b is None:
            h_nom_data3b = combine_hists(nom_f,
                                         f"{var_name}_PROC_YEAR_threeTag_{region}",
                                         years=all_years,
                                         procs=["data_3b_for_mixed", "data", "data_3b"],
                                         debug=args.debug)
        if h_nom_data3b is not None:
            h_nom_data3b.SetDirectory(0)

        h_nom_ttbar3b = combine_hists(nom_f,
                                      f"{var_name}_nominal_TTbar4b_from_d3_YEAR_threeTag_{region}",
                                      years=all_years,
                                      procs=["nominal_TTbar4b_from_d3"],
                                      debug=args.debug)
        if h_nom_ttbar3b is None:
            h_nom_ttbar3b = combine_hists(nom_f,
                                          f"{var_name}_PROC_YEAR_threeTag_{region}",
                                          years=all_years,
                                          procs=["TTbar4b_from_d3"],
                                          debug=args.debug)
        if h_nom_ttbar3b is not None:
            h_nom_ttbar3b.SetDirectory(0)

        for m in range(nMixes):
            h_m = None
            for y in all_years:
                for p in [f"mix_v{m}", f"{args.mix_name}_v{m}"]:
                    k = f"{var_name}_{p}_{y}_fourTag_{region}"
                    h = nom_f.Get(k)
                    if h and not h.IsZombie():
                        if h_m is None:
                            h_m = h.Clone(f"h_mix_m{m}_{region}")
                            h_m.SetDirectory(0)
                        else:
                            h_m.Add(h)
                        break
            if h_m is not None:
                if h_mix_ave_4b is None:
                    h_mix_ave_4b = h_m.Clone(f"h_mix_ave_4b_{region}")
                    h_mix_ave_4b.SetDirectory(0)
                else:
                    h_mix_ave_4b.Add(h_m)
        if h_mix_ave_4b is not None:
            h_mix_ave_4b.Scale(1.0 / nMixes)

        nom_f.Close()

    if region == "SR":
        f_closure = ROOT.TFile(closure_file_out, 'READ')
        if not f_closure.IsZombie():
            if h_mix_ave_4b is None:
                h_c = f_closure.Get(f"{channel}/data_obs")
                if h_c and not h_c.IsZombie():
                    h_mix_ave_4b = h_c.Clone(f"h_mix_ave_4b_{region}")
                    h_mix_ave_4b.SetDirectory(0)
            if h_nom_data3b is None:
                h_c = f_closure.Get(f"{channel}/multijet")
                if h_c and not h_c.IsZombie():
                    h_nom_data3b = h_c.Clone(f"h_nom_data3b_{region}")
                    h_nom_data3b.SetDirectory(0)
            if h_nom_ttbar3b is None:
                h_c = f_closure.Get(f"{channel}/ttbar")
                if h_c and not h_c.IsZombie():
                    h_nom_ttbar3b = h_c.Clone(f"h_nom_ttbar3b_{region}")
                    h_nom_ttbar3b.SetDirectory(0)
            f_closure.Close()

    if h_nom_data4b is None or h_nom_data3b is None or h_mix_ave_4b is None:
        print_log(f"WARNING: Could not load real data 4b, data 3b, or mixed data for {region} in plotDataVsMixedVsBkg. Skipping {region} plot.")
        return

    if h_nom_ttbar3b is None:
        h_nom_ttbar3b = h_nom_data3b.Clone(f"h_nom_ttbar3b_{region}_zero")
        h_nom_ttbar3b.Reset()
        h_nom_ttbar3b.SetDirectory(0)

    target_nbins = h_mix_ave_4b.GetNbinsX()
    for h in [h_nom_data4b, h_nom_data3b, h_nom_ttbar3b]:
        if h.GetNbinsX() > target_nbins and h.GetNbinsX() % target_nbins == 0:
            h.Rebin(h.GetNbinsX() // target_nbins)
        elif h.GetNbinsX() < target_nbins and target_nbins % h.GetNbinsX() == 0:
            h_mix_ave_4b.Rebin(target_nbins // h.GetNbinsX())
            target_nbins = h.GetNbinsX()

    if isinstance(rebin, array.array):
        h_mix_ave_4b = rebin_histogram(h_mix_ave_4b, rebin)
        h_nom_data3b = rebin_histogram(h_nom_data3b, rebin)
        h_nom_ttbar3b = rebin_histogram(h_nom_ttbar3b, rebin)
        h_nom_data4b = rebin_histogram(h_nom_data4b, rebin)
    elif int(rebin) > 1:
        if h_nom_data4b.GetNbinsX() > int(rebin) and h_nom_data4b.GetNbinsX() % int(rebin) == 0:
            h_mix_ave_4b.Rebin(int(rebin))
            h_nom_data3b.Rebin(int(rebin))
            h_nom_ttbar3b.Rebin(int(rebin))
            h_nom_data4b.Rebin(int(rebin))

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
        h_mix_ave_4b = to_uniform(h_mix_ave_4b, f"h_mix_ave_4b_{region}_u")
        h_nom_data3b = to_uniform(h_nom_data3b, f"h_nom_data3b_{region}_u")
        h_nom_ttbar3b = to_uniform(h_nom_ttbar3b, f"h_nom_ttbar3b_{region}_u")
        h_nom_data4b = to_uniform(h_nom_data4b, f"h_nom_data4b_{region}_u")
        x_title = f"{classifier_str.replace('_', ' ')} Classifier Regressed P(Signal) Bin"

    int_data4b = h_nom_data4b.Integral()
    int_mix4b  = h_mix_ave_4b.Integral()
    int_data3b = h_nom_data3b.Integral()
    int_ttbar  = h_nom_ttbar3b.Integral()

    h_total_bkg = h_nom_data3b.Clone(f"h_total_bkg_{region}")
    h_total_bkg.Add(h_nom_ttbar3b)

    c_yellow = ROOT.TColor.GetColor(255, 223, 127)
    c_blue   = ROOT.TColor.GetColor(133, 209, 251)
    c_red    = ROOT.kRed + 1

    h_total_bkg.SetFillColor(c_yellow)
    h_total_bkg.SetLineColor(ROOT.kBlack)
    h_total_bkg.SetLineWidth(1)

    h_nom_ttbar3b.SetFillColor(c_blue)
    h_nom_ttbar3b.SetLineColor(ROOT.kBlack)
    h_nom_ttbar3b.SetLineWidth(1)

    h_leg_data3b = h_nom_data3b.Clone(f"h_leg_data3b_{region}")
    h_leg_data3b.SetFillColor(c_yellow)
    h_leg_data3b.SetLineColor(ROOT.kBlack)
    h_leg_data3b.SetLineWidth(1)

    h_leg_ttbar = h_nom_ttbar3b.Clone(f"h_leg_ttbar_{region}")
    h_leg_ttbar.SetFillColor(c_blue)
    h_leg_ttbar.SetLineColor(ROOT.kBlack)
    h_leg_ttbar.SetLineWidth(1)

    h_mix_ave_4b.SetLineColor(c_red)
    h_mix_ave_4b.SetLineWidth(3)
    h_mix_ave_4b.SetLineStyle(1)
    h_mix_ave_4b.SetFillColor(0)

    h_nom_data4b.SetMarkerStyle(20)
    h_nom_data4b.SetMarkerSize(0.9)
    h_nom_data4b.SetMarkerColor(ROOT.kBlack)
    h_nom_data4b.SetLineColor(ROOT.kBlack)
    h_nom_data4b.SetLineWidth(1)

    canv_name = f"canv_data_vs_mixed_vs_bkg_{region}"
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
    max_val = max(h_nom_data4b.GetMaximum(), h_mix_ave_4b.GetMaximum(), h_total_bkg.GetMaximum())
    h_frame = h_nom_data4b.Clone(f"h_frame_{region}")
    h_frame.Reset()
    h_frame.SetMinimum(1.0)
    h_frame.SetMaximum(max_val * 40.0)
    h_frame.GetYaxis().SetTitle("Events")
    h_frame.GetYaxis().SetTitleSize(0.045)
    h_frame.GetYaxis().SetTitleOffset(1.2)
    h_frame.GetYaxis().SetLabelSize(0.04)
    h_frame.GetXaxis().SetLabelSize(0)
    h_frame.GetXaxis().SetTitle("")
    h_frame.Draw("AXIS")

    h_total_bkg.Draw("HIST SAME")
    h_nom_ttbar3b.Draw("HIST SAME")
    h_mix_ave_4b.Draw("HIST SAME")
    h_nom_data4b.Draw("P E0 SAME")

    legend = ROOT.TLegend(0.48, 0.65, 0.93, 0.90)
    legend.SetBorderSize(0)
    legend.SetFillColorAlpha(ROOT.kWhite, 0.0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.032)
    legend.AddEntry(h_leg_data3b, f"data 3b (N = {int(round(int_data3b))})", "f")
    legend.AddEntry(h_leg_ttbar, f"ttbar (N = {int(round(int_ttbar))})", "f")
    legend.AddEntry(h_mix_ave_4b, f"<mixeddata> (N = {int(round(int_mix4b))})", "l")
    legend.AddEntry(h_nom_data4b, f"data (N = {int(round(int_data4b))})", "ep")
    legend.Draw("SAME")

    lumi_title = f"{lumi} fb^{{-1}} (13 TeV)"
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(61)
    latex.SetTextSize(0.045)
    latex.DrawLatex(0.12, 0.93, "CMS")
    latex.SetTextFont(52)
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.20, 0.93, "Work in Progress")
    latex.SetTextFont(42)
    latex.SetTextSize(0.040)
    latex.SetTextAlign(21)
    latex.DrawLatex(0.53, 0.93, f"{channel} {region}")
    latex.SetTextAlign(31)
    latex.DrawLatex(0.95, 0.93, f"#bf{{{lumi_title}}}")
    h_frame.Draw("AXIS SAME")
    p1.RedrawAxis()

    p2.cd()
    h_ratio_frame = h_nom_data4b.Clone(f"h_ratio_frame_{region}")
    h_ratio_frame.Reset()
    h_ratio_frame.SetMinimum(0.60)
    h_ratio_frame.SetMaximum(1.40)
    h_ratio_frame.GetYaxis().SetTitle("Data / Model")
    h_ratio_frame.GetYaxis().SetNdivisions(505)
    h_ratio_frame.GetYaxis().SetTitleSize(0.10)
    h_ratio_frame.GetYaxis().SetTitleOffset(0.52)
    h_ratio_frame.GetYaxis().SetLabelSize(0.09)
    h_ratio_frame.GetXaxis().SetTitle(x_title)
    h_ratio_frame.GetXaxis().SetTitleSize(0.11)
    h_ratio_frame.GetXaxis().SetTitleOffset(1.1)
    h_ratio_frame.GetXaxis().SetLabelSize(0.09)
    h_ratio_frame.Draw("AXIS")

    line = ROOT.TLine(h_ratio_frame.GetXaxis().GetXmin(), 1.0, h_ratio_frame.GetXaxis().GetXmax(), 1.0)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kGray + 2)
    line.SetLineWidth(1)
    line.Draw("SAME")

    r_data_to_mix = h_nom_data4b.Clone(f"r_data_to_mix_{region}")
    r_data_to_mix.Divide(h_mix_ave_4b)
    r_data_to_mix.SetMarkerStyle(20)
    r_data_to_mix.SetMarkerSize(0.85)
    r_data_to_mix.SetMarkerColor(c_red)
    r_data_to_mix.SetLineColor(c_red)
    r_data_to_mix.Draw("P E0 SAME")

    r_data_to_bkg = h_nom_data4b.Clone(f"r_data_to_bkg_{region}")
    r_data_to_bkg.Divide(h_total_bkg)
    r_data_to_bkg.SetMarkerStyle(21)
    r_data_to_bkg.SetMarkerSize(0.85)
    r_data_to_bkg.SetMarkerColor(ROOT.kBlue + 1)
    r_data_to_bkg.SetLineColor(ROOT.kBlue + 1)
    r_data_to_bkg.Draw("P E0 SAME")

    leg_ratio = ROOT.TLegend(0.14, 0.78, 0.88, 0.96)
    leg_ratio.SetNColumns(2)
    leg_ratio.SetBorderSize(0)
    leg_ratio.SetFillColorAlpha(ROOT.kWhite, 0.0)
    leg_ratio.SetTextFont(42)
    leg_ratio.SetTextSize(0.065)
    leg_ratio.AddEntry(r_data_to_mix, "data / <mixeddata>", "ep")
    leg_ratio.AddEntry(r_data_to_bkg, "data / (data 3b + ttbar)", "ep")
    leg_ratio.Draw("SAME")
    h_ratio_frame.Draw("AXIS SAME")
    p2.RedrawAxis()

    out_base = f"{output_dir}/data_vs_mixed_vs_bkg_{region}_{rebin_label}"
    canv.SaveAs(f"{out_base}.png")
    canv.SaveAs(f"{output_dir}/data_vs_mixed_vs_bkg_{region}.png")
    if region == "SR":
        canv.SaveAs(f"{output_dir}/data_vs_mixed_vs_bkg.png")
    if getattr(args, 'save_all_formats', False):
        canv.SaveAs(f"{out_base}.pdf")
        canv.SaveAs(f"{out_base}.C")
        canv.SaveAs(f"{output_dir}/data_vs_mixed_vs_bkg_{region}.pdf")
    print_log(f"Saved Data vs Mixed vs Bkg comparison plot ({region}): {out_base}.png")
    canv.Close()
