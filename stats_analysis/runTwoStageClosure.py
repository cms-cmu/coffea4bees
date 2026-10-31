from __future__ import print_function
import os
import sys
import json
import argparse
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, os.getcwd())

try:
    from coffea4bees.stats_analysis.two_stage_closure import config
    from coffea4bees.stats_analysis.two_stage_closure import (
        mkpath, combine_hists, BE, BEs, pearsonr, COLORS, CMURED,
        fTest, regionName, color_multijet, color_TTbar,
        make_basis_elements, exists, mkdir, rescale_x_axis,
        writeYears, addYears, addMixes, prepInput, prepInput_uproot,
        multijetEnsemble, closure,
        plotMix, makeInputDiagnosticPlots, plotSubsamplesOverlay,
        plotAverageComparison, plotDataVsMixedVsBkg,
        print_log, make_variable_binning,
    )
except ImportError:
    from two_stage_closure import config
    from two_stage_closure import (
        mkpath, combine_hists, BE, BEs, pearsonr, COLORS, CMURED,
        fTest, regionName, color_multijet, color_TTbar,
        make_basis_elements, exists, mkdir, rescale_x_axis,
        writeYears, addYears, addMixes, prepInput, prepInput_uproot,
        multijetEnsemble, closure,
        plotMix, makeInputDiagnosticPlots, plotSubsamplesOverlay,
        plotAverageComparison, plotDataVsMixedVsBkg,
        print_log, make_variable_binning,
    )


def get_parser():
    parser = argparse.ArgumentParser(description='run two stage closure', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug',                 action="store_true")
    parser.add_argument('-l', '--lumi',            dest="lumi", default="133", help="Luminosity for MC normalization: units [pb]")
    parser.add_argument('--mix_name',              default="3bDvTMix4bDvT")
    parser.add_argument('--nMixes',                type=int, default=15, help="Number of mixes or synthetic datasets")
    parser.add_argument('--classifier',            help="SvB or SvB_MA")
    parser.add_argument('--region',                default="SR", help="SR or SB")
    parser.add_argument('--input_file_data3b',     default="output/histMixedBkg_data_3b_for_mixed.root")
    parser.add_argument('--input_file_TT',         default="output/histMixedBkg_TT.root")
    parser.add_argument('--input_file_mix',        default="output/histMixedData.root")
    parser.add_argument('--input_file_sig',        default="output/histSignal.root")
    parser.add_argument('--channel',               default=None, help="Channel: ttHbb, hh, zh, zz")
    parser.add_argument('--var',                   default="SvB_MA_ps_hh", help="SvB_MA_ps_XX or SvB_MA_ps_XX_fine")
    parser.add_argument('--rebin',                 default=1)
    parser.add_argument('--m4j_xmin',              default=390)
    parser.add_argument('--m4j_xmax',              default=1200)
    parser.add_argument('--variable_binning',      action="store_true")
    parser.add_argument('--outputPath',            default="stats_analysis/closureFitsNew")
    parser.add_argument('--reuse_inputs',          action="store_true")
    parser.add_argument('--skip_closure',          dest="run_closure", action="store_false")
    parser.add_argument('--use_kfold',             action="store_true")
    parser.add_argument('--use_ZZinSB',            action="store_true")
    parser.add_argument('--use_ZZandZHinSB',       action="store_true")
    parser.add_argument('--years',                 nargs='+', default=["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"], help="List of years")
    parser.add_argument('--do_CI',                 action="store_true")
    parser.add_argument('--pure_qcd', '--no_ttbar', dest='pure_qcd', action="store_true", default=False, help="Pure QCD closure mode with zero ttbar")
    parser.add_argument('--auto_scale_mixed',      action="store_true", default=False, help="Auto scale mixed flag (passed from pipeline)")
    parser.add_argument('--signal_scale',          type=float, default=None, help="Scale factor for signal visualization")
    parser.add_argument('--simple_output_dir',     action="store_true", default=False, help="Use simplified output dir outputPath/channel/var")
    parser.add_argument('--maxBasis',              type=int, default=5, help="Max basis order (default: 5)")
    parser.add_argument('--maxBasisEnsemble',      type=int, default=None, help="Max basis order for ensemble variance (defaults to maxBasis)")
    parser.add_argument('--maxBasisClosure',       type=int, default=None, help="Max basis order for closure bias (defaults to maxBasis)")
    parser.add_argument('--unify_background',      action="store_true", default=False, help="Treat sum (Multijet + TTbar) as single background model")
    parser.add_argument('--ignore_failures',       action="store_true", default=False, help="Do not exit with error if bias or ensemble test fails")
    parser.add_argument('--strict_ensemble',       action="store_true", default=False, help="Treat failure to find de-correlating basis in ensemble variance as fatal")
    parser.add_argument('--plot_average_comparison', action="store_true", default=False, help="Plot CMS-style 4-way comparison in SR")
    parser.add_argument('--input_file_nominal_data', default=None, help="Optional ROOT file containing nominal Data 4b and 3b")
    parser.add_argument('--input_file_nominal_bkg', default=None, help="Optional ROOT file containing nominal Background")
    parser.add_argument('--save_all_formats',      action="store_true", default=False, help="Save plots in png, pdf, and C formats")
    parser.add_argument('--match_normalization', '--match-normalization', action="store_true", default=False, help="Scale multijet model to match data_obs integral")
    parser.add_argument('--subsample_indices',     nargs='+', type=int, default=None, help="Explicit list of subsample indices to use")
    parser.add_argument('--include_ensemble_variance', '--include-ensemble-variance', action="store_true", default=False, help="Incorporate subsample-level ensemble variance")
    parser.add_argument('--no_sqrt_n_priors', '--no-sqrt-n-priors', action="store_true", default=False, help="Do not scale variance priors by 1/sqrt(nMixes)")
    parser.add_argument('--basis_type',            default='fourier', choices=['fourier', 'bernstein'], help="Type of basis functions")
    return parser


def init_context(args):
    be_elems, be_tf1s = make_basis_elements(args.basis_type)
    config.BEs = be_elems
    config.BE = be_tf1s
    config.set_args(args)

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
    config.channel = channel

    if args.classifier:
        classifier = args.classifier
    elif not args.var.find("SvB_MA") == -1:
        classifier = "SvB_MA"
    elif not args.var.find("SvB") == -1:
        classifier = "SvB"
    else:
        print(f"ERROR cannot parse classifier from {args.var}")
        print(f"Defaulting to SvB")
        classifier = "SvB"
    config.classifier = classifier

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

    log_file = open(closure_file_out_log, "w")
    config.set_log_file(log_file)

    config.rebin = rebin
    config.rebin_label = rebin_label
    config.output_dir = output_dir
    config.closure_file_out = closure_file_out
    config.closure_file_out_pkl = closure_file_out_pkl
    config.closure_file_out_log = closure_file_out_log

    config.lumi = args.lumi
    config.closure_fit_x_min = 0
    config.maxBasisEnsemble = args.maxBasisEnsemble if args.maxBasisEnsemble is not None else args.maxBasis
    config.maxBasisClosure = args.maxBasisClosure if args.maxBasisClosure is not None else args.maxBasis

    if config.HAS_MPL:
        config.plt.rc('font', family='serif')

    config.ttAverage = False
    config.doSpuriousSignal = True
    config.dataAverage = True
    config.nMixes = args.nMixes
    config.probThreshold = 0.05

    if args.subsample_indices is not None and len(args.subsample_indices) > 0:
        mixes = [f'{args.mix_name}_v{i}' for i in args.subsample_indices]
        config.nMixes = len(mixes)
        print_log(f"[INFO] Using {config.nMixes} explicit subsample indices: {args.subsample_indices}")
    else:
        mixes = [f'{args.mix_name}_v{i}' for i in range(config.nMixes)]
    config.mixes = mixes


def run():
    ROOT = config.ROOT
    channel = config.channel
    args = config.args
    output_dir = config.output_dir
    closure_file_out = config.closure_file_out
    maxBasisClosure = config.maxBasisClosure

    f = ROOT.TFile(closure_file_out, 'UPDATE')

    multijetEnsembles = {}
    multijetEnsembles[channel] = multijetEnsemble(f, channel)

    closures = {}
    closures[channel] = closure(f, channel, multijetEnsembles[channel])

    f.Close()

    multijetEnsembles[channel].print_exit_message()
    closures[channel].print_exit_message()

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

    basis_closure = closures[channel].basis if channel in closures else None
    ss_fprob = None
    ss_passed = None
    ss_basis = None
    if channel in closures and basis_closure is not None and hasattr(closures[channel], "fProb_ss"):
        if basis_closure in closures[channel].fProb_ss:
            ss_fprob = float(closures[channel].fProb_ss[basis_closure])
            ss_passed = bool(ss_fprob < 0.95) if closures[channel].passed else None
            ss_basis = int(basis_closure)

    highest_b = None
    highest_p = None
    all_pvalues = {}
    if channel in closures and hasattr(closures[channel], "bases") and hasattr(closures[channel], "pvalue"):
        cbases = closures[channel].bases
        if cbases:
            highest_b = int(cbases[-1])
            if highest_b in closures[channel].pvalue:
                highest_p = float(closures[channel].pvalue[highest_b])
            for b in cbases:
                if b in closures[channel].pvalue:
                    all_pvalues[int(b)] = float(closures[channel].pvalue[b])

    closure_status_file = f"{output_dir}/closure_status.json"
    rebin_status_val = int(args.rebin) if not isinstance(config.rebin, (list, np.ndarray)) else list(config.rebin)
    status_dict = {
        "channel": channel,
        "var": args.var,
        "rebin": rebin_status_val,
        "n_bins": getattr(closures[channel], "nBins_rebin", None) if channel in closures else None,
        "max_basis": maxBasisClosure,
        "variance_passed": bool(multijetEnsembles[channel].passed) if channel in multijetEnsembles else False,
        "bias_passed": bool(closures[channel].passed) if channel in closures else False,
        "spurious_signal_passed": ss_passed,
        "spurious_signal_basis": ss_basis,
        "spurious_signal_fprob": ss_fprob,
        "passed": bool(closures[channel].passed and (multijetEnsembles[channel].passed or not getattr(args, 'strict_ensemble', False))),
        "selected_basis": int(closures[channel].basis) if channel in closures and closures[channel].basis is not None else None,
        "highest_basis": highest_b,
        "highest_basis_pvalue": highest_p,
        "all_bias_pvalues": all_pvalues,
        "multijet_basis": int(multijetEnsembles[channel].basis) if channel in multijetEnsembles and multijetEnsembles[channel].basis is not None else None,
        "failed_steps": failed_steps,
    }
    with open(closure_status_file, "w") as sf:
        json.dump(status_dict, sf, indent=2)
    print_log(f"Wrote closure status to {closure_status_file}")

    if failed_steps and not getattr(args, 'ignore_failures', False):
        print_log(f"\n[FATAL] Execution stopped because test(s) failed: {', '.join(failed_steps)}.\n")
        if config.log_file and not config.log_file.closed:
            config.log_file.close()
        sys.exit(1)


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"\nRunning with these parameters: {args}")

    init_context(args)

    print_log(f"\nRunning with channel {config.channel} and rebin {config.rebin}")
    print_log(f"   creating:\n")
    print_log(f"\t{config.closure_file_out}")
    print_log(f"\t{config.closure_file_out_log}")
    if args.run_closure:
        print_log(f"\t{config.closure_file_out_pkl}")

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
        rebin_arr = make_variable_binning(args.input_file_sig, args.var, int(args.rebin), f"{config.output_dir}/{os.path.basename(args.input_file_sig).replace('.root', '_rebinned.root')}")
        print_log(f"New rebin value is {list(rebin_arr)}")
        np.savetxt(f"{config.output_dir}/variable_binning_array.txt", rebin_arr)
        config.rebin = rebin_arr

    doPrepInputs = True
    if args.reuse_inputs:
        if os.path.exists(config.closure_file_out):
            doPrepInputs = False
            print_log(f"   reusing inputs from {config.closure_file_out}")
        else:
            print_log(f"WARNING: cannot reuse inputs because {config.closure_file_out} does not exist")

    if not config.HAS_ROOT:
        if args.do_CI:
            print_log("\nRunning lightweight CI closure fallback (uproot) because PyROOT is not available...\n")
            prepInput_uproot()
            print_log("\nCI closure fallback completed successfully.\n")
            if config.log_file and not config.log_file.closed:
                config.log_file.close()
            sys.exit(0)
        else:
            raise ImportError("PyROOT is required to run runTwoStageClosure.py unless --do_CI is specified.")

    if doPrepInputs:
        print_log("\nPreparing the input \n")
        prepInput()

    makeInputDiagnosticPlots(config.channel)

    if args.run_closure:
        print_log("\nRunning the closure \n")
        run()
    else:
        print_log("\nSkipping the closure \n")


if __name__ == "__main__":
    main()
