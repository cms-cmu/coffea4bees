#!/usr/bin/env python3
"""
JCM (Jet Combinatoric Model) Weight Generator

This script produces weights for the Jet Combinatoric Model (JCM) used
in HH→4b analysis to model the combinatorial background from 3-tag events.
It performs a fit to the jet multiplicity distribution and computes weights
to apply to the 3-tag sample to model the 4-tag background.
"""

import sys
import argparse
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from coffea4bees.plots.plots import load_config_4b
from src.plotting.iPlot_config import plot_config
from src.plotting.plots import load_hists, read_axes_and_cuts, makePlot
from src.plotting.helpers import get_cut_dict

from coffea4bees.analysis.jcm_tools.jcm_model import jetCombinatoricModel
from copy import copy

cfg = plot_config()
logger = logging.getLogger('JCM')



def main():
    """Main function to run the JCM weight generation process"""
    parser = argparse.ArgumentParser(
        description='Make Jet Combinatoric Model weights',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--noFitWeight', dest='noFitWeight', default="")
    parser.add_argument('-w', '--weightSet', dest="weightSet", default="",
                        help='Label for the weight set')
    parser.add_argument('-r', dest="weightRegion", default="SB",
                        help='Weight region (e.g. SB for sideband)')
    parser.add_argument('--data4bName', default="data",
                        help='Name of the 4b data process')
    parser.add_argument('-c', dest="cut", default="passPreSel",
                        help='Cut to apply (e.g. passPreSel)')
    parser.add_argument('-fix_e', action="store_true",
                        help='Fix the pairEnhancement parameter to 0')
    parser.add_argument('-fix_d', action="store_true",
                        help='Fix the pairEnhancementDecay parameter to 1')
    parser.add_argument('-i', '--inputFile', nargs="+", dest="inputFile",
                        default='hists.pkl', help='Input file(s). Default: hists.pkl')
    parser.add_argument('-o', '--outputDir', dest='outputDir', default="",
                        help='Output directory for JCM model files and plots')
    parser.add_argument('--ROOTInputs', action="store_true",
                        help='Input file is in ROOT format instead of coffea')
    parser.add_argument('-y', '--year', dest="year", default="RunII",
                        help="Year specifies trigger (and lumiMask for data)")
    parser.add_argument('--debug', action="store_true",
                        help='Enable debug output')
    parser.add_argument('-l', '--lumi', dest="lumi", default="1",
                        help="Luminosity for MC normalization: units [pb]")
    parser.add_argument('--combine_input_files', action="store_true",
                        help='Combine multiple input histogram files')
    parser.add_argument('--scipy_optimize', action="store_true",
                        help='Use scipy.optimize.minimize instead of curve_fit')
    parser.add_argument('-m', '--metadata', dest="metadata",
                        default="coffea4bees/plots/metadata/plotsJCM.yml",
                        help='Metadata file for plots configuration')
    parser.add_argument('--no-plots', dest="no_plots", action="store_true",
                        help='Skip creating plots')
    parser.add_argument('--zero_pseudotag', dest="zero_pseudotag", action="store_true",
                        help='Compute zero pseudotag probabilities and weights in output')
    parser.add_argument('--lowpt', dest="lowpt", action="store_true",
                        help='Use low-pT jet selection for JCM weights')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting JCM weight generation")

    # Create output directory
    if args.outputDir and not os.path.isdir(args.outputDir):
        os.makedirs(args.outputDir)
        logger.info(f"Created output directory: {args.outputDir}")

    # Setup output files
    output_file = os.path.join(args.outputDir, f"jetCombinatoricModel_{args.weightRegion}_{args.weightSet}.txt")
    output_file_yml = output_file.replace(".txt", ".yml")
    logger.info(f"Output files: {output_file} and {output_file_yml}")

    out_txt = open(output_file, "w")
    out_yml = open(output_file_yml, 'w')

    # Select histogram names based on lowpt flag
    if args.lowpt:
        logging.info("Using low-pT jet selection")
        selJets = "selJets_noJCM_lowpt.n"
        tagJets = "tagJets_noJCM_lowpt.n"
        fourTaglabel = "lowpt_fourTag"
        threeTaglabel = "lowpt_threeTag"
    else:
        selJets = "selJets_noJCM.n"
        tagJets = "tagJets_noJCM.n"
        fourTaglabel = "fourTag"
        threeTaglabel = "threeTag"

    try:
        # Load config and histograms
        if not args.ROOTInputs:
            cfg.plotConfig = load_config_4b(args.metadata)
            cfg.hists = load_hists(args.inputFile)
            cfg.combine_input_files = args.combine_input_files
            cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
            cfg.set_hist_key("hists")

        # Load histograms from coffea files
        cutDict = get_cut_dict(args.cut, cfg.cutList)
        year_val = sum if args.year == "RunII" else args.year
        region_val = sum if args.weightRegion in ["sum", sum] else args.weightRegion

        base_dict = {"year": year_val, "region": region_val}
        fourTag_data_dict = {"process": args.data4bName, "tag": fourTaglabel} | base_dict | cutDict
        threeTag_data_dict = {"process": 'data', "tag": threeTaglabel} | base_dict | cutDict

        ttbar_list = ['TTTo2L2Nu', 'TTToSemiLeptonic', 'TTToHadronic']
        fourTag_ttbar_dict = {"process": ttbar_list, "tag": fourTaglabel} | base_dict | cutDict
        threeTag_ttbar_dict = {"process": ttbar_list, "tag": threeTaglabel} | base_dict | cutDict

        hists = cfg.hists[0]['hists']
        hists_data_4b = next((h['hists'] for h in cfg.hists 
                              if selJets in h['hists'] and args.data4bName in h['hists'][selJets].axes["process"]), None)
        if hists_data_4b is None:
            raise ValueError(f"Could not find histograms for data4bName={args.data4bName}")

        data4b = hists_data_4b[selJets][fourTag_data_dict]
        data4b_nTagJets = hists_data_4b[tagJets][fourTag_data_dict]
        data3b = hists[selJets][threeTag_data_dict]
        data3b_nTagJets = hists[tagJets][threeTag_data_dict]
        tt4b = hists[selJets][fourTag_ttbar_dict][sum, :]
        tt4b_nTagJets = hists[tagJets][fourTag_ttbar_dict][sum, :]
        tt3b = hists[selJets][threeTag_ttbar_dict][sum, :]
        tt3b_nTagJets = hists[tagJets][threeTag_ttbar_dict][sum, :]

        # Calculate QCD backgrounds (data - ttbar)
        qcd4b = copy(data4b)
        qcd4b.view().value = data4b.values() - tt4b.values()
        qcd4b.view().variance = data4b.variances() + tt4b.variances()

        qcd3b = copy(data3b)
        qcd3b.view().value = data3b.values() - tt3b.values()
        qcd3b.view().variance = data3b.variances() + tt3b.variances()

        qcd3b_nTightTags = copy(data3b_nTagJets)
        qcd3b_nTightTags.view().value = data3b_nTagJets.values() - tt3b_nTagJets.values()
        qcd3b_nTightTags.view().variance = data3b_nTagJets.variances() + tt3b_nTagJets.variances()

        # Prepare histograms - put tagged jet counts in first bins
        if args.lowpt:
            data4b_values, data4b_vars = np.zeros(len(data4b.values())), np.zeros(len(data4b.variances()))
            data4b_values[0:3], data4b_values[5:14] = data4b_nTagJets.values()[1:4], data4b.values()[1:10]
            data4b_vars[0:3], data4b_vars[5:14] = data4b_nTagJets.variances()[1:4], data4b.variances()[1:10]
            
            tt4b_values, tt4b_vars = np.zeros(len(tt4b.values())), np.zeros(len(tt4b.variances()))
            tt4b_values[0:3], tt4b_values[5:14] = tt4b_nTagJets.values()[1:4], tt4b.values()[1:10]
            tt4b_vars[0:3], tt4b_vars[5:14] = tt4b_nTagJets.variances()[1:4], tt4b.variances()[1:10]
        else:
            data4b_values, data4b_vars = data4b.values().copy(), data4b.variances().copy()
            data4b_values[0:4], data4b_vars[0:4] = data4b_nTagJets.values()[4:8], data4b_nTagJets.variances()[4:8]
            
            tt4b_values, tt4b_vars = tt4b.values().copy(), tt4b.variances().copy()
            tt4b_values[0:4], tt4b_vars[0:4] = tt4b_nTagJets.values()[4:8], tt4b_nTagJets.variances()[4:8]

        data4b.view().value, data4b.view().variance = data4b_values, data4b_vars
        tt4b.view().value, tt4b.view().variance = tt4b_values, tt4b_vars

        # Calculate QCD scale factor and three-tag fraction
        mu_qcd = np.sum(qcd4b.values()) / np.sum(qcd3b.values())
        threeTightTagFraction = (qcd3b_nTightTags.values()[3] / np.sum(qcd3b_nTightTags.values()) 
                                 if np.sum(qcd3b_nTightTags.values()) > 0 else 0.0)

        logger.info(f"QCD scale factor (mu_qcd): {mu_qcd:.6f}")
        logger.info(f"Three tight tag fraction: {threeTightTagFraction:.6f}")

        # Log event counts
        logger.info("Event counts (Unweighted):")
        logger.info(f"  data4b: {np.sum(data4b.values()):.1f}")
        logger.info(f"  data3b: {np.sum(data3b.values()):.1f}")
        logger.info(f"  tt4b:   {np.sum(tt4b.values()):.1f}")
        logger.info(f"  tt3b:   {np.sum(tt3b.values()):.1f}")
        logger.info(f"  qcd3b:  {np.sum(qcd3b.values()):.1f}")

        # Update error calculations
        mu_qcd_bin_by_bin = np.zeros(len(qcd4b.values()))
        qcd3b_non_zero = qcd3b.values() > 0
        mu_qcd_bin_by_bin[qcd3b_non_zero] = np.abs(qcd4b.values()[qcd3b_non_zero] / qcd3b.values()[qcd3b_non_zero])
        mu_qcd_bin_by_bin[mu_qcd_bin_by_bin < 0] = 0

        data3b_error = np.sqrt(data3b.variances()) * mu_qcd_bin_by_bin
        data3b_variances = data3b_error**2

        data4b_variance = data4b.variances()
        data4b_variance[data4b_variance == 0] = 1.17

        combined_variances = data4b.variances() + data3b_variances + tt4b.variances() + tt3b.variances()
        data4b.view().variance = combined_variances

        # Log bin errors
        logger.info("Bin errors overview:")
        logger.info("bin, x | value  | data4b_err, data3b_err, tt4b_err, tt3b_err, increase%")
        for ibin in range(len(data4b.values()) - 1):
            x = data4b.axes[0].centers[ibin] - 0.5
            prev_err = np.sqrt(data4b.variances()[ibin])
            new_err = np.sqrt(combined_variances[ibin])
            increase = 100 * new_err / prev_err if prev_err else 100
            logger.info(f"{ibin:2}, {x:2.0f}| {data4b.values()[ibin]:9.1f} | {increase:5.0f}%")

        # Extract data for fitting (first 15 bins)
        bin_centers = data4b.axes[0].centers[:15]
        if bin_centers[0] == 0.5:
            bin_centers = bin_centers - 0.5
        bin_values = data4b.values()[:15]
        bin_errors = np.sqrt(data4b.variances()[:15])
        bin_errors[bin_errors == 0] = 1.17
        
        tt4b_nTagJets_values = tt4b_nTagJets.values()[:15]
        tt4b_nTagJets_errors = np.sqrt(tt4b_nTagJets.variances()[:15])
        tt4b_values = tt4b.values()[:15]
        qcd3b_values = qcd3b.values()[:15]
        qcd3b_errors = np.sqrt(qcd3b.variances()[:15])

        # Setup and fit model
        JCM_model = jetCombinatoricModel(
            tt4b_nTagJets=tt4b_nTagJets_values,
            tt4b_nTagJets_errors=tt4b_nTagJets_errors,
            qcd3b=qcd3b_values,
            qcd3b_errors=qcd3b_errors,
            tt4b=tt4b_values,
            nbt=3,
        )

        # Set fixed parameters
        if args.fix_e:
            logger.info("Fixing pairEnhancement to 0.0")
            JCM_model.fixParameter_combination({
                "threeTightTagFraction": threeTightTagFraction,
                "pairEnhancement": 0.0,
                "pairEnhancementDecay": 1.0
            })
        elif args.fix_d:
            logger.info("Fixing pairEnhancementDecay to 1.0")
            JCM_model.fixParameter_combination({
                "threeTightTagFraction": threeTightTagFraction,
                "pairEnhancementDecay": 1.0
            })
        else:
            logger.info(f"Fixing threeTightTagFraction to {threeTightTagFraction:.6f}")
            JCM_model.fixParameter_combination({"threeTightTagFraction": threeTightTagFraction})

        # Perform fit
        residuals, pulls = JCM_model.fit(bin_centers, bin_values, bin_errors,
                                         scipy_optimize=args.scipy_optimize)

        logger.info(f"Fit results: chi^2/ndf = {JCM_model.fit_chi2/JCM_model.fit_ndf:.2f}, p-value = {JCM_model.fit_prob:.6f}")
        logger.info("Fit parameters:")
        JCM_model.dump()

        # Write output
        logger.info("Writing output files")
        
        def write_param(text, value):
            out_txt.write(f"{text:<30} {value}\n")
            out_yml.write(f"{text}:\n        {value}\n")

        for parameter in JCM_model.parameters:
            write_param(parameter["name"] + "_" + args.cut, parameter["value"])
            write_param(parameter["name"] + "_" + args.cut + "_err", parameter["error"])
            write_param(parameter["name"] + "_" + args.cut + "_pererr", parameter["percentError"])

        write_param("mu_qcd", mu_qcd)
        write_param("chi^2", JCM_model.fit_chi2)
        write_param("ndf", JCM_model.fit_ndf)
        write_param("chi^2/ndf", JCM_model.fit_chi2 / JCM_model.fit_ndf)
        write_param("p-value", JCM_model.fit_prob)

        # Validation
        try:
            validation_idx = 5
            nXb_true = data4b_nTagJets.values()[validation_idx]
            bin_centers_limited = bin_centers[:10].astype(int)
            nTag_pred = JCM_model.nTagPred(bin_centers_limited + 4)
            nXb_pred = nTag_pred["values"][validation_idx]
            nXb_pred_error = nTag_pred["errors"][validation_idx]
            sigma_pull = (nXb_true - nXb_pred) / nXb_pred_error if nXb_pred_error > 0 else 0
            logger.info(f"Fitted 5b events: {nXb_pred:5.1f} +/- {nXb_pred_error:5f}, Actual: {nXb_true:5.1f} ({sigma_pull:3.1f} sigma)")
            write_param("n5b_pred", nXb_pred)
            write_param("n5b_true", nXb_true)
        except Exception as e:
            logger.warning(f"Could not compute validation: {e}")

        # Write weights
        comb_weights, _ = JCM_model.getCombinatoricWeightList()
        write_param("JCM_weights", comb_weights)

        out_txt.close()
        out_yml.close()
        logger.info("Model output saved successfully")

        # Create plots if requested
        if not args.no_plots and not args.ROOTInputs:
            logger.info("Creating plots")
            
            # Jet multiplicity plot
            nJet_pred = JCM_model.nJetPred_values(bin_centers.astype(int))
            nJet_pred[0:4] = 0
            
            try:
                fig, ax = makePlot(
                    cfg,
                    var=selJets,
                    cut=args.cut,
                    axis_opts={"region": args.weightRegion, "tag": "fourTag"},
                    doRatio=True,
                    xlim=[4, 15],
                    rlim=[0, 2],
                    debug=False
                )
                fit_text = ""
                param_names = {"pseudoTagProb": "f", "pairEnhancement": "e", "pairEnhancementDecay": "d"}
                for param in JCM_model.parameters:
                    if param["name"] != "threeTightTagFraction":
                        fit_text += f"  {param_names[param['name']]} = {param['value']:.2f} +/- {param['error']:.3f}  ({param['percentError']:.1f}%)\n"
                fit_text += f"  χ² / DoF = {JCM_model.fit_chi2:.1f} / {JCM_model.fit_ndf} = {JCM_model.fit_chi2/JCM_model.fit_ndf:.1f}\n"
                fit_text += f"  p-value: {100*JCM_model.fit_prob:.0f}%\n"
                plt.text(10, 6, "Fit Result:", fontsize=20, color='black', fontweight='bold', ha='left', va='center')
                plt.text(10, 5.15, fit_text, fontsize=15, color='black', ha='left', va='center')
                fig.savefig(os.path.join(args.outputDir, f"{selJets}.pdf"))
                logger.info(f"Saved jet multiplicity plot")
            except Exception as e:
                logger.error(f"Failed to create jet multiplicity plot: {e}")

            # Tagged jets plot
            try:
                fig, ax = makePlot(
                    cfg,
                    var=tagJets,
                    cut=args.cut,
                    axis_opts={"region": args.weightRegion, "tag": "fourTag"},
                    doRatio=True,
                    xlim=[4, 8],
                    yscale="log",
                    rlim=[0.8, 1.2],
                    ylim=[0.1, None],
                    debug=True
                )
                fig.savefig(os.path.join(args.outputDir, f"{tagJets}.pdf"))
                logger.info(f"Saved tagged jets plot")
            except Exception as e:
                logger.warning(f"Failed to create tagged jets plot: {e}")

        logger.info("JCM weight generation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error in JCM weight generation: {e}", exc_info=True)
        out_txt.close()
        out_yml.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
