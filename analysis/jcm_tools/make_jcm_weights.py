#!/usr/bin/env python3
"""
JCM (Jet Combinatoric Model) Weight Generator

This script produces weights for the Jet Combinatoric Model (JCM) used
in HH→4b analysis to model the combinatorial background from 3-tag events.
It performs a fit to the jet multiplicity distribution and computes weights
to apply to the 3-tag sample to model the 4-tag background.

Author: Coffea4bees team
"""

import sys
import argparse
import logging
import numpy as np
import os
import matplotlib.pyplot as plt

# Add the current directory to the path
sys.path.insert(0, os.getcwd())
from coffea4bees.plots.plots import load_config_4b
from src.plotting.iPlot_config import plot_config
from coffea4bees.analysis.jcm_tools.jcm_model import jetCombinatoricModel
from coffea4bees.analysis.jcm_tools.helpers import loadHistograms, data_from_Hist, prepHists
from src.plotting.plots import load_hists, read_axes_and_cuts, makePlot

cfg = plot_config()


def write_to_JCM_file(text, value, txt_file, yml_file):
    """Write a parameter and its value to both text and YAML JCM files."""
    txt_file.write(f"{text:<30} {value}\n")
    yml_file.write(f"{text}:\n        {value}\n")


def process_histograms(data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets,
                      tt4b_nTagJets, qcd3b_nTightTags, args, logger):
    """Process histograms and extract data for fitting."""
    # Prepare histograms
    prepHists(data4b, qcd3b, tt4b, data4b_nTagJets, tt4b_nTagJets, lowpt=args.lowpt)

    # Calculate QCD scale factor and three tight tag fraction
    mu_qcd = np.sum(qcd4b.values()) / np.sum(qcd3b.values())
    threeTightTagFraction = (qcd3b.values()[4] / np.sum(qcd3b.values())) if args.lowpt else (qcd3b_nTightTags.values()[3] / np.sum(qcd3b_nTightTags.values()))

    logger.info(f"QCD scale factor (mu_qcd): {mu_qcd:.6f}")
    logger.info(f"Three tight tag fraction: {threeTightTagFraction:.6f}")
    logger.info("Event counts (Unweighted):")
    logger.info(f"  data4b: {np.sum(data4b.values()):.1f}")
    logger.info(f"  data3b: {np.sum(data3b.values()):.1f}")
    logger.info(f"  tt4b:   {np.sum(tt4b.values()):.1f}")
    logger.info(f"  tt3b:   {np.sum(tt3b.values()):.1f}")
    logger.info(f"  qcd3b:  {np.sum(qcd3b.values()):.1f}")

    # Calculate errors with bin-by-bin QCD scaling
    mu_qcd_bin_by_bin = np.zeros(len(qcd4b.values()))
    qcd3b_non_zero = qcd3b.values() > 0
    mu_qcd_bin_by_bin[qcd3b_non_zero] = np.abs(qcd4b.values()[qcd3b_non_zero] / qcd3b.values()[qcd3b_non_zero])
    mu_qcd_bin_by_bin[mu_qcd_bin_by_bin < 0] = 0
    
    data3b_error = np.sqrt(data3b.variances()) * mu_qcd_bin_by_bin
    combined_variances = data4b.variances() + data3b_error**2 + tt4b.variances() + tt3b.variances()
    previous_error = np.sqrt(data4b.variances())
    data4b.view().variance = combined_variances

    # Log error details
    tt4b_error = np.sqrt(tt4b.variances())
    tt3b_error = np.sqrt(tt3b.variances())
    logger.info("Bin errors overview:")
    logger.info("bin, x | value  | data4b_err, data3b_err, tt4b_err, tt3b_err, increase%")
    for ibin in range(len(data4b.values()) - 1):
        x = data4b.axes[0].centers[ibin] - 0.5
        increase = 100 * np.sqrt(combined_variances[ibin]) / previous_error[ibin] if previous_error[ibin] else 100
        logger.info(f"{ibin:2}, {x:2.0f}| {data4b.values()[ibin]:9.1f} | {previous_error[ibin]:5.1f}, {data3b_error[ibin]:5.1f}, {tt4b_error[ibin]:5.1f}, {tt3b_error[ibin]:5.1f}, {increase:5.0f}%")

    # Extract data for fitting
    bin_centers, bin_values, bin_errors = data_from_Hist(data4b)
    _, tt4b_nTagJets_values, tt4b_nTagJets_errors = data_from_Hist(tt4b_nTagJets)
    _, tt4b_values, _ = data_from_Hist(tt4b)
    _, qcd3b_values, qcd3b_errors = data_from_Hist(qcd3b)
    bin_errors[bin_errors == 0] = 1.17  # Minimum Poisson errors

    return (bin_centers, bin_values, bin_errors, tt4b_nTagJets_values,
            tt4b_nTagJets_errors, tt4b_values, qcd3b_values, qcd3b_errors,
            mu_qcd, threeTightTagFraction, data4b_nTagJets)


def setup_and_fit_model(bin_data, args, logger):
    """Set up JCM model, perform fit, and return fitted model."""
    (bin_centers, bin_values, bin_errors, tt4b_nTagJets_values, tt4b_nTagJets_errors,
     tt4b_values, qcd3b_values, qcd3b_errors, _, threeTightTagFraction, _) = bin_data

    # Initialize model
    JCM_model = jetCombinatoricModel(
        tt4b_nTagJets=tt4b_nTagJets_values,
        tt4b_nTagJets_errors=tt4b_nTagJets_errors,
        qcd3b=qcd3b_values,
        qcd3b_errors=qcd3b_errors,
        tt4b=tt4b_values,
    )

    # Set fixed parameters
    if args.fix_e:
        logger.info("Fixing pairEnhancement parameter to 0.0")
        JCM_model.fixParameter_combination({
            "threeTightTagFraction": threeTightTagFraction,
            "pairEnhancement": 0.0,
            "pairEnhancementDecay": 1.0
        })
    elif args.fix_d:
        logger.info("Fixing pairEnhancementDecay parameter to 1.0")
        JCM_model.fixParameter_combination({
            "threeTightTagFraction": threeTightTagFraction,
            "pairEnhancementDecay": 1.0
        })
    else:
        logger.info(f"Fixing threeTightTagFraction to {threeTightTagFraction:.6f}")
        JCM_model.fixParameter_combination({"threeTightTagFraction": threeTightTagFraction})

    # Perform fit
    if args.debug:
        logger.debug("Bin information before fitting:")
        for ibin, center in enumerate(bin_centers):
            logger.debug(f"Bin {ibin}: center={center}, value={bin_values[ibin]}, error={bin_errors[ibin]}")

    residuals, pulls = JCM_model.fit(bin_centers, bin_values, bin_errors, scipy_optimize=args.scipy_optimize)

    # Log fit results
    logger.info(f"Fit results:")
    logger.info(f"chi^2 = {JCM_model.fit_chi2:.2f}  ndf = {JCM_model.fit_ndf} " +
                f"chi^2/ndf = {JCM_model.fit_chi2/JCM_model.fit_ndf:.2f} | " +
                f"p-value = {JCM_model.fit_prob:.6f}")
    logger.info("Pulls (residual/error):")
    for iBin, res in enumerate(residuals):
        logger.info(f"Bin {iBin:2}| {res:5.1f} / {bin_errors[iBin]:5.1f} = {pulls[iBin]:4.1f}")
    logger.info("Fit parameters:")
    JCM_model.dump()

    return JCM_model


def save_model_output(JCM_model, bin_data, args, logger, txt_file, yml_file):
    """Save model parameters and predictions to output files."""
    mu_qcd = bin_data[8]
    bin_centers = bin_data[0]
    data4b_nTagJets = bin_data[10]

    # Write fit parameters
    logger.info("Writing model parameters to output files")
    for parameter in JCM_model.parameters:
        write_to_JCM_file(f"{parameter['name']}_{args.cut}", parameter["value"], txt_file, yml_file)
        write_to_JCM_file(f"{parameter['name']}_{args.cut}_err", parameter["error"], txt_file, yml_file)
        write_to_JCM_file(f"{parameter['name']}_{args.cut}_pererr", parameter["percentError"], txt_file, yml_file)

    # Write fit metrics
    write_to_JCM_file("mu_qcd", mu_qcd, txt_file, yml_file)
    write_to_JCM_file("chi^2", JCM_model.fit_chi2, txt_file, yml_file)
    write_to_JCM_file("ndf", JCM_model.fit_ndf, txt_file, yml_file)
    write_to_JCM_file("chi^2/ndf", JCM_model.fit_chi2 / JCM_model.fit_ndf, txt_file, yml_file)
    write_to_JCM_file("p-value", JCM_model.fit_prob, txt_file, yml_file)

    # 5b event validation
    try:
        n5b_true = data4b_nTagJets.values()[5]
        nTag_pred = JCM_model.nTagPred(bin_centers.astype(int) + 4)
        n5b_pred = nTag_pred["values"][5]
        n5b_pred_error = nTag_pred["errors"][5]
        sigma_pull = (n5b_true-n5b_pred)/n5b_pred_error if n5b_pred_error > 0 else 0
        
        logger.info(f"Fitted number of 5b events: {n5b_pred:5.1f} +/- {n5b_pred_error:5f}")
        logger.info(f"Actual number of 5b events: {n5b_true:5.1f}, ({sigma_pull:3.1f} sigma pull)")
        
        write_to_JCM_file("n5b_pred", n5b_pred, txt_file, yml_file)
        write_to_JCM_file("n5b_true", n5b_true, txt_file, yml_file)
    except Exception as e:
        logger.warning(f"Could not compute 5b event predictions: {e}")

    # Write combinatoric weights
    comb_weights, _ = JCM_model.getCombinatoricWeightList()
    write_to_JCM_file("JCM_weights", comb_weights, txt_file, yml_file)
    logger.debug(f"Combinatoric weight list: {comb_weights}")
    
    txt_file.close()
    yml_file.close()
    logger.info("Model output saved successfully")




def fill_histogram_with_predictions(hist, selJets, args, nJet_pred, logger):
    """Fill histogram with JCM predictions."""
    # Add dummy values to register the JCM process
    dummy_data = {
        'process': ['JCM'],
        'year': ['UL18'], 
        'tag': "lowpt_fourTag" if args.lowpt else "fourTag", 
        'region': "SB",
        'passPreSel': [True], 
        'n': [0],
    }

    # Check if histogram has SvB axes
    hist_axes = hist.axes
    axis_names = [axis.name for axis in hist_axes]
    has_passSvB = 'passSvB' in axis_names
    has_failSvB = 'failSvB' in axis_names
    
    if has_passSvB or has_failSvB:
        dummy_data['passSvB'] = [False]
        dummy_data['failSvB'] = [False]

    hist.fill(**dummy_data)

    # Fill with predicted values
    index_dict = {
        "process": "JCM", 
        "year": "UL18", 
        "tag": "lowpt_fourTag" if args.lowpt else "fourTag", 
        "region": "SB", 
        "passPreSel": True
    }
    if has_passSvB:
        index_dict["passSvB"] = False
    if has_failSvB:
        index_dict["failSvB"] = False

    for iBin in range(len(nJet_pred)):
        index_dict["n"] = iBin
        hist[tuple(index_dict.values())] = (nJet_pred[iBin], 0)


def create_plots(JCM_model, bin_data, args, mu_qcd, selJets, tagJets, logger):
    """Create diagnostic plots for the JCM model."""
    if args.no_plots or args.ROOTInputs:
        logger.info("Skipping plot creation")
        return

    logger.info("Creating plots")
    bin_centers = bin_data[0]

    # Scale QCD by mu_qcd
    for p in ["data_3tag", "TTTo2L2Nu_3tag", "TTToSemiLeptonic_3tag", "TTToHadronic_3tag"]:
        if p in cfg.plotConfig["stack"]["MultiJet"]["sum"]:
            cfg.plotConfig["stack"]["MultiJet"]["sum"][p]["scalefactor"] *= mu_qcd

    # Plot jet multiplicity
    nJet_pred = JCM_model.nJetPred_values(bin_centers.astype(int))
    if args.lowpt:
        nJet_pred[0] = 0
        nJet_pred[1:-3] = nJet_pred[4:]
    else:
        nJet_pred[0:4] = 0

    try:
        fill_histogram_with_predictions(cfg.hists[0]['hists'][selJets], selJets, args, nJet_pred, logger)

        fig, ax = makePlot(
            cfg, var=selJets, cut=args.cut, axis_opts={"region": args.weightRegion},
            doRatio=True, xlim=[0, 10] if args.lowpt else [4, 15], rlim=[0, 2], debug=False
        )

        # Add fit information
        fit_text = ""
        param_names = {"pseudoTagProb": "f", "pairEnhancement": "e", "pairEnhancementDecay": "d"}
        for param in JCM_model.parameters:
            if param["name"] != "threeTightTagFraction":
                fit_text += f"  {param_names[param['name']]} = {round(param['value'],2)} +/- {round(param['error'],3)}  ({round(param['percentError'],1)}%)\n"
        fit_text += f"  $\\chi^2$ / DoF = {round(JCM_model.fit_chi2,1)} / {JCM_model.fit_ndf} = {round(JCM_model.fit_chi2/JCM_model.fit_ndf,1)}\n"
        fit_text += f"  p-value: {round(100*JCM_model.fit_prob)}%\n"

        plt.text(6 if args.lowpt else 10, 6, "Fit Result:", fontsize=20, color='black', 
                fontweight='bold', horizontalalignment='left', verticalalignment='center')
        plt.text(6 if args.lowpt else 10, 5.15, fit_text, fontsize=15, color='black',
                horizontalalignment='left', verticalalignment='center')

        fig.savefig(os.path.join(args.outputDir, "selJets_noJCM_n.pdf"))
        logger.info(f"Saved jet multiplicity plot to {os.path.join(args.outputDir, 'selJets_noJCM_n.pdf')}")
    except Exception as e:
        logger.error(f"Failed to create jet multiplicity plot: {e}")

    # Plot tagged jets
    try:
        nTag_pred = JCM_model.nTagPred(bin_centers.astype(int) + 4)["values"]
        if args.lowpt: 
            nTag_pred[1:-3] = nTag_pred[4:]

        fill_histogram_with_predictions(cfg.hists[0]['hists'][tagJets], tagJets, args, nTag_pred, logger)

        fig, ax = makePlot(
            cfg, var=tagJets, cut=args.cut, axis_opts={"region": args.weightRegion},
            doRatio=True, xlim=[1, 6] if args.lowpt else [4, 8], yscale="log", 
            rlim=[0.8, 1.2], ylim=[0.1, None]
        )

        fig.savefig(os.path.join(args.outputDir, "tagJets_noJCM_n.pdf"))
        logger.info(f"Saved tagged jets plot to {os.path.join(args.outputDir, 'tagJets_noJCM_n.pdf')}")
    except Exception as e:
        logger.warning(f"Failed to create tagged jets plot: {e}")


def main():
    """Main function to run the JCM weight generation process."""
    parser = argparse.ArgumentParser(description='Make Jet Combinatoric Model weights',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', '--weightSet', default="", help='Label for the weight set')
    parser.add_argument('-r', dest="weightRegion", default="SB", help='Weight region (e.g. SB for sideband)')
    parser.add_argument('--data4bName', default="data", help='Name of the 4b data process')
    parser.add_argument('-c', dest="cut", default="passPreSel", help='Cut to apply')
    parser.add_argument('-fix_e', action="store_true", help='Fix pairEnhancement parameter to 0')
    parser.add_argument('-fix_d', action="store_true", help='Fix pairEnhancementDecay parameter to 1')
    parser.add_argument('-i', '--inputFile', nargs="+", default='hists.pkl', help='Input file(s)')
    parser.add_argument('-o', '--outputDir', default="", help='Output directory')
    parser.add_argument('--ROOTInputs', action="store_true", help='Input is ROOT format')
    parser.add_argument('-y', '--year', default="RunII", help="Year for trigger")
    parser.add_argument('--debug', action="store_true", help='Enable debug output')
    parser.add_argument('--scipy_optimize', action="store_true", help='Use scipy.optimize')
    parser.add_argument('-m', '--metadata', default="coffea4bees/plots/metadata/plotsJCM.yml", help='Metadata file')
    parser.add_argument('--no-plots', dest="no_plots", action="store_true", help='Skip plots')
    parser.add_argument('--lowpt', action="store_true", help='Use low pt selection')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('JCM')
    logger.info("Starting JCM weight generation")

    # Create output directory
    if args.outputDir and not os.path.isdir(args.outputDir):
        os.makedirs(args.outputDir)
        logger.info(f"Created output directory: {args.outputDir}")

    # Open output files
    output_name = os.path.join(args.outputDir, f"jetCombinatoricModel_{args.weightRegion}_{args.weightSet}.txt")
    logger.info(f"Output files: {output_name} and .yml version")
    txt_file = open(output_name, "w")
    yml_file = open(output_name.replace(".txt", ".yml"), 'w')

    selJets = "selJets_noJCM_lowpt.n" if args.lowpt else "selJets_noJCM.n"
    tagJets = "tagJets_noJCM_lowpt.n" if args.lowpt else "tagJets_noJCM.n"

    try:
        # Load configuration for coffea inputs
        if not args.ROOTInputs:
            cfg.plotConfig = load_config_4b(args.metadata)
            cfg.hists = load_hists(args.inputFile)
            cfg.combine_input_files = getattr(args, 'combine_input_files', False)
            cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
            cfg.set_hist_key("hists")

        # Load histograms
        histograms = loadHistograms(
            inputFile=args.inputFile[0],
            format='ROOT' if args.ROOTInputs else 'coffea',
            cfg=cfg if not args.ROOTInputs else None,
            cut=args.cut,
            year=args.year,
            weightRegion=args.weightRegion,
            data4bName=args.data4bName,
            taglabel4b="lowpt_fourTag" if args.lowpt else "fourTag",
            taglabel3b="lowpt_threeTag" if args.lowpt else "threeTag",
            selJets=selJets,
            tagJets=tagJets,
        )

        # Process histograms
        bin_data = process_histograms(*histograms, args, logger)

        # Setup model and fit
        JCM_model = setup_and_fit_model(bin_data, args, logger)

        # Save results
        save_model_output(JCM_model, bin_data, args, logger, txt_file, yml_file)

        # Create plots
        create_plots(JCM_model, bin_data, args, bin_data[8], selJets, tagJets, logger)

        logger.info("JCM weight generation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error in JCM weight generation: {e}", exc_info=True)
        txt_file.close()
        yml_file.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
