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
from copy import copy
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # no GUI, renders directly to files
import matplotlib.pyplot as plt
from hist import Hist
from typing import Dict, Tuple, List, Optional, Union, Any
import yaml

# Add the current directory to the path
sys.path.insert(0, os.getcwd())
from coffea4bees.plots.plots import load_config_4b
from src.plotting.iPlot_config import plot_config
cfg = plot_config()

from coffea4bees.analysis.jcm_tools.jcm_model import jetCombinatoricModel
from coffea4bees.analysis.jcm_tools.helpers import (
    loadHistograms,
    data_from_Hist,
    prepHists
)
from src.plotting.plots import load_hists, read_axes_and_cuts, makePlot


def write_to_JCM_file(text: str, value: Any, jetCombinatoricModelFile, jetCombinatoricModelFile_yml) -> None:
    """Write a parameter and its value to both the text and YAML JCM files.

    Args:
        text: The parameter name/key
        value: The parameter value
        jetCombinatoricModelFile: The text file object
        jetCombinatoricModelFile_yml: The YAML file object
    """
    jetCombinatoricModelFile.write(f"{text:<30} {value}\n")
    jetCombinatoricModelFile_yml.write(f"{text}:\n")
    jetCombinatoricModelFile_yml.write(f"        {value}\n")

def process_histograms(data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets,
                       tt4b_nTagJets, qcd3b_nTightTags, args: argparse.Namespace, logger: logging.Logger, jcm_config : dict) -> Tuple:
    """Process the histograms and extract data for fitting

    Args:
        data4b, data3b, tt4b, tt3b, qcd4b, qcd3b: Histogram objects
        data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags: Tag jet histograms
        args: Command line arguments
        logger: Logger instance
        jcm_config: jcm configuration dictionary

    Returns:
        Tuple of (bin_centers, bin_values, bin_errors, tt4b_nTagJets_values,
                 tt4b_nTagJets_errors, tt4b_values, qcd3b_values, qcd3b_errors,
                 mu_qcd, threeTightTagFraction)
    """
    # Prepare histograms
    prepHists(data4b, qcd3b, tt4b, data4b_nTagJets, tt4b_nTagJets, lowpt=args.lowpt)

    # Calculate QCD scale factor
    mu_qcd = np.sum(qcd4b.values()) / np.sum(qcd3b.values())

    if jcm_config.get("threeTightTagFraction", False):
        logger.info(f"Setting QCD scale factor by hand!!!  to {jcm_config.get('threeTightTagFraction')}")
        threeTightTagFraction = float(jcm_config.get('threeTightTagFraction'))
    else:
        threeTightTagFraction = (qcd3b.values()[4] / np.sum(qcd3b.values())) if args.lowpt else (qcd3b_nTightTags.values()[3] / np.sum(qcd3b_nTightTags.values()))

    ignoreTT = jcm_config.get("ignoreTT", False)
    logger.info(f"Ignore TT component: {ignoreTT}")
    logger.info(f"QCD scale factor (mu_qcd): {mu_qcd:.6f}")
    logger.info(f"Three tight tag fraction: {threeTightTagFraction:.6f}")

    # Print event counts
    logger.info("Event counts (Unweighted):")
    logger.info(f"  data4b: {np.sum(data4b.values()):.1f}")
    logger.info(f"  data3b: {np.sum(data3b.values()):.1f}")
    if not ignoreTT:
        logger.info(f"  tt4b:   {np.sum(tt4b.values()):.1f}")
        logger.info(f"  tt3b:   {np.sum(tt3b.values()):.1f}")
    logger.info(f"  qcd3b:  {np.sum(qcd3b.values()):.1f}")

    # Update error calculations for better fit
    mu_qcd_bin_by_bin = np.zeros(len(qcd4b.values()))
    qcd3b_non_zero_filter = qcd3b.values() > 0
    mu_qcd_bin_by_bin[qcd3b_non_zero_filter] = np.abs(
        qcd4b.values()[qcd3b_non_zero_filter] / qcd3b.values()[qcd3b_non_zero_filter]
    )
    mu_qcd_bin_by_bin[mu_qcd_bin_by_bin < 0] = 0
    data3b_error = np.sqrt(data3b.variances()) * mu_qcd_bin_by_bin
    data3b_variances = data3b_error**2

    # Set minimum Poisson errors for data
    data4b_variance = data4b.variances()
    data4b_variance[data4b_variance == 0] = 1.17

    # Combine errors from all sources for a more robust fit
    if ignoreTT:
        combined_variances = data4b.variances() + data3b_variances
    else:
        combined_variances = data4b.variances() + data3b_variances + tt4b.variances() + tt3b.variances()

    combined_error = np.sqrt(combined_variances)
    previous_error = np.sqrt(data4b.variances())
    data4b.view().variance = combined_variances

    # Log error increases for debugging
    if not ignoreTT:
        tt4b_error = np.sqrt(tt4b.variances())
        tt3b_error = np.sqrt(tt3b.variances())

    logger.info("Bin errors overview:")
    logger.info("bin, x | value  | data4b_err, data3b_err, tt4b_err, tt3b_err, increase%")
    for ibin in range(len(data4b.values()) - 1):
        x = data4b.axes[0].centers[ibin] - 0.5
        increase = 100 * np.sqrt(data4b.variances()[ibin]) / previous_error[ibin] if previous_error[ibin] else 100
        if ignoreTT:
            logger.info(f"{ibin:2}, {x:2.0f}| {data4b.values()[ibin]:9.1f} | {previous_error[ibin]:5.1f}, {data3b_error[ibin]:5.1f},      N/A,      N/A, {increase:5.0f}%")
        else:
            logger.info(f"{ibin:2}, {x:2.0f}| {data4b.values()[ibin]:9.1f} | {previous_error[ibin]:5.1f}, {data3b_error[ibin]:5.1f}, {tt4b_error[ibin]:5.1f}, {tt3b_error[ibin]:5.1f}, {increase:5.0f}%")

    # Extract data for fitting
    bin_centers, bin_values, bin_errors = data_from_Hist(data4b)

    # Set minimum Poisson errors for empty bins
    bin_errors[bin_errors == 0] = 1.17

    _, qcd3b_values, qcd3b_errors = data_from_Hist(qcd3b)

    if ignoreTT:
        return (bin_centers, bin_values, bin_errors, None,
                None, None, qcd3b_values, qcd3b_errors,
                mu_qcd, threeTightTagFraction)

    _, tt4b_nTagJets_values, tt4b_nTagJets_errors = data_from_Hist(tt4b_nTagJets)
    _, tt4b_values, _ = data_from_Hist(tt4b)



    return (bin_centers, bin_values, bin_errors, tt4b_nTagJets_values,
            tt4b_nTagJets_errors, tt4b_values, qcd3b_values, qcd3b_errors,
            mu_qcd, threeTightTagFraction)


def setup_model(bin_data: Tuple, args: argparse.Namespace, logger: logging.Logger) -> jetCombinatoricModel:
    """Set up the JCM model for fitting

    Args:
        bin_data: Tuple of data from process_histograms
        args: Command line arguments
        logger: Logger instance

    Returns:
        Configured JCM model ready for fitting
    """
    (_, _, _, tt4b_nTagJets_values, tt4b_nTagJets_errors,
     tt4b_values, qcd3b_values, qcd3b_errors, _, threeTightTagFraction) = bin_data

    # Initialize model with data
    JCM_model = jetCombinatoricModel(
        tt4b_nTagJets=tt4b_nTagJets_values,
        tt4b_nTagJets_errors=tt4b_nTagJets_errors,
        qcd3b=qcd3b_values,
        qcd3b_errors=qcd3b_errors,
        tt4b=tt4b_values,
        lowpt_mode=args.lowpt,
    )

    # Log model setup
    logger.debug(f"Initialized JCM_model with fit parameters names: {[p['name'] for p in JCM_model.fit_parameters]}")
    logger.debug(f"Default parameters: {JCM_model.default_parameters}")
    logger.debug(f"Parameter bounds: {list(zip(JCM_model.parameters_lower_bounds, JCM_model.parameters_upper_bounds))}")

    # Set fixed parameters based on command-line options
    if args.fix_e:
        logger.info("Fixing pairEnhancement parameter to 0.0")
        JCM_model.fixParameter_combination({
            "threeTightTagFraction": threeTightTagFraction,
            "pairEnhancement": 0.0,
            "pairEnhancementDecay": 1.0,
            "tt4bSF": 1.0
        })

    elif args.fix_d:
        logger.info("Fixing pairEnhancementDecay parameter to 1.0")
        JCM_model.fixParameter_combination({
            "threeTightTagFraction": threeTightTagFraction,
            "pairEnhancementDecay": 1.0,
            "tt4bSF": 1.0
        })

    elif not args.float_tt4bSF:
        logger.info("Fixing ttbar4b SF to 1.0")
        JCM_model.fixParameter_combination({
            "threeTightTagFraction": threeTightTagFraction,
            "tt4bSF": 1.0
        })

    else:
        logger.info(f"Fixing threeTightTagFraction to {threeTightTagFraction:.6f}")
        JCM_model.fixParameter_combination({
            "threeTightTagFraction": threeTightTagFraction
        })

    return JCM_model


def perform_fit(JCM_model: jetCombinatoricModel, bin_data: Tuple,
                args: argparse.Namespace, logger: logging.Logger) -> Tuple:
    """Perform the JCM model fit

    Args:
        JCM_model: Configured jetCombinatoricModel instance
        bin_data: Tuple of data from process_histograms
        args: Command line arguments
        logger: Logger instance

    Returns:
        Tuple of (residuals, pulls)
    """
    bin_centers, bin_values, bin_errors = bin_data[0:3]

    # Log detailed bin information if in debug mode
    if args.debug:
        logger.debug("Bin information before fitting:")
        for ibin, center in enumerate(bin_centers):
            logger.debug(f"Bin {ibin}: center={center}, value={bin_values[ibin]}, error={bin_errors[ibin]}")

    # Perform the fit
    residuals, pulls = JCM_model.fit(bin_centers, bin_values, bin_errors,
                                     scipy_optimize=args.scipy_optimize)

    # Log fit results
    logger.info(f"Fit results:")
    logger.info(f"chi^2 = {JCM_model.fit_chi2:.2f}  ndf = {JCM_model.fit_ndf} " +
                f"chi^2/ndf = {JCM_model.fit_chi2/JCM_model.fit_ndf:.2f} | " +
                f"p-value = {JCM_model.fit_prob:.6f}")

    # Print the pulls
    logger.info("Pulls (residual/error):")
    for iBin, res in enumerate(residuals):
        logger.info(f"Bin {iBin:2}| {res:5.1f} / {bin_errors[iBin]:5.1f} = {pulls[iBin]:4.1f}")

    # Print the fit parameters
    logger.info("Fit parameters:")
    JCM_model.dump()

    return residuals, pulls


def save_model_output(JCM_model: jetCombinatoricModel, bin_data: Tuple, args: argparse.Namespace,
                     logger: logging.Logger, output_files: Tuple) -> None:
    """Save the model output to files

    Args:
        JCM_model: Fitted jetCombinatoricModel instance
        bin_data: Tuple of data from process_histograms
        args: Command line arguments
        logger: Logger instance
        output_files: Tuple of file objects (jetCombinatoricModelFile, jetCombinatoricModelFile_yml)
    """
    # Extract only the mu_qcd value from bin_data to avoid unpacking errors
    mu_qcd = bin_data[8] if len(bin_data) > 8 else None
    if mu_qcd is None:
        logger.warning("Could not extract mu_qcd from bin_data")
        mu_qcd = 1.0  # Default fallback value

    bin_centers = bin_data[0] if len(bin_data) > 0 else None

    jetCombinatoricModelFile, jetCombinatoricModelFile_yml = output_files

    # Write parameters to output files
    logger.info(f"Writing model parameters to output files")
    for parameter in JCM_model.parameters:
        write_to_JCM_file(
            parameter["name"] + "_" + args.cut,
            parameter["value"],
            jetCombinatoricModelFile,
            jetCombinatoricModelFile_yml
        )
        write_to_JCM_file(
            parameter["name"] + "_" + args.cut + "_err",
            parameter["error"],
            jetCombinatoricModelFile,
            jetCombinatoricModelFile_yml
        )
        write_to_JCM_file(
            parameter["name"] + "_" + args.cut + "_pererr",
            parameter["percentError"],
            jetCombinatoricModelFile,
            jetCombinatoricModelFile_yml
        )

    # Write fit metrics
    write_to_JCM_file("mu_qcd", mu_qcd, jetCombinatoricModelFile, jetCombinatoricModelFile_yml)
    write_to_JCM_file("chi^2", JCM_model.fit_chi2, jetCombinatoricModelFile, jetCombinatoricModelFile_yml)
    write_to_JCM_file("ndf", JCM_model.fit_ndf, jetCombinatoricModelFile, jetCombinatoricModelFile_yml)
    write_to_JCM_file("chi^2/ndf", JCM_model.fit_chi2 / JCM_model.fit_ndf, jetCombinatoricModelFile, jetCombinatoricModelFile_yml)
    write_to_JCM_file("p-value", JCM_model.fit_prob, jetCombinatoricModelFile, jetCombinatoricModelFile_yml)

    # Add information about validation bin for consistency check
    try:
        # Get the appropriate histogram for validation
        data4b_nTagJets = bin_data[-1] if isinstance(bin_data[-1], Hist) else None

        if data4b_nTagJets is not None and bin_centers is not None:
            # For lowpt: check 1b bin (index 1), for standard: check 5b bin (index 5)
            validation_bin = 1 if args.lowpt else 5
            validation_label = "1b" if args.lowpt else "5b"

            n_true = data4b_nTagJets.values()[validation_bin]

            # For lowpt: call nTagPred with [1] to get prediction for 1 lowpt tag
            # For standard: call nTagPred with bin_centers + 4
            if args.lowpt:
                nTag_pred = JCM_model.nTagPred(np.array([1]), lowpt=True)
                n_pred = nTag_pred["values"][1]  # Prediction for 1 lowpt tag
                n_pred_error = nTag_pred["errors"][1]
            else:
                nTag_pred = JCM_model.nTagPred(bin_centers.astype(int) + 4, lowpt=False)
                n_pred = nTag_pred["values"][validation_bin]
                n_pred_error = nTag_pred["errors"][validation_bin]

            sigma_pull = (n_true - n_pred) / n_pred_error if n_pred_error > 0 else 0

            logger.info(f"Fitted number of {validation_label} events: {n_pred:5.1f} +/- {n_pred_error:5f}")
            logger.info(f"Actual number of {validation_label} events: {n_true:5.1f}, ({sigma_pull:3.1f} sigma pull)")

            write_to_JCM_file(f"n{validation_label}_pred", n_pred, jetCombinatoricModelFile, jetCombinatoricModelFile_yml)
            write_to_JCM_file(f"n{validation_label}_true", n_true, jetCombinatoricModelFile, jetCombinatoricModelFile_yml)
        else:
            logger.warning(f"Missing data for {validation_label} event prediction")
    except (IndexError, AttributeError) as e:
        logger.warning(f"Could not compute validation predictions: {e}")

    # Write the event weights
    comb_weights, zerotag_comb_weights = JCM_model.getCombinatoricWeightList(lowpt=args.lowpt)
    write_to_JCM_file("JCM_weights", comb_weights, jetCombinatoricModelFile, jetCombinatoricModelFile_yml)

    # Consistency check using JCM_model directly
    logger.debug(f"Combinatoric weight list: {comb_weights}")

    # Close files
    jetCombinatoricModelFile.close()
    jetCombinatoricModelFile_yml.close()

    logger.info(f"Model output saved successfully")

def create_plots(
    JCM_model: jetCombinatoricModel,
    bin_data: Tuple,
    args: argparse.Namespace,
    mu_qcd: float,
    jcm_config: dict,
    logger: logging.Logger
) -> None:
    """Create plots for the JCM model

    Args:
        JCM_model: Fitted jetCombinatoricModel instance
        bin_data: Tuple of data from process_histograms
        args: Command line arguments
        mu_qcd: QCD scale factor
        selJets: Variable name for selected jets
        tagJets: Variable name for tagged jets
        logger: Logger instance
    """
    if args.no_plots or args.ROOTInputs:
        logger.info("Skipping plot creation")
        return

    logger.info("Creating plots")
    bin_centers = bin_data[0]

    selJets = jcm_config.get("selJets", "selJets_noJCM.n")
    tagJets = jcm_config.get("tagJets", "tagJets_noJCM.n")
    ignoreTT = jcm_config.get("ignoreTT", False)

    # Scale QCD by mu_qcd
    proc_list = ["data_3tag", "TTTo2L2Nu_3tag", "TTToSemiLeptonic_3tag", "TTToHadronic_3tag"]
    if ignoreTT: proc_list = ["data_3tag"]
    for p in proc_list:
        if p in cfg.plotConfig["stack"]["MultiJet"]["sum"]:
            cfg.plotConfig["stack"]["MultiJet"]["sum"][p]["scalefactor"] *= mu_qcd

    # Plot the jet multiplicity
    nJet_pred = JCM_model.nJetPred_values(bin_centers.astype(int))

    if args.lowpt:
        # For lowpt: first bin (index 0) is for 1 jet, bins start at index 1
        nJet_pred[0] = 0  # Zero out the 0-jet bin
        nJet_pred[1:-3] = nJet_pred[4:]  # Shift predictions
    else:
        # For standard: bins 0-3 are for 4-7 tags, zero them out
        nJet_pred[0:4] = 0

    # Add dummy values to register the JCM process
    dummy_data = {
        'process': ['JCM'],
        'year': ['UL18'],
        'tag': "lowpt_fourTag" if args.lowpt else "fourTag",
        'region': "SB",
        'passPreSel': [True],
        'n': [0],
    }

    # Check if we have the SvB variables and handle accordingly
    try:
        hist_axes = cfg.hists[0]['hists'][selJets].axes
        axis_names = [axis.name for axis in hist_axes]

        logger.debug(f"Histogram axes names: {axis_names}")

        has_passSvB = 'passSvB' in axis_names
        has_failSvB = 'failSvB' in axis_names

        if has_passSvB or has_failSvB:
            dummy_data['passSvB'] = [False]
            dummy_data['failSvB'] = [False]
            logger.debug("SvB variables found in histogram")
        else:
            logger.debug("No SvB variables in histogram")

        cfg.hists[0]['hists'][selJets].fill(**dummy_data)

    except Exception as e:
        logger.warning(f"Error analyzing histogram structure: {e}")
        cfg.hists[0]['hists'][selJets].fill(**dummy_data)
        has_passSvB = False
        has_failSvB = False

    # Overwrite with predicted values
    logger.debug("Setting predicted jet multiplicity values")

    try:
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

        for iBin in range(14):
            index_dict["n"] = iBin
            cfg.hists[0]['hists'][selJets][tuple(index_dict.values())] = (nJet_pred[iBin], 0)

    except Exception as e:
        logger.warning(f"Error setting histogram values, trying alternative approach: {e}")
        for iBin in range(14):
            try:
                hist_view = cfg.hists[0]['hists'][selJets].view()
                for idx, process in enumerate(hist_view.axes[0]):
                    if process == "JCM":
                        process_idx = idx
                        break
                if has_passSvB and has_failSvB:
                    hist_view[process_idx, 0, 1, 1, True, False, False, iBin] = (nJet_pred[iBin], 0)
                else:
                    hist_view[process_idx, 0, 1, 1, True, iBin] = (nJet_pred[iBin], 0)
            except Exception as inner_e:
                logger.error(f"Failed to set values for bin {iBin}: {inner_e}")

    # Plot options for jet multiplicity
    plot_options = {
        "doRatio": True,
        "xlim": [0, 10] if args.lowpt else [4, 15],
        "rlim": [0, 2],
        "debug": False
    }

    # Create jet multiplicity plot
    try:
        logger.info("Creating jet multiplicity plot")
        fig, ax = makePlot(
            cfg,
            var=selJets,
            cut=args.cut,
            axis_opts={"region": args.weightRegion},
            **plot_options
        )

        # Add fit information to the plot
        fit_text = ""
        plot_param_name = {
            "pseudoTagProb": "f",
            "pairEnhancement": "e",
            "pairEnhancementDecay": "d",
            "tt4bSF": "tt4bSF"
        }
        for parameter in JCM_model.parameters:
            if parameter["name"] in ["threeTightTagFraction","tt4bSF"]:
                continue
            fit_text += f"  {plot_param_name[parameter['name']]} = {round(parameter['value'], 2)} +/- {round(parameter['error'], 3)}  ({round(parameter['percentError'], 1)}%)\n"

        fit_text += f"  $\chi^2$ / DoF = {round(JCM_model.fit_chi2, 1)} / {JCM_model.fit_ndf} = {round(JCM_model.fit_chi2 / JCM_model.fit_ndf, 1)}\n"
        fit_text += f"  p-value: {round(100 * JCM_model.fit_prob)}%\n"

        plt.text(6 if args.lowpt else 10, 6, "Fit Result:", fontsize=20, color='black', fontweight='bold',
                horizontalalignment='left', verticalalignment='center')

        plt.text(6 if args.lowpt else 10, 5.15, fit_text, fontsize=15, color='black',
                horizontalalignment='left', verticalalignment='center')

        fig.savefig(os.path.join(args.outputDir, "selJets_noJCM_n.pdf"))
        logger.info(f"Saved jet multiplicity plot to {os.path.join(args.outputDir, 'selJets_noJCM_n.pdf')}")
    except Exception as e:
        logger.error(f"Failed to create jet multiplicity plot: {e}")

    # Plot tagged jets
    try:
        logger.info("Creating tagged jet multiplicity plot")
        cfg.hists[0]['hists'][tagJets].fill(**dummy_data)

        # Get N-tag jet predictions
        if args.lowpt:
            # For lowpt: predict for tag numbers 1, 2, 3, 4, ...
            tag_numbers = np.arange(0, 15)
            nTag_pred = JCM_model.nTagPred(tag_numbers, lowpt=True)["values"]
            # No shifting needed - predictions are already in correct positions
        else:
            nTag_pred = JCM_model.nTagPred(bin_centers.astype(int) + 4, lowpt=False)["values"]

        # Set values using the same approach
        try:
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

            for iBin in range(15):
                index_dict["n"] = iBin
                cfg.hists[0]['hists'][tagJets][tuple(index_dict.values())] = (nTag_pred[iBin], 0)

        except Exception as e:
            logger.warning(f"Error setting histogram values, trying alternative approach: {e}")
            for iBin in range(15):
                try:
                    hist_view = cfg.hists[0]['hists'][tagJets].view()
                    for idx, process in enumerate(hist_view.axes[0]):
                        if process == "JCM":
                            process_idx = idx
                            break
                    if has_passSvB and has_failSvB:
                        hist_view[process_idx, 0, 1, 1, True, False, False, iBin] = (nTag_pred[iBin], 0)
                    else:
                        hist_view[process_idx, 0, 1, 1, True, iBin] = (nTag_pred[iBin], 0)
                except Exception as inner_e:
                    logger.error(f"Failed to set values for bin {iBin}: {inner_e}")

        # Plot options for tagged jets
        plot_options = {
            "doRatio": True,
            "xlim": [1, 6] if args.lowpt else [4, 8],
            "yscale": "log",
            "rlim": [0.8, 1.2],
            "ylim": [0.1, None]
        }

        fig, ax = makePlot(
            cfg,
            var=tagJets,
            cut=args.cut,
            axis_opts={"region": args.weightRegion},
            **plot_options
        )

        fig.savefig(os.path.join(args.outputDir, "tagJets_noJCM_n.pdf"))
        logger.info(f"Saved tagged jets plot to {os.path.join(args.outputDir, 'tagJets_noJCM_n.pdf')}")

    except Exception as e:
        logger.warning(f"Failed to create tagged jets plot: {e}")

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
    parser.add_argument('-c', dest="cut", default="passPreSel",
                        help='Cut to apply (e.g. passPreSel)')
    parser.add_argument('-fix_e', action="store_true",
                        help='Fix the pairEnhancement parameter to 0')
    parser.add_argument('-fix_d', action="store_true",
                        help='Fix the pairEnhancementDecay parameter to 1')
    parser.add_argument('-float_tt4bSF', action="store_true",
                        help='Allow  the ttbar SF  to 1')
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
    parser.add_argument('--jcm_config', default="coffea4bees/analysis/jcm_tools/metadata/nominal_jcm_config.yml")
    parser.add_argument('--zero_pseudotag', dest="zero_pseudotag", action="store_true",
                        help='Compute zero pseudotag probabilities and weights in output')
    parser.add_argument('--lowpt', dest="lowpt", action="store_true",
                        help='Use low pt selection for 4b data')
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('JCM')
    logger.info("Starting JCM weight generation")

    # Create output directory if it doesn't exist
    if args.outputDir and not os.path.isdir(args.outputDir):
        os.makedirs(args.outputDir)
        logger.info(f"Created output directory: {args.outputDir}")

    # Set up output files
    jetCombinatoricModelName = os.path.join(
        args.outputDir,
        f"jetCombinatoricModel_{args.weightRegion}_{args.weightSet}.txt"
    )
    logger.info(f"Output files: {jetCombinatoricModelName} and .yml version")

    jetCombinatoricModelFile = open(jetCombinatoricModelName, "w")
    jetCombinatoricModelFile_yml = open(f'{jetCombinatoricModelName.replace(".txt",".yml")}', 'w')

    jcm_config_yaml = args.jcm_config
    with open(jcm_config_yaml, "r") as f:
        jcm_config = yaml.safe_load(f)

    print("JCM configuration:", jcm_config)

    try:
        if not args.ROOTInputs:
            # Load configuration
            cfg.plotConfig = load_config_4b(args.metadata)
            cfg.hists = load_hists(args.inputFile)
            cfg.combine_input_files = args.combine_input_files
            cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
            cfg.set_hist_key("hists")

        # Load histograms
        histograms = loadHistograms(
            inputFile=args.inputFile[0],
            jcm_config=jcm_config,
            format='ROOT' if args.ROOTInputs else 'coffea',
            cfg=cfg if not args.ROOTInputs else None,
            cut=args.cut,
            year=args.year,
            weightRegion=args.weightRegion,
        )

        # Process histograms and prepare data for fitting
        bin_data = process_histograms(*histograms, args, logger, jcm_config)

        # Set up the model
        JCM_model = setup_model(bin_data, args, logger)

        # Perform the fit
        residuals, pulls = perform_fit(JCM_model, bin_data[:3], args, logger)

        # Save model output
        save_model_output(
            JCM_model,
            bin_data + (histograms[6],),  # Add data4b_nTagJets for 5b calculation
            args,
            logger,
            (jetCombinatoricModelFile, jetCombinatoricModelFile_yml)
        )

        # Create plots
        create_plots(JCM_model, bin_data, args, bin_data[8], jcm_config, logger)

        logger.info(f"JCM weight generation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error in JCM weight generation: {e}", exc_info=True)
        # Clean up files
        jetCombinatoricModelFile.close()
        jetCombinatoricModelFile_yml.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
