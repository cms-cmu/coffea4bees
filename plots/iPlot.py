"""
Interactive plotting utility for HH4b analysis.

This module provides functions for creating and customizing 1D and 2D plots
from histogram data, with support for multiple variables, regions, and processes.
"""

import os
import sys
from typing import Optional, Union, List, Tuple, Dict, Any

# Third-party imports
import hist
import matplotlib.pyplot as plt

# Local imports
sys.path.insert(0, os.getcwd())
from coffea4bees.plots.plots import load_config_4b
from src.plotting.plots import (
    makePlot, make2DPlot, load_hists,
    read_axes_and_cuts, parse_args, print_cfg
)
from src.plotting.iPlot_config import plot_config
cfg = plot_config()

# Constants
DEFAULT_OUTPUT_FILE = "test.pdf"




def ls(option: str = "var", var_match: Optional[str] = None, hist_key: Optional[str] = 'hists') -> None:
    """List available variables in the configuration.

    Args:
        option: The type of labels to list (default: "var")
        var_match: Optional string to filter variables by
    """
    for k in cfg.axisLabelsDict[hist_key][option]:
        if var_match:
            if var_match in k:
                print(k)
        else:
            print(k)


def info() -> None:
    """Print the current configuration."""
    print_cfg(cfg)


def examples() -> None:
    """Print example usage of the plotting functions."""
    examples_text = """
examples:

# Nominal plot of data and background in a region passing a cut
plot("v4j.mass", region="SR", cut="passPreSel")

# Can get a print out of the variables
ls()
plot("*", region="SR", cut="passPreSel")
plot("v4j*", region="SR", cut="passPreSel")

# Can add ratio
plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1)

# Can rebin
plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4)

# Can normalize
plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1)

# Can set logy
plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, yscale="log")

# Can set ranges
plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, rlim=[0.5,1.5])
plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, xlim=[0,1000])
plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, ylim=[0,0.01])

# Can overlay different regions
plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="data", doRatio=1, rebin=4)
plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="HH4b", doRatio=1, rebin=4)
plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="Multijet", doRatio=1, rebin=4)
plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="TTToHadronic", doRatio=1, rebin=4)

# Can overlay different cuts
plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="data", doRatio=1, rebin=4, norm=1)
plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="HH4b", doRatio=1, rebin=4, norm=1)
plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="Multijet", doRatio=1, rebin=4, norm=1)
plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="TTToHadronic", doRatio=1, rebin=4, norm=1)

# Can overlay different variables
plot(["canJet0.pt","canJet1.pt"], region="SR", cut="passPreSel", doRatio=1, process="Multijet")
plot(["canJet0.pt","canJet1.pt","canJet2.pt","canJet3.pt"], region="SR", cut="passPreSel", doRatio=1, process="Multijet")

# Can plot a single process
plot("v4j.mass", region="SR", cut="passPreSel", process="data")

# Can overlay processes
plot("v4j.mass", region="SR", cut="passPreSel", norm=1, process=["data","TTTo2L2Nu","HH4b","Multijet"], doRatio=1)

# Can overlay years
plot("canJet0.pt", region="SR", cut="passPreSel", doRatio=1, process="data", year=["UL16_preVFP","UL16_postVFP","UL17","UL18"])

# Plot 2d hists
plot2d("quadJet_min_dr.close_vs_other_m", process="Multijet", region="SR", cut="failSvB")
plot2d("quadJet_min_dr.close_vs_other_m", process="Multijet", region="SR", cut="failSvB", full=True)

# Unsup4b plots with SB and SRSB as composite regions
plot("v4j.mass", region="SRSB", cut="passPreSel")
plot2d("quadJet_selected.lead_vs_subl_m", process="data3b", region="SRSB")
plot("leadStM_selected", region="SB", cut="passPreSel", process=["data3b","mixeddata"])
plot("v4j.mass", region=["SR", "SB"], cut="passPreSel", process="data3b")
"""
    print(examples_text)


def save_and_open_plot(fig: plt.Figure, output_file: str) -> bool:
    """Save the figure to a file and open it.

    Args:
        fig: The matplotlib figure to save
        output_file: Path where to save the figure

    Returns:
        True if successful, False otherwise
    """
    try:
        fig.savefig(output_file)
        plt.close()
        os.system(f"open {output_file}")
        return True
    except Exception as e:
        print(f"Error saving plot: {e}")
        return False


def handle_wildcards(var: Union[str, List[str]]) -> bool:
    """Handle wildcard matching in variable names.

    Args:
        var: Variable(s) to check for wildcards

    Returns:
        True if wildcards were found and handled, False otherwise
    """
    if isinstance(var, str) and "*" in var:
        ls(var_match=var.replace("*", ""), hist_key=cfg.hist_key)
        return True
    if isinstance(var, list) and var[0].find("*") != -1:
        ls(var_match=var[0].replace("*", ""), hist_key=cfg.hist_key)
        return True
    return False


def plot(var: Union[str, List[str]] = 'selJets.pt', *,
         cut: Union[str, List[str]] = "passPreSel",
         region: Union[str, List[str]] = "SR",
         axis_opts: Dict = {},
         output_file: str = DEFAULT_OUTPUT_FILE,
         **kwargs) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Create a 1D plot of the specified variable.

    Args:
        var: Variable(s) to plot. Can be a string or list of strings.
        cut: Selection cut to apply (default: "passPreSel")
        region: Region to plot (default: "SR")
        axis_opts: Dictionary of axis options
        output_file: Name of the output file (default: "test.pdf")
        **kwargs: Additional plotting options

    Returns:
        Optional tuple of (figure, axes) if debug mode is enabled
    """
    if kwargs.get("debug", False):
        print(f'kwargs = {kwargs}')

    # Handle wildcard matching
    if handle_wildcards(var):
        return

    # Add region to axis_opts
    axis_opts["region"] = region

    opts = {"var": var,
            "cut": cut,
            "axis_opts": axis_opts,
            "outputFolder": cfg.outputFolder
            }
    opts.update(kwargs)

    if len(cfg.hists) > 1:
        opts["fileLabels"] = cfg.fileLabels

    try:
        fig, ax = makePlot(cfg, **opts)


    except ValueError as e:
        print(f"Error creating plot: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    # Save and display the plot
    if not save_and_open_plot(fig, output_file):
        return

    if kwargs.get("debug", False):
        return fig, ax


def plot2d(var: str = 'quadJet_selected.lead_vs_subl_m',
           process: Union[str, List[str]] = "HH4b",
           *,
           cut: Union[str, List[str]] = "passPreSel",
           region: Union[str, List[str]] = "SR",
           axis_opts: Dict = {},
           output_file: str = DEFAULT_OUTPUT_FILE,
           **kwargs) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Create a 2D plot of the specified variable.

    Args:
        var: Variable to plot
        process: Process to plot (default: "HH4b")
        cut: Selection cut to apply (default: "passPreSel")
        region: Region to plot (default: "SR")
        axis_opts: Dictionary of axis options
        output_file: Name of the output file (default: "test.pdf")
        **kwargs: Additional plotting options

    Returns:
        Optional tuple of (figure, axes) if debug mode is enabled
    """
    if kwargs.get("debug", False):
        print(f'kwargs = {kwargs}')

    if handle_wildcards(var):
        return

    # Add region to axis_opts
    axis_opts["region"] = region

    try:
        fig, ax = make2DPlot(cfg, process, var=var, cut=cut,
                             axis_opts=axis_opts, outputFolder=cfg.outputFolder, **kwargs)
    except Exception as e:
        print(f"Error creating 2D plot: {e}")
        return

    if not save_and_open_plot(fig, output_file):
        return

    if kwargs.get("debug", False):
        return fig, ax


def initialize_config() -> None:
    """Initialize the configuration from command line arguments."""
    args = parse_args()
    cfg.plotConfig = load_config_4b(args.metadata)
    cfg.outputFolder = args.outputFolder
    cfg.combine_input_files = args.combine_input_files

    if cfg.outputFolder and not os.path.exists(cfg.outputFolder):
        os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    cfg.set_hist_key("hists")

if __name__ == '__main__':
    initialize_config()
    print_cfg(cfg)
