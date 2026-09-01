import os
import sys
import yaml
import tempfile
import argparse
from typing import Union, List, Tuple
import numpy as np

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
import src.plotting.numpy2_pickle_compat  # noqa: F401
from src.plotting.plots import make2DPlot, load_hists, init_arg_parser
from coffea4bees.plots.plots import load_config_4b
import src.plotting.iPlot_config as cfg

np.seterr(divide='ignore', invalid='ignore')


def load_sr_thresholds(thresholds_path: str, mode_override: str = None) -> dict:
    """Load ttHbb SR/SB thresholds from candidate selection yaml."""
    if not os.path.exists(thresholds_path):
        print(f"Warning: {thresholds_path} not found, using default optimal_balance thresholds.")
        return {
            'h_min': 95.0,
            'h_max': 180.0,
            'm_min': 25.0,
            'arm_max': 400.0,
            'sb_max': 1000.0,
        }

    with open(thresholds_path, 'r') as f:
        cfg_thresh = yaml.safe_load(f)

    sr_cfg = (cfg_thresh or {}).get('sr_ttHbb', {})
    mode = mode_override or sr_cfg.get('mode', 'optimal_balance')

    if mode in ['optimal_balance', 'optimal']:
        return {
            'h_min': float(sr_cfg.get('h_min', 95.0)),
            'h_max': float(sr_cfg.get('h_max', 180.0)),
            'm_min': float(sr_cfg.get('m_min', 25.0)),
            'arm_max': float(sr_cfg.get('arm_max', 400.0)),
            'sb_max': 1000.0,
        }
    else:  # baseline
        return {
            'h_min': float(sr_cfg.get('h_min', 85.0)),
            'h_max': float(sr_cfg.get('h_max', 185.0)),
            'm_min': float(sr_cfg.get('m_min', 25.0)),
            'arm_max': float(sr_cfg.get('arm_max', 1000.0)),
            'sb_max': 1000.0,
        }


def doPlots(varList, sr_params: dict, proc_tag_pairs: List[Tuple[str, str]], regions: list = None,
            draw_sb: bool = True, xlim=(0, 1000), ylim=(0, 1000),
            sr_color='red', sr_linestyle='--', sr_linewidth=2.5,
            sb_color='cyan', sb_linestyle=':', sb_linewidth=2.0,
            fmt: Union[str, List[str]] = "png", debug=False):
    """Generate 2D plots with ttHbb SR and SB boundary overlays."""
    ttHbb_sr_kwargs = {
        'h_min': sr_params.get('h_min', 95.0),
        'h_max': sr_params.get('h_max', 180.0),
        'm_min': sr_params.get('m_min', 25.0),
        'arm_max': sr_params.get('arm_max', 400.0),
        'sb_max': sr_params.get('sb_max', 1000.0),
        'draw_sb': draw_sb,
        'sr_color': sr_color,
        'sr_linestyle': sr_linestyle,
        'sr_linewidth': sr_linewidth,
        'sb_color': sb_color,
        'sb_linestyle': sb_linestyle,
        'sb_linewidth': sb_linewidth,
    }

    if not regions:
        regions = ['inclusive']

    for v in varList:
        print(f"Plotting 2D variable: {v}")

        is_mass_plane = ('lead_vs_subl_m' in v or 'leadstmass_vs_sublstmass' in v or
                         'close_vs_other_m' in v)

        for proc, tag in proc_tag_pairs:
            for region in regions:
                plot_args = {
                    "var": v,
                    "outputFolder": cfg.outputFolder,
                    "ylabel": "Entries",
                    "doRatio": False,
                    "legend": True,
                    "fmt": fmt,
                    "axis_opts": {"region": region, "tag": tag},
                    "plot_ttHbb_sr": is_mass_plane,
                    "ttHbb_sr_params": ttHbb_sr_kwargs if is_mass_plane else None,
                }

                if 'leadstdr_vs_m4j' in v:
                    plot_args["plot_leadst_lines"] = True
                if 'sublstdr_vs_m4j' in v:
                    plot_args["plot_sublst_lines"] = True

                if xlim:
                    plot_args["xlim"] = tuple(xlim)
                if ylim:
                    plot_args["ylim"] = tuple(ylim)

                if debug:
                    print(f"  process: {proc}, tag: {tag}, region: {region}, plot_args: {plot_args}")

                try:
                    fig, ax = make2DPlot(cfg, proc, **plot_args)
                    plt.close(fig)
                except Exception as e:
                    if debug:
                        print(f"  Warning: failed to make 2D plot for {v} (proc={proc}, tag={tag}, reg={region}): {e}")
                    plt.close('all')


def main():
    parser = init_arg_parser()
    parser.description = "Create 2D distribution plots with ttHbb SR/SB boundary overlays."
    parser.set_defaults(
        inputFile=['output/ttHbb_v3/histAll_ttHbb_v3.coffea'],
        metadata='coffea4bees/plots/metadata/plotsAll_ttHbb.yml',
        modifiers='coffea4bees/plots/metadata/plotModifiers.yml',
        outputFolder='plots/ttHbb_v3_2D/',
        list_of_hists=['quadJet_selected.lead_vs_subl_m'],
        fmt='png',
    )

    parser.add_argument('--thresholds', default='coffea4bees/analysis/metadata/candidates_selection_thresholds_ttHbb.yml',
                        help='Path to candidate selection thresholds YAML.')
    parser.add_argument('--mode', choices=['optimal_balance', 'baseline'], default=None,
                        help='Override ttHbb SR mode (default: from thresholds file).')
    parser.add_argument('--no-draw-sb', dest='draw_sb', action='store_false', default=True,
                        help='Disable drawing the outer sideband (SB) boundary box.')
    parser.add_argument('--sr-color', default='red',
                        help='Color for ttHbb SR boundary lines (default: red).')
    parser.add_argument('--sr-linestyle', default='--',
                        help='Line style for ttHbb SR boundary lines (default: --).')
    parser.add_argument('--sr-linewidth', type=float, default=2.5,
                        help='Line width for ttHbb SR boundary lines (default: 2.5).')
    parser.add_argument('--sb-color', default='cyan',
                        help='Color for ttHbb SB boundary box (default: cyan).')
    parser.add_argument('--sb-linestyle', default=':',
                        help='Line style for ttHbb SB boundary box (default: :).')
    parser.add_argument('--sb-linewidth', type=float, default=2.0,
                        help='Line width for ttHbb SB boundary box (default: 2.0).')
    parser.add_argument('--processes', nargs='+', default=None,
                        help='Processes to plot (e.g. ttHbb MultiJet data).')
    parser.add_argument('--regions', nargs='+', default=['inclusive'],
                        help='Regions to plot (default: inclusive).')
    parser.add_argument('--tags', nargs='+', default=None,
                        help='Explicit tags to pair with processes (optional).')
    parser.add_argument('--xlim', nargs=2, type=float, default=[0, 1000],
                        help='X-axis limits [xmin, xmax] (default: 0 1000).')
    parser.add_argument('--ylim', nargs=2, type=float, default=[0, 1000],
                        help='Y-axis limits [ymin, ymax] (default: 0 1000).')
    parser.add_argument('--cuts', '--categories', dest='cuts', nargs='+', default=['inclusive'],
                        help='List of selections/categories to iterate over (default: inclusive).')

    args = parser.parse_args()

    # Load thresholds
    sr_params = load_sr_thresholds(args.thresholds, mode_override=args.mode)
    print(f"Loaded ttHbb SR parameters: {sr_params}")

    # Configure plotting environment
    cfg.plotConfig = load_config_4b(args.metadata)
    cfg.plotConfig['hist_dict'] = {'process': sum, 'selection': 'none', 'year': args.year or 'UL18'}
    cfg.outputFolder = args.outputFolder
    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r')) if os.path.exists(args.modifiers) else {}

    if cfg.outputFolder and not os.path.exists(cfg.outputFolder):
        os.makedirs(cfg.outputFolder, exist_ok=True)

    print(f"Loading input histograms from {args.inputFile}...")
    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels

    varList = args.list_of_hists
    print(f"Plotting variable(s): {varList}")

    # Determine (process, tag) pairs to plot
    proc_tag_pairs = []
    if args.processes:
        for p in args.processes:
            # Check metadata for default tag for this process if not overridden
            if args.tags:
                for t in args.tags:
                    proc_tag_pairs.append((p, t))
            else:
                tag = 'fourTag'
                if 'hists' in cfg.plotConfig and p in cfg.plotConfig['hists']:
                    tag = cfg.plotConfig['hists'][p].get('tag', 'fourTag')
                elif 'stack' in cfg.plotConfig and p in cfg.plotConfig['stack']:
                    tag = cfg.plotConfig['stack'][p].get('tag', 'threeTag')
                elif p.lower() in ['multijet', '3b', 'data_3b']:
                    tag = 'threeTag'
                proc_tag_pairs.append((p, tag))
    else:
        # Default: plot ttHbb (4b) and MultiJet (3b)
        if 'hists' in cfg.plotConfig and 'ttHbb' in cfg.plotConfig['hists']:
            proc_tag_pairs.append(('ttHbb', cfg.plotConfig['hists']['ttHbb'].get('tag', 'fourTag')))
        else:
            proc_tag_pairs.append(('ttHbb', 'fourTag'))

        if 'stack' in cfg.plotConfig and 'MultiJet' in cfg.plotConfig['stack']:
            proc_tag_pairs.append(('MultiJet', cfg.plotConfig['stack']['MultiJet'].get('tag', 'threeTag')))
        else:
            proc_tag_pairs.append(('data', 'threeTag'))

    print(f"Processes and tags to plot: {proc_tag_pairs}")
    regions = args.regions
    cuts = args.cuts

    # Iterate over selections
    for isel in cuts:
        print(f"\n=== Processing category/selection: {isel} ===")
        cfg.plotConfig['hist_dict']['selection'] = isel
        doPlots(
            varList=varList,
            sr_params=sr_params,
            proc_tag_pairs=proc_tag_pairs,
            regions=regions,
            draw_sb=args.draw_sb,
            xlim=args.xlim,
            ylim=args.ylim,
            sr_color=args.sr_color,
            sr_linestyle=args.sr_linestyle,
            sr_linewidth=args.sr_linewidth,
            sb_color=args.sb_color,
            sb_linestyle=args.sb_linestyle,
            sb_linewidth=args.sb_linewidth,
            fmt=args.fmt,
            debug=args.debug,
        )

    print(f"\nSaved plots to: {cfg.outputFolder}")


if __name__ == '__main__':
    main()
