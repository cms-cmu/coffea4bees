import os
import time
import sys
import yaml
import warnings
import hist
import argparse
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
from coffea.util import load
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI, renders directly to files
sys.path.insert(0, os.getcwd())
from coffea4bees.plots.plots import load_config_4b
from src.plotting.plots import makePlot, make2DPlot, load_hists, read_axes_and_cuts, parse_args
from src.plotting.iPlot_config import plot_config
cfg = plot_config()

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore', message='.*All sumw are zero.*')

_GLOBAL_CFG = None

def _init_worker(shared_cfg):
    global _GLOBAL_CFG
    _GLOBAL_CFG = shared_cfg

def _render_1d_task(task):
    global _GLOBAL_CFG
    cat_plotConfig, plot_args, v, region, desc = task
    try:
        import copy
        worker_cfg = copy.copy(_GLOBAL_CFG)
        worker_cfg.plotConfig = cat_plotConfig
        makePlot(worker_cfg, **plot_args)
    except Exception as e:
        print(f"Error plotting {v} {region} {desc}: {str(e)}")
    finally:
        plt.close("all")


def _render_2d_task(task):
    global _GLOBAL_CFG
    cat_plotConfig, process, plot_args, v, region = task
    try:
        import copy
        worker_cfg = copy.copy(_GLOBAL_CFG)
        worker_cfg.plotConfig = cat_plotConfig
        make2DPlot(worker_cfg, process, **plot_args)
    except Exception as e:
        print(f"Error plotting {v} {region} {process}: {str(e)}")
    finally:
        plt.close("all")


def doPlots(varList, debug=False):
    import copy
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    if args.doTest:
        varList = ["SvB_MA.ps_zz", "SvB_MA.ps_zh", "SvB_MA.ps_hh", "quadJet_selected.lead_vs_subl_m", "quadJet_min_dr.close_vs_other_m"]

    #cut = "passPreSel"
    tag = "fourTag"

    regions = cfg.plotConfig.get("regions", ["SR", "SB"])
    categories = cfg.plotConfig.get("categories", [""])

    original_plotConfig = copy.deepcopy(cfg.plotConfig)
    num_workers = getattr(args, "num_workers", 8)

    tasks_1d = []
    tasks_2d = []
    tasks_comp = []

    for category in categories:
        is_cut_category = (
            category.startswith("pass_")
            or category.startswith("fail_")
            or category.startswith("~")
        )

        cat_plotConfig = copy.deepcopy(original_plotConfig)
        if category in ("inclusive", ""):
            outputFolder = os.path.join(args.outputFolder, "inclusive") if category == "inclusive" else args.outputFolder
        elif is_cut_category:
            outputFolder = os.path.join(args.outputFolder, category.strip("_"))
        else:
            if category:
                for key in ["hists", "stack"]:
                    if key in cat_plotConfig:
                        for name, p_cfg in cat_plotConfig[key].items():
                            p_cfg["process"] = p_cfg["process"] + category
            outputFolder = os.path.join(args.outputFolder, category.strip("_")) if category else args.outputFolder

        #
        #  Nominal 1D Plots
        #
        for v in varList:
            vDict = copy.deepcopy(cfg.plotModifiers.get(v, {}))
            if vDict.get("2d", False):
                continue

            vDict["ylabel"] = "Entries"
            vDict["doRatio"] = cat_plotConfig.get("doRatio", True)
            vDict["legend"] = True
            if v.startswith("SvB"):
                vDict.setdefault("yscale", "log")

            if args.doTest:
                vDict["write_yaml"] = True

            for region in regions:
                plot_args = {}
                plot_args["var"] = v
                if is_cut_category:
                    plot_args["cut"] = category
                plot_args["axis_opts"] = {"region": region}
                plot_args["outputFolder"] = outputFolder
                if hasattr(args, "fmt") and args.fmt:
                    plot_args["fmt"] = args.fmt
                if args.year:
                    plot_args["year"] = args.year
                plot_args = plot_args | vDict
                tasks_1d.append((cat_plotConfig, plot_args, v, region, ""))

        #
        #  2D Plots
        #
        for v in varList:
            vDict = copy.deepcopy(cfg.plotModifiers.get(v, {}))
            if not vDict.get("2d", False):
                continue

            vDict["ylabel"] = "Entries"
            vDict["doRatio"] = cat_plotConfig.get("doRatio", True)
            vDict["legend"] = True

            if args.doTest:
                vDict["write_yaml"] = True

            processes = list(cat_plotConfig.get("hists", {}).keys()) + list(cat_plotConfig.get("stack", {}).keys())
            processes = list(dict.fromkeys(processes))

            for process in processes:
                for region in regions:
                    plot_args = {}
                    plot_args["var"] = v
                    if is_cut_category:
                        plot_args["cut"] = category
                    plot_args["axis_opts"] = {"region": region}
                    plot_args["outputFolder"] = outputFolder
                    if hasattr(args, "fmt") and args.fmt:
                        plot_args["fmt"] = args.fmt
                    if args.year:
                        plot_args["year"] = args.year
                    plot_args = plot_args | vDict
                    tasks_2d.append((cat_plotConfig, process, plot_args, v, region))

        #
        #  Comparison Plots (doTest only)
        #
        if args.doTest:
            varListComp = ["v4j.mass", "SvB_MA.ps", "quadJet_selected.xHH"]
            for v in varListComp:
                vDict = copy.deepcopy(cfg.plotModifiers.get(v, {}))
                vDict["ylabel"] = "Entries"
                vDict["doRatio"] = cat_plotConfig.get("doRatio", True)
                vDict["legend"] = True
                vDict["write_yaml"] = True

                processes = list(cat_plotConfig.get("hists", {}).keys()) + list(cat_plotConfig.get("stack", {}).keys())
                processes = list(dict.fromkeys(processes))

                for process in processes:
                    for region in regions:
                        plot_args = {}
                        plot_args["var"] = v
                        plot_args["cut"] = ["failSvB", "passSvB"]
                        plot_args["axis_opts"] = {"region": region}
                        plot_args["outputFolder"] = outputFolder
                        plot_args["process"] = process
                        plot_args["norm"] = True
                        if hasattr(args, "fmt") and args.fmt:
                            plot_args["fmt"] = args.fmt
                        if args.year:
                            plot_args["year"] = args.year
                        plot_args = plot_args | vDict
                        tasks_comp.append((cat_plotConfig, plot_args, v, region, "(comp cuts)"))

                    comp_regions = [r for r in regions if r != "sum"]
                    comp_plot_args = {
                        "var": v,
                        "cut": None,
                        "axis_opts": {"region": comp_regions},
                        "process": process,
                        "outputFolder": outputFolder,
                    }
                    if hasattr(args, "fmt") and args.fmt:
                        comp_plot_args["fmt"] = args.fmt
                    if args.year:
                        comp_plot_args["year"] = args.year
                    comp_plot_args = comp_plot_args | vDict
                    tasks_comp.append((cat_plotConfig, comp_plot_args, v, "comp_regions", "(comp regions)"))

    all_tasks_1d = tasks_1d + tasks_comp
    logging.info(f"Total plots to render: {len(all_tasks_1d)} 1D plots, {len(tasks_2d)} 2D plots (using {num_workers} worker processes)")

    if num_workers > 1 and not debug and (len(all_tasks_1d) + len(tasks_2d)) > 1:
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else None
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx, initializer=_init_worker, initargs=(cfg,)) as executor:
            if all_tasks_1d:
                list(executor.map(_render_1d_task, all_tasks_1d))
            if tasks_2d:
                list(executor.map(_render_2d_task, tasks_2d))
    else:
        _init_worker(cfg)
        for t in all_tasks_1d:
            _render_1d_task(t)
        for t in tasks_2d:
            _render_2d_task(t)


if __name__ == '__main__':

    import logging
    try:
        from src.runner.logging import CustomFormatter
        formatter = CustomFormatter()
    except ImportError:
        formatter = logging.Formatter("[%(levelname)s] %(message)s")

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    else:
        for h in root_logger.handlers:
            h.setFormatter(formatter)

    args = parse_args()

    logging.info("Running with these parameters:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    logging.info(f"Loading metadata from: {args.metadata}")
    cfg.plotConfig = load_config_4b(args.metadata)
    cfg.outputFolder = args.outputFolder
    if cfg.outputFolder:
        logging.info(f"Output directory: {cfg.outputFolder}")
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    logging.info(f"Loading modifiers from: {args.modifiers}")
    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    logging.info(f"Loading histograms from: {args.inputFile}")
    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.combine_input_files = args.combine_input_files if args.combine_input_files else (len(args.inputFile) > 1 and not args.fileLabels)

    # Filter plotConfig to only include processes that are available in the input histograms
    available_processes = []
    if cfg.hists and isinstance(cfg.hists, list):
        for file_data in cfg.hists:
            if 'hists' in file_data:
                for h in file_data['hists'].values():
                    if "process" in h.axes.name:
                        available_processes.extend(list(h.axes["process"]))
    available_processes = list(dict.fromkeys(available_processes))

    if available_processes:
        for key in ["hists", "stack"]:
            if key in cfg.plotConfig:
                new_dict = {}
                for name, p_cfg in cfg.plotConfig[key].items():
                    proc = p_cfg.get("process")
                    if isinstance(proc, list):
                        valid_procs = [p for p in proc if p in available_processes]
                        if valid_procs:
                            p_cfg["process"] = valid_procs if len(valid_procs) > 1 else valid_procs[0]
                            new_dict[name] = p_cfg
                    elif proc in available_processes:
                        new_dict[name] = p_cfg
                cfg.plotConfig[key] = new_dict

    # Auto-infer year if not explicitly provided
    if not args.year and cfg.hists and isinstance(cfg.hists, list) and len(cfg.hists) > 0 and 'hists' in cfg.hists[0]:
        hists_dict = cfg.hists[0]['hists']
        if hists_dict:
            first_hist = next(iter(hists_dict.values()))
            if "year" in first_hist.axes.name:
                available_years = list(first_hist.axes["year"])
                if len(available_years) == 1:
                    args.year = available_years[0]
                elif any("202" in str(y) for y in available_years):
                    args.year = "Run3"
                else:
                    args.year = "RunII"

    if args.year:
        logging.info(f"Plotting for year: {args.year}")

    cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    cfg.set_hist_key("hists")

    if args.list_of_hists:
        varList = args.list_of_hists
    else:
        varList = [h for h in cfg.hists[0]['hists'].keys() if not any(skip in h for skip in args.skip_hists)]

    logging.info(f"Plotting {len(varList)} variables")
    doPlots(varList, debug=args.debug)
