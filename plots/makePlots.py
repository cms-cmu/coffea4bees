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
from src.plotting.plots import makePlot, make2DPlot, load_hists, read_axes_and_cuts, init_arg_parser
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
        return True
    except Exception as e:
        print(f"Error plotting {v} {region} {desc}: {str(e)}")
        return False
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
        return True
    except Exception as e:
        print(f"Error plotting {v} {region} {process}: {str(e)}")
        return False
    finally:
        plt.close("all")


def _load_sr_contours(thresholds_path: str = None, year: str = "RunII"):
    """Load SR/SB contour overlay parameters for ttHbb or HH4b 2D mass-plane plots."""
    cfg_thresh = {}
    if thresholds_path and os.path.exists(thresholds_path):
        with open(thresholds_path, 'r') as f:
            cfg_thresh = yaml.safe_load(f) or {}

    if "sr_ttHbb" in cfg_thresh or (thresholds_path and "ttHbb" in thresholds_path):
        sr_cfg = cfg_thresh.get("sr_ttHbb", {})
        sb_cfg = cfg_thresh.get("sb_ttHbb", {})
        sb_mode = cfg_thresh.get("sb_mode") or sb_cfg.get("mode", "box")
        corner_veto = bool(
            cfg_thresh.get("corner_veto", False)
            or sb_cfg.get("corner_veto", False)
            or (sb_mode in ["corner_veto", "sb_corner_veto", "corner_veto_250"])
        )
        return "ttHbb", {
            "h_min": float(sr_cfg.get("h_min", 95.0)),
            "h_max": float(sr_cfg.get("h_max", 180.0)),
            "m_min": float(sr_cfg.get("m_min", 25.0)),
            "arm_max": float(sr_cfg.get("arm_max", 400.0)),
            "sb_max": float(cfg_thresh.get("sb_max") or sb_cfg.get("sb_max", 500.0)),
            "corner_veto": corner_veto,
            "corner_m_lead": float(cfg_thresh.get("corner_m_lead") or sb_cfg.get("corner_m_lead", 250.0)),
            "corner_m_subl": float(cfg_thresh.get("corner_m_subl") or sb_cfg.get("corner_m_subl", 250.0)),
            "draw_sb": True,
            "sr_color": "red",
            "sr_linestyle": "--",
            "sr_linewidth": 2.5,
            "sb_color": "cyan",
            "sb_linestyle": ":",
            "sb_linewidth": 2.0,
        }

    is_run3 = any(tok in str(year) for tok in ("Run3", "2022", "2023", "2024", "2025"))
    if is_run3:
        sr3 = cfg_thresh.get("sr_run3", {})
        return "hh4b", {
            "mode": "run3",
            "cLead": float(sr3.get("cLead", 125.0)),
            "cSubl": float(sr3.get("cSubl", 120.0)),
            "SR_radius": float(sr3.get("SR_radius", 30.0)),
            "CR_radius": float(sr3.get("CR_radius", 55.0)),
            "draw_sb": True,
        }
    dijet = cfg_thresh.get("dijet", {})
    m_min = dijet.get("mass_min", [52.0, 50.0])
    m_max = dijet.get("mass_max", [180.0, 173.0])
    return "hh4b", {
        "mode": "run2",
        "mass_min": (float(m_min[0]), float(m_min[1])),
        "mass_max": (float(m_max[0]), float(m_max[1])),
        "draw_sb": True,
    }


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
            or category.startswith("pass")
            or category.startswith("fail_")
            or category.startswith("fail")
            or category.startswith("~")
        )
        if is_cut_category:
            cuts_in_cat = [c.strip().lstrip("~") for c in category.replace("+", ",").split(",") if c.strip()]
            available_cuts = set(cfg.cutList or [])
            missing = [c for c in cuts_in_cat if c not in available_cuts]
            if missing:
                logging.warning(f"Skipping category '{category}': missing cut(s) {missing} (available: {cfg.cutList})")
                continue

        cat_plotConfig = copy.deepcopy(original_plotConfig)
        if category in ("inclusive", ""):
            outputFolder = os.path.join(args.outputFolder, "inclusive") if category == "inclusive" else args.outputFolder
        elif is_cut_category:
            outputFolder = os.path.join(args.outputFolder, category.strip("_"))
        else:
            if category:
                for key in ["hists", "hists_2d", "stack"]:
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
        hists_2d_cfg = cat_plotConfig.get("hists_2d")
        vars_2d_filter = cat_plotConfig.get("vars_2d")
        if hists_2d_cfg:
            plotConfig_2d = copy.deepcopy(cat_plotConfig)
            plotConfig_2d["hists"] = copy.deepcopy(hists_2d_cfg)
            plotConfig_2d["stack"] = {}
            processes_2d = list(hists_2d_cfg.keys())
            regions_2d = cat_plotConfig.get("regions_2d", regions)
        else:
            plotConfig_2d = cat_plotConfig
            stack_2d = [k for k, v in cat_plotConfig.get("stack", {}).items() if "process" in v]
            processes_2d = list(dict.fromkeys(list(cat_plotConfig.get("hists", {}).keys()) + stack_2d))
            regions_2d = regions

        contour_kind, contour_params = _load_sr_contours(getattr(args, "thresholds", None), year=args.year or "RunII")

        for v in varList:
            if vars_2d_filter is not None and v not in vars_2d_filter:
                continue
            vDict = copy.deepcopy(cfg.plotModifiers.get(v, {}))
            if not vDict.get("2d", False):
                continue

            vDict["ylabel"] = "Entries"
            vDict["doRatio"] = cat_plotConfig.get("doRatio", True)
            vDict["legend"] = True

            if args.doTest:
                vDict["write_yaml"] = True

            is_mass_plane = ("lead_vs_subl_m" in v or "canH_vs_canTT_m" in v)
            if is_mass_plane and (getattr(args, "thresholds", None) or hists_2d_cfg):
                if contour_kind == "ttHbb" or "canH_vs_canTT_m" in v:
                    vDict.setdefault("plot_ttHbb_sr", True)
                    vDict.setdefault("ttHbb_sr_params", contour_params)
                else:
                    vDict.setdefault("plot_contour", True)
                    vDict.setdefault("hh4b_sr_params", contour_params)

            for process in processes_2d:
                for region in regions_2d:
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
                    tasks_2d.append((plotConfig_2d, process, plot_args, v, region))

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

    results_1d = []
    results_2d = []
    if num_workers > 1 and not debug and (len(all_tasks_1d) + len(tasks_2d)) > 1:
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else None
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx, initializer=_init_worker, initargs=(cfg,)) as executor:
            if all_tasks_1d:
                results_1d = list(executor.map(_render_1d_task, all_tasks_1d))
            if tasks_2d:
                results_2d = list(executor.map(_render_2d_task, tasks_2d))
    else:
        _init_worker(cfg)
        results_1d = [_render_1d_task(t) for t in all_tasks_1d]
        results_2d = [_render_2d_task(t) for t in tasks_2d]

    ok_1d = sum(1 for r in results_1d if r)
    ok_2d = sum(1 for r in results_2d if r)
    logging.info(f"Rendered {ok_1d}/{len(all_tasks_1d)} 1D plots, {ok_2d}/{len(tasks_2d)} 2D plots")
    if len(all_tasks_1d) > 0 and ok_1d == 0:
        raise RuntimeError("All 1D plots failed to render")


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

    parser = init_arg_parser()
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional candidate-selection thresholds YAML for 2D SR/SB contour overlays.",
    )
    args = parser.parse_args()

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
        for key in ["hists", "hists_2d", "stack"]:
            if key in cfg.plotConfig:
                new_dict = {}
                for name, p_cfg in cfg.plotConfig[key].items():
                    if "sum" in p_cfg:
                        new_dict[name] = p_cfg
                        continue
                    proc = p_cfg.get("process")
                    if isinstance(proc, list):
                        valid_procs = [p for p in proc if p in available_processes]
                        if valid_procs:
                            p_cfg["process"] = valid_procs if len(valid_procs) > 1 else valid_procs[0]
                            new_dict[name] = p_cfg
                    elif proc in available_processes:
                        new_dict[name] = p_cfg
                cfg.plotConfig[key] = new_dict

        if isinstance(cfg.plotConfig.get("ratios"), dict):
            valid_hists = cfg.plotConfig.get("hists", {})
            valid_stack = cfg.plotConfig.get("stack", {})
            kept_ratios = {}
            for r_name, r_cfg in cfg.plotConfig["ratios"].items():
                missing_ref = None
                for role in ("numerator", "denominator"):
                    ref = (r_cfg or {}).get(role) or {}
                    ref_type = ref.get("type")
                    ref_key = ref.get("key")
                    if ref_type == "hists" and ref_key not in valid_hists:
                        missing_ref = f"{role} hists.{ref_key}"
                        break
                    if ref_type == "stack" and (not valid_stack or (ref_key and ref_key not in valid_stack)):
                        missing_ref = f"{role} stack.{ref_key}" if ref_key else f"{role} stack (empty)"
                        break
                if missing_ref is not None:
                    logging.warning(f"Dropping ratio '{r_name}': missing {missing_ref} in filtered plotConfig")
                else:
                    kept_ratios[r_name] = r_cfg
            cfg.plotConfig["ratios"] = kept_ratios

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
        # substring match; the plot metadata's `skip_hists:` list adds to -s/--skip
        skip_hists = list(args.skip_hists) + list(cfg.plotConfig.get("skip_hists") or [])
        varList = [h for h in cfg.hists[0]['hists'].keys() if not any(skip in h for skip in skip_hists)]

    logging.info(f"Plotting {len(varList)} variables")
    doPlots(varList, debug=args.debug)
