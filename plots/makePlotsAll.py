import os
import sys
import yaml
import warnings
import hist
import tempfile
import multiprocessing as mp
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI, renders directly to files
sys.path.insert(0, os.getcwd())
from coffea4bees.plots.plots import load_config_4b
from src.plotting.plots import makePlot, make2DPlot, load_hists, read_axes_and_cuts, parse_args, init_arg_parser
from src.plotting.iPlot_config import plot_config
cfg = plot_config()

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore', message='.*All sumw are zero.*')


# ---------------------------------------------------------------------------
# Multiprocessing worker support
# ---------------------------------------------------------------------------

_worker_cfg = None   # set in each worker process by _init_worker


def _init_worker(cfg_state):
    """Initialise a worker process: set up sys.path, matplotlib, and cfg."""
    import os, sys, tempfile, warnings
    os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')
    warnings.filterwarnings('ignore', message='.*All sumw are zero.*')

    sys.path.insert(0, os.getcwd())
    from src.plotting.iPlot_config import plot_config

    global _worker_cfg
    _worker_cfg = plot_config()
    _worker_cfg.hists               = cfg_state['hists']
    _worker_cfg.plotConfig          = cfg_state['plotConfig']
    _worker_cfg.outputFolder        = cfg_state['outputFolder']
    _worker_cfg.plotModifiers       = cfg_state['plotModifiers']
    _worker_cfg.axisLabelsDict      = cfg_state['axisLabelsDict']
    _worker_cfg.cutListDict         = cfg_state['cutListDict']
    _worker_cfg.combine_input_files = cfg_state['combine_input_files']
    _worker_cfg.fileLabels          = cfg_state['fileLabels']
    _worker_cfg.set_hist_key("hists")


def _do_1d_job(job):
    """Worker: render one 1D plot."""
    import matplotlib.pyplot as plt
    from src.plotting.plots import makePlot
    v, region, plot_args = job
    try:
        makePlot(_worker_cfg, **plot_args)
        plt.close("all")
        return True
    except Exception as e:
        plt.close("all")
        print(f"Error plotting {v} {region}: {e}")
        return False


def _do_2d_job(job):
    """Worker: render one 2D plot."""
    import matplotlib.pyplot as plt
    from src.plotting.plots import make2DPlot
    v, process, region, plot_args = job
    try:
        make2DPlot(_worker_cfg, process, **plot_args)
        plt.close("all")
        return True
    except Exception as e:
        plt.close("all")
        print(f"Error plotting {v} {process} {region}: {e}")
        return False


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _build_jobs(varList, outputFolder, debug=False):
    """Build (jobs_1d, jobs_2d) lists without running anything."""
    regions = cfg.axisLabelsDict["hists"]["region"]

    processes = []
    for key, values in cfg.plotConfig.items():
        if key in ["hists", "stack"]:
            for _key in values.keys():
                processes.append(_key)

    jobs_1d, jobs_2d = [], []

    for v in varList:
        vDict = cfg.plotModifiers.get(v, {})
        if vDict.get("2d", False):
            # 2D
            vDict = dict(vDict)
            vDict["ylabel"] = "Entries"
            vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)
            vDict["legend"]  = True
            for process in processes:
                for region in regions:
                    plot_args = {"var": v, "axis_opts": {"region": region},
                                 "outputFolder": outputFolder} | vDict
                    jobs_2d.append((v, process, region, plot_args))
        else:
            # 1D
            vDict = dict(vDict)
            vDict["ylabel"] = "Entries"
            vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)
            vDict["legend"]  = True
            for region in regions:
                plot_args = {"var": v, "axis_opts": {"region": region},
                             "outputFolder": outputFolder} | vDict
                jobs_1d.append((v, region, plot_args))

    return jobs_1d, jobs_2d


def doPlots(varList, outputFolder, n_jobs=1, debug=False):

    jobs_1d, jobs_2d = _build_jobs(varList, outputFolder, debug=debug)
    total = len(jobs_1d) + len(jobs_2d)
    print(f"Plotting {len(jobs_1d)} 1D + {len(jobs_2d)} 2D = {total} plots "
          f"with {n_jobs} worker(s) ...")

    if n_jobs > 1:
        # dict_keys objects aren't picklable — convert to lists
        axis_labels_picklable = {
            hk: {axis: list(vals) for axis, vals in hk_dict.items()}
            for hk, hk_dict in cfg.axisLabelsDict.items()
        }
        cfg_state = {
            'hists':               cfg.hists,
            'plotConfig':          cfg.plotConfig,
            'outputFolder':        outputFolder,
            'plotModifiers':       cfg.plotModifiers,
            'axisLabelsDict':      axis_labels_picklable,
            'cutListDict':         cfg.cutListDict,
            'combine_input_files': cfg.combine_input_files,
            'fileLabels':          cfg.fileLabels,
        }
        with mp.Pool(n_jobs, initializer=_init_worker, initargs=(cfg_state,)) as pool:
            list(pool.imap_unordered(_do_1d_job, jobs_1d, chunksize=4))
            list(pool.imap_unordered(_do_2d_job, jobs_2d, chunksize=2))
    else:
        # Sequential fallback (same behaviour as before)
        for job in jobs_1d:
            v, region, plot_args = job
            try:
                makePlot(cfg, **plot_args)
            except Exception as e:
                print(f"Error plotting {v} {region}: {e}")
            plt.close()

        for job in jobs_2d:
            v, process, region, plot_args = job
            try:
                make2DPlot(cfg, process, **plot_args)
            except Exception as e:
                print(f"Error plotting {v} {process} {region}: {e}")
            plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    parser = init_arg_parser()
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of parallel worker processes (default: 1). '
                             'Try --jobs 4 for a ~4x speedup on multi-core machines.')
    args = parser.parse_args()

    cfg.plotConfig    = load_config_4b(args.metadata)
    cfg.outputFolder  = args.outputFolder
    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))
    cfg.combine_input_files = args.combine_input_files
    cfg.fileLabels    = args.fileLabels

    if cfg.outputFolder:
        os.makedirs(cfg.outputFolder, exist_ok=True)

    cfg.hists = load_hists(args.inputFile)
    cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    cfg.set_hist_key("hists")

    if args.list_of_hists:
        varList = args.list_of_hists
    else:
        varList = [h for h in cfg.hists[0]['hists'].keys()
                   if not any(skip in h for skip in args.skip_hists)]

    doPlots(varList, args.outputFolder, n_jobs=args.jobs, debug=args.debug)
