"""Interactive web-based plot viewer for HH4b analysis.

Serves a local webpage with a pre-generated plot gallery and an
interactive form that mirrors the iPlot.py plot() interface.

Setup (one time):
    python -m venv ~/python-environments/pourover
    source ~/python-environments/pourover/bin/activate
    pip install -r coffea4bees/plots/requirements-pourover.txt
    pip install -e .   # install barista src/ as editable package

Usage:
    python coffea4bees/plots/pourOver.py <coffea_file> -m <metadata.yml> [options]
    python coffea4bees/plots/pourOver.py <coffea_file> -m <metadata.yml> --new [--label TEXT]
    python coffea4bees/plots/pourOver.py --load <archive_dir>

Extra options (beyond the standard iPlot/makePlotsAll args):
    --port PORT          Port to serve on (default: 5000)
    --pregallery         Pre-generate the full plot gallery on startup (default: off)
    --reuse-gallery      Reuse existing gallery plots; only regenerate missing ones
    --output-dir DIR     Directory for generated plots (default: pourover_output)
    --modifiers FILE     Per-variable modifier YAML (optional)
    -j N, --jobs N       Parallel workers for pregallery (default: 1)
    --new                Snapshot inputs into a timestamped archive, register it,
                         generate gallery inside the archive, and serve.
    --load LABEL         Load from a previously created archive by its kebab-case label.
    --registry FILE      Path to the archive registry JSON (default: .pourover_archives/pourover_registry.json).
    --label LABEL        Kebab-case label for this archive (required with --new, e.g. ul18-sb-test).

Example:
    python coffea4bees/plots/pourOver.py \\
        coffea4bees/Run3_MvD/analysis_MvD_new.coffea \\
        -m coffea4bees/plots/metadata/plotsAll_MvD_ttbar_weights.yml
    Then open http://localhost:5000
"""

import os
import re
import sys
import base64
import io
import json
import shutil
import threading
import tempfile
import warnings
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

#os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import yaml
import argparse
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore', message='.*All sumw are zero.*')

sys.path.insert(0, os.getcwd())
from flask import Flask, jsonify, redirect, request, render_template, send_from_directory

from coffea4bees.plots.plots import load_config_4b
from src.plotting.plots import (
    makePlot, make2DPlot, load_hists, read_axes_and_cuts, init_arg_parser,
    _normalize_kwargs,
)
from src.plotting.iPlot_config import plot_config

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
cfg = plot_config()
gallery_items = []          # populated by _pregallery()
gallery_lock = threading.Lock()
plot_lock = threading.Lock()  # matplotlib is not thread-safe
output_dir = None
input_files = []
metadata_file = None
reuse_gallery = False       # set from --reuse-gallery flag
session_label = ""          # set from --label (--new) or manifest (--load)

app = Flask(__name__, template_folder="templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(s):
    if isinstance(s, list):
        s = "_vs_".join(str(x) for x in s)
    return str(s).replace(".", "_").replace("/", "_").replace(" ", "_")


def _fig_to_png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _get_processes():
    processes = []
    for key, values in cfg.plotConfig.items():
        if key in ["hists", "stack"]:
            for _key in values.keys():
                processes.append(_key)
    return processes


def _init_config(args):
    cfg.plotConfig = load_config_4b(args.metadata)
    cfg.outputFolder = None   # web app manages its own output paths
    cfg.combine_input_files = getattr(args, 'combine_input_files', False)
    cfg.fileLabels = getattr(args, 'fileLabels', [])

    cfg.plotModifiers = {}
    mod_file = getattr(args, 'modifiers', None)
    if mod_file and os.path.exists(mod_file):
        cfg.plotModifiers = yaml.safe_load(open(mod_file, 'r'))

    cfg.hists = load_hists(args.inputFile)

    try:
        cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(
            cfg.hists, cfg.plotConfig, hist_keys=['hists', 'hists_ttbar']
        )
    except Exception:
        cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(
            cfg.hists, cfg.plotConfig, hist_keys=['hists']
        )
    cfg.set_hist_key("hists")


# ---------------------------------------------------------------------------
# Multiprocessing worker support for pregallery
# ---------------------------------------------------------------------------

_gallery_worker_cfg = None
_gallery_worker_dir = None


def _gallery_init_worker(cfg_state, gallery_dir_str):
    """Initialise a pregallery worker process."""
    import os, sys, tempfile, warnings
    os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')
    warnings.filterwarnings('ignore', message='.*All sumw are zero.*')

    sys.path.insert(0, os.getcwd())
    from src.plotting.iPlot_config import plot_config

    global _gallery_worker_cfg, _gallery_worker_dir
    _gallery_worker_cfg = plot_config()
    _gallery_worker_cfg.hists               = cfg_state['hists']
    _gallery_worker_cfg.plotConfig          = cfg_state['plotConfig']
    _gallery_worker_cfg.plotModifiers       = cfg_state['plotModifiers']
    _gallery_worker_cfg.axisLabelsDict      = cfg_state['axisLabelsDict']
    _gallery_worker_cfg.cutListDict         = cfg_state['cutListDict']
    _gallery_worker_cfg.combine_input_files = cfg_state['combine_input_files']
    _gallery_worker_cfg.fileLabels          = cfg_state['fileLabels']
    _gallery_worker_cfg.set_hist_key("hists")
    _gallery_worker_dir = gallery_dir_str


def _do_gallery_job(job):
    """Worker: render one pregallery plot and return its item dict (or None)."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    from src.plotting.plots import makePlot

    v, region, stem, plot_kwargs = job
    gallery_dir = Path(_gallery_worker_dir)
    png_path = gallery_dir / f"{stem}.png"
    pdf_path = gallery_dir / f"{stem}.pdf"

    try:
        result = makePlot(_gallery_worker_cfg, var=v, axis_opts={"region": region}, **plot_kwargs)
        if result is None:
            return None
        fig, _ = result
        fig.savefig(str(png_path), format="png", dpi=100, bbox_inches="tight")
        fig.savefig(str(pdf_path), format="pdf")
        plt.close(fig)
        return {
            "var": v,
            "region": str(region),
            "label": f"{v} | {str(region)}",
            "png_url": f"/outputs/gallery/{stem}.png",
            "pdf_url": f"/outputs/gallery/{stem}.pdf",
        }
    except Exception as e:
        plt.close("all")
        print(f"  skip {v} {region}: {e}")
        return None


def _pregallery(out_dir, n_jobs=1, reuse=False):
    """Pre-generate PNG + PDF for every 1D variable × region combination."""
    global gallery_items

    gallery_dir = Path(out_dir) / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    var_list = list(cfg.hists[0]['hists'].keys())
    regions = cfg.axisLabelsDict["hists"].get("region", ["SR"])

    # Build flat job list, optionally skipping already-generated plots
    jobs = []
    reused_items = []
    for v in var_list:
        vDict = cfg.plotModifiers.get(v, {})
        if vDict.get("2d", False):
            continue
        plot_kwargs = {
            "ylabel": "Entries",
            "doRatio": cfg.plotConfig.get("doRatio", True),
            "legend": True,
        }
        plot_kwargs.update({k: val for k, val in vDict.items() if k != "2d"})
        for region in regions:
            stem = f"{_sanitize(v)}_region_{_sanitize(str(region))}"
            if reuse and (gallery_dir / f"{stem}.png").exists() and (gallery_dir / f"{stem}.pdf").exists():
                reused_items.append({
                    "var": v,
                    "region": str(region),
                    "label": f"{v} | {str(region)}",
                    "png_url": f"/outputs/gallery/{stem}.png",
                    "pdf_url": f"/outputs/gallery/{stem}.pdf",
                })
            else:
                jobs.append((v, region, stem, plot_kwargs))

    if reused_items:
        print(f"  Reusing {len(reused_items)} existing plots")

    n_total = len(jobs)
    if n_total:
        print(f"\nPre-generating gallery: {n_total} plots with {n_jobs} worker(s) ...")

    new_items = []

    if not n_total:
        with gallery_lock:
            gallery_items.extend(reused_items)
        print(f"Gallery ready: {len(gallery_items)} plots in {gallery_dir}\n")
        return

    if n_jobs > 1:
        axis_labels_picklable = {
            hk: {axis: list(vals) for axis, vals in hk_dict.items()}
            for hk, hk_dict in cfg.axisLabelsDict.items()
        }
        cfg_state = {
            'hists':               cfg.hists,
            'plotConfig':          cfg.plotConfig,
            'plotModifiers':       cfg.plotModifiers,
            'axisLabelsDict':      axis_labels_picklable,
            'cutListDict':         cfg.cutListDict,
            'combine_input_files': cfg.combine_input_files,
            'fileLabels':          cfg.fileLabels,
        }
        with mp.Pool(n_jobs, initializer=_gallery_init_worker,
                     initargs=(cfg_state, str(gallery_dir))) as pool:
            for i, item in enumerate(pool.imap_unordered(_do_gallery_job, jobs, chunksize=4), 1):
                if item is not None:
                    new_items.append(item)
                if i % 10 == 0 or i == n_total:
                    print(f"  {i}/{n_total} done")
    else:
        for i, (v, region, stem, plot_kwargs) in enumerate(jobs, 1):
            png_path = gallery_dir / f"{stem}.png"
            pdf_path = gallery_dir / f"{stem}.pdf"
            try:
                with plot_lock:
                    result = makePlot(cfg, var=v, axis_opts={"region": region}, **plot_kwargs)
                    if result is None:
                        continue
                    fig, _ = result
                    fig.savefig(str(png_path), format="png", dpi=100, bbox_inches="tight")
                    fig.savefig(str(pdf_path), format="pdf")
                    plt.close(fig)
                new_items.append({
                    "var": v,
                    "region": str(region),
                    "label": f"{v} | {str(region)}",
                    "png_url": f"/outputs/gallery/{stem}.png",
                    "pdf_url": f"/outputs/gallery/{stem}.pdf",
                })
            except Exception as e:
                plt.close("all")
                print(f"  skip {v} {region}: {e}")
            if i % 10 == 0 or i == n_total:
                print(f"  {i}/{n_total} done")

    with gallery_lock:
        gallery_items.extend(reused_items)
        gallery_items.extend(new_items)

    print(f"Gallery ready: {len(gallery_items)} plots in {gallery_dir}\n")


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

def _write_manifest(archive_dir, input_files_rel, metadata_rel, label, args=None):
    """Write manifest.json into archive_dir with paths relative to archive_dir."""
    manifest = {
        "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "inputs": [str(p) for p in input_files_rel],
        "metadata": str(metadata_rel) if metadata_rel else None,
        "combine_input_files": getattr(args, 'combine_input_files', False),
        "fileLabels": getattr(args, 'fileLabels', []) or [],
    }
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _register_archive(registry_path, archive_dir, label):
    """Append an entry for archive_dir to the registry JSON (atomic write)."""
    registry_path = Path(registry_path)
    if registry_path.exists():
        try:
            entries = json.loads(registry_path.read_text())
        except Exception:
            entries = []
    else:
        entries = []
    entries.append({
        "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "archive_dir": str(archive_dir),
    })
    tmp = registry_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.rename(registry_path)


# ---------------------------------------------------------------------------
# Session tracking (live PID → port mapping per archive directory)
# ---------------------------------------------------------------------------

def _sessions_path(registry_path):
    """Derive the sessions file path from the registry path."""
    return Path(registry_path).parent / "pourover_sessions.json"


def _sessions_read(sessions_path):
    sessions_path = Path(sessions_path)
    if not sessions_path.exists():
        return {}
    try:
        return json.loads(sessions_path.read_text())
    except Exception:
        return {}


def _sessions_write(sessions_path, data):
    sessions_path = Path(sessions_path)
    sessions_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sessions_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(sessions_path)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _session_check(archive_dir, sessions_path):
    """If archive_dir has a live session, print its URL and exit."""
    key = str(Path(archive_dir).resolve())
    sessions = _sessions_read(sessions_path)
    # Prune dead entries while we're here
    alive = {k: v for k, v in sessions.items() if _pid_alive(v["pid"])}
    if alive != sessions:
        _sessions_write(sessions_path, alive)
    if key in alive:
        entry = alive[key]
        print(f"\nPourOver is already running for this archive (pid {entry['pid']}).")
        print(f"  → http://localhost:{entry['port']}")
        sys.exit(0)


def _session_register(archive_dir, pid, port, sessions_path):
    """Record a live session for archive_dir."""
    key = str(Path(archive_dir).resolve())
    sessions = _sessions_read(sessions_path)
    # Prune dead entries
    sessions = {k: v for k, v in sessions.items() if _pid_alive(v["pid"])}
    sessions[key] = {"pid": pid, "port": port}
    _sessions_write(sessions_path, sessions)


def _session_remove(archive_dir, sessions_path):
    """Remove any session entry for archive_dir (used by --delete)."""
    key = str(Path(archive_dir).resolve())
    sessions = _sessions_read(sessions_path)
    if key in sessions:
        del sessions[key]
        _sessions_write(sessions_path, sessions)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

def _template_context():
    names = [Path(f).name for f in input_files]
    return dict(input_files=names,
                metadata_file=Path(metadata_file).name if metadata_file else "",
                session_label=session_label)


@app.route("/")
def index():
    return render_template("iplot.html", **_template_context())


@app.route("/gallery-view")
def gallery_view():
    return render_template("gallery.html", **_template_context())


@app.route("/iplot")
def iplot_page():
    return redirect("/")


@app.route("/gallery")
def gallery():
    with gallery_lock:
        return jsonify(list(gallery_items))


@app.route("/axes")
def axes():
    variables = list(cfg.axisLabelsDict["hists"].get("var", []))
    regions   = [str(r) for r in cfg.axisLabelsDict["hists"].get("region", [])]
    cuts      = list(cfg.cutListDict.get("hists", []))
    years     = [str(y) for y in cfg.axisLabelsDict["hists"].get("year", [])]
    processes = _get_processes()
    return jsonify({
        "variables": variables,
        "regions":   regions,
        "cuts":      cuts,
        "years":     years,
        "processes": processes,
    })


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(os.path.abspath(output_dir), filename)


import ast as _ast
import contextlib as _contextlib
import logging as _logging

_PLOT_CMDS = {'plot', 'plot2d', 'plot_roc'}
_TEXT_CMDS = {'ls', 'info', 'examples'}
_ALL_CMDS  = _PLOT_CMDS | _TEXT_CMDS


def _parse_cli_cmd(cmd):
    """Parse a CLI command string.

    Supported functions:
        plot("var", cut="...", region="SR", rebin=2, ...)
        plot2d("var", "process", cut="...", region="SR", ...)
        ls()                  — list all variables
        ls("MvD")             — filter variables containing "MvD"
        ls("region")          — list regions  (or "cut", "var")
        info()                — print config summary
        examples()            — print example commands

    Returns a dict with key "func" and parsed args/kwargs.
    """
    try:
        tree = _ast.parse(cmd.strip(), mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}")

    if not isinstance(tree.body, _ast.Call):
        raise ValueError(f"Expected a function call: {', '.join(sorted(_ALL_CMDS))}")

    call = tree.body
    if not isinstance(call.func, _ast.Name):
        raise ValueError(f"Expected one of: {', '.join(sorted(_ALL_CMDS))}")

    func = call.func.id
    if func not in _ALL_CMDS:
        raise ValueError(f"Unknown command '{func}' — try: {', '.join(sorted(_ALL_CMDS))}")

    try:
        args   = [_ast.literal_eval(a) for a in call.args]
        kwargs = {kw.arg: _ast.literal_eval(kw.value) for kw in call.keywords}
    except ValueError as e:
        raise ValueError(f"Bad argument value: {e}")

    parsed = {"func": func, "args": args, "kwargs": kwargs}

    # For plot/plot2d also build the request payload inline
    if func in _PLOT_CMDS:
        req = {"is2d": func == 'plot2d'}
        if func == 'plot2d':
            if len(args) >= 1: req['var']     = args[0]
            if len(args) >= 2: req['process'] = args[1]
        elif func == 'plot_roc':
            if len(args) >= 1: req['var'] = args[0]
        else:
            if len(args) >= 1: req['var'] = args[0]
        req.update(kwargs)
        parsed["req"] = req

    return parsed


def _execute_text_cmd(func, args, kwargs):
    """Execute a text-output command (ls/info/examples).

    Returns (data_dict, status_code).
    """
    try:
        buf = io.StringIO()
        with _contextlib.redirect_stdout(buf):
            if func == 'ls':
                # First positional arg: if it matches a known axis key treat as
                # option (e.g. "region", "cut"); otherwise treat as var filter.
                hist_key  = kwargs.get('hist_key', cfg.hist_key or 'hists')
                axis_keys = set(cfg.axisLabelsDict.get(hist_key, {}).keys())
                if args and isinstance(args[0], str) and args[0] in axis_keys:
                    option    = args[0]
                    var_match = kwargs.get('var_match')
                else:
                    option    = kwargs.get('option', 'var')
                    var_match = args[0] if args else kwargs.get('var_match')
                entries = cfg.axisLabelsDict.get(hist_key, {}).get(option, [])
                for k in entries:
                    if var_match is None or var_match in k:
                        print(k)

            elif func == 'info':
                from src.plotting.plots import print_cfg
                print_cfg(cfg)

            elif func == 'examples':
                from coffea4bees.plots.iPlot import examples as _iplot_examples
                _iplot_examples()

        text = buf.getvalue().rstrip()
        if not text:
            text = "(no output)"
        return {"text": text}, 200

    except Exception as e:
        return {"error": str(e)}, 400


def _execute_plot(req):
    """Shared plot execution for /plot and /cli endpoints.

    Returns (data_dict, status_code).  data_dict contains either
    {png_b64, png_url, pdf_url} on success or {error} on failure.
    """
    var     = req.get("var", "selJets.pt")
    cut     = req.get("cut") or None
    region  = req.get("region", ["SR"])
    if isinstance(region, str):
        region = [region]
    # Allow "sum" string to be passed as the Python built-in sum function
    region = [sum if r == "sum" else r for r in region]
    is2d    = req.get("is2d", False)
    process = req.get("process") or None

    with plot_lock:
        cfg.set_hist_key("hists")
        if cut and cut in ["passMuMu", "passElMu"]:
            cfg.set_hist_key("hists_ttbar")

    debug = bool(req.get("debug", False))

    _normalize_kwargs(req)

    kwargs = {}
    for k in ["doRatio", "rebin", "norm", "yscale", "xscale", "add_flow", "uniform_bins",
              "year", "year_str", "CMSText",
              "xlabel", "ylabel", "rlabel",
              "legend", "legend_loc", "ratio_legend_loc",
              "do_title",
              "full", "plot_contour", "plot_leadst_lines", "plot_sublst_lines"]:
        v = req.get(k)
        if v is not None:
            kwargs[k] = v
    if debug:
        kwargs["debug"] = True

    for k in ["xlim", "ylim", "rlim", "zlim"]:
        v = req.get(k)
        if v and any(x is not None for x in v):
            kwargs[k] = [x for x in v]

    for k in ["legend_order", "ratio_legend_order"]:
        v = req.get(k)
        if v is not None:
            kwargs[k] = v

    # Convert "sum" string to the Python built-in sum function for axis summing
    if isinstance(kwargs.get("year"), str) and kwargs["year"] == "sum":
        kwargs["year"] = sum

    axis_opts = {"region": region[0] if len(region) == 1 else region}

    try:
        debug_buf = io.StringIO() if debug else None
        debug_cm  = _contextlib.redirect_stdout(debug_buf) if debug else _contextlib.nullcontext()

        # Route logger output from plotting modules to debug_buf
        _debug_loggers = [
            _logging.getLogger("src.plotting.helpers_make_plot"),
            _logging.getLogger("src.plotting.helpers_make_plot_dict"),
        ]
        _log_handler = None
        if debug and debug_buf:
            _log_handler = _logging.StreamHandler(debug_buf)
            _log_handler.setLevel(_logging.DEBUG)
            for _lg in _debug_loggers:
                _lg.addHandler(_log_handler)

        with plot_lock:
            with debug_cm:
                if is2d:
                    if not process:
                        procs = _get_processes()
                        process = procs[0] if procs else "data"
                    fig, _ = make2DPlot(
                        cfg, process, var=var, cut=cut,
                        axis_opts=axis_opts, **kwargs
                    )
                else:
                    if process:
                        kwargs["process"] = process
                    fig, _ = makePlot(
                        cfg, var=var, cut=cut,
                        axis_opts=axis_opts, **kwargs
                    )

        interactive_dir = Path(output_dir) / "interactive"
        interactive_dir.mkdir(parents=True, exist_ok=True)

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"{ts}_{_sanitize(var)}"
        png_path = interactive_dir / f"{stem}.png"
        pdf_path = interactive_dir / f"{stem}.pdf"

        with plot_lock:
            fig.savefig(str(png_path), format="png", dpi=100, bbox_inches="tight")
            fig.savefig(str(pdf_path), format="pdf")
            png_b64 = _fig_to_png_b64(fig)
            plt.close(fig)

        if _log_handler:
            for _lg in _debug_loggers:
                _lg.removeHandler(_log_handler)

        result = {
            "png_b64": png_b64,
            "png_url": f"/outputs/interactive/{stem}.png",
            "pdf_url": f"/outputs/interactive/{stem}.pdf",
        }
        if debug and debug_buf:
            result["debug"] = debug_buf.getvalue()
        return result, 200

    except Exception as e:
        if _log_handler:
            for _lg in _debug_loggers:
                _lg.removeHandler(_log_handler)
        with plot_lock:
            plt.close("all")
        return {"error": str(e)}, 400


def _get_signal_plot(plot_data, s):
    import src.plotting.helpers as _ph
    try:
        return _ph.get_value_nested_dict(plot_data, s)
    except Exception:
        return _ph.make_klambda_hist(s, plot_data)


def _make_roc_figure(roc_data, outputFolder=None):
    for _v in roc_data:
        sig_cumsum = np.cumsum(np.array(roc_data[_v]["sig_values"])[::-1])[::-1]
        bkg_cumsum = np.cumsum(np.array(roc_data[_v]["bkg_values"])[::-1])[::-1]
        TPR = sig_cumsum / sig_cumsum[0]
        FPR = bkg_cumsum / bkg_cumsum[0]
        roc_data[_v]["FPR"] = FPR
        roc_data[_v]["TPR"] = TPR
        roc_data[_v]["auc"] = np.trapz(TPR, FPR)

    fig = plt.figure(figsize=(8, 6))
    for _v in roc_data:
        plt.plot(roc_data[_v]["FPR"], roc_data[_v]["TPR"],
                 color=roc_data[_v]["color"], lw=2,
                 label=f'{roc_data[_v]["label"]} (AUC = {roc_data[_v]["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('Background Efficiency')
    plt.ylabel('Signal Efficiency')
    plt.legend(loc='lower right')
    plt.grid()
    return fig


def _execute_roc(req):
    """Generate a ROC curve: one curve per loaded input file.

    Returns (data_dict, status_code) — same contract as _execute_plot.
    """
    import src.plotting.helpers as plot_helpers
    from src.plotting.helpers_make_plot_dict import get_plot_dict_from_config

    var    = req.get("var", "SvB_MA.ps_hh_fine")
    sig    = req.get("sig") or []
    bkg    = req.get("bkg") or []
    region = req.get("region", "SR")
    cut    = req.get("cut") or None

    if not sig or not bkg:
        return {"error": "plot_roc requires sig= and bkg= process lists"}, 400

    file_labels  = cfg.fileLabels or []
    colors_cycle = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]

    roc_data = {}
    try:
        for i, hist_entry in enumerate(cfg.hists):
            if i < len(file_labels):
                label = file_labels[i]
            elif i < len(input_files):
                label = Path(input_files[i]).stem
            else:
                label = f"file_{i}"
            color = colors_cycle[i % len(colors_cycle)]

            with plot_lock:
                saved_hists = cfg.hists
                cfg.hists = [hist_entry]
                try:
                    plot_data = get_plot_dict_from_config(cfg=cfg, var=var, cut=cut, axis_opts={"region": region})
                finally:
                    cfg.hists = saved_hists

            sig_data = None
            for s in sig:
                try:
                    s_data = _get_signal_plot(plot_data, s)
                except Exception as e:
                    return {"error": f"Signal process '{s}' not found: {e}"}, 400
                if sig_data is None:
                    sig_data = {k: np.array(v) for k, v in s_data.items()
                                if k in ("values", "variances", "under_flow", "over_flow")}
                else:
                    for k in ("values", "variances", "under_flow", "over_flow"):
                        sig_data[k] = sig_data[k] + np.array(s_data[k])

            bkg_data = None
            for b in bkg:
                try:
                    b_data = plot_helpers.get_value_nested_dict(plot_data, b)
                except Exception as e:
                    return {"error": f"Background process '{b}' not found: {e}"}, 400
                if bkg_data is None:
                    bkg_data = {k: np.array(v) for k, v in b_data.items()
                                if k in ("values", "variances", "under_flow", "over_flow")}
                else:
                    for k in ("values", "variances", "under_flow", "over_flow"):
                        bkg_data[k] = bkg_data[k] + np.array(b_data[k])

            sig_data["values"][0] += sig_data["under_flow"]
            bkg_data["values"][0] += bkg_data["under_flow"]

            roc_data[label] = {
                "sig_values": sig_data["values"],
                "bkg_values": bkg_data["values"],
                "label": label,
                "color": color,
            }

        with plot_lock:
            fig = _make_roc_figure(roc_data)

            interactive_dir = Path(output_dir) / "interactive"
            interactive_dir.mkdir(parents=True, exist_ok=True)
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stem = f"{ts}_roc_{_sanitize(var)}"
            png_path = interactive_dir / f"{stem}.png"
            pdf_path = interactive_dir / f"{stem}.pdf"

            fig.savefig(str(png_path), format="png", dpi=100, bbox_inches="tight")
            fig.savefig(str(pdf_path), format="pdf")
            png_b64 = _fig_to_png_b64(fig)
            plt.close(fig)

        return {
            "png_b64": png_b64,
            "png_url": f"/outputs/interactive/{stem}.png",
            "pdf_url": f"/outputs/interactive/{stem}.pdf",
        }, 200

    except Exception as e:
        with plot_lock:
            plt.close("all")
        return {"error": str(e)}, 400


@app.route("/plot", methods=["POST"])
def plot_endpoint():
    data, code = _execute_plot(request.get_json())
    return jsonify(data), code


@app.route("/plot_roc", methods=["POST"])
def plot_roc_endpoint():
    data, code = _execute_roc(request.get_json())
    return jsonify(data), code


@app.route("/cli", methods=["POST"])
def cli_endpoint():
    req = request.get_json()
    cmd = (req.get("cmd") or "").strip()
    if not cmd:
        return jsonify({"error": "Empty command"}), 400
    try:
        parsed = _parse_cli_cmd(cmd)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    func = parsed["func"]

    # Text-output commands
    if func in _TEXT_CMDS:
        data, code = _execute_text_cmd(func, parsed["args"], parsed["kwargs"])
        if code == 200:
            _save_cli_history(cmd)
        return jsonify(data), code

    # Wildcard plot → behave like ls() with the filter
    plot_req = parsed["req"]
    var = plot_req.get("var", "")
    if isinstance(var, str) and "*" in var:
        var_match = var.replace("*", "")
        data, code = _execute_text_cmd("ls", [var_match] if var_match else [], {})
        if code == 200:
            _save_cli_history(cmd)
        return jsonify(data), code

    # Dispatch by command type
    if func == 'plot_roc':
        data, code = _execute_roc(plot_req)
    else:
        data, code = _execute_plot(plot_req)
    if code == 200:
        _save_cli_history(cmd)
        _save_interactive_item(data["png_url"], data["pdf_url"], cmd)
    return jsonify(data), code


def _history_path():
    return Path(output_dir) / "cli_history.json"


def _interactive_manifest_path():
    return Path(output_dir) / "interactive_history.json"


def _save_cli_history(cmd):
    """Append cmd to cli_history.json (chronological order, oldest first)."""
    path = _history_path()
    try:
        history = json.loads(path.read_text()) if path.exists() else []
        if not history or history[-1] != cmd:   # skip exact consecutive duplicates
            history.append(cmd)
        path.write_text(json.dumps(history, indent=2))
    except Exception as e:
        print(f"Warning: could not save CLI history: {e}")


def _save_interactive_item(png_url, pdf_url, cmd):
    """Append an interactive plot entry to interactive_history.json."""
    path = _interactive_manifest_path()
    try:
        items = json.loads(path.read_text()) if path.exists() else []
        items.append({"png_url": png_url, "pdf_url": pdf_url, "cmd": cmd})
        path.write_text(json.dumps(items, indent=2))
    except Exception as e:
        print(f"Warning: could not save interactive history: {e}")


@app.route("/cli/history")
def cli_history():
    """Return saved CLI history (newest first) if --reuse-gallery was set."""
    if not reuse_gallery:
        return jsonify([])
    path = _history_path()
    if not path.exists():
        return jsonify([])
    try:
        history = json.loads(path.read_text())
        return jsonify(list(reversed(history)))   # newest first for the frontend
    except Exception:
        return jsonify([])


@app.route("/interactive/history")
def interactive_history():
    """Return saved interactive plot items (newest first) if --reuse-gallery was set."""
    if not reuse_gallery:
        return jsonify([])
    path = _interactive_manifest_path()
    if not path.exists():
        return jsonify([])
    try:
        items = json.loads(path.read_text())
        return jsonify(list(reversed(items)))   # newest first for the frontend
    except Exception:
        return jsonify([])






# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_KEBAB_RE = re.compile(r'^[a-zA-Z0-9]+(-[a-zA-Z0-9]+)*$')


def _validate_label(label):
    """Return label if valid kebab-case, else raise ValueError."""
    if not _KEBAB_RE.match(label):
        raise ValueError(
            f"Label must be kebab-case (letters, digits, hyphens; "
            f"no leading/trailing hyphens): got {label!r}"
        )
    return label


def _resolve_label(label, registry_path):
    """Look up label in registry; return the archive_dir of the most recent match."""
    try:
        entries = json.loads(Path(registry_path).read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    matches = [e for e in entries if e.get('label') == label]
    if not matches:
        available = sorted({e.get('label', '') for e in entries} - {''})
        raise KeyError(
            f"No archive with label {label!r}. "
            f"Available: {available if available else '(none)'}"
        )
    if len(matches) > 1:
        print(f"Warning: {len(matches)} archives with label {label!r}; using most recent.")
    return matches[-1]['archive_dir']



def _parse_args():
    parser = init_arg_parser()
    # Make inputFile optional so --load can be used without positional args
    for action in parser._actions:
        if getattr(action, 'dest', None) == 'inputFile':
            action.nargs = '*'
            action.default = []
            break
    parser.add_argument('--port', type=int, default=5000, help='Port to serve on (default: 5000)')
    parser.add_argument('--foreground', action='store_true',
                        help='Run server in the foreground instead of detaching to background')
    parser.add_argument('--pregallery', action='store_true',
                        help='Pre-generate the full plot gallery on startup (default: off)')
    parser.add_argument('--reuse-gallery', action='store_true',
                        help='Reuse already-generated gallery plots; only regenerate missing ones')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of parallel workers for pregallery (default: 1)')
    parser.add_argument('--output-dir', default='pourover_output',
                        help='Directory for generated plots (default: pourover_output)')
    parser.add_argument('--new', action='store_true',
                        help='Snapshot inputs, register archive, generate gallery, serve.')
    load_action = parser.add_argument('--load', metavar='LABEL',
                        help='Load from a previously created archive by its kebab-case label.')
    parser.add_argument('--registry', default='.pourover_archives/pourover_registry.json', metavar='FILE',
                        help='Path to the archive registry JSON (default: .pourover_archives/pourover_registry.json).')
    parser.add_argument('--label', default='', metavar='LABEL',
                        help='Kebab-case label for this archive (required with --new, e.g. ul18-sb-test).')
    parser.add_argument('--list', action='store_true',
                        help='List all archives in the registry and exit.')
    parser.add_argument('--rename', nargs=2, metavar=('OLD', 'NEW'),
                        help='Rename all registry entries with label OLD to NEW and exit.')
    parser.add_argument('--delete', metavar='LABEL',
                        help='Delete all archives with this label (removes directories and registry entries) and exit.')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt for --delete.')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("In main")
    import argparse as _argparse

    args = _parse_args()

    # ------------------------------------------------------------------
    # --list: print registry and exit
    # ------------------------------------------------------------------
    if args.list:
        registry = Path(args.registry)
        if not registry.exists():
            print(f"No registry found at {registry}")
            sys.exit(0)
        try:
            entries = json.loads(registry.read_text())
        except Exception as e:
            print(f"Could not read registry: {e}", file=sys.stderr)
            sys.exit(1)
        if not entries:
            print("Registry is empty.")
            sys.exit(0)
        col = max(len(e.get('label', '')) for e in entries)
        for e in entries:
            label   = e.get('label', '(no label)')
            created = e.get('created', '')
            adir    = e.get('archive_dir', '')
            print(f"{label:<{col}}  {created}  {adir}")
        sys.exit(0)

    # ------------------------------------------------------------------
    # --rename OLD NEW: relabel registry entries and manifests, then exit
    # ------------------------------------------------------------------
    if args.rename:
        old_label, new_label = args.rename
        try:
            _validate_label(new_label)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        registry = Path(args.registry)
        if not registry.exists():
            print(f"No registry found at {registry}", file=sys.stderr)
            sys.exit(1)
        try:
            entries = json.loads(registry.read_text())
        except Exception as e:
            print(f"Could not read registry: {e}", file=sys.stderr)
            sys.exit(1)
        matches = [e for e in entries if e.get('label') == old_label]
        if not matches:
            print(f"error: no archives with label {old_label!r}", file=sys.stderr)
            sys.exit(1)
        for e in entries:
            if e.get('label') == old_label:
                e['label'] = new_label
        tmp = registry.with_suffix('.tmp')
        tmp.write_text(json.dumps(entries, indent=2))
        tmp.rename(registry)
        # Update manifest.json inside each matched archive
        for e in matches:
            manifest_path = Path(e['archive_dir']) / 'manifest.json'
            if manifest_path.exists():
                try:
                    m = json.loads(manifest_path.read_text())
                    m['label'] = new_label
                    manifest_path.write_text(json.dumps(m, indent=2))
                except Exception as ex:
                    print(f"Warning: could not update manifest in {e['archive_dir']}: {ex}")
        print(f"Renamed {len(matches)} archive(s): {old_label!r} → {new_label!r}")
        sys.exit(0)

    # ------------------------------------------------------------------
    # --delete LABEL: remove archive directories and registry entries, then exit
    # ------------------------------------------------------------------
    if args.delete:
        registry = Path(args.registry)
        if not registry.exists():
            print(f"No registry found at {registry}", file=sys.stderr)
            sys.exit(1)
        try:
            entries = json.loads(registry.read_text())
        except Exception as e:
            print(f"Could not read registry: {e}", file=sys.stderr)
            sys.exit(1)
        matches = [e for e in entries if e.get('label') == args.delete]
        if not matches:
            print(f"error: no archives with label {args.delete!r}", file=sys.stderr)
            sys.exit(1)
        print(f"Will delete {len(matches)} archive(s) with label {args.delete!r}:")
        for e in matches:
            print(f"  {e['archive_dir']}")
        if not args.yes:
            try:
                answer = input(f"Delete {len(matches)} archive(s)? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)
            if answer not in ('y', 'yes'):
                print("Aborted.")
                sys.exit(1)
        for e in matches:
            adir = Path(e['archive_dir'])
            if adir.exists():
                shutil.rmtree(adir)
                print(f"  Removed {adir}")
            else:
                print(f"  Warning: directory not found, skipping: {adir}")
            _session_remove(adir, _sessions_path(args.registry))
        remaining = [e for e in entries if e.get('label') != args.delete]
        tmp = registry.with_suffix('.tmp')
        tmp.write_text(json.dumps(remaining, indent=2))
        tmp.rename(registry)
        print(f"Deleted {len(matches)} archive(s) with label {args.delete!r}")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Mutual exclusion / required-arg validation
    # ------------------------------------------------------------------
    def _arg_error(msg):
        print(f"error: {msg}", file=sys.stderr)
        sys.exit(2)

    if args.new and args.load:
        _arg_error("--new and --load are mutually exclusive.")
    if args.load and args.inputFile:
        _arg_error("Do not pass inputFile with --load; inputs come from the archive.")
    if not args.load and not args.inputFile:
        _arg_error("inputFile is required (unless using --load).")
    if args.new and not args.label:
        _arg_error("--label is required with --new (kebab-case, e.g. ul18-sb-test).")
    if args.new:
        try:
            _validate_label(args.label)
        except ValueError as e:
            _arg_error(str(e))

    # ------------------------------------------------------------------
    # --load mode: start from an existing archive
    # ------------------------------------------------------------------
    if args.load:
        try:
            archive_dir = Path(_resolve_label(args.load, args.registry))
        except (FileNotFoundError, KeyError) as e:
            _arg_error(str(e))
        if not archive_dir.exists():
            _arg_error(f"Archive directory does not exist: {archive_dir}")
        manifest_path = archive_dir / "manifest.json"
        if not manifest_path.exists():
            _arg_error(f"No manifest.json found in {archive_dir} — not a valid --new archive.")

        manifest = json.loads(manifest_path.read_text())
        resolved_inputs = [str(archive_dir / p) for p in manifest["inputs"]]
        resolved_meta   = str(archive_dir / manifest["metadata"]) if manifest.get("metadata") else args.metadata

        # Fabricate an args namespace for _init_config
        load_args = _argparse.Namespace(
            inputFile=resolved_inputs,
            metadata=resolved_meta,
            combine_input_files=manifest.get('combine_input_files', False),
            fileLabels=manifest.get('fileLabels', []),
            modifiers=getattr(args, 'modifiers', None),
        )

        output_dir    = str(archive_dir)
        input_files   = resolved_inputs
        metadata_file = resolved_meta
        reuse_gallery = True
        session_label = manifest.get('label', '')

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        _session_check(archive_dir, _sessions_path(args.registry))

        print(f"Loading archive: {archive_dir}")
        print(f"  Label: {manifest.get('label', '')}")
        print(f"  Created: {manifest.get('created', '')}")
        _init_config(load_args)
        print(f"  Loaded {len(cfg.hists)} file(s)")
        print(f"  Metadata: {metadata_file}")

        if args.pregallery:
            _pregallery(output_dir, n_jobs=args.jobs, reuse=True)

    # ------------------------------------------------------------------
    # --new mode: snapshot inputs, register, generate, serve
    # ------------------------------------------------------------------
    elif args.new:
        registry = Path(args.registry)
        if registry.exists():
            try:
                existing = json.loads(registry.read_text())
                if any(e.get('label') == args.label for e in existing):
                    _arg_error(f"Label {args.label!r} already exists in the registry. "
                               f"Use --rename to rename it or --delete to remove it first.")
            except Exception:
                pass  # malformed registry; let _register_archive handle it

        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path(".pourover_archives") / f"pourover_archive_{ts}"
        inputs_dir  = archive_dir / "inputs"
        inputs_dir.mkdir(parents=True)

        # Copy input coffea files into archive
        copied_inputs = []
        for src in args.inputFile:
            dst = inputs_dir / Path(src).name
            shutil.copy(src, dst)
            copied_inputs.append(Path("inputs") / Path(src).name)

        # Copy metadata YAML if provided
        copied_meta = None
        if args.metadata:
            dst = inputs_dir / Path(args.metadata).name
            shutil.copy(args.metadata, dst)
            copied_meta = Path("inputs") / Path(args.metadata).name

        _write_manifest(archive_dir, copied_inputs, copied_meta, args.label, args=args)
        _register_archive(args.registry, archive_dir, args.label)

        _session_check(archive_dir, _sessions_path(args.registry))

        print(f"Archive created: {archive_dir}")
        if args.label:
            print(f"  Label: {args.label}")

        output_dir    = str(archive_dir)
        input_files   = args.inputFile
        metadata_file = args.metadata
        reuse_gallery = False
        session_label = args.label

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("Loading histograms and config ...")
        _init_config(args)
        print(f"  Loaded {len(cfg.hists)} file(s): {input_files}")
        print(f"  Metadata: {metadata_file}")

        if args.pregallery:
            _pregallery(output_dir, n_jobs=args.jobs, reuse=False)

    # ------------------------------------------------------------------
    # Normal mode
    # ------------------------------------------------------------------
    else:
        output_dir    = args.output_dir
        input_files   = args.inputFile
        metadata_file = args.metadata
        reuse_gallery = args.reuse_gallery
        session_label = ""

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("Loading histograms and config ...")
        _init_config(args)
        print(f"  Loaded {len(cfg.hists)} file(s): {input_files}")
        print(f"  Metadata: {metadata_file}")

        if args.pregallery:
            _pregallery(output_dir, n_jobs=args.jobs, reuse=args.reuse_gallery)

    import socket
    def _find_free_port(preferred):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', preferred)) != 0:
                return preferred  # preferred port is free
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"\nPort {args.port} is in use; using port {port} instead.")

    _sessions_path_val = _sessions_path(args.registry)

    def _run_server():
        if output_dir:
            _session_register(output_dir, os.getpid(), port, _sessions_path_val)
        try:
            app.run(host='0.0.0.0', port=port, threaded=False, debug=False)
        finally:
            if output_dir:
                _session_remove(output_dir, _sessions_path_val)

    if args.foreground:
        print(f"\nStarting web server on http://localhost:{port}")
        print("Press Ctrl+C to stop.\n")
        _run_server()
    else:
        # Fork: parent prints URL and returns to the shell; child runs the server
        log_path = Path(output_dir) / "server.log" if output_dir else Path(os.devnull)
        pid = os.fork()
        if pid > 0:
            # Parent
            url = f"http://localhost:{port}"
            print(f"\nPourOver running in background (pid {pid})")
            print(f"  → {url}")
            if output_dir:
                print(f"  Logs: {log_path}")
            print(f"  Stop: kill {pid}")
            # Copy URL to clipboard (best-effort)
            import subprocess as _sp, platform as _platform
            _copied = False
            try:
                if _platform.system() == "Darwin":
                    _sp.run(["pbcopy"], input=url.encode(), check=True)
                    _copied = True
                else:
                    for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
                        if _sp.run(["which", cmd[0]], capture_output=True).returncode == 0:
                            _sp.run(cmd, input=url.encode(), check=True)
                            _copied = True
                            break
            except Exception:
                pass
            if _copied:
                print("  (URL copied to clipboard)")
            sys.exit(0)
        # Child: detach from terminal, redirect output to log file
        os.setsid()
        log_fd = open(log_path, 'a')
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())
        log_fd.close()
        _run_server()
