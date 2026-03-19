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

Extra options (beyond the standard iPlot/makePlotsAll args):
    --port PORT          Port to serve on (default: 5000)
    --no-pregallery      Skip pre-generating the gallery on startup
    --reuse-gallery      Reuse existing gallery plots; only regenerate missing ones
    --output-dir DIR     Directory for generated plots (default: pourover_output)
    --modifiers FILE     Per-variable modifier YAML (optional)
    -j N, --jobs N       Parallel workers for pregallery (default: 1)

Example:
    python coffea4bees/plots/pourOver.py \\
        coffea4bees/Run3_MvD/analysis_MvD_new.coffea \\
        -m coffea4bees/plots/metadata/plotsAll_MvD_ttbar_weights.yml
    Then open http://localhost:5000
"""

import os
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

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import yaml
import argparse
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore', message='.*All sumw are zero.*')

sys.path.insert(0, os.getcwd())
from flask import Flask, jsonify, request, render_template, send_from_directory

from coffea4bees.plots.plots import load_config_4b
from src.plotting.plots import (
    makePlot, make2DPlot, load_hists, read_axes_and_cuts, init_arg_parser
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
# Flask routes
# ---------------------------------------------------------------------------

def _template_context():
    names = [Path(f).name for f in input_files]
    return dict(input_files=names,
                metadata_file=Path(metadata_file).name if metadata_file else "")


@app.route("/")
def index():
    return render_template("gallery.html", **_template_context())


@app.route("/iplot")
def iplot_page():
    return render_template("iplot.html", **_template_context())


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

_PLOT_CMDS = {'plot', 'plot2d'}
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

    kwargs = {}
    for k in ["doRatio", "rebin", "norm", "yscale", "add_flow", "uniform_bins"]:
        v = req.get(k)
        if v is not None:
            kwargs[k] = v
    if debug:
        kwargs["debug"] = True

    for k in ["xlim", "ylim", "rlim"]:
        v = req.get(k)
        if v and any(x is not None for x in v):
            kwargs[k] = [x for x in v]

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


@app.route("/plot", methods=["POST"])
def plot_endpoint():
    data, code = _execute_plot(request.get_json())
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

    # Normal plot
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


@app.route("/archive", methods=["POST"])
def archive():
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"pourover_archive_{ts}")
    archive_dir.mkdir()

    # Copy inputs
    inputs_dir = archive_dir / "inputs"
    inputs_dir.mkdir()
    for f in input_files:
        shutil.copy(f, inputs_dir / Path(f).name)
    if metadata_file:
        shutil.copy(metadata_file, inputs_dir / Path(metadata_file).name)

    # Copy plots
    for subdir in ["gallery", "interactive"]:
        src = Path(output_dir) / subdir
        if src.exists():
            shutil.copytree(str(src), str(archive_dir / subdir))

    # Write reproduce.sh
    cmd = f"python coffea4bees/plots/pourOver.py {' '.join(input_files)} -m {metadata_file}"
    script = archive_dir / "reproduce.sh"
    script.write_text(f"#!/bin/bash\n# Run from barista root directory\n{cmd}\n")
    os.chmod(str(script), 0o755)

    # Write self-contained index.html
    _write_archive_html(archive_dir, ts)

    return jsonify({"archive_path": str(archive_dir.resolve())})


def _write_archive_html(archive_dir, ts):
    gallery_dir = archive_dir / "gallery"
    if not gallery_dir.exists():
        return

    with gallery_lock:
        items_snapshot = list(gallery_items)

    items_html = ""
    for item in items_snapshot:
        stem     = f"{_sanitize(item['var'])}_region_{_sanitize(item['region'])}"
        png_file = gallery_dir / f"{stem}.png"
        if not png_file.exists():
            continue
        with open(str(png_file), "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        pdf_name = f"{stem}.pdf"
        items_html += f"""
        <div class="thumb" data-label="{item['label']}">
          <img src="data:image/png;base64,{b64}" title="{item['label']}" />
          <div class="label">{item['label']}</div>
          <a href="gallery/{pdf_name}" download>⬇ PDF</a>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Plot Archive {ts}</title>
  <style>
    body {{ font-family: sans-serif; background: #1a1a2e; color: #eee; padding: 20px; margin: 0; }}
    h1 {{ color: #e0aaff; margin-bottom: 4px; }}
    .meta {{ color: #a0c4ff; font-size: 13px; margin-bottom: 16px; }}
    input[type=text] {{ background: #0f3460; border: 1px solid #444; color: #eee;
                        padding: 6px 10px; border-radius: 4px; width: 300px; margin-bottom: 12px; }}
    .grid {{ display: flex; flex-wrap: wrap; gap: 14px; }}
    .thumb {{ background: #16213e; border-radius: 8px; padding: 10px; width: 280px; }}
    .thumb img {{ width: 100%; border-radius: 4px; cursor: pointer; }}
    .thumb img:hover {{ opacity: 0.9; }}
    .label {{ font-size: 11px; margin: 6px 0 4px; color: #a0c4ff; word-break: break-all; }}
    .thumb a {{ font-size: 11px; color: #e0aaff; text-decoration: none; }}
    .thumb a:hover {{ text-decoration: underline; }}
    .hidden {{ display: none !important; }}
    /* lightbox */
    #lb {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,0.85);
           align-items:center; justify-content:center; z-index:999; }}
    #lb.active {{ display:flex; }}
    #lb img {{ max-width:90vw; max-height:90vh; border-radius:6px; }}
    #lb-close {{ position:absolute; top:16px; right:24px; font-size:28px;
                 cursor:pointer; color:#fff; }}
  </style>
</head>
<body>
  <h1>Plot Archive</h1>
  <div class="meta">
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
    Inputs: {', '.join(Path(f).name for f in input_files)}<br>
    Metadata: {Path(metadata_file).name if metadata_file else ''}
  </div>
  <input type="text" id="filter" placeholder="Filter variables..." oninput="filterGallery()" />
  <div class="grid" id="grid">{items_html}</div>

  <div id="lb" onclick="closeLb()">
    <span id="lb-close">✕</span>
    <img id="lb-img" src="" />
  </div>

  <script>
    function filterGallery() {{
      const q = document.getElementById('filter').value.toLowerCase();
      document.querySelectorAll('.thumb').forEach(el => {{
        el.classList.toggle('hidden', !el.dataset.label.toLowerCase().includes(q));
      }});
    }}
    document.querySelectorAll('.thumb img').forEach(img => {{
      img.addEventListener('click', e => {{
        document.getElementById('lb-img').src = e.target.src;
        document.getElementById('lb').classList.add('active');
      }});
    }});
    function closeLb() {{ document.getElementById('lb').classList.remove('active'); }}
    document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeLb(); }});
  </script>
</body>
</html>"""

    (archive_dir / "index.html").write_text(html)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    parser = init_arg_parser()
    parser.add_argument('--port', type=int, default=5000, help='Port to serve on (default: 5000)')
    parser.add_argument('--no-pregallery', action='store_true',
                        help='Skip pre-generating the plot gallery on startup')
    parser.add_argument('--reuse-gallery', action='store_true',
                        help='Reuse already-generated gallery plots; only regenerate missing ones')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of parallel workers for pregallery (default: 1)')
    parser.add_argument('--output-dir', default='pourover_output',
                        help='Directory for generated plots (default: pourover_output)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = _parse_args()

    output_dir    = args.output_dir
    input_files   = args.inputFile
    metadata_file = args.metadata
    reuse_gallery = args.reuse_gallery

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading histograms and config ...")
    _init_config(args)
    print(f"  Loaded {len(cfg.hists)} file(s): {input_files}")
    print(f"  Metadata: {metadata_file}")

    if not args.no_pregallery:
        _pregallery(output_dir, n_jobs=args.jobs, reuse=args.reuse_gallery)

    print(f"\nStarting web server on http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host='0.0.0.0', port=args.port, threaded=False, debug=False)
