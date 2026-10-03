#!/usr/bin/env python3
"""Generates a self-contained HTML summary dashboard for Stage F_2 Two-Stage Closure.

Displays the closure evaluation table (Variance, Bias, Spurious Signal with Fourier bases,
and overall verdict) and organizes all diagnostic plots with interactive search, tabs,
collapsible projection/diagonalized groupings (hidden by default), and an image lightbox.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create two-stage closure summary HTML dashboard.")
    parser.add_argument("--summary", "-s", required=True, help="Path to closure_summary.json")
    parser.add_argument("--output_dir", "-d", required=True, help="Base Stage F_2 output directory")
    parser.add_argument("--output", "-o", default=None, help="Output HTML path (default: <output_dir>/closure_summary.html)")
    parser.add_argument("--title", "-t", default="Two-Stage Closure Summary", help="Dashboard title")
    return parser.parse_args()


def is_hidden_plot(filename: str) -> bool:
    """Returns True if the plot belongs to high-dimensional scans or diagonalized bases."""
    fn = filename.lower()
    if "diagonalized" in fn:
        return True
    if "projection" in fn:
        return True
    if "_parameters_basis" in fn:
        return True
    return False


def categorize_plot(filename: str) -> tuple[int, str]:
    """Assigns an ordering rank and category label for visible diagnostic plots."""
    fn = filename.lower()
    if "data_vs_mixed" in fn or "comparison" in fn:
        return (0, "Key Diagnostics & Comparisons")
    if "1_bias_pvalues" in fn:
        return (0, "Key Diagnostics & Comparisons")
    if "0_variance_pearsonr" in fn:
        return (0, "Key Diagnostics & Comparisons")
    if "subsamples_" in fn:
        return (0, "Key Diagnostics & Comparisons")
    if fn.startswith("0_variance_"):
        return (1, "Stage 0 — Multijet Ensemble Fits & Residuals")
    if fn.startswith("1_basis_") or fn.startswith("1_bias_") or "bias" in fn:
        return (2, "Stage 1 — Bias Fits & Basis Vectors")
    if fn.startswith("2_spurious_signal"):
        return (3, "Stage 2 — Spurious Signal Fits")
    if "subsample" in fn or fn.startswith("mix_"):
        return (4, "Pre-Fit Subsamples & Diagnostics")
    if "normalized_basis" in fn or "additive_basis" in fn:
        return (5, "Normalized Basis Function Shapes")
    return (6, "Other Diagnostic Plots")


def categorize_hidden_plot(filename: str) -> tuple[int, str]:
    """Assigns an ordering rank and category label for hidden/collapsed parameter projections."""
    fn = filename.lower()
    if "0_variance" in fn:
        return (0, "Stage 0 — Variance Parameter Projections")
    if "1_basis" in fn or "1_bias" in fn:
        return (1, "Stage 1 — Bias Parameter Projections")
    if "2_spurious_signal" in fn:
        return (2, "Stage 2 — Spurious Signal Parameter Projections")
    if "diagonalized" in fn:
        return (3, "Diagonalized Basis Vectors")
    return (4, "Other Parameter Projections")


def collect_plots(rebin_dir: Path, base_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Scans a rebin candidate directory and separates plots into visible and hidden groups."""
    visible: dict[str, list[dict]] = {}
    hidden: dict[str, list[dict]] = {}
    if not rebin_dir.exists():
        return {"visible": visible, "hidden": hidden}

    for p in sorted(rebin_dir.rglob("*.png")):
        rel_path = p.relative_to(base_dir).as_posix()
        fn = p.name
        if is_hidden_plot(fn):
            rank, cat = categorize_hidden_plot(fn)
            item = {"name": p.stem, "filename": fn, "path": rel_path, "rank": rank, "cat": cat}
            hidden.setdefault(cat, []).append(item)
        else:
            rank, cat = categorize_plot(fn)
            item = {"name": p.stem, "filename": fn, "path": rel_path, "rank": rank, "cat": cat}
            visible.setdefault(cat, []).append(item)

    vis_order = [
        "Key Diagnostics & Comparisons",
        "Stage 0 — Multijet Ensemble Fits & Residuals",
        "Stage 1 — Bias Fits & Basis Vectors",
        "Stage 2 — Spurious Signal Fits",
        "Pre-Fit Subsamples & Diagnostics",
        "Normalized Basis Function Shapes",
        "Other Diagnostic Plots",
    ]
    hid_order = [
        "Stage 0 — Variance Parameter Projections",
        "Stage 1 — Bias Parameter Projections",
        "Stage 2 — Spurious Signal Parameter Projections",
        "Diagonalized Basis Vectors",
        "Other Parameter Projections",
    ]
    ordered_vis = {c: visible[c] for c in vis_order if c in visible and visible[c]}
    ordered_hid = {c: hidden[c] for c in hid_order if c in hidden and hidden[c]}
    return {"visible": ordered_vis, "hidden": ordered_hid}


def parse_spurious_signal_from_log(cand_dir: Path) -> tuple[bool | None, float | None]:
    """Parses candidate log file to extract spurious signal test outcome if not in status json."""
    if not cand_dir.exists():
        return None, None
    for log_path in cand_dir.glob("*.log"):
        try:
            with open(log_path, "r") as f:
                content = f.read()
            if "SS f-test" in content:
                for line in content.splitlines():
                    if "SS f-test =" in line:
                        m = re.search(r"SS f-test\s*=\s*(\d+)%", line)
                        fprob = float(m.group(1)) / 100.0 if m else None
                        if "Do not need to include spurious signal systematic" in line:
                            return True, fprob
                        elif "STRONG EVIDENCE" in line:
                            return False, fprob
        except Exception:
            pass
    return None, None


def collect_ttbar_plots(ttbar_dir: Path, base_dir: Path) -> list[dict]:
    items = []
    if not ttbar_dir.exists():
        return items
    for p in sorted(ttbar_dir.glob("*.png")):
        items.append({
            "name": p.stem,
            "filename": p.name,
            "path": p.relative_to(base_dir).as_posix(),
        })
    return items


def generate_html(summary_data: dict, out_dir: Path, title: str) -> str:
    passing_rebins = summary_data.get("passing_rebins", [])
    evaluations = summary_data.get("all_evaluations", [])

    # Extract common metadata
    channel = evaluations[0].get("channel", "ttHbb") if evaluations else "ttHbb"
    var = evaluations[0].get("var", "SvB_MA_ps_ttHbb") if evaluations else "SvB_MA_ps_ttHbb"

    # Find candidates and plots
    candidates_info = []
    for ev in evaluations:
        r_val = ev.get("rebin")
        if r_val is None:
            continue
        # Search for candidate directory under closure_fits
        pattern = f"rebin{r_val}"
        matched_dirs = list(out_dir.glob(f"closure_fits/*/*/{pattern}/SR/{channel}"))
        cand_dir = matched_dirs[0] if matched_dirs else None
        plots_data = collect_plots(cand_dir, out_dir) if cand_dir else {"visible": {}, "hidden": {}}
        vis_plots = sum(len(v) for v in plots_data["visible"].values())
        hid_plots = sum(len(v) for v in plots_data["hidden"].values())

        # Spurious signal info resolution
        ss_passed = ev.get("spurious_signal_passed")
        ss_basis = ev.get("spurious_signal_basis")
        ss_fprob = ev.get("spurious_signal_fprob")
        if ss_passed is None and cand_dir:
            parsed_pass, parsed_fprob = parse_spurious_signal_from_log(cand_dir)
            if parsed_pass is not None:
                ss_passed = parsed_pass
                ss_fprob = parsed_fprob
                ss_basis = ev.get("selected_basis")

        candidates_info.append({
            "rebin": r_val,
            "n_bins": ev.get("n_bins", "?"),
            "variance_passed": ev.get("variance_passed", False),
            "bias_passed": ev.get("bias_passed", False),
            "spurious_signal_passed": ss_passed,
            "spurious_signal_basis": ss_basis,
            "spurious_signal_fprob": ss_fprob,
            "passed": ev.get("passed", False),
            "selected_basis": ev.get("selected_basis", None),
            "multijet_basis": ev.get("multijet_basis", None),
            "failed_steps": ev.get("failed_steps", []),
            "plots": plots_data,
            "visible_plots": vis_plots,
            "hidden_plots": hid_plots,
            "total_plots": vis_plots + hid_plots,
            "dir_found": cand_dir is not None,
        })

    # TTbar comparison plots
    ttbar_dir = out_dir / "plots_ttbar_MC_vs_d3"
    ttbar_plots = collect_ttbar_plots(ttbar_dir, out_dir)
    ttbar_cutflow_html = "ttbar_MC_vs_d3_cutflow.html" if (out_dir / "ttbar_MC_vs_d3_cutflow.html").exists() else None

    # HTML building
    doc = []
    doc.append("<!DOCTYPE html>")
    doc.append("<html lang='en'>")
    doc.append("<head>")
    doc.append("  <meta charset='UTF-8'>")
    doc.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    doc.append(f"  <title>{html.escape(title)}</title>")
    doc.append("""  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --pass: #10b981;
      --pass-bg: rgba(16, 185, 129, 0.15);
      --fail: #ef4444;
      --fail-bg: rgba(239, 68, 68, 0.15);
      --accent: #38bdf8;
      --accent-bg: rgba(56, 189, 248, 0.1);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }
    .container { max-width: 1440px; margin: 0 auto; }
    header {
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    h1 { font-size: 1.8rem; margin-bottom: 8px; color: #fff; display: flex; align-items: center; gap: 12px; }
    .badge-bar { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
    .badge {
      font-size: 0.82rem;
      padding: 3px 10px;
      border-radius: 9999px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--accent);
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .badge.pass { background: var(--pass-bg); border-color: var(--pass); color: var(--pass); font-weight: 600; }
    .badge.fail { background: var(--fail-bg); border-color: var(--fail); color: var(--fail); font-weight: 600; }
    
    .basis-tag {
      display: inline-block;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.8rem;
      padding: 2px 7px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid var(--border);
      color: #e2e8f0;
      margin-left: 6px;
      vertical-align: middle;
    }
    .basis-tag.no-basis {
      color: var(--text-muted);
      border-style: dashed;
      background: transparent;
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .card-title { font-size: 1.2rem; font-weight: 600; margin-bottom: 14px; color: #fff; }
    
    table { width: 100%; border-collapse: collapse; margin-top: 10px; text-align: left; }
    th, td { padding: 12px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }
    th { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); background: rgba(0,0,0,0.15); }
    tr:hover td { background: rgba(255,255,255,0.02); }
    
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; overflow-x: auto; }
    .tab-btn {
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text-muted);
      padding: 8px 16px;
      cursor: pointer;
      font-size: 0.95rem;
      font-weight: 500;
      transition: all 0.2s ease;
      white-space: nowrap;
    }
    .tab-btn:hover { background: var(--card-bg); color: #fff; }
    .tab-btn.active {
      background: var(--accent-bg);
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 600;
    }
    
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
    
    .filter-bar {
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    .filter-input {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 14px;
      color: #fff;
      font-size: 0.9rem;
      width: 360px;
    }
    .filter-input:focus { outline: none; border-color: var(--accent); }
    
    .cat-section { margin-bottom: 28px; }
    .cat-title {
      font-size: 1.05rem;
      font-weight: 600;
      color: var(--accent);
      margin-bottom: 12px;
      padding-bottom: 6px;
      border-bottom: 1px solid rgba(56, 189, 248, 0.2);
    }
    
    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }
    .plot-card {
      background: rgba(0,0,0,0.25);
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      cursor: pointer;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .plot-card:hover {
      transform: translateY(-2px);
      border-color: var(--accent);
    }
    .plot-card img {
      width: 100%;
      height: 240px;
      object-fit: contain;
      background: #000;
      display: block;
    }
    .plot-info {
      padding: 10px;
      font-size: 0.85rem;
      font-family: monospace;
      color: var(--text-muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    
    /* Collapsible / Hidden Sections */
    details.collapsible-section {
      background: rgba(15, 23, 42, 0.6);
      border: 1px dashed var(--border);
      border-radius: 8px;
      margin-top: 24px;
      margin-bottom: 24px;
      padding: 14px 18px;
      transition: all 0.2s ease;
    }
    details.collapsible-section[open] {
      background: rgba(0, 0, 0, 0.3);
      border-style: solid;
      border-color: var(--accent);
    }
    details.collapsible-section summary {
      cursor: pointer;
      font-size: 1.05rem;
      font-weight: 600;
      color: var(--text-muted);
      list-style: none;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
    }
    details.collapsible-section summary::-webkit-details-marker {
      display: none;
    }
    details.collapsible-section summary:hover {
      color: var(--text);
    }
    details.collapsible-section summary::before {
      content: "▶";
      color: var(--accent);
      display: inline-block;
      margin-right: 10px;
      font-size: 0.85rem;
      transition: transform 0.2s ease;
    }
    details.collapsible-section[open] summary::before {
      transform: rotate(90deg);
    }
    .collapsible-hint {
      font-size: 0.82rem;
      font-weight: 400;
      color: var(--text-muted);
      background: var(--card-bg);
      padding: 3px 10px;
      border-radius: 4px;
      border: 1px solid var(--border);
    }

    /* Lightbox */
    .lightbox {
      display: none;
      position: fixed;
      z-index: 9999;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.9);
      justify-content: center;
      align-items: center;
      flex-direction: column;
    }
    .lightbox.active { display: flex; }
    .lightbox img {
      max-width: 90%;
      max-height: 85%;
      object-fit: contain;
      border: 1px solid var(--border);
      border-radius: 4px;
    }
    .lightbox-caption {
      margin-top: 12px;
      color: #fff;
      font-family: monospace;
      font-size: 1rem;
    }
    .lightbox-close {
      position: absolute;
      top: 20px;
      right: 30px;
      font-size: 2rem;
      color: #fff;
      cursor: pointer;
    }
  </style>""")
    doc.append("</head>")
    doc.append("<body>")
    doc.append("<div class='container'>")

    # Header
    doc.append("  <header>")
    doc.append(f"    <h1><span>📊</span> {html.escape(title)}</h1>")
    doc.append("    <div class='badge-bar'>")
    doc.append(f"      <span class='badge'>Channel: {html.escape(channel)}</span>")
    doc.append(f"      <span class='badge'>Variable: {html.escape(var)}</span>")
    if passing_rebins:
        doc.append(f"      <span class='badge pass'>Passing Rebins: {', '.join(str(r) for r in passing_rebins)}</span>")
    else:
        doc.append("      <span class='badge fail'>No Rebin Candidates Passed</span>")
    doc.append("    </div>")
    doc.append("  </header>")

    # Summary Table Card
    doc.append("  <div class='card'>")
    doc.append("    <div class='card-title'>Candidate Evaluation Verdicts</div>")
    doc.append("    <table>")
    doc.append("      <thead>")
    doc.append("        <tr>")
    doc.append("          <th>Rebin</th>")
    doc.append("          <th>Bins</th>")
    doc.append("          <th>Variance</th>")
    doc.append("          <th>Closure Bias</th>")
    doc.append("          <th>Spurious Signal</th>")
    doc.append("          <th>Overall Result</th>")
    doc.append("          <th>Diagnostic Plots</th>")
    doc.append("        </tr>")
    doc.append("      </thead>")
    doc.append("      <tbody>")

    for cand in candidates_info:
        r = cand["rebin"]

        # Variance column (Pass/Fail + Fourier basis)
        if cand["variance_passed"]:
            v_b = cand.get("multijet_basis")
            v_str = f"basis {v_b}" if v_b is not None else ""
            var_cell = f"<span class='badge pass'>PASS</span><span class='basis-tag'>{html.escape(v_str)}</span>"
        else:
            var_cell = "<span class='badge fail'>FAIL</span>"

        # Bias column (Pass/Fail + Fourier basis)
        if cand["bias_passed"]:
            b_b = cand.get("selected_basis")
            b_str = f"basis {b_b}" if b_b is not None else ""
            bias_cell = f"<span class='badge pass'>PASS</span><span class='basis-tag'>{html.escape(b_str)}</span>"
        else:
            bias_cell = "<span class='badge fail'>FAIL</span><span class='basis-tag no-basis'>no basis</span>"

        # Spurious Signal column (Pass/Fail + Fourier basis)
        ss_p = cand.get("spurious_signal_passed")
        ss_b = cand.get("spurious_signal_basis")
        if ss_p is True:
            ss_str = f"basis {ss_b}" if ss_b is not None else ""
            ss_cell = f"<span class='badge pass'>PASS</span><span class='basis-tag'>{html.escape(ss_str)}</span>"
        elif ss_p is False:
            ss_str = f"basis {ss_b}" if ss_b is not None else ""
            ss_cell = f"<span class='badge fail'>FAIL</span><span class='basis-tag'>{html.escape(ss_str)}</span>"
        else:
            ss_cell = "<span style='color: var(--text-muted); font-size: 0.9rem;'>—</span>"

        # Overall verdict
        overall_cell = "<span class='badge pass'>ELIGIBLE</span>" if cand["passed"] else "<span class='badge fail'>EXCLUDED</span>"

        # Plot counts
        vis_count = cand["visible_plots"]
        hid_count = cand["hidden_plots"]
        if cand["total_plots"] > 0:
            plots_cell = f"<strong>{vis_count}</strong> visible <span style='color: var(--text-muted); font-size: 0.8rem;'>({hid_count} hidden)</span>"
        else:
            plots_cell = "<span style='color: var(--text-muted);'>None found</span>"

        doc.append("        <tr>")
        doc.append(f"          <td><strong>Rebin {r}</strong></td>")
        doc.append(f"          <td>{cand['n_bins']}</td>")
        doc.append(f"          <td>{var_cell}</td>")
        doc.append(f"          <td>{bias_cell}</td>")
        doc.append(f"          <td>{ss_cell}</td>")
        doc.append(f"          <td>{overall_cell}</td>")
        doc.append(f"          <td>{plots_cell}</td>")
        doc.append("        </tr>")

    doc.append("      </tbody>")
    doc.append("    </table>")
    doc.append("  </div>")

    # Filter bar
    doc.append("  <div class='filter-bar'>")
    doc.append("    <input type='text' class='filter-input' placeholder='🔍 Filter plots by name (e.g. pvalues, logy, mix)...' onkeyup='filterPlots(this.value)'>")
    doc.append("    <span style='color: var(--text-muted); font-size: 0.85rem;'>Parameter projections and diagonalized bases are hidden by default</span>")
    doc.append("  </div>")

    # Tabs for each rebin candidate
    doc.append("  <div class='tabs'>")
    first = True
    for cand in candidates_info:
        r = cand["rebin"]
        active_cls = " active" if first else ""
        pass_indicator = "✓" if cand["passed"] else "✗"
        doc.append(f"    <button class='tab-btn{active_cls}' onclick='switchTab(\"tab-rebin-{r}\", this)'>Rebin {r} ({pass_indicator})</button>")
        first = False

    if ttbar_plots:
        doc.append("    <button class='tab-btn' onclick='switchTab(\"tab-ttbar\", this)'>TTbar MC vs Data-Driven (D3)</button>")
    doc.append("  </div>")

    # Tab panes
    first = True
    for cand in candidates_info:
        r = cand["rebin"]
        active_cls = " active" if first else ""
        doc.append(f"  <div id='tab-rebin-{r}' class='tab-pane{active_cls}'>")

        plots_data = cand["plots"]
        vis_plots = plots_data.get("visible", {})
        hid_plots = plots_data.get("hidden", {})

        if not vis_plots and not hid_plots:
            doc.append("    <div class='card'><p style='color: var(--text-muted);'>No plots found for this rebin candidate.</p></div>")
        else:
            # Render visible categories
            for cat, items in vis_plots.items():
                doc.append("    <div class='cat-section'>")
                doc.append(f"      <div class='cat-title'>{html.escape(cat)} ({len(items)})</div>")
                doc.append("      <div class='gallery-grid'>")
                for it in items:
                    doc.append(f"        <div class='plot-card' onclick='openLightbox(\"{it['path']}\", \"{html.escape(it['filename'])}\")'>")
                    doc.append(f"          <img src='{it['path']}' alt='{html.escape(it['name'])}' loading='lazy'>")
                    doc.append(f"          <div class='plot-info'>{html.escape(it['filename'])}</div>")
                    doc.append("        </div>")
                doc.append("      </div>")
                doc.append("    </div>")

            # Render hidden / collapsible group if any exist
            if hid_plots:
                total_hid = sum(len(v) for v in hid_plots.values())
                doc.append("    <details class='collapsible-section'>")
                doc.append("      <summary>")
                doc.append(f"        <span>📁 Detailed Parameter Projections & Diagonalized Basis ({total_hid} plots)</span>")
                doc.append("        <span class='collapsible-hint'>Hidden by default — click to expand</span>")
                doc.append("      </summary>")
                doc.append("      <div class='collapsible-body' style='margin-top: 16px;'>")
                for subcat, items in hid_plots.items():
                    doc.append("        <div class='cat-section'>")
                    doc.append(f"          <div class='cat-title' style='color: var(--text-muted); font-size: 0.95rem; border-color: var(--border);'>{html.escape(subcat)} ({len(items)})</div>")
                    doc.append("          <div class='gallery-grid'>")
                    for it in items:
                        doc.append(f"            <div class='plot-card' onclick='openLightbox(\"{it['path']}\", \"{html.escape(it['filename'])}\")'>")
                        doc.append(f"              <img src='{it['path']}' alt='{html.escape(it['name'])}' loading='lazy'>")
                        doc.append(f"              <div class='plot-info'>{html.escape(it['filename'])}</div>")
                        doc.append("            </div>")
                    doc.append("          </div>")
                    doc.append("        </div>")
                doc.append("      </div>")
                doc.append("    </details>")

        doc.append("  </div>")
        first = False

    # TTbar tab pane if present
    if ttbar_plots:
        doc.append("  <div id='tab-ttbar' class='tab-pane'>")
        if ttbar_cutflow_html:
            doc.append(f"    <div class='card'><a href='{ttbar_cutflow_html}' target='_blank' style='color: var(--accent); font-weight: 600;'>🔗 View Detailed TTbar MC vs D3 Cutflow Comparison Table</a></div>")
        doc.append("    <div class='gallery-grid'>")
        for it in ttbar_plots:
            doc.append(f"      <div class='plot-card' onclick='openLightbox(\"{it['path']}\", \"{html.escape(it['filename'])}\")'>")
            doc.append(f"        <img src='{it['path']}' alt='{html.escape(it['name'])}' loading='lazy'>")
            doc.append(f"        <div class='plot-info'>{html.escape(it['filename'])}</div>")
            doc.append("      </div>")
        doc.append("    </div>")
        doc.append("  </div>")

    # Lightbox Modal
    doc.append("""  <div id="lightbox" class="lightbox" onclick="closeLightbox(event)">
    <span class="lightbox-close">&times;</span>
    <img id="lightbox-img" src="" alt="Enlarged Plot">
    <div id="lightbox-caption" class="lightbox-caption"></div>
  </div>""")

    # JavaScript
    doc.append("""  <script>
    function switchTab(tabId, btn) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(tabId);
      if (target) {
        target.classList.add('active');
        const filterInput = document.querySelector('.filter-input');
        if (filterInput && filterInput.value) {
          filterPlots(filterInput.value);
        }
      }
    }

    function filterPlots(query) {
      const q = query.toLowerCase().trim();
      const activePane = document.querySelector('.tab-pane.active');
      if (!activePane) return;

      const cards = activePane.querySelectorAll('.plot-card');
      let hiddenSectionHasMatch = false;

      cards.forEach(card => {
        const text = card.querySelector('.plot-info').innerText.toLowerCase();
        const match = (q === '' || text.includes(q));
        card.style.display = match ? 'block' : 'none';
        if (match && q !== '' && card.closest('details.collapsible-section')) {
          hiddenSectionHasMatch = true;
        }
      });

      // Update cat section visibility
      activePane.querySelectorAll('.cat-section').forEach(sec => {
        if (q === '') {
          sec.style.display = 'block';
        } else {
          const anyVisible = Array.from(sec.querySelectorAll('.plot-card')).some(c => c.style.display !== 'none');
          sec.style.display = anyVisible ? 'block' : 'none';
        }
      });

      // Automatically open collapsible section if query matches inside it
      const details = activePane.querySelector('details.collapsible-section');
      if (details) {
        if (q !== '' && hiddenSectionHasMatch) {
          details.open = true;
        }
      }
    }

    function openLightbox(src, caption) {
      document.getElementById('lightbox-img').src = src;
      document.getElementById('lightbox-caption').innerText = caption;
      document.getElementById('lightbox').classList.add('active');
    }

    function closeLightbox(e) {
      if (e.target.id === 'lightbox' || e.target.classList.contains('lightbox-close')) {
        document.getElementById('lightbox').classList.remove('active');
      }
    }

    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        document.getElementById('lightbox').classList.remove('active');
      }
    });
  </script>""")

    doc.append("</div>")
    doc.append("</body>")
    doc.append("</html>")

    return "\n".join(doc)


def main():
    args = parse_args()
    summary_path = Path(args.summary)
    out_dir = Path(args.output_dir)

    if not summary_path.exists():
        print(f"Error: summary file {summary_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    html_content = generate_html(summary_data, out_dir, args.title)

    out_file = Path(args.output) if args.output else out_dir / "closure_summary.html"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(html_content)

    print(f"Successfully generated Two-Stage Closure Summary HTML: {out_file}")


if __name__ == "__main__":
    main()
