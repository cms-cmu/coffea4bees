#!/usr/bin/env python3
"""Generates the HTML summary page for Stage F_2 Two-Stage Closure.

The page is barista's plot gallery (src/plotting/make_gallery.py: filter, size slider, by group /
by variable views, lightbox), the same page as every makePlots gallery, with the closure verdict
table (Variance, Bias, Spurious Signal with Fourier bases, overall) on top.

Supports interactive flipping between Signal Region (SR, default) and Sideband (SB) within a single
dashboard per classifier (e.g. summary_SvB_MA.html, summary_SvB.html).
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
    parser.add_argument("--summary", "-s", default=None, help="Path to closure_summary.json (single region mode)")
    parser.add_argument("--summary-sr", default=None, help="Path to SR closure_summary.json")
    parser.add_argument("--summary-sb", default=None, help="Path to SB closure_summary.json")
    parser.add_argument("--output_dir", "-d", required=True, help="Base Stage F_2 output directory")
    parser.add_argument("--output", "-o", default=None, help="Output HTML path")
    parser.add_argument("--region", "-r", default="SR", help="Closure region for single region mode (default: SR)")
    parser.add_argument("--classifier", "-c", default=None, help="Classifier name (e.g. SvB_MA or SvB)")
    parser.add_argument("--channel", default="", help="Analysis channel (e.g. ttHbb or HH4b)")
    parser.add_argument("--title", "-t", default=None, help="Dashboard title")
    return parser.parse_args()


def is_hidden_plot(filename: str) -> bool:
    """Returns True if the plot belongs to high-dimensional scans, diagonalized bases, or basis shapes."""
    fn = filename.lower()
    if "diagonalized" in fn:
        return True
    if "projection" in fn:
        return True
    if "_parameters_basis" in fn:
        return True
    if "normalized_basis" in fn or "additive_basis" in fn:
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
    return (5, "Other Diagnostic Plots")


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
    if "normalized_basis" in fn or "additive_basis" in fn:
        return (4, "Normalized & Additive Basis Shapes")
    return (5, "Other Parameter Projections")


def collect_plots(rebin_dir: Path, base_dir: Path, region: str = "SR") -> dict[str, dict[str, list[dict]]]:
    """Scans a rebin candidate directory and separates plots into visible and hidden groups."""
    visible: dict[str, list[dict]] = {}
    hidden: dict[str, list[dict]] = {}
    if not rebin_dir.exists():
        return {"visible": visible, "hidden": hidden}

    for p in sorted(rebin_dir.rglob("*.png")):
        fn = p.name
        fn_l = fn.lower()
        # Region separation: exclude SB comparison plots from SR and vice-versa
        if region == "SR" and ("_sb" in fn_l):
            continue
        if region == "SB" and ("_sr" in fn_l or fn_l == "data_vs_mixed_vs_bkg.png"):
            continue

        rel_path = p.relative_to(base_dir).as_posix()
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


def collect_candidates(summary_data: dict, out_dir: Path, region: str = "SR", clf: str = None) -> dict:
    if not summary_data:
        return {"channel": "ttHbb", "var": clf or "SvB_MA", "region": region, "passing_rebins": [],
                "candidates": [], "ttbar_plots": [], "ttbar_cutflow_html": None}

    passing_rebins = summary_data.get("passing_rebins", [])
    evaluations = summary_data.get("all_evaluations", [])

    # Extract common metadata from first valid evaluation
    valid_evs = [ev for ev in evaluations if ev.get("var") or ev.get("channel")]
    channel = valid_evs[0].get("channel", "ttHbb") if valid_evs else "ttHbb"
    var = valid_evs[0].get("var", f"{clf}_ps_ttHbb") if valid_evs else (f"{clf}_ps_ttHbb" if clf else "SvB_MA_ps_ttHbb")
    if not clf:
        clf = "SvB_MA" if "SvB_MA" in var else ("SvB" if "SvB" in var else "*")

    candidates_info = []
    for ev in evaluations:
        r_val = ev.get("rebin")
        if r_val is None:
            continue
        pattern = f"rebin{r_val}"
        matched_dirs = list(out_dir.glob(f"closure_fits/*/{clf}/{pattern}/{region}/{channel}"))
        cand_dir = None
        for d in matched_dirs:
            if any(d.glob("*.png")):
                cand_dir = d
                break
        if cand_dir is None and matched_dirs:
            cand_dir = matched_dirs[0]
        plots_data = collect_plots(cand_dir, out_dir, region=region) if cand_dir else {"visible": {}, "hidden": {}}
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

        # Highest Fourier mode / basis p-values for closure bias test
        highest_b = ev.get("highest_basis")
        highest_p = ev.get("highest_basis_pvalue")
        all_pvals = ev.get("all_bias_pvalues", {})
        if (highest_p is None or not all_pvals) and cand_dir:
            bias_yml = cand_dir / "1_bias_results.yml"
            if bias_yml.exists():
                try:
                    import yaml
                    with open(bias_yml, "r") as byf:
                        bias_res = yaml.safe_load(byf) or {}
                    int_bases = sorted([int(k) for k in bias_res.keys() if str(k).lstrip('-').isdigit()])
                    if int_bases:
                        hb = int_bases[-1]
                        hb_info = bias_res.get(hb) or bias_res.get(str(hb))
                        if isinstance(hb_info, dict) and "pvalue" in hb_info:
                            highest_b = hb
                            highest_p = float(hb_info["pvalue"])
                        all_pvals = {b: float(bias_res.get(b, {}).get("pvalue", float("nan"))) for b in int_bases if isinstance(bias_res.get(b), dict)}
                except Exception:
                    pass

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
            "highest_basis": highest_b,
            "highest_basis_pvalue": highest_p,
            "all_bias_pvalues": all_pvals,
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

    return {"channel": channel, "var": var, "region": region, "passing_rebins": passing_rebins,
            "candidates": candidates_info, "ttbar_plots": ttbar_plots,
            "ttbar_cutflow_html": ttbar_cutflow_html}


def _chip(passed, basis=None, none_text="—", fail_extra=None, title=None):
    if passed is None:
        return f"<span class='na'>{none_text}</span>"
    if passed:
        b = "" if basis is None or basis == -1 else f"<span class='basis'>basis {basis}</span>"
        return f"<span class='chip pass'>PASS</span>{b}"
    t_attr = f" title='{html.escape(str(title))}'" if title else ""
    b = f"<span class='basis fail_info'{t_attr}>{html.escape(str(fail_extra))}</span>" if fail_extra else ""
    return f"<span class='chip fail'>FAIL</span>{b}"


def gallery_items(info_map: dict[str, dict]) -> list[dict]:
    """Generates gallery items with 'region' metadata for both SR and SB."""
    items = []
    seen_paths = set()

    for reg in ["SR", "SB"]:
        if reg not in info_map:
            continue
        info = info_map[reg]
        for cand in info["candidates"]:
            tag = f"rebin{int(cand['rebin']):02d}: {'PASS' if cand['passed'] else 'FAIL'}"
            for kind, groups in (("", cand["plots"]["visible"]), (" · projections", cand["plots"]["hidden"])):
                for cat, plots in groups.items():
                    for it in plots:
                        p_str = it["path"]
                        item = {
                            "path": p_str,
                            "group": f"{tag} · {cat}{kind}",
                            "name": it["name"],
                            "kind": "img",
                            "region": reg,
                        }
                        # Priority for summary pins
                        if reg == "SR":
                            if it["name"] in ("data_vs_mixed_vs_bkg_SR", "data_vs_mixed_vs_bkg"):
                                item["summary"] = 0
                            elif it["name"] == "1_bias_pvalues":
                                item["summary"] = 1
                            elif it["name"] == "0_variance_pearsonr_multijet_variance":
                                item["summary"] = 2
                        elif reg == "SB":
                            if it["name"] == "data_vs_mixed_vs_bkg_SB":
                                item["summary"] = 0
                            elif it["name"] == "1_bias_pvalues":
                                item["summary"] = 1
                            elif it["name"] == "0_variance_pearsonr_multijet_variance":
                                item["summary"] = 2

                        items.append(item)
                        seen_paths.add(p_str)

    # Common TTbar comparison plots (shown in both regions)
    for reg in ["SR", "SB"]:
        if reg in info_map:
            for it in info_map[reg].get("ttbar_plots", []):
                p_str = it["path"]
                if p_str not in seen_paths:
                    items.append({
                        "path": p_str,
                        "group": "TTbar MC vs data-driven (d3)",
                        "name": it["name"],
                        "kind": "img",
                        "region": "ALL",
                    })
                    seen_paths.add(p_str)
            break

    return items


def verdict_section(info: dict) -> str:
    rows = []
    for c in info["candidates"]:
        bias_fail_str = None
        bias_fail_title = None
        if not c['bias_passed']:
            hb = c.get('highest_basis')
            hp = c.get('highest_basis_pvalue')
            all_p = c.get('all_bias_pvalues', {})
            title_parts = [f"b{b}: {p*100:.1f}%" for b, p in sorted(all_p.items()) if not (isinstance(p, float) and p != p)]
            if title_parts:
                bias_fail_title = f"Basis p-values: {', '.join(title_parts)}"
            if hb is not None and hp is not None:
                bias_fail_str = f"max b{hb}: p={hp*100:.1f}%"

        rows.append(
            f"<tr data-rebin='rebin{int(c['rebin']):02d}:' title='show only rebin {c['rebin']}'>"
            f"<td><b>{c['rebin']}</b></td><td>{c['n_bins']}</td>"
            f"<td>{_chip(c['variance_passed'], c.get('multijet_basis'))}</td>"
            f"<td>{_chip(c['bias_passed'], c.get('selected_basis') if c['bias_passed'] else None, fail_extra=bias_fail_str, title=bias_fail_title)}</td>"
            f"<td>{_chip(c.get('spurious_signal_passed'), c.get('spurious_signal_basis'))}</td>"
            f"<td><span class='chip {'pass' if c['passed'] else 'fail'}'>{'ELIGIBLE' if c['passed'] else 'EXCLUDED'}</span></td>"
            f"<td class='num'>{c['visible_plots']} <span class='na'>+ {c['hidden_plots']} projections</span></td></tr>")

    passing = info["passing_rebins"]
    region_display = info.get("region", "SR")
    meta = (f"channel <b>{html.escape(str(info['channel']))}</b> &nbsp;·&nbsp; region <b>{html.escape(str(region_display))}</b> &nbsp;·&nbsp; variable <b>{html.escape(str(info['var']))}</b>"
            f" &nbsp;·&nbsp; passing rebins: " + (f"<b class='ok'>{', '.join(str(r) for r in passing)}</b>" if passing else "<b class='bad'>none</b>"))
    if info.get("ttbar_cutflow_html"):
        meta += f" &nbsp;·&nbsp; <a href='{info['ttbar_cutflow_html']}' target='_blank'>ttbar MC vs d3 cutflow</a>"

    return f"""<section class="verdict">
  <div class="vmeta">{meta}</div>
  <table>
    <thead><tr><th>rebin</th><th>bins</th><th>variance</th><th>closure bias</th><th>spurious signal</th><th>overall</th><th>plots</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <div class="vnote">click a row to show only that rebin; parameter projections and diagonalized bases are collapsed (click a heading to open)</div>
</section>
"""


VERDICT_CSS = """
.region-toggle-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 12px 16px 0;
  padding: 8px 14px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.rtoggle-group {
  display: flex;
  gap: 8px;
}
.rtab-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 14px;
  font-size: 13px;
  font-weight: 600;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: var(--bg);
  color: var(--muted);
  cursor: pointer;
  transition: all 0.15s ease;
}
.rtab-btn:hover {
  background: #ffffff;
  color: var(--fg);
  border-color: #cbd5e1;
}
.rtab-btn.active {
  background: #2563eb;
  color: #ffffff;
  border-color: #2563eb;
  box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
}
.rtab-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}
.rtoggle-hint {
  font-size: 12px;
  color: var(--muted);
}
section.verdict { margin:8px 16px 0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
section.verdict .vmeta { color:var(--muted); margin-bottom:8px; font-size:12.5px; }
section.verdict .vmeta b { color:var(--fg); font-weight:500; }
section.verdict .vmeta b.ok { color:#1f8a4c; } section.verdict .vmeta b.bad { color:#c8372d; }
section.verdict .vmeta a, section.verdict .vnote a { color:var(--accent); }
section.verdict table { border-collapse:collapse; width:100%; font-size:13px; }
section.verdict th { text-align:left; font-weight:500; color:var(--muted); border-bottom:1px solid var(--line); padding:5px 8px; }
section.verdict td { border-bottom:1px solid var(--line); padding:5px 8px; white-space:nowrap; }
section.verdict tbody tr { cursor:pointer; } section.verdict tbody tr:hover td { background:var(--bg); }
section.verdict tbody tr.sel td { background:#e8f0fd; }
section.verdict .chip { display:inline-block; min-width:42px; text-align:center; padding:1px 7px; border-radius:4px; font-size:12px; font-weight:600; }
section.verdict .chip.pass { background:#e3f4ea; color:#1f8a4c; } section.verdict .chip.fail { background:#fbe6e4; color:#c8372d; }
section.verdict .basis { color:var(--muted); margin-left:6px; font-size:12px; }
section.verdict .basis.fail_info { color:#c8372d; margin-left:6px; font-size:12px; font-weight:500; cursor:help; }
section.verdict .na { color:var(--muted); }
section.verdict .vnote { color:var(--muted); font-size:12px; margin-top:6px; }
"""

REGION_FILTER_JS = """window.currentRegion = 'SR';

window.switchRegion = function(reg) {
  window.currentRegion = reg;
  document.querySelectorAll('.rtab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.region === reg);
  });
  document.querySelectorAll('.verdict-panel').forEach(p => {
    p.style.display = (p.id === 'verdict-' + reg) ? 'block' : 'none';
  });
  document.querySelectorAll('section.verdict tbody tr').forEach(r => r.classList.remove('sel'));
  q.value = '';
  render();
};

function filtered(){
  const terms = q.value.toLowerCase().split(/\\s+/).filter(Boolean);
  return ITEMS.filter(it => {
    if (it.region && it.region !== 'ALL' && it.region !== window.currentRegion) return false;
    const hay = (it.group+'/'+it.name).toLowerCase();
    return terms.every(t => hay.includes(t));
  });
}
function _orig_filtered_unused(){
"""

VERDICT_ROW_CLICK_JS = """render();
document.querySelectorAll('section.verdict tbody tr').forEach(tr => {
  tr.onclick = function() {
    const on = this.classList.contains('sel');
    document.querySelectorAll('section.verdict tbody tr').forEach(r => r.classList.remove('sel'));
    q.value = on ? '' : this.dataset.rebin + ' ';
    if (!on) this.classList.add('sel');
    render();
  };
});
</script>"""


def generate_html(info_map: dict[str, dict], out_dir: Path, title: str) -> str:
    """The page of barista's plot galleries with interactive SR/SB toggle and verdict tables on top."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # barista root
    from src.plotting.make_gallery import PAGE

    has_sr = "SR" in info_map and info_map["SR"].get("candidates")
    has_sb = "SB" in info_map and info_map["SB"].get("candidates")

    # Build Verdict HTML block
    verdict_blocks = []
    if has_sr and has_sb:
        toggle_bar = """
        <div class="region-toggle-bar">
          <div class="rtoggle-group">
            <button type="button" class="rtab-btn active" id="tabSR" data-region="SR" onclick="switchRegion('SR')">
              <span class="rtab-dot"></span> Signal Region (SR)
            </button>
            <button type="button" class="rtab-btn" id="tabSB" data-region="SB" onclick="switchRegion('SB')">
              <span class="rtab-dot"></span> Sideband (SB)
            </button>
          </div>
          <div class="rtoggle-hint">Toggle between Signal Region (SR) and Sideband (SB) closure fits & diagnostics</div>
        </div>
        """
        verdict_blocks.append(toggle_bar)
        verdict_blocks.append(f'<div id="verdict-SR" class="verdict-panel">{verdict_section(info_map["SR"])}</div>')
        verdict_blocks.append(f'<div id="verdict-SB" class="verdict-panel" style="display:none;">{verdict_section(info_map["SB"])}</div>')
    elif has_sr:
        verdict_blocks.append(f'<div id="verdict-SR" class="verdict-panel">{verdict_section(info_map["SR"])}</div>')
    elif has_sb:
        verdict_blocks.append(f'<div id="verdict-SB" class="verdict-panel">{verdict_section(info_map["SB"])}</div>')
    else:
        # Fallback if candidates empty
        first_info = next(iter(info_map.values()), {})
        verdict_blocks.append(f'<div class="verdict-panel">{verdict_section(first_info)}</div>')

    full_verdict_html = "\n".join(verdict_blocks)
    items_json = json.dumps(gallery_items(info_map), separators=(",", ":"))

    page = (PAGE.replace("__TITLE__", html.escape(title))
                .replace("__ITEMS__", items_json))

    hooks = {
        "</style>": VERDICT_CSS + "</style>",
        '<main id="main"></main>': full_verdict_html + '<main id="main"></main>',
        "function filtered(){": REGION_FILTER_JS,
        "render();\n</script>": VERDICT_ROW_CLICK_JS,
        # projection groups start collapsed
        "if(!$('#expandAll').checked && !cls) s.classList.add('closed');":
            "if((!$('#expandAll').checked || title.endsWith(' · projections')) && !cls) s.classList.add('closed');",
    }
    for old, new in hooks.items():
        if old not in page:
            raise RuntimeError(f"make_gallery.PAGE changed: {old!r} not found")
        page = page.replace(old, new, 1)
    return page


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)

    info_map: dict[str, dict] = {}
    clf = args.classifier

    # Mode 1: Explicit --summary-sr and/or --summary-sb
    if args.summary_sr or args.summary_sb:
        if args.summary_sr and Path(args.summary_sr).exists():
            with open(args.summary_sr, "r") as f:
                data_sr = json.load(f)
            info_map["SR"] = collect_candidates(data_sr, out_dir, region="SR", clf=clf)
        if args.summary_sb and Path(args.summary_sb).exists():
            with open(args.summary_sb, "r") as f:
                data_sb = json.load(f)
            info_map["SB"] = collect_candidates(data_sb, out_dir, region="SB", clf=clf)

    # Mode 2: Auto-detect by classifier name under output_dir
    elif clf:
        sr_file = out_dir / f"summary_{clf}_SR.json"
        sb_file = out_dir / f"summary_{clf}_SB.json"
        if sr_file.exists():
            with open(sr_file, "r") as f:
                data_sr = json.load(f)
            info_map["SR"] = collect_candidates(data_sr, out_dir, region="SR", clf=clf)
        if sb_file.exists():
            with open(sb_file, "r") as f:
                data_sb = json.load(f)
            info_map["SB"] = collect_candidates(data_sb, out_dir, region="SB", clf=clf)

    # Mode 3: Legacy single --summary path
    if not info_map and args.summary:
        summary_path = Path(args.summary)
        if summary_path.exists():
            with open(summary_path, "r") as f:
                data = json.load(f)
            reg = args.region or "SR"
            info_map[reg] = collect_candidates(data, out_dir, region=reg, clf=clf)

    if not info_map:
        print(f"Warning: No valid summary JSON files found for classifier {clf} in {out_dir}; generating placeholder page.", file=sys.stderr)
        out_file = Path(args.output) if args.output else out_dir / (f"summary_{clf}.html" if clf else "closure_summary.html")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        clf_str = f"{args.channel} - {clf}" if args.channel else (clf or 'Summary')
        title = args.title or f"Stage F_2 Two-Stage Closure ({clf_str})"
        placeholder = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; margin: 0; }}
.card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; max-width: 800px; margin: 2rem auto; padding: 2rem; }}
h2 {{ color: #94a3b8; margin-top: 0; }}
p {{ color: #cbd5e1; line-height: 1.5; }}
</style>
</head>
<body>
<div class="card">
    <h2>Two-Stage Closure: {clf or 'Classifier'} (Not Available)</h2>
    <p>Standalone {clf or 'classifier'} was not evaluated for this analysis roast or has no closure data available. Only the primary classifier is available.</p>
</div>
</body>
</html>"""
        with open(out_file, "w") as f:
            f.write(placeholder)
        print(f"Wrote placeholder closure summary: {out_file}")
        return

    clf_str = f"{args.channel} - {clf}" if args.channel else (clf or 'Summary')
    title = args.title or f"Stage F_2 Two-Stage Closure ({clf_str})"
    html_content = generate_html(info_map, out_dir, title)

    out_file = Path(args.output) if args.output else out_dir / (f"summary_{clf}.html" if clf else "closure_summary.html")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(html_content)

    print(f"Successfully generated Two-Stage Closure Summary HTML: {out_file}")


if __name__ == "__main__":
    main()
