#!/usr/bin/env python3
"""Generates the HTML summary page for Stage F_2 Two-Stage Closure.

The page is barista's plot gallery (src/plotting/make_gallery.py: filter, size slider, by group /
by variable views, lightbox), the same page as every makePlots gallery, with the closure verdict
table (Variance, Bias, Spurious Signal with Fourier bases, overall) on top. Plots are grouped per
rebin candidate and category; parameter projections and diagonalized bases start collapsed; the
key diagnostics of every candidate are pinned side by side. Click a table row to show one rebin.
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


def collect_candidates(summary_data: dict, out_dir: Path) -> dict:
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

    return {"channel": channel, "var": var, "passing_rebins": passing_rebins,
            "candidates": candidates_info, "ttbar_plots": ttbar_plots,
            "ttbar_cutflow_html": ttbar_cutflow_html}


KEY_PLOTS = ["data_vs_mixed_vs_bkg", "1_bias_pvalues", "0_variance_pearsonr_multijet_variance"]


def _chip(passed, basis=None, none_text="—"):
    if passed is None:
        return f"<span class='na'>{none_text}</span>"
    b = "" if basis is None or basis == -1 else f"<span class='basis'>basis {basis}</span>"
    return f"<span class='chip {'pass' if passed else 'fail'}'>{'PASS' if passed else 'FAIL'}</span>{b}"


def gallery_items(info: dict) -> list[dict]:
    """The plots as make_gallery items: one group per (rebin, category), parameter projections and
    diagonalized bases in their own groups (collapsed by default), key diagnostics pinned."""
    items = []
    for cand in info["candidates"]:
        tag = f"rebin{int(cand['rebin']):02d}: {'PASS' if cand['passed'] else 'FAIL'}"
        for kind, groups in (("", cand["plots"]["visible"]), (" · projections", cand["plots"]["hidden"])):
            for cat, plots in groups.items():
                for it in plots:
                    item = {"path": it["path"], "group": f"{tag} · {cat}{kind}", "name": it["name"], "kind": "img"}
                    if it["name"] in KEY_PLOTS:
                        item["summary"] = KEY_PLOTS.index(it["name"])
                    items.append(item)
    for it in info["ttbar_plots"]:
        items.append({"path": it["path"], "group": "TTbar MC vs data-driven (d3)", "name": it["name"], "kind": "img"})
    return items


def verdict_section(info: dict) -> str:
    rows = []
    for c in info["candidates"]:
        rows.append(
            f"<tr data-rebin='rebin{int(c['rebin']):02d}:' title='show only rebin {c['rebin']}'>"
            f"<td><b>{c['rebin']}</b></td><td>{c['n_bins']}</td>"
            f"<td>{_chip(c['variance_passed'], c.get('multijet_basis'))}</td>"
            f"<td>{_chip(c['bias_passed'], c.get('selected_basis') if c['bias_passed'] else None)}</td>"
            f"<td>{_chip(c.get('spurious_signal_passed'), c.get('spurious_signal_basis'))}</td>"
            f"<td><span class='chip {'pass' if c['passed'] else 'fail'}'>{'ELIGIBLE' if c['passed'] else 'EXCLUDED'}</span></td>"
            f"<td class='num'>{c['visible_plots']} <span class='na'>+ {c['hidden_plots']} projections</span></td></tr>")
    passing = info["passing_rebins"]
    meta = (f"channel <b>{html.escape(str(info['channel']))}</b> &nbsp;·&nbsp; variable <b>{html.escape(str(info['var']))}</b>"
            f" &nbsp;·&nbsp; passing rebins: " + (f"<b class='ok'>{', '.join(str(r) for r in passing)}</b>" if passing else "<b class='bad'>none</b>"))
    if info["ttbar_cutflow_html"]:
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
section.verdict { margin:12px 16px 0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
section.verdict .vmeta { color:var(--muted); margin-bottom:8px; }
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
section.verdict .na { color:var(--muted); }
section.verdict .vnote { color:var(--muted); font-size:12px; margin-top:6px; }
"""

VERDICT_JS = """<script>
document.querySelectorAll('section.verdict tbody tr').forEach(tr => tr.onclick = () => {
  const on = tr.classList.contains('sel');
  document.querySelectorAll('section.verdict tbody tr').forEach(r => r.classList.remove('sel'));
  q.value = on ? '' : tr.dataset.rebin + ' '; if(!on) tr.classList.add('sel'); render();
});
</script>
"""


def generate_html(summary_data: dict, out_dir: Path, title: str) -> str:
    """The page of barista's plot galleries (src/plotting/make_gallery.py) with the verdict table on top."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # barista root
    from src.plotting.make_gallery import PAGE

    info = collect_candidates(summary_data, out_dir)
    page = (PAGE.replace("__TITLE__", html.escape(title))
                .replace("__ITEMS__", json.dumps(gallery_items(info), separators=(",", ":"))))
    hooks = {
        "</style>": VERDICT_CSS + "</style>",
        '<main id="main"></main>': verdict_section(info) + '<main id="main"></main>',
        "</body>": VERDICT_JS + "</body>",
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
