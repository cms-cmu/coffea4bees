"""Mixed-data validation report (Snakefile_MakeMixedData_6_validation.smk, step M.6).

    mixeddata_validation_report.py study   STUDY.coffea  OUTDIR  [--n-subsamples 16]

study:   plots from processor_study_mixed_data (hemisphere match distance, thrust delta-phi, jet
         multiplicity before/after mixing, selected jets raw vs mixed-JCM-weighted vs subsample v0)
         and the subsample overlap matrix (heatmap + csv + summary.yml).
(The M.6 cutflow page is the shared src/tools/cutflow_closure.py: --multijet sample4b
--multijet-process mixeddata_all --pseudodata ttbar_PSData --compare mix_v<k>.)
"""
import argparse
import csv
import html
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── study ────────────────────────────────────────────────────────────────────

def _project(h, tag="fourTag", region=None):
    """Sum a study histogram over process/year (and region unless given), fourTag only."""
    sel = {}
    for ax in h.axes:
        if ax.name == "tag":
            sel["tag"] = tag
        elif ax.name == "region":
            sel["region"] = region if region is not None else sum
        elif ax.name in ("process", "year"):
            sel[ax.name] = sum
    return h[sel]


def _step(ax, h1, label, density=False, **kw):
    vals = h1.values()
    edges = h1.axes[0].edges
    if density and vals.sum() > 0:
        vals = vals / vals.sum()
    ax.stairs(vals, edges, label=label, **kw)


def study(path, outdir, n_sub):
    from coffea.util import load
    os.makedirs(outdir, exist_ok=True)
    o = load(path)
    hists = o["hists"]
    made = []

    def save(fig, name):
        f = os.path.join(outdir, name)
        fig.savefig(f, dpi=110, bbox_inches="tight")
        plt.close(fig)
        made.append(os.path.basename(f))

    for region in ("SR", "SB"):
        for key, xlabel in (("matchDist", "hemisphere match distance"),
                            ("thrustDeltaPhi", r"$\Delta\phi$(thrust axes)")):
            if key not in hists:
                continue
            fig, ax = plt.subplots(figsize=(6, 4.5))
            _step(ax, _project(hists[key], region=region), "mixeddata_all", color="k")
            ax.set_xlabel(xlabel); ax.set_ylabel("events"); ax.set_title(f"4b mixed data, {region}")
            save(fig, f"study_{key}_{region}.png")

        if "selJets.nOld" in hists and "selJets.n" in hists:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            _step(ax, _project(hists["selJets.nOld"], region=region), "before mixing (3b event)", color="0.5")
            _step(ax, _project(hists["selJets.n"], region=region), "after mixing", color="k")
            ax.set_xlabel("selected jets"); ax.set_ylabel("events"); ax.legend(); ax.set_title(f"4b mixed data, {region}")
            save(fig, f"study_nSelJets_before_after_{region}.png")

        for var, xlabel in (("n", "selected jets"), ("pt", "selected jet $p_T$ [GeV]"), ("eta", r"selected jet $\eta$")):
            keys = [(f"selJets.{var}", "raw", "k"), (f"selJets_JCM.{var}", "mixed-JCM weighted", "#e42536"),
                    (f"selJets_v0.{var}", "subsample v0", "#1f4e9c")]
            if not all(k in hists for k, _, _ in keys):
                continue
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for k, label, color in keys:
                _step(ax, _project(hists[k], region=region), label, density=True, color=color)
            ax.set_xlabel(xlabel); ax.set_ylabel("normalised"); ax.legend(); ax.set_title(f"4b mixed data, {region} (shapes)")
            save(fig, f"study_selJets_{var}_raw_JCM_v0_{region}.png")

    # subsample overlap: sum the per-dataset upper-triangle counts
    m = np.zeros((n_sub, n_sub))
    same = {"same_hemis": 0, "same_hemis_SB": 0, "same_hemis_SR": 0}
    n_ds = 0
    for k, v in o.items():
        if isinstance(v, dict) and "subsample_counts" in v:
            n_ds += 1
            for key, c in v["subsample_counts"].items():
                i, j = map(int, key.split("_"))
                if i < n_sub and j < n_sub:
                    m[i, j] += float(c)
            for s in same:
                same[s] += int(v.get(s, 0) or 0)
    if n_ds == 0:
        raise ValueError(f"{path}: no subsample_counts found (not a processor_study_mixed_data output?)")
    full = m + np.triu(m, 1).T
    sizes = np.diag(full).copy()
    frac = np.divide(full, sizes[:, None], out=np.zeros_like(full), where=sizes[:, None] > 0)
    np.fill_diagonal(frac, np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    im = ax.imshow(frac * 100, cmap="viridis", origin="lower")
    ax.set_xticks(range(n_sub)); ax.set_yticks(range(n_sub))
    ax.set_xlabel("subsample j"); ax.set_ylabel("subsample i")
    ax.set_title("events of subsample i also in subsample j [%]")
    fig.colorbar(im, ax=ax)
    save(fig, "subsample_overlap_matrix.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(n_sub), sizes / 1e6, color="#FFDF7F", edgecolor="k")
    ax.set_xlabel("subsample"); ax.set_ylabel("4b events [M]"); ax.set_title("subsample sizes")
    save(fig, "subsample_sizes.png")

    with open(os.path.join(outdir, "subsample_overlap_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i\\j"] + list(range(n_sub)))
        for i in range(n_sub):
            w.writerow([i] + [int(x) for x in full[i]])
    shared = np.nansum(np.where(np.isnan(frac), 0, frac), axis=1)
    summary = {
        "n_subsamples": n_sub,
        "subsample_events": [int(x) for x in sizes],
        "shared_with_others_sum_of_pairwise_fraction": [round(float(x), 4) for x in shared],
        "max_pairwise_fraction": round(float(np.nanmax(frac)), 4),
        "mean_pairwise_fraction": round(float(np.nanmean(frac)), 4),
        **same,
    }
    with open(os.path.join(outdir, "summary.yml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    made += ["subsample_overlap_matrix.csv", "summary.yml"]
    _study_index(outdir, made, summary)
    print("wrote", len(made) + 1, "files to", outdir)


STUDY_SECTIONS = [
    ("Subsample overlap", "subsample_", "mixeddata_4b pseudo-experiments: the fraction of subsample i's "
     "events also in subsample j (off-diagonal), and each subsample's size. Disjoint subsamples need "
     "N·w ≤ 1 for the mixed-data JCM weight w; events above that wrap to slice (event+v) mod floor(1/w) and are shared."),
    ("Hemisphere matching", "study_matchDist", "distance between each 3b event's hemisphere and the "
     "library hemisphere it was swapped for."),
    ("Thrust axes", "study_thrustDeltaPhi", "Δφ between the thrust axes of the two mixed-in hemispheres."),
    ("Jet multiplicity before / after mixing", "study_nSelJets", "selected jets in the original 3b "
     "event vs in the mixed event."),
    ("Selected jets: raw vs mixed-JCM weighted vs subsample v0", "study_selJets_", "normalised shapes."),
]


def _study_index(outdir, files, summary):
    """One self-contained page for the study directory: the numbers, then every plot by topic."""
    pngs = sorted(f for f in files if f.endswith(".png"))
    sizes = summary["subsample_events"]
    shared = summary["shared_with_others_sum_of_pairwise_fraction"]
    rows = "".join(f"<tr><td>v{i}</td><td>{n:,}</td><td>{100 * s:.1f}%</td></tr>"
                   for i, (n, s) in enumerate(zip(sizes, shared)))
    facts = [("subsamples", summary["n_subsamples"]),
             ("largest pairwise overlap", f"{100 * summary['max_pairwise_fraction']:.1f}%"),
             ("mean pairwise overlap", f"{100 * summary['mean_pairwise_fraction']:.2f}%"),
             ("same-event hemisphere pairs (SR / SB / all)",
              f"{summary['same_hemis_SR']:,} / {summary['same_hemis_SB']:,} / {summary['same_hemis']:,}")]
    fact_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in facts)
    used, sections = set(), []
    for title, prefix, text in STUDY_SECTIONS:
        imgs = [p for p in pngs if p.startswith(prefix) and p not in used]
        used.update(imgs)
        if imgs:
            figs = "".join(f"<figure><a href='{p}'><img src='{p}' loading='lazy'></a>"
                           f"<figcaption>{html.escape(p[:-4])}</figcaption></figure>" for p in imgs)
            sections.append(f"<h2>{html.escape(title)}</h2><p>{html.escape(text)}</p><div class='grid'>{figs}</div>")
    rest = [p for p in pngs if p not in used]
    if rest:
        figs = "".join(f"<figure><a href='{p}'><img src='{p}' loading='lazy'></a>"
                       f"<figcaption>{html.escape(p[:-4])}</figcaption></figure>" for p in rest)
        sections.append(f"<h2>Other</h2><div class='grid'>{figs}</div>")
    style = ("<style>body{font-family:sans-serif;margin:1.5em;max-width:1400px}"
             "table{border-collapse:collapse;margin:0.5em 2em 1em 0;display:inline-table;vertical-align:top}"
             "td,th{border:1px solid #ccc;padding:2px 10px;text-align:right}th{background:#eee;text-align:left}"
             ".grid{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:420px}"
             "figure img{width:100%;border:1px solid #ddd}figcaption{font-size:12px;color:#555}</style>")
    page = (f"<html><head><meta charset='utf-8'><title>mixed-data study</title>{style}</head><body>"
            f"<h1>Mixed-data study (processor_study_mixed_data)</h1>"
            f"<p>Four-tag mixed data. Files: <a href='summary.yml'>summary.yml</a> · "
            f"<a href='subsample_overlap_matrix.csv'>subsample_overlap_matrix.csv</a> "
            f"(events in both subsample i and j; diagonal = subsample size)</p>"
            f"<table>{fact_rows}</table>"
            f"<table><tr><th>subsample</th><th>4b events</th><th>shared with others</th></tr>{rows}</table>"
            f"{''.join(sections)}</body></html>")
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["study"])
    ap.add_argument("input")
    ap.add_argument("outdir")
    ap.add_argument("--n-subsamples", type=int, default=16)
    a = ap.parse_args()
    study(a.input, a.outdir, a.n_subsamples)


if __name__ == "__main__":
    main()
