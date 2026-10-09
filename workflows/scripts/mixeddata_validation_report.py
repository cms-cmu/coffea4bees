"""Mixed-data validation report (Snakefile_MakeMixedData_6_validation.smk, step M.6).

    mixeddata_validation_report.py study   STUDY.coffea  OUTDIR  [--n-subsamples 16]
    mixeddata_validation_report.py seeds   REG_v0.yml ... REG_vN-1.yml --outdir OUTDIR [--years ...]

study:   plots from processor_study_mixed_data (hemisphere match distance, thrust delta-phi, jet
         multiplicity before/after mixing, selected jets raw vs mixed-JCM-weighted vs subsample v0)
         and the subsample overlap matrix (heatmap + csv + summary.yml).
seeds:   4b mixing (mixing.source: fourTag). Reads the per-seed mixed picoAODs listed in M.2's per-seed
         registries: seed-overlap matrix (events of seed i with the SAME two replacement hemispheres
         in seed j; and per hemisphere), sizes, chosen-rank and match-distance distributions, and
         the self-match check (a replacement hemisphere from the event's own library entry -- the
         source-event veto must make this 0; the report raises otherwise).
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


# ── seeds (4b mixing) ────────────────────────────────────────────────────────

_SEED_BRANCHES = ["event", "run", "luminosityBlock",
                  "posHemiNew_event", "posHemiNew_run", "posHemiNew_luminosityBlock",
                  "negHemiNew_event", "negHemiNew_run", "negHemiNew_luminosityBlock",
                  "posHemiNew_match_rank", "negHemiNew_match_rank",
                  "posHemiNew_match_dist", "negHemiNew_match_dist"]
_GOLD = np.uint64(0x9E3779B97F4A7C15)


def _event_key(event, run, lumi):
    """One uint64 per (run, lumi, event): (run << 32 | lumi) mixed with the event number."""
    rl = (np.asarray(run).astype(np.uint64) << np.uint64(32)) | np.asarray(lumi).astype(np.uint32).astype(np.uint64)
    with np.errstate(over="ignore"):
        return (rl * _GOLD) ^ np.asarray(event).astype(np.int64).view(np.uint64)


def _read_seed(registry, years):
    import uproot
    from src.tools.make_dataset_yml import parse_dataset_key
    with open(registry) as f:
        reg = yaml.safe_load(f) or {}
    files = []
    for key, ds in reg.items():
        year, _ = parse_dataset_key(key)
        if year in years:
            files += (ds or {}).get("files") or []
    if not files:
        raise ValueError(f"{registry}: no files for {years}")
    cols = {b: [] for b in _SEED_BRANCHES}
    for fp in sorted(files):
        with uproot.open(fp) as f:
            arr = f["Events"].arrays(_SEED_BRANCHES, library="np")
        for b in _SEED_BRANCHES:
            cols[b].append(arr[b])
    c = {b: np.concatenate(v) for b, v in cols.items()}
    src = _event_key(c["event"], c["run"], c["luminosityBlock"])
    pos = _event_key(c["posHemiNew_event"], c["posHemiNew_run"], c["posHemiNew_luminosityBlock"])
    neg = _event_key(c["negHemiNew_event"], c["negHemiNew_run"], c["negHemiNew_luminosityBlock"])
    order = np.argsort(src, kind="stable")
    return {"src": src[order], "pos": pos[order], "neg": neg[order],
            "self_match": int(np.sum((pos == src) | (neg == src))),
            "rank": np.concatenate([c["posHemiNew_match_rank"], c["negHemiNew_match_rank"]]).astype(int),
            "dist": np.concatenate([c["posHemiNew_match_dist"], c["negHemiNew_match_dist"]]).astype(float),
            "n_files": len(files)}


def seeds(registries, outdir, years):
    os.makedirs(outdir, exist_ok=True)
    made = []

    def save(fig, name):
        fig.savefig(os.path.join(outdir, name), dpi=110, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    data = [_read_seed(r, years) for r in registries]
    n = len(data)
    sizes = np.array([len(d["src"]) for d in data])
    pair = np.full((n, n), np.nan)      # same (pos, neg) replacement: an identical mixed event
    hemi = np.full((n, n), np.nan)      # per hemisphere: same library hemisphere on that side
    common = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            _, ii, jj = np.intersect1d(data[i]["src"], data[j]["src"], assume_unique=False, return_indices=True)
            common[i, j] = len(ii)
            if len(ii) == 0:
                continue
            same_p = data[i]["pos"][ii] == data[j]["pos"][jj]
            same_n = data[i]["neg"][ii] == data[j]["neg"][jj]
            pair[i, j] = float(np.mean(same_p & same_n))
            hemi[i, j] = float((np.sum(same_p) + np.sum(same_n)) / (2 * len(ii)))

    for m, name, title in ((pair, "seed_overlap_matrix.png", "events of seed i with the same two replacement hemispheres in seed j [%]"),
                           (hemi, "seed_overlap_hemispheres.png", "hemispheres of seed i with the same replacement in seed j [%]")):
        fig, ax = plt.subplots(figsize=(7.5, 6.3))
        im = ax.imshow(m * 100, cmap="viridis", origin="lower")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xlabel("seed j"); ax.set_ylabel("seed i"); ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax)
        save(fig, name)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(n), sizes / 1e6, color="#FFDF7F", edgecolor="k")
    ax.set_xlabel("seed"); ax.set_ylabel("mixed 4b events [M]"); ax.set_title("sample sizes")
    save(fig, "seed_sizes.png")

    k = int(max(d["rank"].max() for d in data)) + 1
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s_, d in enumerate(data[:4]):
        ax.stairs(np.bincount(d["rank"], minlength=k) / len(d["rank"]), np.arange(k + 1) - 0.5, label=f"seed {s_}")
    ax.set_xlabel("chosen rank"); ax.set_ylabel("fraction of hemispheres"); ax.legend()
    save(fig, "seed_rank.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    hi = float(np.nanpercentile(np.concatenate([d["dist"] for d in data[:4]]), 99.5))
    for s_, d in enumerate(data[:4]):
        ax.hist(d["dist"], bins=80, range=(0, hi), histtype="step", density=True, label=f"seed {s_}")
    ax.set_xlabel("hemisphere match distance"); ax.set_ylabel("normalised"); ax.legend()
    save(fig, "seed_matchDist.png")

    with open(os.path.join(outdir, "seed_overlap_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i\\j"] + list(range(n)))
        for i in range(n):
            w.writerow([i] + ["" if np.isnan(x) else round(float(x), 5) for x in pair[i]])
    self_match = [d["self_match"] for d in data]
    summary = {
        "n_seeds": n,
        "years": list(years),
        "seed_events": [int(x) for x in sizes],
        "self_match_events": self_match,
        "mean_pair_overlap": round(float(np.nanmean(pair)), 5),
        "max_pair_overlap": round(float(np.nanmax(pair)), 5),
        "mean_hemisphere_overlap": round(float(np.nanmean(hemi)), 4),
        "mean_common_source_events_fraction": round(float(np.mean(common[~np.eye(n, dtype=bool)] / np.repeat(sizes, n - 1))), 4),
        "mean_rank": [round(float(d["rank"].mean()), 3) for d in data],
        "mean_match_dist": [round(float(np.mean(d["dist"])), 4) for d in data],
    }
    with open(os.path.join(outdir, "summary.yml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    made += ["seed_overlap_matrix.csv", "summary.yml"]
    _seeds_index(outdir, made, summary)
    print(yaml.safe_dump(summary, sort_keys=False))
    if any(self_match):
        raise SystemExit(f"self-matched hemispheres found {self_match}: the source-event veto did not work")


def _seeds_index(outdir, files, summary):
    pngs = [f for f in files if f.endswith(".png")]
    rows = "".join(f"<tr><td>v{i}</td><td>{n:,}</td><td>{r}</td><td>{d}</td><td>{sm}</td></tr>"
                   for i, (n, r, d, sm) in enumerate(zip(summary["seed_events"], summary["mean_rank"],
                                                        summary["mean_match_dist"], summary["self_match_events"])))
    facts = [("seeds", summary["n_seeds"]), ("years", ", ".join(summary["years"])),
             ("identical mixed event in two seeds (mean / max)",
              f"{100 * summary['mean_pair_overlap']:.2f}% / {100 * summary['max_pair_overlap']:.2f}%"),
             ("same replacement hemisphere in two seeds (mean)", f"{100 * summary['mean_hemisphere_overlap']:.1f}%"),
             ("source events common to two seeds (mean)", f"{100 * summary['mean_common_source_events_fraction']:.1f}%")]
    fact_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in facts)
    figs = "".join(f"<figure><a href='{p}'><img src='{p}' loading='lazy'></a>"
                   f"<figcaption>{html.escape(p[:-4])}</figcaption></figure>" for p in pngs)
    style = ("<style>body{font-family:sans-serif;margin:1.5em;max-width:1400px}"
             "table{border-collapse:collapse;margin:0.5em 2em 1em 0;display:inline-table;vertical-align:top}"
             "td,th{border:1px solid #ccc;padding:2px 10px;text-align:right}th{background:#eee;text-align:left}"
             ".grid{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:420px}"
             "figure img{width:100%;border:1px solid #ddd}figcaption{font-size:12px;color:#555}</style>")
    page = (f"<html><head><meta charset='utf-8'><title>4b-mixing seeds</title>{style}</head><body>"
            f"<h1>4b mixing: seed study</h1>"
            f"<p>Each seed draws every hemisphere's replacement uniformly from its nearest library "
            f"neighbours (own event vetoed). Files: <a href='summary.yml'>summary.yml</a> · "
            f"<a href='seed_overlap_matrix.csv'>seed_overlap_matrix.csv</a></p>"
            f"<table>{fact_rows}</table>"
            f"<table><tr><th>seed</th><th>events</th><th>mean rank</th><th>mean match dist</th>"
            f"<th>self-matches</th></tr>{rows}</table>"
            f"<div class='grid'>{figs}</div></body></html>")
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["study", "seeds"])
    ap.add_argument("input", nargs="+", help="study: STUDY.coffea OUTDIR; seeds: the per-seed registries")
    ap.add_argument("--outdir", default=None, help="seeds: output directory")
    ap.add_argument("--years", nargs="+", default=None, help="seeds: data years to read (default: all in the registries)")
    ap.add_argument("--n-subsamples", type=int, default=16)
    a = ap.parse_args()
    if a.mode == "study":
        if len(a.input) != 2:
            ap.error("study takes STUDY.coffea OUTDIR")
        study(a.input[0], a.input[1], a.n_subsamples)
    else:
        if not a.outdir:
            ap.error("seeds needs --outdir")
        years = a.years or ["2022_preEE", "2022_EE", "2023_preBPix", "2023_BPix",
                            "UL16_preVFP", "UL16_postVFP", "UL17", "UL18"]
        seeds(a.input, a.outdir, years)


if __name__ == "__main__":
    main()
