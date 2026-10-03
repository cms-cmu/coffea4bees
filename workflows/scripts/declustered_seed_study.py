"""D.5 seed study of a multi-seed library declustering: how independent are the seeds, and what
are n_seeds replicas worth? (the declustered analogue of mixeddata_validation_report.py seeds)

    declustered_seed_study.py REG_seed0.yml ... REG_seedN-1.yml --outdir OUTDIR [--years ...]
        [--cutflow D4/cutflow_declustered.yml] [--hists D4/histAll_declustered.coffea]
        [--process-prefix syn] [--vars ...]

Draws (per-seed picoAODs listed in D3's per-seed registries; only run, luminosityBlock, event,
Jet_pt and Jet_lib_index are read). Jet_lib_index is the library row the jet's splitting was
drawn from (-1: never declustered); an event's draws are its distinct rows. Per seed pair (i, j),
over the source events both seeds kept:
  draw overlap        sum |draws_i & draws_j| / sum |draws_i|: how often two seeds pick the same
                      splitting for the same event (random among the k nearest: ~1/k, plus the
                      targets with a single candidate within the distance cap)
  identical events    fraction of seed i's events whose jets are identical in seed j
  common events       fraction of seed i's events that seed j also kept
Per seed, from the registries' library_lookups: the fraction of lookups with a single candidate
(drawn identically by every seed) and the last-resort same-event matches. And the reuse of
library rows over all seeds.

Effective number of replicas. A seed's count in a bin is N_s = sum_e 1[event e lands in the bin]
with probability p_e over the declustering; the source events are Poisson. Then
Var_total(N_s) = <N> (data-like), the seed-to-seed variance (same source events) is
B = sum p_e (1 - p_e), and the part all seeds share is A = sum p_e^2 = <N> - B. With correlation
rho = A / <N> = 1 - Var_seeds / <N>, the mean of n seeds has the precision of
N_eff = n / (1 + (n - 1) rho) independent replicas (1: every event lands in the bin whatever the
draw, n: as if independent). Estimated per cut x year from the D.4 cutflow (counts4_unit, the
<prefix>_v<s>_<year> rows) and per bin (SR / SB, fourTag, summed over years) of --vars from the
D.4 histograms. Var_seeds is an estimate from n seeds (relative error ~ sqrt(2 / (n - 1))).
"""
import argparse
import csv
import html
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

BRANCHES = ["run", "luminosityBlock", "event", "Jet_pt", "Jet_lib_index"]
DEFAULT_VARS = ["quadJet_selected.lead.mass", "quadJet_selected.subl.mass", "quadJet_selected.xHH",
                "canJet0.pt", "canJet3.pt"]


def _registry_files(path, years):
    """{year: [files]} and the summed library_lookups of one per-seed registry."""
    from src.tools.make_dataset_yml import parse_dataset_key
    with open(path) as f:
        reg = yaml.safe_load(f) or {}
    files, lookups = {}, {}
    for key, entry in reg.items():
        year, _ = parse_dataset_key(key)
        entry = entry or {}
        for k, v in (entry.get("library_lookups") or {}).items():
            lookups[k] = lookups.get(k, 0) + int(v)
        if year is None or (years and year not in years):
            continue
        files.setdefault(year, []).extend(entry.get("files") or [])
    return files, lookups


def _read(files):
    import awkward as ak
    import uproot
    parts = [b for b in uproot.iterate({f: "Events" for f in files}, BRANCHES, library="ak", step_size="200 MB")]
    ev = ak.concatenate(parts) if parts else None
    if ev is None or len(ev) == 0:
        raise SystemExit(f"no events in {files[:2]}...")
    if "Jet_lib_index" not in ev.fields:
        raise SystemExit(f"{files[0]}: no Jet_lib_index branch (declustered before the seed study existed)")
    key = np.asarray(ev.run, dtype=np.int64) * (1 << 40) + np.asarray(ev.event, dtype=np.int64)
    order = np.argsort(key, kind="stable")
    key = key[order]
    if np.any(key[1:] == key[:-1]):
        raise SystemExit(f"duplicate (run, event) among {files[:2]}...")
    pt = ev.Jet_pt[order]
    lib = ev.Jet_lib_index[order]
    s = ak.sort(lib[lib >= 0], axis=1)
    first = ak.concatenate([ak.values_astype(ak.ones_like(s[:, :1]), bool), s[:, 1:] != s[:, :-1]], axis=1)
    return {"key": key, "pt": pt, "draws": s[first]}


def _pair(a, b):
    import awkward as ak
    common, ia, ib = np.intersect1d(a["key"], b["key"], assume_unique=True, return_indices=True)
    pa, pb = a["pt"][ia], b["pt"][ib]
    same_n = np.asarray(ak.num(pa) == ak.num(pb))
    n_ident = int(np.sum(ak.all(pa[same_n] == pb[same_n], axis=1)))
    da, db = a["draws"][ia], b["draws"][ib]
    pairs = ak.cartesian([da, db], nested=False)
    n_same = int(ak.sum(pairs["0"] == pairs["1"]))
    return {"common": len(common), "identical": n_ident, "same_draws": n_same,
            "draws_a": int(ak.sum(ak.num(da))), "draws_b": int(ak.sum(ak.num(db)))}


def draws_study(registries, years, outdir):
    import awkward as ak
    n = len(registries)
    per_seed = [_registry_files(r, years) for r in registries]
    years = years or sorted(set().union(*(set(f) for f, _ in per_seed)))
    tot = {k: np.zeros((n, n)) for k in ("common", "identical", "same_draws", "draws", "events")}
    reuse = {}
    for year in years:
        print(f"seed study: reading {year} for {n} seeds")
        seeds = []
        for s, (files, _) in enumerate(per_seed):
            if not files.get(year):
                raise SystemExit(f"{registries[s]}: no files for {year}")
            seeds.append(_read(files[year]))
        rows = np.concatenate([ak.to_numpy(ak.flatten(d["draws"])) for d in seeds]).astype(np.int64)
        reuse[year] = np.bincount(np.bincount(rows)) if len(rows) else np.zeros(1, dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                r = _pair(seeds[i], seeds[j])          # symmetric but for the normalisations
                for a_, b_, own in ((i, j, "draws_a"), (j, i, "draws_b")):
                    for k in ("common", "identical", "same_draws"):
                        tot[k][a_, b_] += r[k]
                    tot["draws"][a_, b_] += r[own]
                    tot["events"][a_, b_] += len(seeds[a_]["key"])
        del seeds
    with np.errstate(invalid="ignore", divide="ignore"):
        m = {"draw_overlap": tot["same_draws"] / tot["draws"],
             "identical_events": tot["identical"] / tot["events"],
             "common_events": tot["common"] / tot["events"]}
    for k in m:
        np.fill_diagonal(m[k], np.nan)

    lookups = [lk for _, lk in per_seed]
    n_lookups = [sum(lk.get(l, 0) for l in ("exact", "child_content", "parent_content", "coarse")) for lk in lookups]
    single = [lk.get("single_candidate", 0) / nl if nl else float("nan") for lk, nl in zip(lookups, n_lookups)]
    self_match = [lk.get("self_match", 0) for lk in lookups]
    return m, reuse, {"years": list(years), "single_candidate_fraction": single, "self_matches": self_match,
                      "lookups": lookups}


def _rho(x):
    """x: (n_seeds, ...) counts -> mean, rho = 1 - Var_seeds/mean (clipped to [0, 1]), N_eff."""
    n = x.shape[0]
    mean = x.mean(axis=0)
    var = x.var(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.clip(1 - var / mean, 0, 1)
    return mean, rho, n / (1 + (n - 1) * rho)


def neff_cutflow(path, prefix, n):
    with open(path) as f:
        counts = (yaml.safe_load(f) or {}).get("counts4_unit") or {}
    pat = re.compile(rf"^{re.escape(prefix)}_v(\d+)_(.+)$")
    table = {}
    for name, cuts in counts.items():
        mt = pat.match(name)
        if not mt:
            continue
        s, year = int(mt.group(1)), mt.group(2)
        for cut, v in cuts.items():
            table.setdefault((year, cut), {})[s] = float(v)
    rows = []
    for (year, cut), by_seed in table.items():
        if sorted(by_seed) != list(range(n)):
            raise SystemExit(f"{path}: {prefix} seeds {sorted(by_seed)} for {year} {cut}, expected 0..{n - 1}")
        mean, rho, ne = _rho(np.array([by_seed[s] for s in range(n)]))
        rows.append({"year": year, "cut": cut, "mean": float(mean), "rho": float(rho), "n_eff": float(ne)})
    totals = {}
    for (year, cut), by_seed in table.items():
        t = totals.setdefault(cut, np.zeros(n))
        t += np.array([by_seed[s] for s in range(n)])
    for cut, t in totals.items():
        mean, rho, ne = _rho(t)
        rows.append({"year": "all", "cut": cut, "mean": float(mean), "rho": float(rho), "n_eff": float(ne)})
    return rows


def neff_hists(path, prefix, n, variables):
    from coffea.util import load
    hists = load(path)["hists"]
    out = {}
    for var in variables:
        if var not in hists:
            print(f"seed study: {var} not in {path}, skipped")
            continue
        h = hists[var]
        procs = list(h.axes["process"])
        names = [f"{prefix}_v{s}" for s in range(n)]
        if not all(p in procs for p in names):
            raise SystemExit(f"{path} {var}: missing {[p for p in names if p not in procs]}")
        for region in ("SR", "SB"):
            x = np.array([h[{"process": p, "tag": "fourTag", "region": region, "year": sum}].values()
                          for p in names], dtype=float)
            mean, rho, ne = _rho(x)
            out[(var, region)] = {"edges": h.axes[-1].edges, "mean": mean, "rho": rho, "n_eff": ne}
    return out


def _matrix_plot(m, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(100 * m, cmap="viridis")
    fig.colorbar(im, ax=ax, label="%")
    ticks = range(0, len(m), max(1, len(m) // 8))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel("seed j"); ax.set_ylabel("seed i"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def write_outputs(outdir, n, draws, cutflow_rows, hist_neff):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    made, summary = [], {"n_seeds": n}
    if draws:
        m, reuse, info = draws
        titles = {"draw_overlap": "draws of seed i also drawn by seed j for the same event [%]",
                  "identical_events": "events of seed i identical in seed j [%]",
                  "common_events": "events of seed i also kept by seed j [%]"}
        for k, t in titles.items():
            _matrix_plot(m[k], t, os.path.join(outdir, f"seed_{k}.png"))
            made.append(f"seed_{k}.png")
            summary[f"mean_{k}"] = round(float(np.nanmean(m[k])), 6)
            summary[f"max_{k}"] = round(float(np.nanmax(m[k])), 6)
        with open(os.path.join(outdir, "seed_overlap_matrix.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed_i", "seed_j", *titles])
            for i in range(n):
                for j in range(n):
                    if i != j:
                        w.writerow([i, j, *(f"{m[k][i, j]:.6f}" for k in titles)])
        made.append("seed_overlap_matrix.csv")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for year, counts in reuse.items():
            uses = np.arange(len(counts))
            ax.step(uses[1:], counts[1:] / max(counts[1:].sum(), 1), where="mid", label=year)
        ax.set_yscale("log"); ax.set_xlabel(f"draws of a library row (all {n} seeds)")
        ax.set_ylabel("fraction of rows drawn"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "library_reuse.png"), dpi=120); plt.close(fig)
        made.append("library_reuse.png")
        summary.update({"years": info["years"],
                        "single_candidate_fraction": [round(x, 5) for x in info["single_candidate_fraction"]],
                        "self_matches": info["self_matches"]})
    if cutflow_rows:
        summary["n_eff_cutflow"] = {f"{r['year']} {r['cut']}": {"mean": round(r["mean"], 1), "rho": round(r["rho"], 4),
                                                              "n_eff": round(r["n_eff"], 2)} for r in cutflow_rows}
    if hist_neff:
        fig, axes = plt.subplots(len(hist_neff), 1, figsize=(7, 2.6 * len(hist_neff)), squeeze=False)
        summary["n_eff_hists"] = {}
        for ax, ((var, region), r) in zip(axes[:, 0], hist_neff.items()):
            ok = r["mean"] >= 100          # bins with a usable variance estimate
            ax.stairs(np.where(ok, r["n_eff"], np.nan), r["edges"])
            ax.axhline(n, ls=":", c="k"); ax.set_ylim(0, n * 1.05)
            ax.set_ylabel("N_eff"); ax.set_title(f"{var} {region} (bins with >= 100 events per seed)", fontsize=9)
            w = r["mean"][ok]
            rho_w = float(np.sum(w * r["rho"][ok]) / np.sum(w)) if np.sum(w) else float("nan")
            summary["n_eff_hists"][f"{var} {region}"] = {"rho_weighted": round(rho_w, 4),
                                                         "n_eff_at_weighted_rho": round(n / (1 + (n - 1) * rho_w), 2)}
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "n_eff_bins.png"), dpi=110); plt.close(fig)
        made.append("n_eff_bins.png")
    with open(os.path.join(outdir, "summary.yml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    made.append("summary.yml")
    _index(outdir, made, summary, cutflow_rows)


def _index(outdir, made, summary, cutflow_rows):
    n = summary["n_seeds"]
    pct = lambda k: f"{100 * summary[f'mean_{k}']:.2f}% / {100 * summary[f'max_{k}']:.2f}%" if f"mean_{k}" in summary else "-"
    facts = [("seeds", n), ("years", ", ".join(summary.get("years", []))),
             ("same draw for the same event in two seeds (mean / max)", pct("draw_overlap")),
             ("identical event in two seeds (mean / max)", pct("identical_events")),
             ("source events common to two seeds (mean / max)", pct("common_events"))]
    if "single_candidate_fraction" in summary:
        facts.append(("lookups with a single candidate (per seed, mean)",
                      f"{100 * np.nanmean(summary['single_candidate_fraction']):.2f}%"))
        facts.append(("same-event fallback matches (all seeds)", sum(summary["self_matches"])))
    rows = "".join(f"<tr><td>{html.escape(str(a))}</td><td>{html.escape(str(b))}</td></tr>" for a, b in facts)
    cf = ""
    if cutflow_rows:
        cf = ("<h3>Effective replicas per cut (D.4 cutflow, four-tag, unweighted)</h3><table><tr><th>year</th><th>cut</th>"
              "<th>events / seed</th><th>&rho;</th><th>N<sub>eff</sub></th></tr>"
              + "".join(f"<tr><td>{r['year']}</td><td>{r['cut']}</td><td>{r['mean']:,.0f}</td><td>{r['rho']:.3f}</td>"
                        f"<td>{r['n_eff']:.1f}</td></tr>" for r in cutflow_rows) + "</table>")
    imgs = "".join(f"<p><img src='{f}' style='max-width:900px'></p>" for f in made if f.endswith(".png"))
    style = "<style>body{font-family:sans-serif;margin:2em}td,th{padding:2px 10px;border-bottom:1px solid #ddd}</style>"
    page = (f"<html><head><meta charset='utf-8'><title>declustering seeds</title>{style}</head><body>"
            f"<h2>Library declustering: {n} seeds</h2>"
            f"<p>Each seed draws every jet's splitting at random from its nearest library neighbours. "
            f"&rho; = 1 &minus; Var<sub>seeds</sub>/&lang;N&rang; is the part of a bin's content all seeds "
            f"share; the {n} seeds together are worth N<sub>eff</sub> = {n}/(1+{n - 1}&rho;) independent replicas. "
            f"<a href='summary.yml'>summary.yml</a> <a href='seed_overlap_matrix.csv'>seed_overlap_matrix.csv</a></p>"
            f"<table>{rows}</table>{cf}{imgs}</body></html>")
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("registries", nargs="+", help="per-seed registries, seed 0 first")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--years", nargs="*", default=None, help="years whose picoAODs are read (default: all)")
    ap.add_argument("--cutflow", default=None, help="D.4 cutflow_declustered.yml")
    ap.add_argument("--hists", default=None, help="D.4 histAll_declustered.coffea")
    ap.add_argument("--process-prefix", default="syn", help="sample prefix of the seeds (<prefix>_v<s>)")
    ap.add_argument("--vars", nargs="*", default=DEFAULT_VARS)
    ap.add_argument("--no-draws", action="store_true", help="skip reading the picoAODs")
    a = ap.parse_args()
    n = len(a.registries)
    if n < 2:
        raise SystemExit("the seed study needs at least 2 seeds")
    draws = None if a.no_draws else draws_study(a.registries, a.years, a.outdir)
    cutflow_rows = neff_cutflow(a.cutflow, a.process_prefix, n) if a.cutflow else []
    hist_neff = neff_hists(a.hists, a.process_prefix, n, a.vars) if a.hists else {}
    write_outputs(a.outdir, n, draws, cutflow_rows, hist_neff)
    print(f"seed study written to {a.outdir}")


if __name__ == "__main__":
    main()
