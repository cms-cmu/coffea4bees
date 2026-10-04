"""Poisson bootstrap of the SvB distribution of a multi-seed declustered (or mixed) sample:
what is the n-seed average worth in statistics? (after jet_clustering/bootstrap_correlation.py)

    svb_bootstrap.py DUMP_year1.coffea [DUMP_year2.coffea ...] --outdir OUTDIR
        [--prefix syn] [--toys 30] [--bins 40] [--rng-seed 1]

Input: processor_HH4b outputs made with dump_SvB_in_SR: true -- processOutput["SvB_in_SR"][dataset] =
{run, luminosityBlock, event, SvB_MA_ps} of the four-tag SR events, dataset <prefix>_v<s>_<year...>.

Every INPUT event (run, luminosityBlock, event) -- the same in all seeds, it is the 4b data event
that was declustered -- gets one Poisson(1) weight per toy, shared by all its declustered versions.
The library draws are held fixed. Per toy, the n-seed MEAN SvB histogram is filled with those weights;
the toy-to-toy variance per bin, around the unweighted mean, is the statistical variance of the
n-seed template: sum_e c_e^2 for c_e = (seeds in which event e lands in the bin) / n, which is the
input data's own fluctuation plus the declustering noise divided by n.

    N_eff(bin) = <N> / Var_boot    (<N> = one replica's Poisson variance)

the number of independent replicas the n-seed mean is worth: 1 if every seed puts each event in the
same bin, n if the seeds were independent. Pooled: sum <N> / sum Var_boot. Cross-checks: the same
bootstrap of ONE seed (should give N_eff ~ 1) and the seed-to-seed spread around the mean.
"""
import argparse
import html
import os
import re

import numpy as np
import yaml


def load(paths, prefix):
    """{seed: (run, lumi, event, svb)} concatenated over the dumps' datasets <prefix>_v<s>_*."""
    from coffea.util import load as cload
    pat = re.compile(rf"^{re.escape(prefix)}_v(\d+)_")
    per_seed = {}
    for p in paths:
        out = cload(p)
        dump = out.get("SvB_in_SR") if isinstance(out, dict) else None
        if not dump:
            raise SystemExit(f"{p}: no SvB_in_SR (run the processor with dump_SvB_in_SR: true)")
        for ds, cols in dump.items():
            m = pat.match(ds)
            if not m:
                continue
            s = int(m.group(1))
            acc = per_seed.setdefault(s, {k: [] for k in ("run", "luminosityBlock", "event", "SvB_MA_ps")})
            for k in acc:
                acc[k].extend(cols[k])
    if not per_seed:
        raise SystemExit(f"no {prefix}_v<s>_* datasets in {paths}")
    seeds = sorted(per_seed)
    if seeds != list(range(len(seeds))):
        raise SystemExit(f"seeds {seeds}: expected 0..n-1")
    return {s: {k: np.asarray(v) for k, v in per_seed[s].items()} for s in seeds}


def bootstrap(data, edges, n_toys, rng):
    n = len(data)
    run = np.concatenate([data[s]["run"] for s in data]).astype(np.int64)
    lumi = np.concatenate([data[s]["luminosityBlock"] for s in data]).astype(np.int64)
    evt = np.concatenate([data[s]["event"] for s in data]).astype(np.int64)
    svb = np.concatenate([data[s]["SvB_MA_ps"] for s in data]).astype(np.float64)
    seed = np.concatenate([np.full(len(data[s]["run"]), s) for s in data])
    ids = np.stack([run, lumi, evt], axis=1)
    _, inv = np.unique(ids, axis=0, return_inverse=True)
    inv = inv.ravel()
    n_inputs = inv.max() + 1
    nb = len(edges) - 1
    b = np.clip(np.digitize(svb, edges) - 1, 0, nb - 1)

    per_seed = np.stack([np.bincount(b[seed == s], minlength=nb) for s in range(n)]).astype(float)
    mean = per_seed.mean(axis=0)                     # the n-seed template (unweighted)
    W = rng.poisson(1.0, size=(n_inputs, n_toys)).astype(float)
    toys = np.stack([np.bincount(b, weights=W[inv, t], minlength=nb) for t in range(n_toys)]) / n
    var_boot = np.mean((toys - mean) ** 2, axis=0)

    # one seed alone, same input weights: should be ~ Poisson (N_eff ~ 1)
    m0 = seed == 0
    toys0 = np.stack([np.bincount(b[m0], weights=W[inv[m0], t], minlength=nb) for t in range(n_toys)])
    var_boot0 = np.mean((toys0 - per_seed[0]) ** 2, axis=0)
    seed_spread = per_seed.var(axis=0, ddof=1)       # declustering noise of ONE seed (data fixed)
    return {"n_seeds": n, "n_inputs": int(n_inputs), "n_entries": int(len(svb)), "per_seed": per_seed,
            "mean": mean, "var_boot": var_boot, "var_boot0": var_boot0, "seed0": per_seed[0],
            "seed_spread": seed_spread}


def summarize(r, edges, min_count):
    with np.errstate(invalid="ignore", divide="ignore"):
        neff = r["mean"] / r["var_boot"]
        neff0 = r["seed0"] / r["var_boot0"]
    ok = r["mean"] >= min_count
    pooled = float(r["mean"][ok].sum() / r["var_boot"][ok].sum())
    pooled0 = float(r["seed0"][ok].sum() / r["var_boot0"][ok].sum())
    # the shared part: Var_boot = A + B/n with B = seed spread -> A, rho = A / <N>
    A = r["var_boot"] - r["seed_spread"] / r["n_seeds"]
    rho_pooled = float(A[ok].sum() / r["mean"][ok].sum())
    return neff, neff0, ok, {
        "n_seeds": r["n_seeds"], "toys": None, "input_events": r["n_inputs"], "sr_entries_all_seeds": r["n_entries"],
        "events_per_seed": float(r["mean"].sum()),
        "n_eff_pooled": round(pooled, 3),
        "n_eff_one_seed_pooled (expect ~1)": round(pooled0, 3),
        "rho_shared_pooled": round(rho_pooled, 4),
        "bins_used": int(ok.sum()), "min_count_per_bin": min_count,
        "per_bin": [{"lo": float(edges[i]), "hi": float(edges[i + 1]), "mean": round(float(r["mean"][i]), 2),
                     "sigma_boot": round(float(np.sqrt(r["var_boot"][i])), 3),
                     "sigma_naive_indep": round(float(np.sqrt(r["mean"][i] / r["n_seeds"])), 3),
                     "sigma_one_replica": round(float(np.sqrt(r["mean"][i])), 3),
                     "n_eff": None if not np.isfinite(neff[i]) else round(float(neff[i]), 2)}
                    for i in range(len(edges) - 1)],
    }


def plot(r, edges, neff, neff0, ok, outdir, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = 0.5 * (edges[1:] + edges[:-1])
    n = r["n_seeds"]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.4, 1.4]})
    a1.stairs(r["mean"], edges, color="k", label=f"mean of {n} seeds")
    a1.stairs(r["seed0"], edges, color="tab:orange", alpha=0.7, label="seed 0")
    a1.errorbar(c, r["mean"], yerr=np.sqrt(r["var_boot"]), fmt="none", ecolor="tab:blue", capsize=2,
                label=r"$\sigma$ bootstrap (input events)")
    a1.set_yscale("log"); a1.set_ylabel("SR four-tag events"); a1.legend(fontsize=8); a1.set_title(title, fontsize=9)
    with np.errstate(invalid="ignore", divide="ignore"):
        a2.plot(c, np.sqrt(r["var_boot"] / r["mean"]), "o", ms=3, label=r"$\sigma_{boot}$ / $\sqrt{N}$ (one replica)")
        a2.plot(c, np.sqrt(r["var_boot"] / (r["mean"] / n)), "s", ms=3, label=rf"$\sigma_{{boot}}$ / $\sqrt{{N/{n}}}$ (independent seeds)")
        a2.plot(c, np.sqrt(r["seed_spread"] / r["mean"]), "^", ms=3, label=r"seed spread / $\sqrt{N}$")
    a2.axhline(1, ls=":", c="k"); a2.set_ylabel("ratio"); a2.legend(fontsize=7)
    a3.plot(c[ok], neff[ok], "o", ms=3, label=f"N_eff, mean of {n} seeds")
    a3.plot(c[ok], neff0[ok], "x", ms=4, label="N_eff, one seed (expect 1)")
    a3.axhline(n, ls=":", c="k"); a3.axhline(1, ls=":", c="grey")
    a3.set_ylim(0, n * 1.1); a3.set_ylabel("N_eff"); a3.set_xlabel("SvB_MA ps"); a3.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "svb_bootstrap.png"), dpi=120); plt.close(fig)


def plot_cms(values, sig_boot, sig_naive, edges, path, run_label, region="SR", xlabel="SvB",
             ratio_range=(0.9, 1.1)):
    """John's bootstrap figure: the average with the bootstrap RMS as the yellow band (stack + ratio),
    the naive (independent-samples) uncertainty as black points; ratio to the average."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import mplhep as hep
        plt.style.use(hep.style.CMS)
    except ImportError:
        hep = None
    yellow, c = "#F9DF8E", 0.5 * (edges[1:] + edges[:-1])
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax.stairs(values, edges, fill=True, color=yellow, label="Average (Bootstrap Uncertainties)")
    ax.stairs(values, edges, color="k", lw=1.5)
    ax.errorbar(c, values, yerr=sig_naive, fmt="o", color="k", ms=6, label="Average (Naive Uncertainties)")
    ax.set_yscale("log"); ax.legend(loc="upper right", fontsize=18)
    ax.set_title(region, fontsize=24)
    if hep is not None:
        hep.cms.label("Internal", data=True, ax=ax, rlabel=f"{run_label} (13 TeV)" if run_label == "RunII"
                      else f"{run_label} (13.6 TeV)", loc=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rb, rn = sig_boot / values, sig_naive / values
    ok = values > 0
    rx.bar(c[ok], 2 * rb[ok], bottom=1 - rb[ok], width=np.diff(edges)[ok], color=yellow, alpha=0.9, lw=0)
    rx.errorbar(c[ok], np.ones(ok.sum()), yerr=rn[ok], fmt="o", color="k", ms=4)
    rx.axhline(1, ls="--", c="k", lw=2)
    rx.set_ylim(*ratio_range); rx.set_ylabel("Ratio"); rx.set_xlabel(xlabel)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="syn")
    ap.add_argument("--toys", type=int, default=30)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--min-count", type=float, default=20, help="bins with fewer mean events are left out of the pooled N_eff")
    ap.add_argument("--rng-seed", type=int, default=1)
    ap.add_argument("--title", default="SvB Poisson bootstrap")
    ap.add_argument("--run-label", default=None, help="RunII / Run3 for the CMS label (default: from the dataset years)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    data = load(a.dumps, a.prefix)
    if a.run_label is None:
        a.run_label = "Run3" if any("202" in os.path.basename(p) for p in a.dumps) else "RunII"
    edges = np.linspace(0, 1, a.bins + 1)
    r = bootstrap(data, edges, a.toys, np.random.default_rng(a.rng_seed))
    neff, neff0, ok, summary = summarize(r, edges, a.min_count)
    summary["toys"] = a.toys
    plot(r, edges, neff, neff0, ok, a.outdir, a.title)
    n = r["n_seeds"]
    # test: ONE sample used n times -- the same events n times, so the bootstrap RMS is that sample's
    # full sqrt(N) while the naive independent-samples error is sqrt(n N)/n = sqrt(N/n)
    plot_cms(r["seed0"], np.sqrt(r["var_boot0"]), np.sqrt(n * r["seed0"]) / n, edges,
             os.path.join(a.outdir, "svb_bootstrap_one_sample_x%d" % n), a.run_label)
    # the real n-seed average: bootstrap RMS vs naive sqrt(sum N)/n
    plot_cms(r["mean"], np.sqrt(r["var_boot"]), np.sqrt(n * r["mean"]) / n, edges,
             os.path.join(a.outdir, "svb_bootstrap_average"), a.run_label)
    with open(os.path.join(a.outdir, "summary.yml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
                   for k, v in summary.items() if k != "per_bin")
    with open(os.path.join(a.outdir, "index.html"), "w") as f:
        f.write(f"<html><head><meta charset='utf-8'><title>SvB bootstrap</title><style>body{{font-family:sans-serif;"
                f"margin:2em}}td{{padding:2px 10px;border-bottom:1px solid #ddd}}</style></head><body>"
                f"<h2>{html.escape(a.title)}</h2><p>Each input 4b event gets one Poisson(1) weight per toy, shared by all "
                f"its declustered versions; the toy-to-toy variance of the {r['n_seeds']}-seed mean SvB histogram (SR, "
                f"four-tag) is its statistical variance. N_eff = &lang;N&rang;/Var. <a href='summary.yml'>summary.yml</a>"
                f"</p><table>{rows}</table>"
                f"<h3>One sample used {r['n_seeds']} times (test)</h3><p><img src='svb_bootstrap_one_sample_x{r['n_seeds']}.png' style='max-width:700px'></p>"
                f"<h3>Average of the {r['n_seeds']} declustered seeds</h3><p><img src='svb_bootstrap_average.png' style='max-width:700px'></p>"
                f"<h3>Diagnostics</h3><p><img src='svb_bootstrap.png' style='max-width:800px'></p></body></html>")
    print(f"svb bootstrap: N_eff pooled {summary['n_eff_pooled']} (one seed {summary['n_eff_one_seed_pooled (expect ~1)']}), "
          f"written to {a.outdir}")


if __name__ == "__main__":
    main()
